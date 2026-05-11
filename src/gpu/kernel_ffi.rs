// CUDA runtime FFI via cudarc 0.19.
//
// Adaptation notes (cudarc 0.19 vs the plan's assumed 0.12 API):
//
// • API renamed: CudaDevice → CudaContext; no device.load_ptx/get_func.
// • `CudaContext::load_module` is gated on the `nvrtc` feature — we enable it.
//   The NVRTC compiler itself is only dlopen'd at runtime, so compile-time
//   success does not require nvcc on the build host.
// • `cuda-12030` feature selects the CUDA 12.3 ABI (CI host must have CUDA ≥12.3).
// • With `fallback-dynamic-loading` the CUDA driver is also dlopen'd at runtime,
//   meaning `cargo check --features cuda-runtime` succeeds without a CUDA install.
// • `CudaModule::cu_module` is `pub(crate)`, so we cannot touch it directly.
//   We call `module.get_global(name, &stream)` for symbol access and use
//   `stream.memcpy_htod` / `stream.memcpy_dtoh` for all device ↔ host transfers.
//   For double-buffer offset writes we use `CudaViewMut::try_slice_mut`.
// • `CudaFunction` is `Clone` in 0.19 — no `Mutex` needed.
// • The `grind` kernel takes no Rust-side arguments (all state lives in
//   `__device__` globals), so `launch_builder` is used with zero `.arg()` calls.
// • `CudaContext::default_stream()` is infallible in 0.19.

#![cfg(feature = "cuda-runtime")]

use crate::error::{MinerError, Result};
use crate::gpu::ptx::PTX;
use crate::gpu::Hit;
use alloy::primitives::{B256, U256};
use cudarc::driver::{
    safe::{CudaContext, CudaFunction, CudaModule, CudaStream},
    LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

// ─── Raw layout matching the CUDA kernel's HitRecord struct ──────────────────

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HitRaw {
    nonce: [u8; 32],
    hash: [u8; 32],
    epoch_id: u32,
    _pad: [u8; 4],
}

// ─── GpuRuntime ──────────────────────────────────────────────────────────────

pub struct GpuRuntime {
    pub context: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    grind_fn: CudaFunction,
    /// Host mirror of which double-buffer slot is live on the device.
    active_idx_host: u32,
}

impl GpuRuntime {
    /// Initialise context, load PTX, resolve kernel function.
    pub fn init(device_id: u32) -> Result<Self> {
        let context = CudaContext::new(device_id as usize)
            .map_err(|e| MinerError::Gpu(format!("CudaContext::new: {e:?}")))?;

        // PTX bytes are a UTF-8 null-terminated string produced by nvcc.
        let ptx_str = std::str::from_utf8(PTX)
            .map_err(|_| MinerError::Gpu("PTX is not valid UTF-8".into()))?;

        let module: Arc<CudaModule> = context
            .load_module(Ptx::from_src(ptx_str))
            .map_err(|e| MinerError::Gpu(format!("load_module: {e:?}")))?;

        let grind_fn: CudaFunction = module
            .load_function("grind")
            .map_err(|e| MinerError::Gpu(format!("load_function(grind): {e:?}")))?;

        let stream = context.default_stream();

        Ok(Self {
            context,
            stream,
            module,
            grind_fn,
            active_idx_host: 0,
        })
    }

    /// Swap to new mining parameters without restarting the kernel.
    ///
    /// Writes into the inactive double-buffer slot, then flips `d_active_idx`.
    pub fn hot_swap(&mut self, challenge: B256, target: U256, epoch_id: u32) -> Result<()> {
        let next = 1u32 - self.active_idx_host;

        // target → 4 × u64 big-endian (matches the kernel's u64[4] layout).
        let bytes = target.to_be_bytes::<32>();
        let target_words_bytes: [u8; 32] = unsafe {
            // SAFETY: target_words is plain [u64; 4] with no pointer/padding;
            // we transmute it to bytes for the device copy.
            let words: [u64; 4] = [
                u64::from_be_bytes(bytes[0..8].try_into().unwrap()),
                u64::from_be_bytes(bytes[8..16].try_into().unwrap()),
                u64::from_be_bytes(bytes[16..24].try_into().unwrap()),
                u64::from_be_bytes(bytes[24..32].try_into().unwrap()),
            ];
            std::mem::transmute(words)
        };
        let challenge_bytes: [u8; 32] = challenge.0;
        let epoch_bytes: [u8; 4] = epoch_id.to_ne_bytes();

        let slot_off_32 = (next as usize) * 32;
        let slot_off_4 = (next as usize) * 4;

        // c_challenge[next] = challenge
        {
            let mut sym = self
                .module
                .get_global("c_challenge", &self.stream)
                .map_err(|e| MinerError::Gpu(format!("get_global(c_challenge): {e:?}")))?;
            let mut slot = sym
                .try_slice_mut(slot_off_32..slot_off_32 + 32)
                .ok_or_else(|| MinerError::Gpu("c_challenge slice OOB".into()))?;
            self.stream
                .memcpy_htod(&challenge_bytes, &mut slot)
                .map_err(|e| MinerError::Gpu(format!("htod c_challenge: {e:?}")))?;
        }
        // c_target[next] = target (as bytes)
        {
            let mut sym = self
                .module
                .get_global("c_target", &self.stream)
                .map_err(|e| MinerError::Gpu(format!("get_global(c_target): {e:?}")))?;
            let mut slot = sym
                .try_slice_mut(slot_off_32..slot_off_32 + 32)
                .ok_or_else(|| MinerError::Gpu("c_target slice OOB".into()))?;
            self.stream
                .memcpy_htod(&target_words_bytes, &mut slot)
                .map_err(|e| MinerError::Gpu(format!("htod c_target: {e:?}")))?;
        }
        // c_epoch_id[next] = epoch_id
        {
            let mut sym = self
                .module
                .get_global("c_epoch_id", &self.stream)
                .map_err(|e| MinerError::Gpu(format!("get_global(c_epoch_id): {e:?}")))?;
            let mut slot = sym
                .try_slice_mut(slot_off_4..slot_off_4 + 4)
                .ok_or_else(|| MinerError::Gpu("c_epoch_id slice OOB".into()))?;
            self.stream
                .memcpy_htod(&epoch_bytes, &mut slot)
                .map_err(|e| MinerError::Gpu(format!("htod c_epoch_id: {e:?}")))?;
        }
        // flip active index
        {
            let mut sym = self
                .module
                .get_global("d_active_idx", &self.stream)
                .map_err(|e| MinerError::Gpu(format!("get_global(d_active_idx): {e:?}")))?;
            let next_bytes: [u8; 4] = next.to_ne_bytes();
            self.stream
                .memcpy_htod(&next_bytes, &mut sym)
                .map_err(|e| MinerError::Gpu(format!("htod d_active_idx: {e:?}")))?;
        }
        // reset hit counter (nonce counter is intentionally left alone)
        {
            let mut sym = self
                .module
                .get_global("d_hit_count", &self.stream)
                .map_err(|e| MinerError::Gpu(format!("get_global(d_hit_count): {e:?}")))?;
            let zero: [u8; 4] = 0u32.to_ne_bytes();
            self.stream
                .memcpy_htod(&zero, &mut sym)
                .map_err(|e| MinerError::Gpu(format!("htod d_hit_count: {e:?}")))?;
        }

        self.active_idx_host = next;
        Ok(())
    }

    /// Set the stop flag so the persistent kernel exits its polling loop.
    pub fn signal_stop(&self) -> Result<()> {
        let mut sym = self
            .module
            .get_global("d_should_stop", &self.stream)
            .map_err(|e| MinerError::Gpu(format!("get_global(d_should_stop): {e:?}")))?;
        let one: [u8; 4] = 1u32.to_ne_bytes();
        self.stream
            .memcpy_htod(&one, &mut sym)
            .map_err(|e| MinerError::Gpu(format!("htod d_should_stop: {e:?}")))
    }

    /// Launch the persistent `grind` kernel asynchronously.
    ///
    /// The kernel runs until `d_should_stop` is set to 1.
    pub fn launch_persistent(&self, blocks: u32, threads: u32) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.grind_fn)
                .launch(cfg)
                .map_err(|e| MinerError::Gpu(format!("launch grind: {e:?}")))?;
        }
        Ok(())
    }

    /// Drain hits that have accumulated in device memory since the last call.
    pub fn poll_hits(&mut self) -> Result<Vec<Hit>> {
        let count: u32 = {
            let sym = self
                .module
                .get_global("d_hit_count", &self.stream)
                .map_err(|e| MinerError::Gpu(format!("get_global(d_hit_count): {e:?}")))?;
            let mut buf = [0u8; 4];
            self.stream
                .memcpy_dtoh(&sym, &mut buf)
                .map_err(|e| MinerError::Gpu(format!("dtoh d_hit_count: {e:?}")))?;
            u32::from_ne_bytes(buf)
        };
        if count == 0 {
            return Ok(Vec::new());
        }

        let n = (count as usize).min(16);
        let byte_len = n * std::mem::size_of::<HitRaw>();

        let hits_bytes: Vec<u8> = {
            let sym = self
                .module
                .get_global("d_hits", &self.stream)
                .map_err(|e| MinerError::Gpu(format!("get_global(d_hits): {e:?}")))?;
            let view = sym
                .try_slice(..byte_len)
                .ok_or_else(|| MinerError::Gpu("d_hits slice OOB".into()))?;
            let mut buf = vec![0u8; byte_len];
            self.stream
                .memcpy_dtoh(&view, &mut buf)
                .map_err(|e| MinerError::Gpu(format!("dtoh d_hits: {e:?}")))?;
            buf
        };

        // Reset the device counter so the ring-buffer slots can be reused.
        {
            let mut sym = self
                .module
                .get_global("d_hit_count", &self.stream)
                .map_err(|e| MinerError::Gpu(format!("get_global(d_hit_count) reset: {e:?}")))?;
            let zero: [u8; 4] = 0u32.to_ne_bytes();
            self.stream
                .memcpy_htod(&zero, &mut sym)
                .map_err(|e| MinerError::Gpu(format!("htod d_hit_count reset: {e:?}")))?;
        }

        // SAFETY: HitRaw is repr(C); all fields are plain integer arrays.
        let raw_hits: Vec<HitRaw> = {
            let mut out = Vec::with_capacity(n);
            unsafe {
                let src = hits_bytes.as_ptr().cast::<HitRaw>();
                for i in 0..n {
                    out.push(std::ptr::read_unaligned(src.add(i)));
                }
            }
            out
        };

        Ok(raw_hits
            .into_iter()
            .map(|r| Hit {
                nonce: U256::from_be_bytes(r.nonce),
                hash: B256::from(r.hash),
                epoch_id: u64::from(r.epoch_id),
            })
            .collect())
    }

    /// Read the global nonce counter (for hash-rate estimation).
    pub fn read_nonce_counter(&self) -> Result<u64> {
        let sym = self
            .module
            .get_global("d_nonce_counter", &self.stream)
            .map_err(|e| MinerError::Gpu(format!("get_global(d_nonce_counter): {e:?}")))?;
        let mut buf = [0u8; 8];
        self.stream
            .memcpy_dtoh(&sym, &mut buf)
            .map_err(|e| MinerError::Gpu(format!("dtoh d_nonce_counter: {e:?}")))?;
        Ok(u64::from_ne_bytes(buf))
    }

    /// Test helper: pre-seed `d_nonce_counter` so the next batch starts at `nonce.low_u64()`,
    /// resets stop/hit flags, then launches a 1-block × 1-thread grid.
    /// The kernel will grind BATCH_PER_THREAD nonces from that starting point then exit.
    pub fn force_test_nonce(&mut self, nonce: U256) -> Result<()> {
        let counter: u64 = u64::try_from(nonce).unwrap_or(u64::MAX);

        // Reset d_should_stop = 0
        {
            let mut sym = self
                .module
                .get_global("d_should_stop", &self.stream)
                .map_err(|e| MinerError::Gpu(format!("get_global(d_should_stop): {e:?}")))?;
            let zero: [u8; 4] = 0u32.to_ne_bytes();
            self.stream
                .memcpy_htod(&zero, &mut sym)
                .map_err(|e| MinerError::Gpu(format!("htod d_should_stop: {e:?}")))?;
        }
        // Reset d_hit_count = 0
        {
            let mut sym = self
                .module
                .get_global("d_hit_count", &self.stream)
                .map_err(|e| MinerError::Gpu(format!("get_global(d_hit_count): {e:?}")))?;
            let zero: [u8; 4] = 0u32.to_ne_bytes();
            self.stream
                .memcpy_htod(&zero, &mut sym)
                .map_err(|e| MinerError::Gpu(format!("htod d_hit_count reset: {e:?}")))?;
        }
        // Set d_nonce_counter = counter
        {
            let mut sym = self
                .module
                .get_global("d_nonce_counter", &self.stream)
                .map_err(|e| MinerError::Gpu(format!("get_global(d_nonce_counter): {e:?}")))?;
            let bytes: [u8; 8] = counter.to_ne_bytes();
            self.stream
                .memcpy_htod(&bytes, &mut sym)
                .map_err(|e| MinerError::Gpu(format!("htod d_nonce_counter: {e:?}")))?;
        }
        // Launch 1 block × 1 thread; kernel will grind its batch then check stop flag.
        self.launch_persistent(1, 1)?;
        // Signal stop so kernel exits after its first batch.
        self.signal_stop()
    }

    /// Blocking poll: retry `poll_hits` until at least one hit is returned or `timeout` elapses.
    pub fn poll_hits_blocking(&mut self, timeout: std::time::Duration) -> Result<Vec<Hit>> {
        let start = std::time::Instant::now();
        loop {
            let hits = self.poll_hits()?;
            if !hits.is_empty() {
                return Ok(hits);
            }
            if start.elapsed() >= timeout {
                return Ok(Vec::new());
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}
