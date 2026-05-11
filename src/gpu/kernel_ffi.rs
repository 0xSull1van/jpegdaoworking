// CUDA runtime FFI via cudarc 0.19.
//
// One-shot mining kernel architecture:
//   • Each kernel launch processes a fixed grid of work-items (BATCH_SIZE).
//   • Each thread computes EXACTLY ONE keccak hash for nonce = nonce_start + gid.
//   • If a thread finds a valid hash, it sets `found = 1` via atomicCAS and
//     writes the winning nonce to `result`. Other threads see `found=1` on entry
//     and exit immediately.
//   • Host launches kernels in a loop, incrementing nonce_start by BATCH_SIZE
//     between launches and reading the result/found flags after each sync.
//
// Why this design (vs the previous persistent kernel):
//   • No global atomic on a shared nonce counter — every thread derives its
//     unique nonce from gid implicitly.
//   • Compiler optimises a one-shot thread far better than an infinite outer
//     loop with hold-live state across iterations.
//   • Driver scheduling depriorities of long-running kernels (observed on
//     Blackwell sm_120) no longer apply.
//
// All kernel args are passed by reference per launch — no `__device__` globals
// are used in this kernel. Buffers are allocated once in `init`.

#![cfg(feature = "cuda-runtime")]

use crate::error::{MinerError, Result};
use crate::gpu::ptx::PTX;
use alloy::primitives::{B256, U256};
use cudarc::driver::{
    safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, PushKernelArg},
    LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub struct GpuRuntime {
    pub context: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    mine_fn: CudaFunction,

    // Device buffers — allocated once, reused for every launch.
    challenge_buf: CudaSlice<u64>, // 4 lanes, LE-interpreted (state lanes 0..3 raw)
    target_buf: CudaSlice<u64>,    // 4 lanes, big-endian (target[0] = MSB)
    result_buf: CudaSlice<u64>,    // 1 element — winning nonce
    found_buf: CudaSlice<i32>,     // 1 element — atomic flag

    // Host-side cache of the active epoch id (for tagging Hit events).
    pub epoch_id_host: u32,
}

impl GpuRuntime {
    pub fn init(device_id: u32) -> Result<Self> {
        let context = CudaContext::new(device_id as usize)
            .map_err(|e| MinerError::Gpu(format!("CudaContext::new: {e:?}")))?;

        let ptx_str = std::str::from_utf8(PTX)
            .map_err(|_| MinerError::Gpu("PTX is not valid UTF-8".into()))?;

        let module: Arc<CudaModule> = context
            .load_module(Ptx::from_src(ptx_str))
            .map_err(|e| MinerError::Gpu(format!("load_module: {e:?}")))?;

        let mine_fn: CudaFunction = module
            .load_function("mine")
            .map_err(|e| MinerError::Gpu(format!("load_function(mine): {e:?}")))?;

        let stream = context.default_stream();

        // Pre-allocate the small fixed-size device buffers.
        let challenge_buf = stream
            .alloc_zeros::<u64>(4)
            .map_err(|e| MinerError::Gpu(format!("alloc challenge_buf: {e:?}")))?;
        let target_buf = stream
            .alloc_zeros::<u64>(4)
            .map_err(|e| MinerError::Gpu(format!("alloc target_buf: {e:?}")))?;
        let result_buf = stream
            .alloc_zeros::<u64>(1)
            .map_err(|e| MinerError::Gpu(format!("alloc result_buf: {e:?}")))?;
        let found_buf = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| MinerError::Gpu(format!("alloc found_buf: {e:?}")))?;

        Ok(Self {
            context,
            stream,
            module,
            mine_fn,
            challenge_buf,
            target_buf,
            result_buf,
            found_buf,
            epoch_id_host: 0,
        })
    }

    /// Update challenge + target on the device for upcoming launches.
    /// Safe to call between launches; in flight launch uses whatever was set
    /// when it was queued.
    ///
    /// `challenge` is the 32-byte raw keccak256 challenge; interpreted as 4 LE
    /// uint64 lanes (matching `st[0..3] = challenge[0..3]` in the kernel).
    /// `target` is the 256-bit difficulty, stored as 4 BE u64 (target[0] = MSB).
    pub fn hot_swap(&mut self, challenge: B256, target: U256, epoch_id: u32) -> Result<()> {
        // Challenge → 4 LE u64 lanes from the raw 32-byte challenge.
        let cb: [u8; 32] = challenge.0;
        let challenge_lanes: [u64; 4] = [
            u64::from_le_bytes(cb[0..8].try_into().unwrap()),
            u64::from_le_bytes(cb[8..16].try_into().unwrap()),
            u64::from_le_bytes(cb[16..24].try_into().unwrap()),
            u64::from_le_bytes(cb[24..32].try_into().unwrap()),
        ];

        // Target → 4 BE u64 lanes (most-significant first).
        let tb = target.to_be_bytes::<32>();
        let target_lanes: [u64; 4] = [
            u64::from_be_bytes(tb[0..8].try_into().unwrap()),
            u64::from_be_bytes(tb[8..16].try_into().unwrap()),
            u64::from_be_bytes(tb[16..24].try_into().unwrap()),
            u64::from_be_bytes(tb[24..32].try_into().unwrap()),
        ];

        self.stream
            .memcpy_htod(&challenge_lanes, &mut self.challenge_buf)
            .map_err(|e| MinerError::Gpu(format!("htod challenge: {e:?}")))?;
        self.stream
            .memcpy_htod(&target_lanes, &mut self.target_buf)
            .map_err(|e| MinerError::Gpu(format!("htod target: {e:?}")))?;
        self.epoch_id_host = epoch_id;

        // Sync to ensure the writes land before any subsequent launch sees them.
        self.stream
            .synchronize()
            .map_err(|e| MinerError::Gpu(format!("hot_swap sync: {e:?}")))?;
        Ok(())
    }

    /// Launch the mine kernel for one batch of nonces starting at `nonce_start`.
    /// Returns Ok(Some(nonce)) if a solution was found this launch, Ok(None) otherwise.
    ///
    /// `batch_size` is the total number of work-items spawned (grid * block).
    /// `block_size` is threads per block (must divide batch_size; typical 256).
    pub fn launch_mine(
        &mut self,
        nonce_start: u64,
        batch_size: u64,
        block_size: u32,
    ) -> Result<Option<u64>> {
        // Reset found + result buffers.
        let zero_i32: [i32; 1] = [0];
        let zero_u64: [u64; 1] = [0];
        self.stream
            .memcpy_htod(&zero_i32, &mut self.found_buf)
            .map_err(|e| MinerError::Gpu(format!("reset found: {e:?}")))?;
        self.stream
            .memcpy_htod(&zero_u64, &mut self.result_buf)
            .map_err(|e| MinerError::Gpu(format!("reset result: {e:?}")))?;

        // Grid dimensions. CUDA's grid_dim.x max ≈ 2^31 - 1, so grid_count fits
        // comfortably for any reasonable batch_size.
        let grid_count: u32 = (batch_size / block_size as u64) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_count, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(&self.mine_fn);
            builder
                .arg(&self.challenge_buf)
                .arg(&nonce_start)
                .arg(&self.target_buf)
                .arg(&mut self.result_buf)
                .arg(&mut self.found_buf);
            builder
                .launch(cfg)
                .map_err(|e| MinerError::Gpu(format!("launch mine: {e:?}")))?;
        }

        // Wait for kernel completion + read result.
        self.stream
            .synchronize()
            .map_err(|e| MinerError::Gpu(format!("launch sync: {e:?}")))?;

        let mut found_h: [i32; 1] = [0];
        self.stream
            .memcpy_dtoh(&self.found_buf, &mut found_h)
            .map_err(|e| MinerError::Gpu(format!("dtoh found: {e:?}")))?;

        if found_h[0] != 0 {
            let mut result_h: [u64; 1] = [0];
            self.stream
                .memcpy_dtoh(&self.result_buf, &mut result_h)
                .map_err(|e| MinerError::Gpu(format!("dtoh result: {e:?}")))?;
            Ok(Some(result_h[0]))
        } else {
            Ok(None)
        }
    }
}
