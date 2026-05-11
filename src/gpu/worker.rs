#![cfg(feature = "cuda-runtime")]

use super::{Grinder, Hit};
use crate::error::Result;
use crate::gpu::kernel_ffi::GpuRuntime;
use alloy::primitives::{B256, U256};
use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// Live state shared between GpuWorker and the launch-loop background thread.
struct WorkerState {
    /// Current challenge for hit tagging. Updated by hot_swap, read on every hit.
    challenge: B256,
    /// Current epoch_id — written into emitted Hit events.
    epoch_id: u64,
    /// Pending challenge/target/epoch from hot_swap — applied to device at the
    /// next launch iteration boundary.
    pending: Option<(B256, U256, u64)>,
}

pub struct GpuWorker {
    state: Arc<Mutex<WorkerState>>,
    _hit_tx: mpsc::Sender<Hit>,
    hit_rx: Mutex<Option<mpsc::Receiver<Hit>>>,
    hashrate: Arc<RwLock<f64>>,
    shutdown: Arc<AtomicBool>,
}

impl GpuWorker {
    /// Start the GPU launch loop in a dedicated OS thread.
    ///
    /// `batch_size`  — total nonces per kernel launch (grid_size * block_size).
    ///                 Default 268M for 1-2 GH/s cards, up to 2G for 5090.
    /// `block_size`  — threads per block (256 is typical).
    pub async fn start(
        device_id: u32,
        batch_size: u64,
        block_size: u32,
        _poll_ms_unused: u64,
    ) -> Result<Self> {
        let rt = GpuRuntime::init(device_id)?;

        let state = Arc::new(Mutex::new(WorkerState {
            challenge: B256::ZERO,
            epoch_id: 0,
            pending: None,
        }));
        let (tx, rx) = mpsc::channel::<Hit>(16);
        let hashrate = Arc::new(RwLock::new(0.0));
        let shutdown = Arc::new(AtomicBool::new(false));

        // Launch loop in a dedicated OS thread (CUDA context is thread-local,
        // and the kernel calls block on stream.synchronize() — std::thread
        // works better here than tokio::spawn since tokio expects async work).
        let state_c = state.clone();
        let tx_c = tx.clone();
        let rate_c = hashrate.clone();
        let shut_c = shutdown.clone();
        std::thread::Builder::new()
            .name("hashminer-gpu-launch".into())
            .spawn(move || {
                if let Err(e) =
                    launch_loop(rt, batch_size, block_size, state_c, tx_c, rate_c, shut_c)
                {
                    tracing::error!(error = ?e, "GPU launch loop failed");
                }
            })
            .map_err(|e| crate::error::MinerError::Gpu(format!("spawn launch thread: {e}")))?;

        Ok(Self {
            state,
            _hit_tx: tx,
            hit_rx: Mutex::new(Some(rx)),
            hashrate,
            shutdown,
        })
    }
}

/// Synchronous launch loop. Holds the GpuRuntime exclusively (CUDA context
/// is thread-bound). Applies hot-swaps lazily between launches.
fn launch_loop(
    mut rt: GpuRuntime,
    batch_size: u64,
    block_size: u32,
    state: Arc<Mutex<WorkerState>>,
    tx: mpsc::Sender<Hit>,
    rate: Arc<RwLock<f64>>,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    // Random initial nonce_start so two miners restarting concurrently don't
    // hash the same range — matches friend's reference miner.
    let mut nonce_start: u64 = rand::thread_rng().gen_range(0..(1u64 << 62));

    let mut total_hashes: u64 = 0;
    let mut sample_t = Instant::now();
    let mut current_challenge = B256::ZERO;
    let mut current_epoch: u64 = 0;

    while !shutdown.load(Ordering::Relaxed) {
        // Apply any pending hot_swap before launching.
        let pending_opt = state.lock().pending.take();
        if let Some((challenge, target, epoch_id)) = pending_opt {
            rt.hot_swap(challenge, target, epoch_id as u32)?;
            current_challenge = challenge;
            current_epoch = epoch_id;
            let mut s = state.lock();
            s.challenge = challenge;
            s.epoch_id = epoch_id;
            tracing::debug!(epoch_id, "GPU swapped challenge/target");
            // Reset nonce_start to a fresh random region on epoch change so
            // restarts don't waste effort on previously-tried nonces.
            nonce_start = rand::thread_rng().gen_range(0..(1u64 << 62));
        }

        // Skip launches until we have a valid challenge (target != 0 implicitly).
        if current_challenge == B256::ZERO {
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }

        let maybe_nonce = rt.launch_mine(nonce_start, batch_size, block_size)?;
        total_hashes = total_hashes.wrapping_add(batch_size);

        if let Some(nonce) = maybe_nonce {
            let hit = Hit {
                nonce: U256::from(nonce),
                hash: B256::ZERO, // we don't reconstruct hash here; submitter verifies via CPU
                epoch_id: current_epoch,
            };
            tracing::info!(nonce, epoch = current_epoch, "GPU found nonce");
            // Use blocking_send since we're in a sync OS thread context.
            if let Err(e) = tx.blocking_send(hit) {
                tracing::warn!(error = %e, "hit channel closed, exiting launch loop");
                break;
            }
        }

        nonce_start = nonce_start.wrapping_add(batch_size);

        // Hashrate sampling every ~1 second.
        let now = Instant::now();
        let dt = now.duration_since(sample_t).as_secs_f64();
        if dt >= 1.0 {
            *rate.write() = total_hashes as f64 / dt;
            total_hashes = 0;
            sample_t = now;
        }
    }

    tracing::info!("GPU launch loop exiting");
    Ok(())
}

#[async_trait]
impl Grinder for GpuWorker {
    async fn hot_swap(&self, challenge: B256, target: U256, epoch_id: u64) -> Result<()> {
        self.state.lock().pending = Some((challenge, target, epoch_id));
        Ok(())
    }

    fn take_hit_rx(&self) -> mpsc::Receiver<Hit> {
        self.hit_rx.lock().take().expect("take_hit_rx called twice")
    }

    fn hashrate(&self) -> f64 {
        *self.hashrate.read()
    }

    async fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}
