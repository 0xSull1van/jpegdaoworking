#![cfg(feature = "cuda-runtime")]

use super::{Grinder, Hit};
use crate::error::Result;
use crate::gpu::kernel_ffi::GpuRuntime;
use alloy::primitives::{B256, U256};
use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

pub struct GpuWorker {
    rt: Arc<Mutex<GpuRuntime>>,
    _hit_tx: mpsc::Sender<Hit>,
    hit_rx: Mutex<Option<mpsc::Receiver<Hit>>>,
    hashrate: Arc<RwLock<f64>>,
    shutdown: Arc<AtomicBool>,
}

impl GpuWorker {
    pub async fn start(device_id: u32, blocks: u32, threads: u32, poll_ms: u64) -> Result<Self> {
        let rt = GpuRuntime::init(device_id)?;
        rt.launch_persistent(blocks, threads)?;
        let rt = Arc::new(Mutex::new(rt));
        let (tx, rx) = mpsc::channel::<Hit>(16);
        let hashrate = Arc::new(RwLock::new(0.0));
        let shutdown = Arc::new(AtomicBool::new(false));

        // Poll task.
        let rt2 = rt.clone();
        let tx2 = tx.clone();
        let rate = hashrate.clone();
        let shut = shutdown.clone();
        tokio::spawn(async move {
            let mut last_counter: u64 = 0;
            let mut last_t = Instant::now();
            while !shut.load(Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_millis(poll_ms)).await;
                let hits_and_counter = {
                    let mut g = rt2.lock();
                    let hits = g.poll_hits().unwrap_or_default();
                    let counter = g.read_nonce_counter().unwrap_or(0);
                    (hits, counter)
                };
                for h in hits_and_counter.0 {
                    let _ = tx2.send(h).await;
                }
                let counter = hits_and_counter.1;
                let now = Instant::now();
                let dt = now.duration_since(last_t).as_secs_f64();
                if dt > 0.5 {
                    let delta = counter.saturating_sub(last_counter) as f64;
                    *rate.write() = delta / dt;
                    last_counter = counter;
                    last_t = now;
                }
            }
        });

        Ok(Self {
            rt,
            _hit_tx: tx,
            hit_rx: Mutex::new(Some(rx)),
            hashrate,
            shutdown,
        })
    }
}

#[async_trait]
impl Grinder for GpuWorker {
    async fn hot_swap(&self, challenge: B256, target: U256, epoch_id: u64) -> Result<()> {
        let mut g = self.rt.lock();
        g.hot_swap(challenge, target, epoch_id as u32)
    }

    fn take_hit_rx(&self) -> mpsc::Receiver<Hit> {
        self.hit_rx.lock().take().expect("take_hit_rx called twice")
    }

    fn hashrate(&self) -> f64 {
        *self.hashrate.read()
    }

    async fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        let _ = self.rt.lock().signal_stop();
    }
}
