use super::{Grinder, Hit};
use crate::chain::challenge::compute_inner_hash;
use crate::error::Result;
use alloy::primitives::{B256, U256};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct FakeGrinder {
    state: Arc<Mutex<State>>,
    hit_tx: mpsc::Sender<Hit>,
    hit_rx: Mutex<Option<mpsc::Receiver<Hit>>>,
}

struct State {
    challenge: B256,
    target: U256,
    epoch_id: u64,
    next_nonce: u64,
    hashrate: f64,
}

impl FakeGrinder {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(16);
        Self {
            state: Arc::new(Mutex::new(State {
                challenge: B256::ZERO,
                target: U256::ZERO,
                epoch_id: 0,
                next_nonce: 0,
                hashrate: 0.0,
            })),
            hit_tx: tx,
            hit_rx: Mutex::new(Some(rx)),
        }
    }

    /// Synthetic grind: find the first valid nonce ≥ `next_nonce` and emit it.
    ///
    /// Used by tests to drive a single solution through the pipeline without GPU hardware.
    pub async fn drive_one_hit(&self) {
        let (challenge, target, epoch_id, start) = {
            let g = self.state.lock();
            (g.challenge, g.target, g.epoch_id, g.next_nonce)
        };
        for n in start.. {
            let h = compute_inner_hash(challenge, U256::from(n));
            if U256::from_be_bytes(*h) < target {
                let _ = self
                    .hit_tx
                    .send(Hit {
                        nonce: U256::from(n),
                        hash: h,
                        epoch_id,
                    })
                    .await;
                self.state.lock().next_nonce = n + 1;
                return;
            }
        }
    }
}

impl Default for FakeGrinder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Grinder for FakeGrinder {
    async fn hot_swap(&self, challenge: B256, target: U256, epoch_id: u64) -> Result<()> {
        let mut g = self.state.lock();
        g.challenge = challenge;
        g.target = target;
        g.epoch_id = epoch_id;
        g.next_nonce = 0;
        Ok(())
    }

    fn take_hit_rx(&self) -> mpsc::Receiver<Hit> {
        self.hit_rx
            .lock()
            .take()
            .expect("take_hit_rx called twice")
    }

    fn hashrate(&self) -> f64 {
        self.state.lock().hashrate
    }

    async fn shutdown(&self) {}
}
