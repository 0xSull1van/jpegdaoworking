use alloy::primitives::TxHash;
use serde::Serialize;
use std::time::SystemTime;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum Event {
    Hashrate { hashrate_hps: f64 },
    EpochSwap { epoch: u64, block: u64, diff: String, challenge: String, latency_ms: u64 },
    HitFound { epoch: u64, nonce: String },
    HitStale { epoch: u64, current_epoch: u64 },
    HitDropped { reason: String },
    TxSubmitted { relay: String, tx: TxHash },
    TxIncluded { tx: TxHash, block: u64, reward: String },
    TxReverted { tx: TxHash, reason: String },
    StateChange { from: String, to: String },
}

#[derive(Clone)]
pub struct MetricsBus {
    tx: mpsc::Sender<(SystemTime, Event)>,
}

impl MetricsBus {
    pub fn channel(capacity: usize) -> (Self, mpsc::Receiver<(SystemTime, Event)>) {
        let (tx, rx) = mpsc::channel(capacity);
        (Self { tx }, rx)
    }
    pub fn emit(&self, e: Event) {
        let _ = self.tx.try_send((SystemTime::now(), e));
    }
}

pub mod stdout;
pub mod jsonl;
pub mod redact;
