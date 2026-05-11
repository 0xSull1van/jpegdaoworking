use alloy::primitives::{B256, U256};
use async_trait::async_trait;
use tokio::sync::mpsc;

/// One valid (nonce, hash, epoch_id) triple emitted by the kernel.
#[derive(Debug, Clone)]
pub struct Hit {
    pub nonce: U256,
    pub hash: B256,
    pub epoch_id: u64,
}

#[async_trait]
pub trait Grinder: Send + Sync + 'static {
    /// Apply a new (challenge, target, epoch_id) without restarting the kernel.
    async fn hot_swap(&self, challenge: B256, target: U256, epoch_id: u64) -> crate::error::Result<()>;
    /// Subscribe to the hit stream. Single-consumer.
    fn take_hit_rx(&self) -> mpsc::Receiver<Hit>;
    /// Current observed hashrate in hashes/sec, averaged over last ~1s.
    fn hashrate(&self) -> f64;
    /// Cooperative shutdown.
    async fn shutdown(&self);
}
