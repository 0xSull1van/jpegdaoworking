use alloy::primitives::{B256, U256};
use async_trait::async_trait;
use tokio::sync::mpsc;

/// One valid `(nonce, hash, epoch_id)` triple emitted by the kernel (or FakeGrinder).
#[derive(Debug, Clone)]
pub struct Hit {
    pub nonce: U256,
    pub hash: B256,
    pub epoch_id: u64,
}

/// Common interface over the GPU kernel worker and the CPU fake used in tests.
#[async_trait]
pub trait Grinder: Send + Sync + 'static {
    /// Apply a new `(challenge, target, epoch_id)` without restarting the kernel.
    async fn hot_swap(&self, challenge: B256, target: U256, epoch_id: u64) -> crate::error::Result<()>;
    /// Subscribe to the hit stream. Single-consumer; panics on second call.
    fn take_hit_rx(&self) -> mpsc::Receiver<Hit>;
    /// Current observed hashrate in hashes/sec, averaged over last ~1 s.
    fn hashrate(&self) -> f64;
    /// Cooperative shutdown — waits until the kernel or background task exits.
    async fn shutdown(&self);
}

pub mod ptx;
pub mod fake;

#[cfg(feature = "cuda-runtime")]
pub mod kernel_ffi;

#[cfg(feature = "cuda-runtime")]
pub mod worker;
