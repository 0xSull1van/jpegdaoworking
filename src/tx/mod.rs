use alloy::primitives::{TxHash, U256};
use async_trait::async_trait;
use crate::gpu::Hit;

#[derive(Debug, Clone)]
pub enum SubmitOutcome {
    Included { tx: TxHash, block: u64, reward_wei: U256, relay: String },
    Reverted { tx: TxHash, reason: String },
    Dropped  { reason: String },
}

#[async_trait]
pub trait Submitter: Send + Sync + 'static {
    /// Submit a hit. Returns when included, reverted, or dropped.
    /// Implementations gate concurrency internally (sequential gate).
    async fn submit(&self, hit: Hit) -> crate::error::Result<SubmitOutcome>;
}
