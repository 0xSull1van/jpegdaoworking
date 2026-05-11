use alloy::primitives::{Address, B256, U256};
use async_trait::async_trait;
use tokio::sync::watch;

/// What ChainWatcher publishes every time epoch or difficulty changes.
#[derive(Debug, Clone)]
pub struct ChallengeUpdate {
    pub challenge: B256,
    pub target: U256,
    pub epoch: u64,
    pub block_number: u64,
}

/// Snapshot of contract state (single `miningState()` call).
#[derive(Debug, Clone, Copy)]
pub struct MiningState {
    pub era: u64,
    pub reward_wei: U256,
    pub difficulty: U256,
    pub minted_wei: U256,
    pub remaining_wei: U256,
    pub epoch: u64,
    pub epoch_blocks_left: u64,
}

#[async_trait]
pub trait ChainSource: Send + Sync + 'static {
    /// Latest block number observed.
    fn head(&self) -> u64;
    /// Compute (or fetch) current challenge for the given miner address.
    async fn challenge_for(&self, miner: Address) -> crate::error::Result<B256>;
    /// Read `miningState()` snapshot.
    async fn mining_state(&self) -> crate::error::Result<MiningState>;
    /// Subscribe to a stream of (challenge, target, epoch) updates.
    fn subscribe(&self, miner: Address) -> watch::Receiver<ChallengeUpdate>;
    /// Read mints already in the given block (for EV gate).
    async fn mints_in_block(&self, block: u64) -> crate::error::Result<u64>;
    /// Has the genesis sale closed? Mining reverts until true.
    async fn genesis_complete(&self) -> crate::error::Result<bool>;
}

pub mod challenge;
pub mod contract;
pub mod fake;
