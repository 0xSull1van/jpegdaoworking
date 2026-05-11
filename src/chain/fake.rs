use super::{ChainSource, ChallengeUpdate, MiningState};
use crate::chain::{
    challenge::compute_challenge,
    contract::{CONTRACT, EPOCH_BLOCKS},
};
use crate::error::Result;
use alloy::primitives::{Address, B256, U256};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::watch;

#[derive(Clone)]
pub struct FakeChain {
    inner: Arc<Mutex<Inner>>,
    update_tx: watch::Sender<ChallengeUpdate>,
}

struct Inner {
    head: u64,
    difficulty: U256,
    minted: U256,
    #[allow(dead_code)]
    total_mints: u64,
    mints_in_block: std::collections::HashMap<u64, u64>,
    genesis_complete: bool,
}

impl FakeChain {
    pub fn new(initial_head: u64, difficulty: U256, miner: Address) -> Self {
        let inner = Inner {
            head: initial_head,
            difficulty,
            minted: U256::ZERO,
            total_mints: 0,
            mints_in_block: Default::default(),
            genesis_complete: true,
        };
        let epoch = initial_head / EPOCH_BLOCKS;
        let challenge = compute_challenge(1, CONTRACT, miner, epoch);
        let (tx, _) = watch::channel(ChallengeUpdate {
            challenge,
            target: difficulty,
            epoch,
            block_number: initial_head,
        });
        Self {
            inner: Arc::new(Mutex::new(inner)),
            update_tx: tx,
        }
    }

    /// Advance the head by N blocks and re-publish challenge if epoch rolled.
    pub fn advance_blocks(&self, n: u64, miner: Address) {
        let mut g = self.inner.lock();
        let old_epoch = g.head / EPOCH_BLOCKS;
        g.head += n;
        let new_epoch = g.head / EPOCH_BLOCKS;
        drop(g);
        if new_epoch != old_epoch {
            let challenge = compute_challenge(1, CONTRACT, miner, new_epoch);
            let g = self.inner.lock();
            let _ = self.update_tx.send(ChallengeUpdate {
                challenge,
                target: g.difficulty,
                epoch: new_epoch,
                block_number: g.head,
            });
        }
    }

    pub fn set_difficulty(&self, d: U256, miner: Address) {
        let mut g = self.inner.lock();
        g.difficulty = d;
        let epoch = g.head / EPOCH_BLOCKS;
        let challenge = compute_challenge(1, CONTRACT, miner, epoch);
        let _ = self.update_tx.send(ChallengeUpdate {
            challenge,
            target: d,
            epoch,
            block_number: g.head,
        });
    }
}

#[async_trait]
impl ChainSource for FakeChain {
    fn head(&self) -> u64 {
        self.inner.lock().head
    }

    async fn challenge_for(&self, miner: Address) -> Result<B256> {
        let g = self.inner.lock();
        Ok(compute_challenge(1, CONTRACT, miner, g.head / EPOCH_BLOCKS))
    }

    async fn mining_state(&self) -> Result<MiningState> {
        let g = self.inner.lock();
        let epoch = g.head / EPOCH_BLOCKS;
        Ok(MiningState {
            era: g.total_mints / 100_000,
            reward_wei: U256::from(100_u64) * U256::from(10u64).pow(U256::from(18u64)),
            difficulty: g.difficulty,
            minted_wei: g.minted,
            remaining_wei: U256::ZERO,
            epoch,
            epoch_blocks_left: EPOCH_BLOCKS - (g.head % EPOCH_BLOCKS),
        })
    }

    fn subscribe(&self, _miner: Address) -> watch::Receiver<ChallengeUpdate> {
        self.update_tx.subscribe()
    }

    async fn mints_in_block(&self, block: u64) -> Result<u64> {
        Ok(self.inner.lock().mints_in_block.get(&block).copied().unwrap_or(0))
    }

    async fn genesis_complete(&self) -> Result<bool> {
        Ok(self.inner.lock().genesis_complete)
    }
}
