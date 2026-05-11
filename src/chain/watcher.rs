//! Live `ChainWatcher`: polls `miningState()` via HTTP (or WS) every 4 s,
//! publishes `ChallengeUpdate` events when the epoch changes.
//!
//! ## alloy 0.8 type-machinery note
//!
//! `alloy::providers::Provider` is generic over a `Transport` type parameter,
//! so `dyn Provider + Send + Sync` is not a legal object type.  `ChainWatcher`
//! is therefore **generic** over `P: Provider<BoxTransport>`.  Callers can
//! construct a `RootProvider<BoxTransport>` via `crate::rpc::connect` and pass
//! it directly; it is `Clone + Send + Sync + 'static` and already contains an
//! internal `Arc`, so no outer wrapping is needed.

use super::{ChainSource, ChallengeUpdate, MiningState};
use crate::chain::{
    challenge::compute_challenge,
    contract::{Hash, CHAIN_ID, CONTRACT, EPOCH_BLOCKS},
};
use crate::error::{MinerError, Result};
use alloy::primitives::{Address, B256, U256};
use alloy::providers::Provider;
use alloy::transports::BoxTransport;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::sync::Arc;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

/// Live chain-watcher backed by an alloy `Provider<BoxTransport>`.
pub struct ChainWatcher<P> {
    provider: P,
    #[allow(dead_code)]
    miner: Address,
    head: Arc<RwLock<u64>>,
    last_state: Arc<RwLock<Option<MiningState>>>,
    update_tx: watch::Sender<ChallengeUpdate>,
    /// Background polling task — kept alive for the lifetime of this struct.
    _bg: JoinHandle<()>,
}

impl<P> ChainWatcher<P>
where
    P: Provider<BoxTransport> + Clone + Send + Sync + 'static,
{
    /// Connect, fetch initial state, and start the background poll loop.
    pub async fn start(provider: P, miner: Address) -> Result<Self> {
        let block_number = provider
            .get_block_number()
            .await
            .map_err(|e| MinerError::Rpc(e.to_string()))?;

        let head = Arc::new(RwLock::new(block_number));
        let contract = Hash::new(CONTRACT, provider.clone());

        // Fetch the initial mining state.
        let st = contract
            .miningState()
            .call()
            .await
            .map_err(|e| MinerError::Contract(e.to_string()))?;

        let initial = MiningState {
            era: st.era.to::<u64>(),
            reward_wei: st.reward,
            difficulty: st.difficulty,
            minted_wei: st.minted,
            remaining_wei: st.remaining,
            epoch: st.epoch.to::<u64>(),
            epoch_blocks_left: st.epochBlocksLeft_.to::<u64>(),
        };

        let last_state = Arc::new(RwLock::new(Some(initial)));
        let challenge = compute_challenge(CHAIN_ID, CONTRACT, miner, initial.epoch);

        let (tx, _) = watch::channel(ChallengeUpdate {
            challenge,
            target: initial.difficulty,
            epoch: initial.epoch,
            block_number,
        });

        let bg = tokio::spawn(Self::poll_loop(
            provider.clone(),
            miner,
            head.clone(),
            last_state.clone(),
            tx.clone(),
        ));

        Ok(Self {
            provider,
            miner,
            head,
            last_state,
            update_tx: tx,
            _bg: bg,
        })
    }

    async fn poll_loop(
        provider: P,
        miner: Address,
        head: Arc<RwLock<u64>>,
        last_state: Arc<RwLock<Option<MiningState>>>,
        update_tx: watch::Sender<ChallengeUpdate>,
    ) {
        let contract = Hash::new(CONTRACT, provider.clone());
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(4));
        let mut last_epoch: Option<u64> = None;

        loop {
            interval.tick().await;

            let bn = match provider.get_block_number().await {
                Ok(n) => {
                    *head.write() = n;
                    n
                }
                Err(e) => {
                    warn!("get_block_number: {e}");
                    continue;
                }
            };

            match contract.miningState().call().await {
                Ok(st) => {
                    let epoch = st.epoch.to::<u64>();
                    let difficulty = st.difficulty;
                    let snap = MiningState {
                        era: st.era.to::<u64>(),
                        reward_wei: st.reward,
                        difficulty,
                        minted_wei: st.minted,
                        remaining_wei: st.remaining,
                        epoch,
                        epoch_blocks_left: st.epochBlocksLeft_.to::<u64>(),
                    };
                    *last_state.write() = Some(snap);

                    if last_epoch != Some(epoch) {
                        let challenge = compute_challenge(CHAIN_ID, CONTRACT, miner, epoch);
                        info!(epoch, ?challenge, "challenge swap");
                        let _ = update_tx.send(ChallengeUpdate {
                            challenge,
                            target: difficulty,
                            epoch,
                            block_number: bn,
                        });
                        last_epoch = Some(epoch);
                    }
                }
                Err(e) => warn!("miningState call: {e}"),
            }
        }
    }
}

#[async_trait]
impl<P> ChainSource for ChainWatcher<P>
where
    P: Provider<BoxTransport> + Clone + Send + Sync + 'static,
{
    fn head(&self) -> u64 {
        *self.head.read()
    }

    async fn challenge_for(&self, miner: Address) -> Result<B256> {
        let epoch = *self.head.read() / EPOCH_BLOCKS;
        Ok(compute_challenge(CHAIN_ID, CONTRACT, miner, epoch))
    }

    async fn mining_state(&self) -> Result<MiningState> {
        (*self.last_state.read())
            .ok_or_else(|| MinerError::Rpc("no miningState yet".into()))
    }

    fn subscribe(&self, _miner: Address) -> watch::Receiver<ChallengeUpdate> {
        self.update_tx.subscribe()
    }

    async fn mints_in_block(&self, block: u64) -> Result<u64> {
        let contract = Hash::new(CONTRACT, self.provider.clone());
        let ret = contract
            .mintsInBlock(U256::from(block))
            .call()
            .await
            .map_err(|e| MinerError::Contract(e.to_string()))?;
        Ok(ret._0.to::<u64>())
    }

    async fn genesis_complete(&self) -> Result<bool> {
        let contract = Hash::new(CONTRACT, self.provider.clone());
        let ret = contract
            .genesisComplete()
            .call()
            .await
            .map_err(|e| MinerError::Contract(e.to_string()))?;
        Ok(ret._0)
    }
}
