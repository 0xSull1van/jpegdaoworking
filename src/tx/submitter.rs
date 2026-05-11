use super::{Submitter, SubmitOutcome};
use crate::chain::ChainSource;
use crate::chain::contract::CHAIN_ID;
use crate::error::{MinerError, Result};
use crate::gpu::Hit;
use crate::tx::{
    builder::{build_mine_tx, MineTxParams},
    ev_gate::{EvGate, EvParams},
    nonce_manager::NonceGate,
    relay::Relay,
};
use crate::wallet::MinerSigner;
use alloy::eips::eip2718::Encodable2718;
use alloy::network::{EthereumWallet, TransactionBuilder};
use alloy::primitives::{Bytes, TxHash, U256};
use alloy::providers::{Provider, RootProvider};
use alloy::rpc::types::TransactionReceipt;
use alloy::transports::BoxTransport;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{info, warn};

pub struct TxSubmitter<C: ChainSource> {
    chain: Arc<C>,
    signer: Arc<MinerSigner>,
    read_provider: RootProvider<BoxTransport>,
    relays: Vec<Relay>,
    nonce_gate: Arc<NonceGate>,
    seq_gate: Arc<Mutex<()>>, // one in-flight tx at a time
    ev: EvParams,
    confirmations: u64,
}

impl<C: ChainSource> TxSubmitter<C> {
    pub fn new(
        chain: Arc<C>,
        signer: Arc<MinerSigner>,
        read_provider: RootProvider<BoxTransport>,
        relays: Vec<Relay>,
        ev: EvParams,
        confirmations: u64,
    ) -> Self {
        let nonce_gate = Arc::new(NonceGate::new(read_provider.clone(), signer.address()));
        Self {
            chain,
            signer,
            read_provider,
            relays,
            nonce_gate,
            seq_gate: Arc::new(Mutex::new(())),
            ev,
            confirmations,
        }
    }
}

#[async_trait]
impl<C: ChainSource> Submitter for TxSubmitter<C> {
    async fn submit(&self, hit: Hit) -> Result<SubmitOutcome> {
        let _guard = self.seq_gate.lock().await;

        // Stale-epoch re-check.
        let st = self.chain.mining_state().await?;
        if hit.epoch_id != st.epoch {
            return Ok(SubmitOutcome::Dropped {
                reason: format!("stale epoch {} vs {}", hit.epoch_id, st.epoch),
            });
        }

        // Gas estimate using a baseline; real value pulled from latest base fee + tip.
        let gas_limit: u64 = 120_000; // mine() is small, mostly storage
        let base_fee = self
            .read_provider
            .get_gas_price()
            .await
            .map_err(|e| MinerError::Rpc(e.to_string()))?;
        let max_priority: u128 = 3_000_000_000; // 3 gwei default tip
        let max_fee = base_fee + max_priority;
        let gas_cost = U256::from(max_fee) * U256::from(gas_limit);

        // EV gate.
        let allowed = EvGate {
            chain: &*self.chain,
            params: EvParams {
                max_mints_per_block: self.ev.max_mints_per_block,
                min_ratio: self.ev.min_ratio,
            },
        }
        .allow(st.reward_wei, gas_cost)
        .await?;
        if !allowed {
            return Ok(SubmitOutcome::Dropped {
                reason: "EV gate".into(),
            });
        }

        let tx_nonce = self.nonce_gate.reserve().await?;
        let req = build_mine_tx(MineTxParams {
            from: self.signer.address(),
            nonce: tx_nonce,
            mine_nonce: hit.nonce,
            max_fee_per_gas: max_fee,
            max_priority_fee_per_gas: max_priority,
            gas_limit,
            chain_id: CHAIN_ID,
        })?;

        // Sign.
        let wallet = EthereumWallet::from(self.signer.signer().clone());
        let signed = req
            .build(&wallet)
            .await
            .map_err(|e| MinerError::Tx(e.to_string()))?;
        let raw: Bytes = signed.encoded_2718().into();

        // Fan out to all relays in parallel; first OK response wins.
        let relay_futs: Vec<_> = self
            .relays
            .iter()
            .map(|r| {
                let r_name = r.name.clone();
                let raw = raw.clone();
                async move { (r_name, r.send_raw(raw).await) }
            })
            .collect();

        let mut winner_tx: Option<(String, TxHash)> = None;
        let mut errors = Vec::new();
        for fut in relay_futs {
            match fut.await {
                (n, Ok(h)) => {
                    info!(relay = %n, ?h, "submitted");
                    if winner_tx.is_none() {
                        winner_tx = Some((n, h));
                    }
                }
                (n, Err(e)) => {
                    warn!(relay = %n, error = %e, "relay failed");
                    errors.push((n, e));
                }
            }
        }

        let (relay, tx) = match winner_tx {
            Some(w) => w,
            None => {
                self.nonce_gate.resync().await.ok();
                return Ok(SubmitOutcome::Dropped {
                    reason: format!("all relays failed: {errors:?}"),
                });
            }
        };

        // Wait for receipt with timeout.
        let outcome = wait_for_receipt(
            self.read_provider.clone(),
            tx,
            self.confirmations,
            Duration::from_secs(90),
        )
        .await?;
        self.nonce_gate.resync().await.ok();
        match outcome {
            Some(r) if r.status() => Ok(SubmitOutcome::Included {
                tx,
                block: r.block_number.unwrap_or_default(),
                reward_wei: st.reward_wei,
                relay,
            }),
            Some(r) => Ok(SubmitOutcome::Reverted {
                tx,
                reason: format!("receipt status=0 block={:?}", r.block_number),
            }),
            None => Ok(SubmitOutcome::Dropped {
                reason: "receipt timeout".into(),
            }),
        }
    }
}

async fn wait_for_receipt(
    provider: RootProvider<BoxTransport>,
    tx: TxHash,
    confirmations: u64,
    timeout: Duration,
) -> Result<Option<TransactionReceipt>> {
    let start = std::time::Instant::now();
    loop {
        if start.elapsed() > timeout {
            return Ok(None);
        }
        if let Some(r) = provider
            .get_transaction_receipt(tx)
            .await
            .map_err(|e| MinerError::Rpc(e.to_string()))?
        {
            let head = provider
                .get_block_number()
                .await
                .map_err(|e| MinerError::Rpc(e.to_string()))?;
            let included_at = r.block_number.unwrap_or(head);
            if head.saturating_sub(included_at) >= confirmations {
                return Ok(Some(r));
            }
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}
