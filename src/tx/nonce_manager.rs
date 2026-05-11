use alloy::primitives::Address;
use alloy::providers::{Provider, RootProvider};
use alloy::transports::BoxTransport;
use crate::error::{MinerError, Result};
use parking_lot::Mutex;

/// Sequential gate: serialize submissions on the local nonce so we never have two in-flight.
pub struct NonceGate {
    provider: RootProvider<BoxTransport>,
    miner: Address,
    next: Mutex<Option<u64>>,
}

impl NonceGate {
    pub fn new(provider: RootProvider<BoxTransport>, miner: Address) -> Self {
        Self { provider, miner, next: Mutex::new(None) }
    }

    /// Reserve the next Ethereum tx nonce. Caller MUST consume or call `resync` on failure.
    pub async fn reserve(&self) -> Result<u64> {
        let next_held = *self.next.lock();
        let n = match next_held {
            Some(n) => n,
            None => self
                .provider
                .get_transaction_count(self.miner)
                .await
                .map_err(|e| MinerError::Rpc(e.to_string()))?,
        };
        *self.next.lock() = Some(n + 1);
        Ok(n)
    }

    /// Called after receipt or permanent failure to keep the local counter aligned with chain.
    pub async fn resync(&self) -> Result<()> {
        let n = self
            .provider
            .get_transaction_count(self.miner)
            .await
            .map_err(|e| MinerError::Rpc(e.to_string()))?;
        *self.next.lock() = Some(n);
        Ok(())
    }
}
