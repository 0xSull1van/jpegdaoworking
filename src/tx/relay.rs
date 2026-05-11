use crate::error::{MinerError, Result};
use alloy::primitives::{Bytes, TxHash};
use alloy::providers::{Provider, ProviderBuilder, RootProvider};
use alloy::transports::BoxTransport;
use url::Url;

pub struct Relay {
    pub name: String,
    provider: RootProvider<BoxTransport>,
}

impl Relay {
    /// Construct a relay asynchronously — validates the URL and connects.
    pub async fn new(name: impl Into<String>, url: &str) -> Result<Self> {
        let _ = Url::parse(url).map_err(|e| MinerError::Config(e.to_string()))?;
        let provider = ProviderBuilder::new()
            .on_builtin(url)
            .await
            .map_err(|e| MinerError::Rpc(e.to_string()))?;
        Ok(Self { name: name.into(), provider })
    }

    /// Broadcast a signed, RLP-encoded transaction and return its hash.
    pub async fn send_raw(&self, raw: Bytes) -> Result<TxHash> {
        self.provider
            .send_raw_transaction(&raw)
            .await
            .map(|p| *p.tx_hash())
            .map_err(|e| MinerError::Tx(format!("{}: {}", self.name, e)))
    }
}

/// Default relay set: MEV-Blocker, Flashbots Protect, then a public fallback.
pub async fn default_relays(public_fallback: &str) -> Result<Vec<Relay>> {
    Ok(vec![
        Relay::new("mev-blocker", "https://rpc.mevblocker.io/fast").await?,
        Relay::new("flashbots", "https://rpc.flashbots.net/fast").await?,
        Relay::new("public", public_fallback).await?,
    ])
}
