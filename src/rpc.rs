//! RPC provider construction — type-erased via `BoxTransport` so callers
//! don't need to parameterise over the transport kind.

use alloy::providers::{ProviderBuilder, RootProvider};
use alloy::transports::BoxTransport;
use crate::error::{MinerError, Result};

/// A type-erased Ethereum read provider backed by `BoxTransport`.
///
/// `on_builtin` understands `http://`, `https://`, `ws://`, and `wss://`
/// URLs and returns a `RootProvider<BoxTransport>` in all cases, making
/// this the simplest uniform interface without an enum.
pub type ReadProvider = RootProvider<BoxTransport>;

/// Connect to an Ethereum node at `url`.
///
/// Supports `http://`, `https://`, `ws://`, and `wss://` schemes.
pub async fn connect(url: &str) -> Result<ReadProvider> {
    ProviderBuilder::new()
        .on_builtin(url)
        .await
        .map_err(|e| MinerError::Rpc(e.to_string()))
}
