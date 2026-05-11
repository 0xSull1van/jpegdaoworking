use crate::error::{MinerError, Result};
use alloy::primitives::{Address, B256};
use alloy::signers::local::PrivateKeySigner;
use zeroize::Zeroizing;

pub struct MinerSigner {
    inner: PrivateKeySigner,
    address: Address,
}

impl MinerSigner {
    pub fn from_key(key: Zeroizing<[u8; 32]>) -> Result<Self> {
        let signer = PrivateKeySigner::from_bytes(&B256::from(*key))
            .map_err(|e| MinerError::Keystore(format!("invalid privkey: {e}")))?;
        let address = signer.address();
        Ok(Self { inner: signer, address })
    }

    pub fn address(&self) -> Address {
        self.address
    }

    pub fn signer(&self) -> &PrivateKeySigner {
        &self.inner
    }
}
