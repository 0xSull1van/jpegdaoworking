use thiserror::Error;

#[derive(Debug, Error)]
pub enum MinerError {
    #[error("io: {0}")]                     Io(#[from] std::io::Error),
    #[error("config: {0}")]                 Config(String),
    #[error("rpc: {0}")]                    Rpc(String),
    #[error("contract: {0}")]               Contract(String),
    #[error("keystore: {0}")]               Keystore(String),
    #[error("gpu: {0}")]                    Gpu(String),
    #[error("tx: {0}")]                     Tx(String),
    #[error("revert: {0}")]                 Revert(String),
    #[error("genesis not yet complete")]    GenesisNotComplete,
    #[error("wrong chain id, expected 1")]  WrongChain,
    #[error("alloy: {0}")]                  Alloy(String),
}

pub type Result<T> = std::result::Result<T, MinerError>;

pub const EXIT_OK: i32 = 0;
pub const EXIT_GENERIC: i32 = 1;
pub const EXIT_GPU_FATAL: i32 = 2;
pub const EXIT_KEYSTORE: i32 = 3;
pub const EXIT_WRONG_CHAIN: i32 = 4;
