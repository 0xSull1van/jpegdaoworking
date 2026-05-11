use thiserror::Error;

#[derive(Debug, Error)]
pub enum MinerError {
    #[error("io: {0}")] Io(#[from] std::io::Error),
    #[error("config: {0}")] Config(String),
}

pub type Result<T> = std::result::Result<T, MinerError>;
