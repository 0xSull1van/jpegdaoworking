use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub chain: ChainCfg,
    pub relays: RelayCfg,
    pub wallet: WalletCfg,
    pub mining: MiningCfg,
    pub gpu: GpuCfg,
    pub metrics: MetricsCfg,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChainCfg {
    pub read_rpc_ws: Option<String>,
    #[serde(default)]
    pub read_rpc_http: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RelayCfg {
    pub private: Vec<String>,
    pub public_fallback: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WalletCfg {
    pub keystore_path: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MiningCfg {
    pub max_tip_gwei: f64,
    pub ev_min_ratio: f64,
    pub confirmations: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GpuCfg {
    pub device_id: u32,
    pub threads_per_block: u32,
    pub blocks_per_sm: u32,
    pub batch_per_thread: u32,
    pub poll_interval_ms: u64,
    #[serde(default = "default_sms")]
    pub sm_count: u32,
}

fn default_sms() -> u32 {
    144 // RTX 4090
}

#[derive(Debug, Deserialize, Clone)]
pub struct MetricsCfg {
    pub jsonl_path: PathBuf,
    pub stdout_hz: u32,
}

impl Config {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&s)?)
    }
}
