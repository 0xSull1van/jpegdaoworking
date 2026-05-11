use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

#[derive(Default)]
pub struct Stats {
    pub hashrate: f64,
    pub era: u64,
    pub epoch: u64,
    pub diff: String,
    pub hits: u64,
    pub tx: u64,
    pub wins: u64,
    pub balance: String,
}

pub fn run_stdout_loop(stats: Arc<RwLock<Stats>>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            let s = stats.read();
            info!(target: "hashminer",
                hashrate=%format_hps(s.hashrate), diff=%s.diff, era=s.era, epoch=s.epoch,
                hits=s.hits, tx=s.tx, wins=s.wins, balance=%s.balance,
                "tick");
        }
    })
}

fn format_hps(h: f64) -> String {
    if h > 1e9 {
        format!("{:.2}GH/s", h / 1e9)
    } else if h > 1e6 {
        format!("{:.2}MH/s", h / 1e6)
    } else if h > 1e3 {
        format!("{:.2}kH/s", h / 1e3)
    } else {
        format!("{:.0}H/s", h)
    }
}
