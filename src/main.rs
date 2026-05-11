use clap::{Parser, Subcommand};
use hashminer::chain::watcher::ChainWatcher;
use hashminer::chain::ChainSource;
use hashminer::rpc;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "hashminer", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Demo: print miningState every 4s and log epoch swaps.
    ChainWatch {
        #[arg(long, env = "HASHMINER_RPC")]
        rpc: String,
        #[arg(
            long,
            default_value = "0x0000000000000000000000000000000000000001"
        )]
        miner: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,hashminer=info")),
        )
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::ChainWatch { rpc, miner } => {
            let miner_addr: alloy::primitives::Address = miner.parse()?;
            let provider = rpc::connect(&rpc).await?;
            let watcher = ChainWatcher::start(provider, miner_addr).await?;
            let mut rx = watcher.subscribe(miner_addr);
            loop {
                rx.changed().await?;
                let u = rx.borrow().clone();
                println!(
                    "epoch={} block={} diff={:#x} challenge={}",
                    u.epoch, u.block_number, u.target, u.challenge
                );
            }
        }
    }
}
