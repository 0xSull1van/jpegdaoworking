use clap::{Parser, Subcommand};
use hashminer::chain::{ChainSource, watcher::ChainWatcher};
use hashminer::config::Config;
use hashminer::gpu::Grinder;
use hashminer::metrics::{Event, MetricsBus};
use hashminer::rpc;
use hashminer::tx::{Submitter, relay::default_relays, submitter::TxSubmitter, ev_gate::EvParams};
use hashminer::wallet::{MinerSigner, keystore::{create_keystore, read_password_from_env_or_prompt, unlock}};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "hashminer", version, about = "Native CUDA miner for hash256 on-chain PoW token")]
struct Cli {
    /// Path to TOML config file
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the miner with the configured wallet against mainnet (or wherever the RPC points).
    Run,
    /// Demo: subscribe to challenge updates and print them. No mining, no submitting.
    ChainWatch {
        /// WebSocket or HTTP RPC endpoint URL
        #[arg(long, env = "HASHMINER_RPC")]
        rpc: String,
        /// Miner address to watch (used for challenge derivation)
        #[arg(long, default_value = "0x0000000000000000000000000000000000000001")]
        miner: String,
    },
    /// Import a hex private key and write an encrypted v3 keystore JSON file.
    ImportKey {
        /// Directory where the keystore JSON will be written.
        #[arg(long)]
        out: PathBuf,
        /// Optional fixed filename (otherwise a UUID is generated).
        #[arg(long)]
        name: Option<String>,
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
    match cli.cmd.unwrap_or(Cmd::Run) {
        Cmd::Run => run(cli.config).await,
        Cmd::ChainWatch { rpc, miner } => chain_watch(&rpc, miner.parse()?).await,
        Cmd::ImportKey { out, name } => import_key(out, name.as_deref()).await,
    }
}

async fn chain_watch(rpc_url: &str, miner: alloy::primitives::Address) -> anyhow::Result<()> {
    let provider = rpc::connect(rpc_url).await?;
    let watcher = ChainWatcher::start(provider, miner).await?;
    let mut rx = watcher.subscribe(miner);
    loop {
        rx.changed().await?;
        let u = rx.borrow().clone();
        println!(
            "epoch={} block={} diff={:#x} challenge={}",
            u.epoch, u.block_number, u.target, u.challenge
        );
    }
}

async fn import_key(out: PathBuf, name: Option<&str>) -> anyhow::Result<()> {
    use zeroize::Zeroizing;

    let pk_hex = Zeroizing::new(
        rpassword::prompt_password("Private key (0x prefix optional, hidden): ")?,
    );
    let trimmed = pk_hex.trim_start_matches("0x").trim().to_owned();
    let bytes_vec =
        hex::decode(&trimmed).map_err(|e| anyhow::anyhow!("invalid hex: {e}"))?;
    if bytes_vec.len() != 32 {
        anyhow::bail!("private key must be 32 bytes (got {})", bytes_vec.len());
    }
    let mut key = Zeroizing::new([0u8; 32]);
    key.copy_from_slice(&bytes_vec);
    drop(bytes_vec);

    let pw1 = rpassword::prompt_password("Encryption password: ")?;
    let pw2 = rpassword::prompt_password("Confirm password: ")?;
    if pw1 != pw2 {
        anyhow::bail!("passwords do not match");
    }
    let pw = Zeroizing::new(pw1);
    // pw2 is a plain String; drop it promptly to limit its lifetime in memory.
    drop(pw2);

    let path = create_keystore(&out, &key, &pw, name)?;
    let signer = MinerSigner::from_key(key)?;
    println!("Keystore written: {}", path.display());
    println!("Derived address:  {}", signer.address());
    println!("Save the password somewhere safe — without it the keystore is unrecoverable.");
    Ok(())
}

async fn run(config_path: PathBuf) -> anyhow::Result<()> {
    let cfg = Config::load(&config_path)
        .map_err(|e| anyhow::anyhow!("failed to load config {:?}: {}", config_path, e))?;

    let cancel = CancellationToken::new();
    let (metrics, metrics_rx) = MetricsBus::channel(1024);

    // Wallet.
    let pw = read_password_from_env_or_prompt()?;
    let key = unlock(&cfg.wallet.keystore_path, &pw)?;
    let signer = Arc::new(MinerSigner::from_key(key)?);
    tracing::info!(addr = %signer.address(), "miner address");

    // Chain (read provider).
    let read_url = cfg.chain.read_rpc_ws.clone()
        .or_else(|| cfg.chain.read_rpc_http.first().cloned())
        .ok_or_else(|| anyhow::anyhow!("config: at least one read RPC URL required"))?;
    let provider = rpc::connect(&read_url).await?;
    let watcher = Arc::new(ChainWatcher::start(provider.clone(), signer.address()).await?);

    // Pre-flight checks.
    if !watcher.genesis_complete().await? {
        anyhow::bail!("genesis not complete; cannot mine yet");
    }

    // Relays.
    let relays = default_relays(&cfg.relays.public_fallback).await?;
    let ev = EvParams {
        max_mints_per_block: 10,
        min_ratio: cfg.mining.ev_min_ratio,
    };
    let submitter = Arc::new(TxSubmitter::new(
        watcher.clone(),
        signer.clone(),
        provider.clone(),
        relays,
        ev,
        cfg.mining.confirmations,
    ));

    // GPU grinder. Behind a feature flag — use FakeGrinder when feature off.
    #[cfg(feature = "cuda-runtime")]
    let grinder: Arc<dyn Grinder> = Arc::new(
        hashminer::gpu::worker::GpuWorker::start(
            cfg.gpu.device_id,
            cfg.gpu.blocks_per_sm * cfg.gpu.sm_count,
            cfg.gpu.threads_per_block,
            cfg.gpu.poll_interval_ms,
        )
        .await?,
    );
    #[cfg(not(feature = "cuda-runtime"))]
    let grinder: Arc<dyn Grinder> = Arc::new(hashminer::gpu::fake::FakeGrinder::new());

    // Wire challenge updates → grinder hot-swap.
    {
        let grinder = grinder.clone();
        let mut rx = watcher.subscribe(signer.address());
        let metrics = metrics.clone();
        tokio::spawn(async move {
            loop {
                if rx.changed().await.is_err() {
                    break;
                }
                let u = rx.borrow().clone();
                let _ = grinder.hot_swap(u.challenge, u.target, u.epoch).await;
                metrics.emit(Event::EpochSwap {
                    epoch: u.epoch,
                    block: u.block_number,
                    diff: format!("{:#x}", u.target),
                    challenge: format!("{}", u.challenge),
                    latency_ms: 0,
                });
            }
        });
    }

    // Wire hits → submitter.
    {
        let submitter = submitter.clone();
        let metrics = metrics.clone();
        let mut hit_rx = grinder.take_hit_rx();
        tokio::spawn(async move {
            while let Some(hit) = hit_rx.recv().await {
                metrics.emit(Event::HitFound {
                    epoch: hit.epoch_id,
                    nonce: format!("{}", hit.nonce),
                });
                match submitter.submit(hit).await {
                    Ok(out) => match out {
                        hashminer::tx::SubmitOutcome::Included {
                            tx,
                            block,
                            reward_wei,
                            relay: _,
                        } => metrics.emit(Event::TxIncluded {
                            tx,
                            block,
                            reward: format!("{}", reward_wei),
                        }),
                        hashminer::tx::SubmitOutcome::Reverted { tx, reason } => {
                            metrics.emit(Event::TxReverted { tx, reason })
                        }
                        hashminer::tx::SubmitOutcome::Dropped { reason } => {
                            metrics.emit(Event::HitDropped { reason })
                        }
                    },
                    Err(e) => metrics.emit(Event::HitDropped {
                        reason: format!("{e}"),
                    }),
                }
            }
        });
    }

    // Metrics → JSONL.
    {
        let path = cfg.metrics.jsonl_path.clone();
        tokio::spawn(async move {
            if let Err(e) = hashminer::metrics::jsonl::run_appender(path, metrics_rx).await {
                tracing::error!(error = %e, "metrics appender failed");
            }
        });
    }

    // Signal handling.
    let cancel2 = cancel.clone();
    tokio::spawn(async move {
        let _ = signal::ctrl_c().await;
        tracing::info!("ctrl+c received, shutting down");
        cancel2.cancel();
    });

    cancel.cancelled().await;
    grinder.shutdown().await;
    Ok(())
}
