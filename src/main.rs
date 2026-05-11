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
    Run {
        /// Test mode: override the real on-chain target with this hex value.
        /// Use this to force fast hits and verify the full submit pipeline
        /// (sign → relay → tx → receipt) without waiting hours for a real hit.
        /// On-chain the tx will revert with `InsufficientWork` (expected — your
        /// real hash won't satisfy the actual difficulty). Cost ≈ $0.5 in gas
        /// per test tx. Example: --test-target 0x000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        #[arg(long)]
        test_target: Option<String>,
    },
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
    match cli.cmd.unwrap_or(Cmd::Run { test_target: None }) {
        Cmd::Run { test_target } => {
            let parsed = match test_target {
                None => None,
                Some(s) => {
                    let cleaned = s.trim_start_matches("0x");
                    let v = alloy::primitives::U256::from_str_radix(cleaned, 16)
                        .map_err(|e| anyhow::anyhow!("--test-target hex parse: {e}"))?;
                    tracing::warn!(
                        "TEST MODE: overriding on-chain target with {:#x}. Submitted txs will revert with InsufficientWork.",
                        v
                    );
                    Some(v)
                }
            };
            run(cli.config, parsed).await
        }
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

async fn run(
    config_path: PathBuf,
    test_target_override: Option<alloy::primitives::U256>,
) -> anyhow::Result<()> {
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
            // CRITICAL: tokio::watch does NOT notify subscribers about the initial channel
            // value — only about subsequent send()s. The watcher publishes the initial
            // challenge at creation time (before main subscribes); without an explicit
            // read of borrow() here the kernel would run with target=0 (no hits possible)
            // until the next epoch change, ~20 minutes away.
            let initial = rx.borrow_and_update().clone();
            let effective_target = test_target_override.unwrap_or(initial.target);
            tracing::info!(
                epoch = initial.epoch,
                block = initial.block_number,
                challenge = %initial.challenge,
                "initial challenge → grinder"
            );
            let _ = grinder
                .hot_swap(initial.challenge, effective_target, initial.epoch)
                .await;
            metrics.emit(Event::EpochSwap {
                epoch: initial.epoch,
                block: initial.block_number,
                diff: format!("{:#x}", effective_target),
                challenge: format!("{}", initial.challenge),
                latency_ms: 0,
            });

            loop {
                if rx.changed().await.is_err() {
                    break;
                }
                let u = rx.borrow_and_update().clone();
                let effective_target = test_target_override.unwrap_or(u.target);
                let _ = grinder.hot_swap(u.challenge, effective_target, u.epoch).await;
                metrics.emit(Event::EpochSwap {
                    epoch: u.epoch,
                    block: u.block_number,
                    diff: format!("{:#x}", effective_target),
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

    // Heartbeat: log hashrate + chain state every 5 seconds so the operator can see
    // the miner is alive between rare epoch-swap / hit events.
    {
        let grinder = grinder.clone();
        let watcher = watcher.clone();
        let metrics = metrics.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
            // First tick fires immediately; skip it so the very first line isn't 0 H/s.
            interval.tick().await;
            loop {
                interval.tick().await;
                let hps = grinder.hashrate();
                let hps_str = if hps > 1e9 {
                    format!("{:.2} GH/s", hps / 1e9)
                } else if hps > 1e6 {
                    format!("{:.2} MH/s", hps / 1e6)
                } else if hps > 1e3 {
                    format!("{:.2} kH/s", hps / 1e3)
                } else {
                    format!("{:.0} H/s", hps)
                };
                match watcher.mining_state().await {
                    Ok(st) => {
                        tracing::info!(
                            hashrate = %hps_str,
                            era = st.era,
                            epoch = st.epoch,
                            blocks_left = st.epoch_blocks_left,
                            "heartbeat"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(hashrate = %hps_str, error = %e, "heartbeat (no chain state)");
                    }
                }
                metrics.emit(Event::Hashrate { hashrate_hps: hps });
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
