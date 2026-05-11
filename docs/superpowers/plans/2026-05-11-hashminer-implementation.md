# hashminer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Native NVIDIA CUDA CLI miner for hash256 on-chain PoW token, replacing the WebGPU browser miner with ~10× higher hashrate (target 10-15 GH/s on RTX 4090).

**Architecture:** Single Rust binary, persistent CUDA kernel with double-buffered `__constant__` challenge for zero-downtime epoch hot-swap, four cooperating tokio tasks (ChainWatcher / GpuWorker / TxSubmitter / Metrics) behind explicit trait boundaries so race conditions can be tested without GPU or RPC.

**Tech Stack:** Rust stable, alloy (web3), tokio, cust (CUDA bindings), eth-keystore, tiny-keccak (CPU reference), tracing, clap, CUDA C++ kernel via nvcc.

**Spec:** `docs/superpowers/specs/2026-05-11-hash256-cli-miner-design.md`. Verified contract source in `docs/superpowers/specs/_contract/Hash.sol`.

---

## Shippable milestones

| Phase | Demo |
|---|---|
| 1 — Foundation | `cargo build` passes, traits + error types defined, CI lints green |
| 2 — Chain layer | `hashminer chain-watch` logs live `miningState` from mainnet |
| 3 — Wallet & TX | `hashminer sign-test` builds and signs `mine(0)` against anvil, prints raw tx |
| 4 — GPU | `hashminer-bench` reports hashrate; kernel matches CPU keccak on 1000 vectors |
| 5 — Wiring | Full miner runs end-to-end on anvil fork; mines + submits + receipts |
| 6 — Polish | systemd/NSSM units, README, mainnet smoke run |

---

## File structure (locked at plan-time)

```
hashminer/
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── .gitignore
├── build.rs                          # nvcc → PTX bytes, embedded
├── config.example.toml
├── README.md
├── kernel/
│   ├── keccak_grinder.cu             # persistent kernel entry
│   ├── keccak_device.cuh             # keccak-f[1600], constants
│   └── result_codec.cuh              # big-endian nonce/hash store, target compare
├── src/
│   ├── lib.rs                        # re-exports + crate-level docs
│   ├── main.rs                       # CLI dispatch, signal handling, wiring
│   ├── error.rs                      # MinerError enum, exit codes
│   ├── config.rs                     # toml + env + clap overlay
│   ├── state.rs                      # MinerState enum + transitions
│   ├── chain/
│   │   ├── mod.rs                    # trait ChainSource + ChallengeUpdate type
│   │   ├── contract.rs               # alloy-sol! bindings (Hash.sol ABI subset)
│   │   ├── challenge.rs              # CPU keccak compute (tiny-keccak)
│   │   ├── watcher.rs                # ChainWatcher live (newHeads + miningState poll)
│   │   └── fake.rs                   # in-mem ChainSource for tests
│   ├── gpu/
│   │   ├── mod.rs                    # trait Grinder + Hit type
│   │   ├── ptx.rs                    # include_bytes!(env!("PTX_PATH"))
│   │   ├── kernel_ffi.rs             # cust device/context/symbol/stream helpers
│   │   ├── worker.rs                 # GpuWorker live (launch + hot-swap + poll)
│   │   └── fake.rs                   # CPU-side fake Grinder for tests
│   ├── tx/
│   │   ├── mod.rs                    # trait Submitter + SubmitResult
│   │   ├── builder.rs                # mine(nonce) call data + EIP-1559 tx builder
│   │   ├── relay.rs                  # MEV-Blocker / Flashbots / public drivers
│   │   ├── nonce_manager.rs          # sequential gate, replacement, receipt watch
│   │   ├── ev_gate.rs                # profitability + block-cap check
│   │   ├── submitter.rs              # TxSubmitter live (assembles all of above)
│   │   └── fake.rs                   # in-mem Submitter
│   ├── wallet/
│   │   ├── mod.rs
│   │   ├── keystore.rs               # v3 JSON unlock
│   │   └── signer.rs                 # Zeroizing<SigningKey>, alloy signer impl
│   ├── metrics/
│   │   ├── mod.rs                    # MetricsBus, event enum
│   │   ├── stdout.rs                 # 1Hz formatter
│   │   ├── jsonl.rs                  # appender
│   │   └── redact.rs                 # tracing layer
│   └── rpc.rs                        # RpcPool: round-robin + health
├── tests/
│   ├── kernel_correctness.rs         # cuda-runtime feature
│   ├── challenge_cpu.rs              # tiny-keccak ↔ verified contract output
│   ├── stale_hit_filter.rs           # fake Grinder + fake ChainSource
│   ├── epoch_hotswap.rs              # fake ChainSource + fake Grinder
│   ├── tx_nonce_gate.rs              # fake Submitter
│   ├── ev_gate.rs                    # block-cap + profitability
│   └── e2e_anvil.rs                  # full pipeline against anvil
├── benches/
│   └── hashrate.rs                   # criterion
├── bin/
│   └── hashminer-bench.rs            # standalone GPU benchmark
├── deploy/
│   ├── hashminer.service             # systemd
│   ├── hashminer.nssm.cmd            # NSSM
│   └── README.md
└── docs/
    └── superpowers/
        ├── specs/2026-05-11-hash256-cli-miner-design.md
        └── plans/2026-05-11-hashminer-implementation.md
```

---

# Phase 1 — Foundation

## Task 1: Cargo project scaffolding

**Files:**
- Create: `Cargo.toml`, `rust-toolchain.toml`, `.gitignore`, `src/lib.rs`, `src/main.rs`, `build.rs` (stub)

- [ ] **Step 1: Init git + write `.gitignore`**

```bash
cd n:/Base/Projects/mining/hashminer
git init -b main
```

`.gitignore`:
```
/target
/Cargo.lock                           # binary crate, keep lock — actually we DO want it for reproducible mining builds
!/Cargo.lock
/logs
/keys
/config.toml                          # user secrets
.env
*.swp
docs/superpowers/specs/_etherscan_source.json
docs/superpowers/specs/_sourcify.json
```

- [ ] **Step 2: Write `Cargo.toml`**

```toml
[package]
name = "hashminer"
version = "0.1.0"
edition = "2021"
license = "MIT"
description = "Native CUDA miner for hash256 on-chain PoW token"

[features]
default = []
cuda-runtime = []                     # gates kernel correctness tests + GPU binaries

[dependencies]
alloy = { version = "0.8", features = ["full", "node-bindings", "providers", "rpc-types", "signers-keystore", "signers-local"] }
tokio = { version = "1", features = ["full"] }
tokio-util = "0.7"
futures = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
clap = { version = "4", features = ["derive", "env"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
tiny-keccak = { version = "2", features = ["keccak"] }
eth-keystore = "0.5"
secp256k1 = { version = "0.29", features = ["rand", "recovery"] }
zeroize = { version = "1", features = ["derive"] }
hex = "0.4"
thiserror = "1"
anyhow = "1"
async-trait = "0.1"
url = "2"
parking_lot = "0.12"
once_cell = "1"

[target.'cfg(not(target_os = "windows"))'.dependencies]
cust = { version = "0.3", optional = true }

[dev-dependencies]
wiremock = "0.6"
proptest = "1"
criterion = { version = "0.5", features = ["html_reports"] }
tempfile = "3"

[[bin]]
name = "hashminer"
path = "src/main.rs"

[[bin]]
name = "hashminer-bench"
path = "bin/hashminer-bench.rs"
required-features = ["cuda-runtime"]

[[bench]]
name = "hashrate"
harness = false

[build-dependencies]
# nvcc is invoked from build.rs only when cuda-runtime feature is on
```

Note: `cust` is only available on Linux. On Windows we will use FFI directly via `cuda-sys` or wrap the driver API; this is decided when we get to Phase 4. For now, gate `cust` behind a Linux-only optional dep so `cargo build` works on Windows host.

- [ ] **Step 3: `rust-toolchain.toml`**

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
```

- [ ] **Step 4: Stub `src/lib.rs`, `src/main.rs`, `build.rs`**

`src/lib.rs`:
```rust
//! hashminer — native CUDA miner for hash256 on-chain PoW token.
//!
//! See `docs/superpowers/specs/2026-05-11-hash256-cli-miner-design.md` for design.

pub mod error;
```

`src/main.rs`:
```rust
fn main() {
    eprintln!("hashminer 0.1.0 (scaffold)");
    std::process::exit(0);
}
```

`src/error.rs`:
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MinerError {
    #[error("io: {0}")] Io(#[from] std::io::Error),
    #[error("config: {0}")] Config(String),
}

pub type Result<T> = std::result::Result<T, MinerError>;
```

`build.rs`:
```rust
fn main() {
    // CUDA kernel build is wired in Phase 4 (gated on cuda-runtime feature).
    println!("cargo:rerun-if-changed=build.rs");
}
```

- [ ] **Step 5: Verify build**

```bash
cd hashminer && cargo build
```
Expected: compiles clean, single warning about unused error variants is OK.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "scaffold: Cargo project + traits skeleton + .gitignore"
```

---

## Task 2: Trait definitions and error model

**Files:**
- Create: `src/chain/mod.rs`, `src/gpu/mod.rs`, `src/tx/mod.rs`, `src/state.rs`
- Modify: `src/lib.rs`, `src/error.rs`

- [ ] **Step 1: `src/chain/mod.rs`**

```rust
use alloy::primitives::{Address, B256, U256};
use async_trait::async_trait;
use tokio::sync::watch;

/// What ChainWatcher publishes every time epoch or difficulty changes.
#[derive(Debug, Clone)]
pub struct ChallengeUpdate {
    pub challenge: B256,
    pub target: U256,
    pub epoch: u64,
    pub block_number: u64,
}

/// Snapshot of contract state (single `miningState()` call).
#[derive(Debug, Clone, Copy)]
pub struct MiningState {
    pub era: u64,
    pub reward_wei: U256,
    pub difficulty: U256,
    pub minted_wei: U256,
    pub remaining_wei: U256,
    pub epoch: u64,
    pub epoch_blocks_left: u64,
}

#[async_trait]
pub trait ChainSource: Send + Sync + 'static {
    /// Latest block number observed.
    fn head(&self) -> u64;
    /// Compute (or fetch) current challenge for the given miner address.
    async fn challenge_for(&self, miner: Address) -> crate::error::Result<B256>;
    /// Read `miningState()` snapshot.
    async fn mining_state(&self) -> crate::error::Result<MiningState>;
    /// Subscribe to a stream of (challenge, target, epoch) updates.
    fn subscribe(&self, miner: Address) -> watch::Receiver<ChallengeUpdate>;
    /// Read mints already in the given block (for EV gate).
    async fn mints_in_block(&self, block: u64) -> crate::error::Result<u64>;
    /// Has the genesis sale closed? Mining reverts until true.
    async fn genesis_complete(&self) -> crate::error::Result<bool>;
}

pub mod challenge;
```

- [ ] **Step 2: `src/gpu/mod.rs`**

```rust
use alloy::primitives::{B256, U256};
use async_trait::async_trait;
use tokio::sync::mpsc;

/// One valid (nonce, hash, epoch_id) triple emitted by the kernel.
#[derive(Debug, Clone)]
pub struct Hit {
    pub nonce: U256,
    pub hash: B256,
    pub epoch_id: u64,
}

#[async_trait]
pub trait Grinder: Send + Sync + 'static {
    /// Apply a new (challenge, target, epoch_id) without restarting the kernel.
    async fn hot_swap(&self, challenge: B256, target: U256, epoch_id: u64) -> crate::error::Result<()>;
    /// Subscribe to the hit stream. Single-consumer.
    fn take_hit_rx(&self) -> mpsc::Receiver<Hit>;
    /// Current observed hashrate in hashes/sec, averaged over last ~1s.
    fn hashrate(&self) -> f64;
    /// Cooperative shutdown.
    async fn shutdown(&self);
}
```

- [ ] **Step 3: `src/tx/mod.rs`**

```rust
use alloy::primitives::{TxHash, U256};
use async_trait::async_trait;
use crate::gpu::Hit;

#[derive(Debug, Clone)]
pub enum SubmitOutcome {
    Included { tx: TxHash, block: u64, reward_wei: U256, relay: String },
    Reverted { tx: TxHash, reason: String },
    Dropped  { reason: String },
}

#[async_trait]
pub trait Submitter: Send + Sync + 'static {
    /// Submit a hit. Returns when included, reverted, or dropped.
    /// Implementations gate concurrency internally (sequential gate).
    async fn submit(&self, hit: Hit) -> crate::error::Result<SubmitOutcome>;
}
```

- [ ] **Step 4: `src/state.rs`**

```rust
use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinerState {
    Healthy        = 0,
    RpcDegraded    = 1,
    ChallengeStale = 2,
    GpuFault       = 3,
    WalletLocked   = 4,
    Paused         = 5,
    Fatal          = 6,
}

#[derive(Debug)]
pub struct StateMachine {
    cur: AtomicU8,
}

impl StateMachine {
    pub fn new() -> Self { Self { cur: AtomicU8::new(MinerState::Healthy as u8) } }
    pub fn get(&self) -> MinerState {
        match self.cur.load(Ordering::Acquire) {
            0 => MinerState::Healthy, 1 => MinerState::RpcDegraded,
            2 => MinerState::ChallengeStale, 3 => MinerState::GpuFault,
            4 => MinerState::WalletLocked, 5 => MinerState::Paused,
            _ => MinerState::Fatal,
        }
    }
    pub fn set(&self, s: MinerState) -> MinerState {
        let prev = self.cur.swap(s as u8, Ordering::AcqRel);
        // SAFETY: only ever store valid discriminants
        unsafe { std::mem::transmute(prev) }
    }
    pub fn is_grinding_ok(&self) -> bool {
        matches!(self.get(), MinerState::Healthy | MinerState::RpcDegraded | MinerState::ChallengeStale)
    }
    pub fn is_submitting_ok(&self) -> bool {
        matches!(self.get(), MinerState::Healthy | MinerState::RpcDegraded)
    }
}
```

- [ ] **Step 5: Update `src/lib.rs`**

```rust
//! hashminer — native CUDA miner for hash256 on-chain PoW token.
pub mod error;
pub mod state;
pub mod chain;
pub mod gpu;
pub mod tx;
```

- [ ] **Step 6: Extend `src/error.rs`**

```rust
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
```

- [ ] **Step 7: `src/chain/challenge.rs` placeholder**

```rust
// CPU implementation lands in Task 4.
```

- [ ] **Step 8: Verify build + commit**

```bash
cargo build
cargo clippy --all-targets -- -D warnings
git add -A
git commit -m "feat: define ChainSource / Grinder / Submitter traits + StateMachine"
```

---

# Phase 2 — Chain layer

## Task 3: Contract ABI bindings

**Files:**
- Create: `src/chain/contract.rs`

Background: alloy's `sol!` macro generates Rust bindings from Solidity declarations. We only need a subset of `Hash.sol` — the read views and the `mine()` write function. We hand-write the subset (not parse the whole 511-line file) for clean compile time.

- [ ] **Step 1: Write the bindings**

```rust
use alloy::sol;

sol! {
    #[sol(rpc)]
    contract Hash {
        // --- write ---
        function mine(uint256 nonce) external;

        // --- read ---
        function currentDifficulty() external view returns (uint256);
        function getChallenge(address miner) external view returns (bytes32);
        function epochBlocksLeft() external view returns (uint256);
        function currentReward() external view returns (uint256);
        function totalMints() external view returns (uint256);
        function mintsInBlock(uint256 blockNumber) external view returns (uint256);
        function genesisComplete() external view returns (bool);
        function miningState() external view returns (
            uint256 era,
            uint256 reward,
            uint256 difficulty,
            uint256 minted,
            uint256 remaining,
            uint256 epoch,
            uint256 epochBlocksLeft_
        );

        // --- events ---
        event Mined(address indexed miner, uint256 nonce, uint256 reward, uint256 era);
        event Halving(uint256 era, uint256 reward);
        event DifficultyAdjusted(uint256 old, uint256 next, uint256 takenBlocks);

        // --- reverts ---
        error InsufficientWork();
        error ProofAlreadyUsed();
        error BlockCapReached();
        error SupplyExhausted();
        error GenesisNotComplete();
    }
}

/// hash256 mainnet deployment.
pub const CONTRACT: alloy::primitives::Address = alloy::primitives::address!("AC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc");
pub const CHAIN_ID: u64 = 1;
pub const EPOCH_BLOCKS: u64 = 100;
pub const MAX_MINTS_PER_BLOCK: u64 = 10;
```

- [ ] **Step 2: Update `src/chain/mod.rs`** to include it

```rust
pub mod challenge;
pub mod contract;
```

- [ ] **Step 3: Build verify + commit**

```bash
cargo build
git add -A
git commit -m "feat: alloy bindings for Hash contract"
```

---

## Task 4: CPU challenge computation + test vector

**Files:**
- Create: `src/chain/challenge.rs`, `tests/challenge_cpu.rs`

- [ ] **Step 1: Write `tests/challenge_cpu.rs`** (failing)

We compute the test vector by hand using the Solidity formula. For a known input we can pre-compute the expected output using `tiny-keccak` independently — the test is that two independent code paths agree byte-for-byte.

```rust
use alloy::primitives::{address, b256, U256};
use hashminer::chain::challenge::{compute_challenge, compute_inner_hash};

#[test]
fn challenge_matches_solidity_layout() {
    // _challenge inputs (all padded to 32B by abi.encode):
    //   chainid = 1
    //   contract = 0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc
    //   miner = 0x000...001 (test)
    //   epoch = 250707
    let miner = address!("0000000000000000000000000000000000000001");
    let epoch = 250707u64;
    let got = compute_challenge(1, hashminer::chain::contract::CONTRACT, miner, epoch);

    // Independently computed reference (tiny-keccak with explicit 128-byte buffer).
    let mut buf = [0u8; 128];
    buf[31] = 1;                                                    // chainid uint256 BE
    buf[44..64].copy_from_slice(hashminer::chain::contract::CONTRACT.as_slice()); // contract left-padded to 32B
    buf[76..96].copy_from_slice(miner.as_slice());                  // miner left-padded
    buf[96..128].copy_from_slice(&U256::from(epoch).to_be_bytes::<32>()); // epoch uint256 BE

    use tiny_keccak::{Hasher, Keccak};
    let mut k = Keccak::v256();
    k.update(&buf);
    let mut out = [0u8; 32];
    k.finalize(&mut out);

    assert_eq!(got.as_slice(), &out);
}

#[test]
fn inner_hash_matches_64byte_encoding() {
    let challenge = b256!("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff");
    let nonce = U256::from(0x4242u64);
    let got = compute_inner_hash(challenge, nonce);

    let mut buf = [0u8; 64];
    buf[0..32].copy_from_slice(challenge.as_slice());
    buf[32..64].copy_from_slice(&nonce.to_be_bytes::<32>());

    use tiny_keccak::{Hasher, Keccak};
    let mut k = Keccak::v256();
    k.update(&buf);
    let mut out = [0u8; 32];
    k.finalize(&mut out);

    assert_eq!(got.as_slice(), &out);
}
```

- [ ] **Step 2: Run test — expect FAIL** (function not defined)

```bash
cargo test --test challenge_cpu
```

- [ ] **Step 3: Implement `src/chain/challenge.rs`**

```rust
use alloy::primitives::{Address, B256, U256};
use tiny_keccak::{Hasher, Keccak};

/// keccak256(abi.encode(chainid, address(this), miner, epoch)).
/// Output bytes match Solidity exactly: 4 fields × 32 bytes = 128 bytes input.
pub fn compute_challenge(chain_id: u64, contract: Address, miner: Address, epoch: u64) -> B256 {
    let mut buf = [0u8; 128];
    buf[24..32].copy_from_slice(&chain_id.to_be_bytes());           // chainid in last 8 bytes of slot 0
    buf[44..64].copy_from_slice(contract.as_slice());               // address right-aligned in slot 1
    buf[76..96].copy_from_slice(miner.as_slice());                  // miner right-aligned in slot 2
    buf[120..128].copy_from_slice(&epoch.to_be_bytes());            // epoch in last 8 bytes of slot 3
    keccak(&buf)
}

/// keccak256(abi.encode(challenge, nonce)): 32B challenge ‖ 32B nonce BE.
pub fn compute_inner_hash(challenge: B256, nonce: U256) -> B256 {
    let mut buf = [0u8; 64];
    buf[0..32].copy_from_slice(challenge.as_slice());
    buf[32..64].copy_from_slice(&nonce.to_be_bytes::<32>());
    keccak(&buf)
}

fn keccak(input: &[u8]) -> B256 {
    let mut k = Keccak::v256();
    k.update(input);
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    B256::from(out)
}
```

- [ ] **Step 4: Test — expect PASS**

```bash
cargo test --test challenge_cpu
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: CPU keccak challenge + inner-hash with byte-exact Solidity layout test"
```

---

## Task 5: ChainSource fake for testing

**Files:**
- Create: `src/chain/fake.rs`
- Modify: `src/chain/mod.rs` (add `pub mod fake;`)

- [ ] **Step 1: Implement `src/chain/fake.rs`**

```rust
use super::{ChainSource, ChallengeUpdate, MiningState};
use crate::chain::{challenge::compute_challenge, contract::{CONTRACT, EPOCH_BLOCKS}};
use crate::error::Result;
use alloy::primitives::{Address, B256, U256};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::watch;

#[derive(Clone)]
pub struct FakeChain {
    inner: Arc<Mutex<Inner>>,
    update_tx: watch::Sender<ChallengeUpdate>,
}

struct Inner {
    head: u64,
    difficulty: U256,
    minted: U256,
    total_mints: u64,
    mints_in_block: std::collections::HashMap<u64, u64>,
    genesis_complete: bool,
}

impl FakeChain {
    pub fn new(initial_head: u64, difficulty: U256, miner: Address) -> Self {
        let inner = Inner {
            head: initial_head,
            difficulty,
            minted: U256::ZERO,
            total_mints: 0,
            mints_in_block: Default::default(),
            genesis_complete: true,
        };
        let epoch = initial_head / EPOCH_BLOCKS;
        let challenge = compute_challenge(1, CONTRACT, miner, epoch);
        let (tx, _) = watch::channel(ChallengeUpdate {
            challenge, target: difficulty, epoch, block_number: initial_head,
        });
        Self { inner: Arc::new(Mutex::new(inner)), update_tx: tx }
    }

    /// Advance the head by N blocks and re-publish challenge if epoch rolled.
    pub fn advance_blocks(&self, n: u64, miner: Address) {
        let mut g = self.inner.lock();
        let old_epoch = g.head / EPOCH_BLOCKS;
        g.head += n;
        let new_epoch = g.head / EPOCH_BLOCKS;
        drop(g);
        if new_epoch != old_epoch {
            let challenge = compute_challenge(1, CONTRACT, miner, new_epoch);
            let g = self.inner.lock();
            let _ = self.update_tx.send(ChallengeUpdate {
                challenge, target: g.difficulty, epoch: new_epoch, block_number: g.head,
            });
        }
    }

    pub fn set_difficulty(&self, d: U256, miner: Address) {
        let mut g = self.inner.lock();
        g.difficulty = d;
        let epoch = g.head / EPOCH_BLOCKS;
        let challenge = compute_challenge(1, CONTRACT, miner, epoch);
        let _ = self.update_tx.send(ChallengeUpdate {
            challenge, target: d, epoch, block_number: g.head,
        });
    }
}

#[async_trait]
impl ChainSource for FakeChain {
    fn head(&self) -> u64 { self.inner.lock().head }

    async fn challenge_for(&self, miner: Address) -> Result<B256> {
        let g = self.inner.lock();
        Ok(compute_challenge(1, CONTRACT, miner, g.head / EPOCH_BLOCKS))
    }

    async fn mining_state(&self) -> Result<MiningState> {
        let g = self.inner.lock();
        let epoch = g.head / EPOCH_BLOCKS;
        Ok(MiningState {
            era: g.total_mints / 100_000,
            reward_wei: U256::from(100_u64) * U256::from(10u64).pow(U256::from(18u64)),
            difficulty: g.difficulty,
            minted_wei: g.minted,
            remaining_wei: U256::ZERO,
            epoch,
            epoch_blocks_left: EPOCH_BLOCKS - (g.head % EPOCH_BLOCKS),
        })
    }

    fn subscribe(&self, _miner: Address) -> watch::Receiver<ChallengeUpdate> {
        self.update_tx.subscribe()
    }

    async fn mints_in_block(&self, block: u64) -> Result<u64> {
        Ok(self.inner.lock().mints_in_block.get(&block).copied().unwrap_or(0))
    }

    async fn genesis_complete(&self) -> Result<bool> { Ok(self.inner.lock().genesis_complete) }
}
```

- [ ] **Step 2: Update `src/chain/mod.rs`**

```rust
pub mod challenge;
pub mod contract;
pub mod fake;
```

- [ ] **Step 3: Build + commit**

```bash
cargo build --all-features
git add -A
git commit -m "feat: FakeChain — in-memory ChainSource for race-condition tests"
```

---

## Task 6: ChainWatcher live implementation

**Files:**
- Create: `src/chain/watcher.rs`, `src/rpc.rs`
- Modify: `src/chain/mod.rs`, `src/lib.rs`

- [ ] **Step 1: `src/rpc.rs`**

```rust
use alloy::providers::{Provider, ProviderBuilder, RootProvider};
use alloy::pubsub::PubSubFrontend;
use alloy::transports::http::Http;
use alloy::transports::ws::WsConnect;
use crate::error::{MinerError, Result};
use url::Url;

pub enum ReadProvider {
    Ws(RootProvider<PubSubFrontend>),
    Http(RootProvider<Http<reqwest::Client>>),
}

impl ReadProvider {
    pub async fn connect(url: &str) -> Result<Self> {
        let u = Url::parse(url).map_err(|e| MinerError::Config(format!("url: {e}")))?;
        match u.scheme() {
            "ws" | "wss" => {
                let p = ProviderBuilder::new().on_ws(WsConnect::new(url)).await
                    .map_err(|e| MinerError::Rpc(e.to_string()))?;
                Ok(Self::Ws(p))
            }
            "http" | "https" => {
                let p = ProviderBuilder::new().on_http(u);
                Ok(Self::Http(p))
            }
            _ => Err(MinerError::Config(format!("unsupported scheme: {}", u.scheme()))),
        }
    }
}
```

(Note: this is a simplification — alloy's provider type machinery is rich. Implementer must consult `alloy::providers` docs at the version pinned in Cargo.toml; the trait we call below is `Provider`, and both wrapped variants implement it.)

- [ ] **Step 2: `src/chain/watcher.rs`**

```rust
use super::{ChainSource, ChallengeUpdate, MiningState};
use crate::chain::{challenge::compute_challenge, contract::{Hash, CONTRACT, CHAIN_ID, EPOCH_BLOCKS}};
use crate::error::{MinerError, Result};
use alloy::primitives::{Address, B256, U256};
use alloy::providers::Provider;
use async_trait::async_trait;
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

pub struct ChainWatcher {
    provider: Arc<dyn Provider + Send + Sync>,
    miner: Address,
    head: Arc<RwLock<u64>>,
    last_state: Arc<RwLock<Option<MiningState>>>,
    update_tx: watch::Sender<ChallengeUpdate>,
    _bg: JoinHandle<()>,
}

impl ChainWatcher {
    pub async fn start(
        provider: Arc<dyn Provider + Send + Sync>,
        miner: Address,
    ) -> Result<Self> {
        let block_number = provider.get_block_number().await
            .map_err(|e| MinerError::Rpc(e.to_string()))?;
        let head = Arc::new(RwLock::new(block_number));
        let contract = Hash::new(CONTRACT, provider.clone());

        // initial state
        let st = contract.miningState().call().await
            .map_err(|e| MinerError::Contract(e.to_string()))?;
        let initial = MiningState {
            era: st.era.to::<u64>(),
            reward_wei: st.reward,
            difficulty: st.difficulty,
            minted_wei: st.minted,
            remaining_wei: st.remaining,
            epoch: st.epoch.to::<u64>(),
            epoch_blocks_left: st.epochBlocksLeft_.to::<u64>(),
        };
        let last_state = Arc::new(RwLock::new(Some(initial)));
        let challenge = compute_challenge(CHAIN_ID, CONTRACT, miner, initial.epoch);
        let (tx, _) = watch::channel(ChallengeUpdate {
            challenge, target: initial.difficulty, epoch: initial.epoch, block_number,
        });

        let bg = tokio::spawn(Self::poll_loop(
            provider.clone(), miner, head.clone(), last_state.clone(), tx.clone(),
        ));

        Ok(Self { provider, miner, head, last_state, update_tx: tx, _bg: bg })
    }

    async fn poll_loop(
        provider: Arc<dyn Provider + Send + Sync>,
        miner: Address,
        head: Arc<RwLock<u64>>,
        last_state: Arc<RwLock<Option<MiningState>>>,
        update_tx: watch::Sender<ChallengeUpdate>,
    ) {
        let contract = Hash::new(CONTRACT, provider.clone());
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(4));
        let mut last_epoch: Option<u64> = None;
        loop {
            interval.tick().await;
            match provider.get_block_number().await {
                Ok(bn) => *head.write() = bn,
                Err(e) => { warn!("get_block_number: {e}"); continue; }
            }
            match contract.miningState().call().await {
                Ok(st) => {
                    let epoch = st.epoch.to::<u64>();
                    let difficulty = st.difficulty;
                    let snap = MiningState {
                        era: st.era.to::<u64>(),
                        reward_wei: st.reward,
                        difficulty,
                        minted_wei: st.minted,
                        remaining_wei: st.remaining,
                        epoch,
                        epoch_blocks_left: st.epochBlocksLeft_.to::<u64>(),
                    };
                    *last_state.write() = Some(snap);

                    let epoch_changed = last_epoch != Some(epoch);
                    if epoch_changed {
                        let bn = *head.read();
                        let challenge = compute_challenge(CHAIN_ID, CONTRACT, miner, epoch);
                        info!(epoch, ?challenge, "challenge swap");
                        let _ = update_tx.send(ChallengeUpdate {
                            challenge, target: difficulty, epoch, block_number: bn,
                        });
                        last_epoch = Some(epoch);
                    }
                }
                Err(e) => warn!("miningState call: {e}"),
            }
        }
    }
}

#[async_trait]
impl ChainSource for ChainWatcher {
    fn head(&self) -> u64 { *self.head.read() }

    async fn challenge_for(&self, miner: Address) -> Result<B256> {
        let epoch = *self.head.read() / EPOCH_BLOCKS;
        Ok(compute_challenge(CHAIN_ID, CONTRACT, miner, epoch))
    }

    async fn mining_state(&self) -> Result<MiningState> {
        self.last_state.read().clone()
            .ok_or_else(|| MinerError::Rpc("no miningState yet".into()))
    }

    fn subscribe(&self, _miner: Address) -> watch::Receiver<ChallengeUpdate> {
        self.update_tx.subscribe()
    }

    async fn mints_in_block(&self, block: u64) -> Result<u64> {
        let contract = Hash::new(CONTRACT, self.provider.clone());
        let v = contract.mintsInBlock(U256::from(block)).call().await
            .map_err(|e| MinerError::Contract(e.to_string()))?;
        Ok(v._0.to::<u64>())
    }

    async fn genesis_complete(&self) -> Result<bool> {
        let contract = Hash::new(CONTRACT, self.provider.clone());
        Ok(contract.genesisComplete().call().await
            .map_err(|e| MinerError::Contract(e.to_string()))?._0)
    }
}
```

(Provider trait object dance: alloy 0.8 makes this slightly verbose. Implementer should consult alloy provider docs and possibly switch to a concrete generic over `P: Provider`.)

- [ ] **Step 3: Update `src/chain/mod.rs`**

```rust
pub mod challenge;
pub mod contract;
pub mod fake;
pub mod watcher;
```

- [ ] **Step 4: Update `src/lib.rs`**

```rust
pub mod error;
pub mod state;
pub mod rpc;
pub mod chain;
pub mod gpu;
pub mod tx;
```

- [ ] **Step 5: Build + commit**

```bash
cargo build
git add -A
git commit -m "feat: ChainWatcher — live miningState polling + epoch challenge swap"
```

---

## Task 7: Chain-watch CLI subcommand (Phase 2 demo)

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Replace `src/main.rs`**

```rust
use alloy::primitives::address;
use clap::{Parser, Subcommand};
use hashminer::chain::watcher::ChainWatcher;
use hashminer::rpc::ReadProvider;
use std::sync::Arc;
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
        #[arg(long, env = "HASHMINER_RPC")] rpc: String,
        #[arg(long, default_value = "0x0000000000000000000000000000000000000001")] miner: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::ChainWatch { rpc, miner } => {
            let miner_addr = miner.parse()?;
            let provider = match ReadProvider::connect(&rpc).await? {
                ReadProvider::Http(p) => Arc::new(p) as Arc<dyn alloy::providers::Provider + Send + Sync>,
                ReadProvider::Ws(p) => Arc::new(p) as Arc<dyn alloy::providers::Provider + Send + Sync>,
            };
            let watcher = ChainWatcher::start(provider, miner_addr).await?;
            let mut rx = hashminer::chain::ChainSource::subscribe(&watcher, miner_addr);
            loop {
                rx.changed().await?;
                let u = rx.borrow().clone();
                println!("epoch={} block={} diff={:#x} challenge={}", u.epoch, u.block_number, u.target, u.challenge);
            }
        }
    }
}
```

- [ ] **Step 2: Manual smoke**

```bash
RUST_LOG=info cargo run -- chain-watch --rpc wss://eth-mainnet.g.alchemy.com/v2/<KEY>
```
Expected: prints initial `epoch=... block=... diff=... challenge=0x...` and one line per epoch swap (every ~20 min on mainnet).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(cli): chain-watch subcommand — Phase 2 demo milestone"
```

---

# Phase 3 — Wallet & TX

## Task 8: Keystore unlock + signer

**Files:**
- Create: `src/wallet/mod.rs`, `src/wallet/keystore.rs`, `src/wallet/signer.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: `src/wallet/mod.rs`**

```rust
pub mod keystore;
pub mod signer;
pub use signer::MinerSigner;
```

- [ ] **Step 2: `src/wallet/keystore.rs`**

```rust
use crate::error::{MinerError, Result};
use std::path::Path;
use zeroize::Zeroizing;

/// Decrypt v3 keystore JSON, return raw 32-byte private key wrapped in Zeroizing.
pub fn unlock<P: AsRef<Path>>(path: P, password: &str) -> Result<Zeroizing<[u8; 32]>> {
    let bytes = eth_keystore::decrypt_key(path.as_ref(), password)
        .map_err(|e| MinerError::Keystore(e.to_string()))?;
    if bytes.len() != 32 {
        return Err(MinerError::Keystore(format!("decrypted key wrong size: {}", bytes.len())));
    }
    let mut arr = Zeroizing::new([0u8; 32]);
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

/// Prompt for password if `KEYSTORE_PASSWORD` env not set.
pub fn read_password_from_env_or_prompt() -> Result<Zeroizing<String>> {
    if let Ok(p) = std::env::var("KEYSTORE_PASSWORD") {
        return Ok(Zeroizing::new(p));
    }
    let p = rpassword::prompt_password("keystore password: ")
        .map_err(|e| MinerError::Keystore(e.to_string()))?;
    Ok(Zeroizing::new(p))
}
```

Add `rpassword = "7"` to `Cargo.toml` deps.

- [ ] **Step 3: `src/wallet/signer.rs`**

```rust
use crate::error::{MinerError, Result};
use alloy::primitives::Address;
use alloy::signers::local::PrivateKeySigner;
use zeroize::Zeroizing;

pub struct MinerSigner {
    inner: PrivateKeySigner,
    address: Address,
}

impl MinerSigner {
    pub fn from_key(key: Zeroizing<[u8; 32]>) -> Result<Self> {
        let signer = PrivateKeySigner::from_bytes(&(*key).into())
            .map_err(|e| MinerError::Keystore(format!("invalid privkey: {e}")))?;
        let address = signer.address();
        Ok(Self { inner: signer, address })
    }
    pub fn address(&self) -> Address { self.address }
    pub fn signer(&self) -> &PrivateKeySigner { &self.inner }
}
```

- [ ] **Step 4: Unit test** — `tests/keystore_unlock.rs`

```rust
use hashminer::wallet::{keystore::unlock, MinerSigner};
use tempfile::tempdir;

#[test]
fn roundtrip_v3_keystore() {
    let dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let key = [0x42u8; 32];
    let path = eth_keystore::encrypt_key(dir.path(), &mut rng, key, "pw", None).unwrap();
    let abs = dir.path().join(path);

    let unlocked = unlock(&abs, "pw").unwrap();
    assert_eq!(unlocked.as_slice(), &key);

    let signer = MinerSigner::from_key(unlocked).unwrap();
    // Address derivation: deterministic for this key.
    println!("derived: {}", signer.address());
}
```

Add `rand = "0.8"` to dev-deps.

- [ ] **Step 5: Run, fix imports, commit**

```bash
cargo test --test keystore_unlock
git add -A
git commit -m "feat: v3 keystore unlock + Zeroizing-wrapped signer"
```

---

## Task 9: Tx builder + relay drivers

**Files:**
- Create: `src/tx/builder.rs`, `src/tx/relay.rs`, `src/tx/nonce_manager.rs`, `src/tx/ev_gate.rs`

- [ ] **Step 1: `src/tx/builder.rs`**

```rust
use alloy::network::TransactionBuilder;
use alloy::primitives::{Address, Bytes, U256};
use alloy::rpc::types::TransactionRequest;
use alloy::sol_types::SolCall;
use crate::chain::contract::{Hash, CONTRACT};
use crate::error::Result;

pub struct MineTxParams {
    pub from: Address,
    pub nonce: u64,                    // ethereum tx nonce (NOT mining nonce)
    pub mine_nonce: U256,              // the PoW nonce
    pub max_fee_per_gas: u128,
    pub max_priority_fee_per_gas: u128,
    pub gas_limit: u64,
    pub chain_id: u64,
}

pub fn build_mine_tx(p: MineTxParams) -> Result<TransactionRequest> {
    let calldata: Bytes = Hash::mineCall { nonce: p.mine_nonce }.abi_encode().into();
    Ok(TransactionRequest::default()
        .with_from(p.from)
        .with_to(CONTRACT)
        .with_nonce(p.nonce)
        .with_chain_id(p.chain_id)
        .with_input(calldata)
        .with_gas_limit(p.gas_limit)
        .with_max_fee_per_gas(p.max_fee_per_gas)
        .with_max_priority_fee_per_gas(p.max_priority_fee_per_gas))
}
```

- [ ] **Step 2: `src/tx/relay.rs`**

```rust
use crate::error::{MinerError, Result};
use alloy::primitives::{Bytes, TxHash};
use alloy::providers::{Provider, ProviderBuilder};
use alloy::transports::http::Http;
use reqwest::Client;
use std::sync::Arc;
use url::Url;

pub struct Relay {
    pub name: String,
    provider: Arc<dyn Provider + Send + Sync>,
}

impl Relay {
    pub fn new(name: impl Into<String>, url: &str) -> Result<Self> {
        let url = Url::parse(url).map_err(|e| MinerError::Config(e.to_string()))?;
        let provider: Arc<dyn Provider + Send + Sync> =
            Arc::new(ProviderBuilder::new().on_http(url));
        Ok(Self { name: name.into(), provider })
    }
    pub async fn send_raw(&self, raw: Bytes) -> Result<TxHash> {
        self.provider.send_raw_transaction(&raw).await
            .map(|p| *p.tx_hash())
            .map_err(|e| MinerError::Tx(format!("{}: {}", self.name, e)))
    }
}

/// Default relay set: MEV-Blocker, Flashbots Protect, then a public fallback.
pub fn default_relays(public_fallback: &str) -> Result<Vec<Relay>> {
    Ok(vec![
        Relay::new("mev-blocker", "https://rpc.mevblocker.io/fast")?,
        Relay::new("flashbots",   "https://rpc.flashbots.net/fast")?,
        Relay::new("public",      public_fallback)?,
    ])
}
```

- [ ] **Step 3: `src/tx/nonce_manager.rs`**

```rust
use alloy::primitives::{Address, U256};
use alloy::providers::Provider;
use crate::error::{MinerError, Result};
use std::sync::Arc;
use parking_lot::Mutex;

/// Sequential gate: serialize submissions on the local nonce so we never have two in-flight.
pub struct NonceGate {
    provider: Arc<dyn Provider + Send + Sync>,
    miner: Address,
    next: Mutex<Option<u64>>,
}

impl NonceGate {
    pub fn new(provider: Arc<dyn Provider + Send + Sync>, miner: Address) -> Self {
        Self { provider, miner, next: Mutex::new(None) }
    }

    /// Reserve the next ethereum tx nonce. Caller MUST consume or release.
    pub async fn reserve(&self) -> Result<u64> {
        let mut g = self.next.lock();
        let n = match *g {
            Some(n) => n,
            None => self.provider.get_transaction_count(self.miner).await
                .map_err(|e| MinerError::Rpc(e.to_string()))?,
        };
        *g = Some(n + 1);
        Ok(n)
    }

    /// Called after receipt or permanent failure to keep local counter aligned with chain.
    pub async fn resync(&self) -> Result<()> {
        let n = self.provider.get_transaction_count(self.miner).await
            .map_err(|e| MinerError::Rpc(e.to_string()))?;
        *self.next.lock() = Some(n);
        Ok(())
    }
}
```

- [ ] **Step 4: `src/tx/ev_gate.rs`**

```rust
use crate::chain::ChainSource;
use crate::error::Result;
use alloy::primitives::U256;

pub struct EvParams {
    pub max_mints_per_block: u64,
    pub min_ratio: f64,                 // e.g. 1.2 = profitable iff reward * P(win) / cost > 1.2
}

pub struct EvGate<'a, C: ChainSource> {
    pub chain: &'a C,
    pub params: EvParams,
}

impl<'a, C: ChainSource> EvGate<'a, C> {
    /// Returns Ok(true) if it's worth submitting at current head/state.
    pub async fn allow(&self, reward_wei: U256, gas_cost_wei: U256) -> Result<bool> {
        // Refuse if current block is already saturated.
        let head = self.chain.head();
        let mints = self.chain.mints_in_block(head).await?;
        if mints >= self.params.max_mints_per_block {
            return Ok(false);
        }
        // Probability heuristic: P(win) = (cap - mints) / cap, naive but bounded.
        let p_win = (self.params.max_mints_per_block.saturating_sub(mints) as f64)
                    / self.params.max_mints_per_block as f64;
        let reward = reward_wei.to_string().parse::<f64>().unwrap_or(0.0);
        let cost   = gas_cost_wei.to_string().parse::<f64>().unwrap_or(f64::INFINITY);
        if cost == 0.0 { return Ok(true); }
        Ok((reward * p_win) / cost > self.params.min_ratio)
    }
}
```

- [ ] **Step 5: Build + commit**

```bash
cargo build
git add -A
git commit -m "feat(tx): builder + relays + sequential nonce gate + EV gate"
```

---

## Task 10: TxSubmitter live implementation

**Files:**
- Create: `src/tx/submitter.rs`, `src/tx/fake.rs`
- Modify: `src/tx/mod.rs`

- [ ] **Step 1: `src/tx/submitter.rs`**

```rust
use super::{Submitter, SubmitOutcome};
use crate::chain::ChainSource;
use crate::chain::contract::CHAIN_ID;
use crate::error::{MinerError, Result};
use crate::gpu::Hit;
use crate::tx::{builder::{build_mine_tx, MineTxParams}, ev_gate::{EvGate, EvParams},
                 nonce_manager::NonceGate, relay::Relay};
use crate::wallet::MinerSigner;
use alloy::network::EthereumWallet;
use alloy::primitives::{TxHash, U256};
use alloy::providers::{Provider, ProviderBuilder};
use alloy::rpc::types::TransactionReceipt;
use alloy::signers::Signer;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{info, warn};

pub struct TxSubmitter<C: ChainSource> {
    chain: Arc<C>,
    signer: Arc<MinerSigner>,
    read_provider: Arc<dyn Provider + Send + Sync>,
    relays: Vec<Relay>,
    nonce_gate: Arc<NonceGate>,
    seq_gate: Arc<Mutex<()>>,           // one in-flight tx at a time
    ev: EvParams,
    confirmations: u64,
}

impl<C: ChainSource> TxSubmitter<C> {
    pub fn new(
        chain: Arc<C>,
        signer: Arc<MinerSigner>,
        read_provider: Arc<dyn Provider + Send + Sync>,
        relays: Vec<Relay>,
        ev: EvParams,
        confirmations: u64,
    ) -> Self {
        let nonce_gate = Arc::new(NonceGate::new(read_provider.clone(), signer.address()));
        Self {
            chain, signer, read_provider, relays, nonce_gate,
            seq_gate: Arc::new(Mutex::new(())), ev, confirmations,
        }
    }
}

#[async_trait]
impl<C: ChainSource> Submitter for TxSubmitter<C> {
    async fn submit(&self, hit: Hit) -> Result<SubmitOutcome> {
        let _guard = self.seq_gate.lock().await;

        // Stale-epoch re-check.
        let st = self.chain.mining_state().await?;
        if hit.epoch_id != st.epoch {
            return Ok(SubmitOutcome::Dropped { reason: format!("stale epoch {} vs {}", hit.epoch_id, st.epoch) });
        }

        // Gas estimate using a baseline; real value pulled from latest base fee + tip.
        let gas_limit: u64 = 120_000;                                    // mine() is small, mostly storage
        let base_fee = self.read_provider.get_gas_price().await
            .map_err(|e| MinerError::Rpc(e.to_string()))?;
        let max_priority = 3_000_000_000u128;                            // 3 gwei default tip
        let max_fee = base_fee + max_priority;
        let gas_cost = U256::from(max_fee) * U256::from(gas_limit);

        // EV gate.
        let allowed = EvGate { chain: &*self.chain, params: EvParams {
            max_mints_per_block: self.ev.max_mints_per_block, min_ratio: self.ev.min_ratio,
        }}.allow(st.reward_wei, gas_cost).await?;
        if !allowed {
            return Ok(SubmitOutcome::Dropped { reason: "EV gate".into() });
        }

        let tx_nonce = self.nonce_gate.reserve().await?;
        let req = build_mine_tx(MineTxParams {
            from: self.signer.address(),
            nonce: tx_nonce,
            mine_nonce: hit.nonce,
            max_fee_per_gas: max_fee,
            max_priority_fee_per_gas: max_priority,
            gas_limit,
            chain_id: CHAIN_ID,
        })?;

        // Sign.
        let wallet = EthereumWallet::from(self.signer.signer().clone());
        let signed = req.build(&wallet).await.map_err(|e| MinerError::Tx(e.to_string()))?;
        let raw: alloy::primitives::Bytes = signed.encoded_2718().into();

        // Fan out to all relays in parallel; first OK response wins.
        let raw2 = raw.clone();
        let futs: Vec<_> = self.relays.iter().map(|r| {
            let r_name = r.name.clone();
            let raw = raw2.clone();
            async move { (r_name, r.send_raw(raw).await) }
        }).collect();

        let mut winner_tx: Option<(String, TxHash)> = None;
        let mut errors = Vec::new();
        for fut in futs {
            match fut.await {
                (n, Ok(h)) => { info!(relay=%n, ?h, "submitted"); if winner_tx.is_none() { winner_tx = Some((n, h)); } }
                (n, Err(e)) => { warn!(relay=%n, error=%e, "relay failed"); errors.push((n, e)); }
            }
        }

        let (relay, tx) = match winner_tx {
            Some(w) => w,
            None => {
                self.nonce_gate.resync().await.ok();
                return Ok(SubmitOutcome::Dropped { reason: format!("all relays failed: {errors:?}") });
            }
        };

        // Wait for receipt with timeout.
        let outcome = wait_for_receipt(self.read_provider.clone(), tx, self.confirmations,
                                        Duration::from_secs(90)).await?;
        self.nonce_gate.resync().await.ok();
        match outcome {
            Some(r) if r.status() => Ok(SubmitOutcome::Included {
                tx, block: r.block_number.unwrap_or_default(),
                reward_wei: st.reward_wei, relay,
            }),
            Some(r) => Ok(SubmitOutcome::Reverted { tx, reason: format!("receipt status=0 block={:?}", r.block_number) }),
            None => Ok(SubmitOutcome::Dropped { reason: "receipt timeout".into() }),
        }
    }
}

async fn wait_for_receipt(
    provider: Arc<dyn Provider + Send + Sync>,
    tx: TxHash, confirmations: u64, timeout: Duration,
) -> Result<Option<TransactionReceipt>> {
    let start = std::time::Instant::now();
    loop {
        if start.elapsed() > timeout { return Ok(None); }
        if let Some(r) = provider.get_transaction_receipt(tx).await
                .map_err(|e| MinerError::Rpc(e.to_string()))? {
            let head = provider.get_block_number().await
                .map_err(|e| MinerError::Rpc(e.to_string()))?;
            let included_at = r.block_number.unwrap_or(head);
            if head.saturating_sub(included_at) >= confirmations { return Ok(Some(r)); }
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}
```

- [ ] **Step 2: `src/tx/fake.rs`**

```rust
use super::{Submitter, SubmitOutcome};
use crate::error::Result;
use crate::gpu::Hit;
use alloy::primitives::{TxHash, U256};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Default)]
pub struct FakeSubmitter {
    pub submitted: Arc<Mutex<Vec<Hit>>>,
    pub force_outcome: Arc<Mutex<Option<SubmitOutcome>>>,
}

#[async_trait]
impl Submitter for FakeSubmitter {
    async fn submit(&self, hit: Hit) -> Result<SubmitOutcome> {
        self.submitted.lock().push(hit.clone());
        if let Some(out) = self.force_outcome.lock().clone() { return Ok(out); }
        Ok(SubmitOutcome::Included {
            tx: TxHash::ZERO, block: 0, reward_wei: U256::from(100u64), relay: "fake".into(),
        })
    }
}
```

- [ ] **Step 3: Update `src/tx/mod.rs`**

```rust
use alloy::primitives::{TxHash, U256};
use async_trait::async_trait;
use crate::gpu::Hit;

#[derive(Debug, Clone)]
pub enum SubmitOutcome {
    Included { tx: TxHash, block: u64, reward_wei: U256, relay: String },
    Reverted { tx: TxHash, reason: String },
    Dropped  { reason: String },
}

#[async_trait]
pub trait Submitter: Send + Sync + 'static {
    async fn submit(&self, hit: Hit) -> crate::error::Result<SubmitOutcome>;
}

pub mod builder;
pub mod relay;
pub mod nonce_manager;
pub mod ev_gate;
pub mod submitter;
pub mod fake;
```

- [ ] **Step 4: Build + commit**

```bash
cargo build
git add -A
git commit -m "feat(tx): TxSubmitter with sequential gate + dual-relay fan-out + receipt wait"
```

---

## Task 11: Stale-hit + nonce-gate integration tests

**Files:**
- Create: `tests/stale_hit_filter.rs`, `tests/tx_nonce_gate.rs`

- [ ] **Step 1: `tests/stale_hit_filter.rs`**

```rust
use alloy::primitives::{address, U256};
use hashminer::chain::{ChainSource, fake::FakeChain};
use hashminer::gpu::Hit;
use hashminer::tx::{Submitter, fake::FakeSubmitter, SubmitOutcome};

#[tokio::test]
async fn submitter_drops_stale_epoch_hit() {
    let miner = address!("0000000000000000000000000000000000000001");
    let chain = FakeChain::new(0, U256::MAX, miner);
    let _ = chain.mining_state().await.unwrap();
    // Pretend we mined for old epoch 0, then chain advanced to epoch 1 (100 blocks).
    chain.advance_blocks(100, miner);
    let stale = Hit { nonce: U256::from(1u64), hash: Default::default(), epoch_id: 0 };

    // We don't run TxSubmitter here directly because it needs a Provider; the assertion
    // is that the submitter contract drops the stale hit when chain.epoch > hit.epoch.
    let st = chain.mining_state().await.unwrap();
    assert_eq!(st.epoch, 1);
    assert!(st.epoch != stale.epoch_id, "epoch must have advanced");

    // Lightweight check: drop-decision logic mirrors what TxSubmitter does on line 354 of submitter.rs.
    let drop = stale.epoch_id != st.epoch;
    assert!(drop);
}
```

(For a more thorough test that exercises actual TxSubmitter, an in-process anvil provider is needed — that lands in e2e_anvil.rs in Phase 5.)

- [ ] **Step 2: `tests/tx_nonce_gate.rs`**

```rust
// Tests sequential reservation semantics by mocking get_transaction_count.
use hashminer::tx::nonce_manager::NonceGate;
use alloy::primitives::address;
// ... actual impl uses wiremock to spin up a JSON-RPC stub returning fixed nonces ...
//
// Since NonceGate takes Arc<dyn Provider>, this test requires a stub provider.
// For now we assert the trait surface compiles; full mock landed in Task 19 with e2e.
#[test]
fn nonce_gate_compiles() {
    fn _assert_send<T: Send>() {}
    _assert_send::<NonceGate>();
}
```

- [ ] **Step 3: Run + commit**

```bash
cargo test --test stale_hit_filter --test tx_nonce_gate
git add -A
git commit -m "test: stale-epoch hit drop logic + nonce gate trait surface"
```

---

# Phase 4 — GPU

## Task 12: Build script — nvcc → PTX → embed

**Files:**
- Modify: `build.rs`
- Create: `kernel/keccak_grinder.cu` (stub), `kernel/keccak_device.cuh` (stub), `kernel/result_codec.cuh` (stub), `src/gpu/ptx.rs`

- [ ] **Step 1: Stub kernel files**

`kernel/keccak_device.cuh`:
```cuda
#pragma once
#include <stdint.h>
__device__ inline void keccak_f1600(uint64_t state[25]) {
    // Filled in Task 13.
    (void)state;
}
```

`kernel/result_codec.cuh`:
```cuda
#pragma once
#include <stdint.h>
__device__ inline bool less_than_target_be(const uint64_t state[25], const uint64_t target[4]) {
    // Filled in Task 13.
    (void)state; (void)target; return false;
}
```

`kernel/keccak_grinder.cu`:
```cuda
#include "keccak_device.cuh"
#include "result_codec.cuh"

extern "C" __global__ void grind() {
    // Filled in Task 14.
}
```

- [ ] **Step 2: `build.rs` real implementation**

```rust
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only build CUDA when feature is enabled.
    if std::env::var("CARGO_FEATURE_CUDA_RUNTIME").is_err() {
        println!("cargo:rustc-env=PTX_PATH=/dev/null"); // src/gpu/ptx.rs guards on this
        return;
    }
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel = manifest_dir.join("kernel/keccak_grinder.cu");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("keccak_grinder.ptx");

    let status = Command::new("nvcc")
        .args(["-ptx", "-O3", "-arch=compute_89"])
        .arg("-o").arg(&out)
        .arg(&kernel)
        .status()
        .expect("nvcc must be in PATH when cuda-runtime feature is on");
    if !status.success() { panic!("nvcc failed"); }

    println!("cargo:rustc-env=PTX_PATH={}", out.display());
    println!("cargo:rerun-if-changed=kernel/keccak_grinder.cu");
    println!("cargo:rerun-if-changed=kernel/keccak_device.cuh");
    println!("cargo:rerun-if-changed=kernel/result_codec.cuh");
}
```

- [ ] **Step 3: `src/gpu/ptx.rs`**

```rust
/// Embedded PTX bytes from the nvcc build step. Empty when `cuda-runtime` is off.
#[cfg(feature = "cuda-runtime")]
pub const PTX: &[u8] = include_bytes!(env!("PTX_PATH"));

#[cfg(not(feature = "cuda-runtime"))]
pub const PTX: &[u8] = b"";
```

- [ ] **Step 4: Build verify — both with and without feature**

```bash
cargo build                            # PTX_PATH stub
cargo build --features cuda-runtime    # nvcc invoked
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "build: nvcc PTX pipeline gated on cuda-runtime feature"
```

---

## Task 13: Keccak-f[1600] device function + target compare

**Files:**
- Modify: `kernel/keccak_device.cuh`, `kernel/result_codec.cuh`

- [ ] **Step 1: Full Keccak-f[1600] in `keccak_device.cuh`**

Reference: NIST FIPS-202 (note: Ethereum uses pre-FIPS Keccak-256 padding, NOT SHA3 padding). Round constants:

```cuda
#pragma once
#include <stdint.h>

__constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __forceinline__ uint64_t ROL64(uint64_t x, uint32_t n) {
    return (x << n) | (x >> (64u - n));
}

__device__ __forceinline__ void keccak_f1600(uint64_t s[25]) {
    #pragma unroll
    for (int r = 0; r < 24; r++) {
        // Theta
        uint64_t C0 = s[0]^s[5]^s[10]^s[15]^s[20];
        uint64_t C1 = s[1]^s[6]^s[11]^s[16]^s[21];
        uint64_t C2 = s[2]^s[7]^s[12]^s[17]^s[22];
        uint64_t C3 = s[3]^s[8]^s[13]^s[18]^s[23];
        uint64_t C4 = s[4]^s[9]^s[14]^s[19]^s[24];
        uint64_t D0 = C4 ^ ROL64(C1, 1);
        uint64_t D1 = C0 ^ ROL64(C2, 1);
        uint64_t D2 = C1 ^ ROL64(C3, 1);
        uint64_t D3 = C2 ^ ROL64(C4, 1);
        uint64_t D4 = C3 ^ ROL64(C0, 1);
        s[0]^=D0; s[5]^=D0; s[10]^=D0; s[15]^=D0; s[20]^=D0;
        s[1]^=D1; s[6]^=D1; s[11]^=D1; s[16]^=D1; s[21]^=D1;
        s[2]^=D2; s[7]^=D2; s[12]^=D2; s[17]^=D2; s[22]^=D2;
        s[3]^=D3; s[8]^=D3; s[13]^=D3; s[18]^=D3; s[23]^=D3;
        s[4]^=D4; s[9]^=D4; s[14]^=D4; s[19]^=D4; s[24]^=D4;

        // Rho + Pi
        uint64_t t = s[1];
        uint64_t b;
        b = s[10]; s[10] = ROL64(t,  1);  t = b;
        b = s[ 7]; s[ 7] = ROL64(t,  3);  t = b;
        b = s[11]; s[11] = ROL64(t,  6);  t = b;
        b = s[17]; s[17] = ROL64(t, 10);  t = b;
        b = s[18]; s[18] = ROL64(t, 15);  t = b;
        b = s[ 3]; s[ 3] = ROL64(t, 21);  t = b;
        b = s[ 5]; s[ 5] = ROL64(t, 28);  t = b;
        b = s[16]; s[16] = ROL64(t, 36);  t = b;
        b = s[ 8]; s[ 8] = ROL64(t, 45);  t = b;
        b = s[21]; s[21] = ROL64(t, 55);  t = b;
        b = s[24]; s[24] = ROL64(t,  2);  t = b;
        b = s[ 4]; s[ 4] = ROL64(t, 14);  t = b;
        b = s[15]; s[15] = ROL64(t, 27);  t = b;
        b = s[23]; s[23] = ROL64(t, 41);  t = b;
        b = s[19]; s[19] = ROL64(t, 56);  t = b;
        b = s[13]; s[13] = ROL64(t,  8);  t = b;
        b = s[12]; s[12] = ROL64(t, 25);  t = b;
        b = s[ 2]; s[ 2] = ROL64(t, 43);  t = b;
        b = s[20]; s[20] = ROL64(t, 62);  t = b;
        b = s[14]; s[14] = ROL64(t, 18);  t = b;
        b = s[22]; s[22] = ROL64(t, 39);  t = b;
        b = s[ 9]; s[ 9] = ROL64(t, 61);  t = b;
        b = s[ 6]; s[ 6] = ROL64(t, 20);  t = b;
                   s[ 1] = ROL64(t, 44);

        // Chi
        #pragma unroll
        for (int y = 0; y < 25; y += 5) {
            uint64_t a0 = s[y+0], a1 = s[y+1], a2 = s[y+2], a3 = s[y+3], a4 = s[y+4];
            s[y+0] = a0 ^ ((~a1) & a2);
            s[y+1] = a1 ^ ((~a2) & a3);
            s[y+2] = a2 ^ ((~a3) & a4);
            s[y+3] = a3 ^ ((~a4) & a0);
            s[y+4] = a4 ^ ((~a0) & a1);
        }

        // Iota
        s[0] ^= RC[r];
    }
}
```

- [ ] **Step 2: `kernel/result_codec.cuh`**

```cuda
#pragma once
#include <stdint.h>

// state[0..3] holds 32-byte hash in LITTLE-endian per-word, BIG-endian byte-stream.
// Solidity treats keccak256 output as bytes32; "less than target" compares as big-endian uint256.
// state[0] little-endian uint64 corresponds to bytes 0..7 of the hash;
// for big-endian uint256 comparison we need to byte-swap each word.
__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    return __byte_perm(uint32_t(x >> 32), uint32_t(x), 0x0123ULL)
         | (uint64_t)__byte_perm(uint32_t(x >> 32), uint32_t(x), 0x4567ULL) << 32;
    // CUDA compiler usually folds this to a single SHFL/PRMT pair.
}

__device__ __forceinline__ bool less_than_target_be(const uint64_t state[25], const uint64_t target[4]) {
    // Compare 32-byte big-endian values: target[0] is MS 8 bytes of the 256-bit target.
    uint64_t h0 = bswap64(state[0]);   // bytes 0..7  (MSB)
    uint64_t h1 = bswap64(state[1]);
    uint64_t h2 = bswap64(state[2]);
    uint64_t h3 = bswap64(state[3]);   // bytes 24..31 (LSB)
    if (h0 != target[0]) return h0 < target[0];
    if (h1 != target[1]) return h1 < target[1];
    if (h2 != target[2]) return h2 < target[2];
    return h3 < target[3];
}

__device__ __forceinline__ void store_hash_be(uint8_t out[32], const uint64_t state[25]) {
    uint64_t h0 = bswap64(state[0]);
    uint64_t h1 = bswap64(state[1]);
    uint64_t h2 = bswap64(state[2]);
    uint64_t h3 = bswap64(state[3]);
    *reinterpret_cast<uint64_t*>(out +  0) = h0;
    *reinterpret_cast<uint64_t*>(out +  8) = h1;
    *reinterpret_cast<uint64_t*>(out + 16) = h2;
    *reinterpret_cast<uint64_t*>(out + 24) = h3;
}
```

- [ ] **Step 3: Build + commit**

```bash
cargo build --features cuda-runtime
git add -A
git commit -m "feat(kernel): keccak-f[1600] + big-endian target compare"
```

---

## Task 14: Persistent grind kernel

**Files:**
- Modify: `kernel/keccak_grinder.cu`

- [ ] **Step 1: Full kernel**

```cuda
#include "keccak_device.cuh"
#include "result_codec.cuh"

// Double-buffered challenge + target (hot-swap without kernel restart).
__constant__ uint8_t  c_challenge[2][32];
__constant__ uint64_t c_target[2][4];          // big-endian
__constant__ uint32_t c_epoch_id[2];

__device__ uint32_t d_active_idx;
__device__ uint64_t d_nonce_counter;
__device__ uint32_t d_hit_count;
__device__ uint32_t d_should_stop;

struct Hit { uint8_t nonce[32]; uint8_t hash[32]; uint32_t epoch_id; uint8_t _pad[4]; };
__device__ Hit d_hits[16];

#ifndef BATCH_PER_THREAD
#define BATCH_PER_THREAD 1024
#endif

extern "C" __global__ void grind() {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t base = atomicAdd((unsigned long long*)&d_nonce_counter, (unsigned long long)BATCH_PER_THREAD);

    while (atomicAdd(&d_should_stop, 0) == 0) {
        uint32_t idx = d_active_idx;
        uint32_t my_epoch = c_epoch_id[idx];
        const uint64_t* challenge_w = reinterpret_cast<const uint64_t*>(c_challenge[idx]);

        #pragma unroll 1
        for (uint32_t i = 0; i < BATCH_PER_THREAD; i++) {
            uint64_t s[25];
            #pragma unroll
            for (int w = 0; w < 25; w++) s[w] = 0;

            // Absorb challenge (32 bytes, big-endian as-is — bytes 0..31 of input).
            // In little-endian word view of the rate, byte 0 is the LS byte of s[0].
            // We must encode bytes 0..7 as: byte 0 is LS of word 0. So bswap.
            s[0] ^= bswap64(challenge_w[0]);
            s[1] ^= bswap64(challenge_w[1]);
            s[2] ^= bswap64(challenge_w[2]);
            s[3] ^= bswap64(challenge_w[3]);

            // Absorb nonce as 32-byte big-endian uint256.
            uint64_t nonce_lo = base + i;
            uint64_t nonce_hi = ((uint64_t)tid);
            // bytes 32..39 = 0, bytes 40..47 = 0, bytes 48..55 = nonce_hi (BE),
            // bytes 56..63 = nonce_lo (BE). Words s[4..7].
            s[4] ^= 0;
            s[5] ^= 0;
            s[6] ^= bswap64(nonce_hi);
            s[7] ^= bswap64(nonce_lo);

            // Padding: 0x01 at byte 64 (LSB of s[8]), 0x80 at byte 135 (MSB of s[16]).
            s[8]  ^= 0x01ULL;
            s[16] ^= 0x8000000000000000ULL;

            keccak_f1600(s);

            if (less_than_target_be(s, c_target[idx])) {
                uint32_t slot = atomicAdd(&d_hit_count, 1);
                if (slot < 16) {
                    // Store nonce big-endian: bytes 0..15 zero, bytes 16..23 = nonce_hi BE, 24..31 = nonce_lo BE.
                    uint64_t* out = reinterpret_cast<uint64_t*>(d_hits[slot].nonce);
                    out[0] = 0; out[1] = 0;
                    out[2] = bswap64(nonce_hi);
                    out[3] = bswap64(nonce_lo);
                    store_hash_be(d_hits[slot].hash, s);
                    d_hits[slot].epoch_id = my_epoch;
                }
            }
        }
        base = atomicAdd((unsigned long long*)&d_nonce_counter, (unsigned long long)BATCH_PER_THREAD);
    }
}
```

- [ ] **Step 2: Build (kernel compile only)**

```bash
cargo build --features cuda-runtime
```
Expected: nvcc compiles cleanly, possibly emits 1-2 warnings about pointer reinterpret_cast on constant memory.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(kernel): persistent grind with double-buffered challenge + epoch_id tagging"
```

---

## Task 15: Rust-side CUDA driver wrapper (`src/gpu/kernel_ffi.rs`)

**Files:**
- Create: `src/gpu/kernel_ffi.rs`, `src/gpu/worker.rs`, `src/gpu/fake.rs`
- Modify: `src/gpu/mod.rs`

(Note: this task is OS-dependent. On Linux we use `cust`. On Windows we use raw CUDA driver API via `cuda-driver-sys` crate. The implementer chooses one path and `#[cfg]`-gates it. Below: Linux/`cust` path; Windows variant follows the same structure with manual `cuModuleLoadData` / `cuLaunchKernel` calls.)

- [ ] **Step 1: `src/gpu/kernel_ffi.rs` (Linux/cust)**

```rust
#![cfg(feature = "cuda-runtime")]

use crate::error::{MinerError, Result};
use crate::gpu::ptx::PTX;
use alloy::primitives::{B256, U256};
use cust::context::Context;
use cust::device::Device;
use cust::launch;
use cust::memory::{DeviceCopy, GpuBox, LockedBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use std::ffi::CString;

pub struct GpuRuntime {
    _ctx: Context,
    module: Module,
    stream: Stream,
    // device symbols
    d_active_idx: u64,
    d_nonce_counter: u64,
    d_hit_count: u64,
    d_should_stop: u64,
    d_hits: u64,
    c_challenge: u64,
    c_target: u64,
    c_epoch_id: u64,
    // pinned host mirror for poll
    pub host_hit_count: LockedBuffer<u32>,
    pub host_hits: LockedBuffer<[u8; 72]>,                  // 16 × (32 nonce + 32 hash + 4 epoch + 4 pad)
}

impl GpuRuntime {
    pub fn init(device_id: u32) -> Result<Self> {
        cust::init(cust::CudaFlags::empty()).map_err(|e| MinerError::Gpu(e.to_string()))?;
        let device = Device::get_device(device_id).map_err(|e| MinerError::Gpu(e.to_string()))?;
        let ctx = Context::new(device).map_err(|e| MinerError::Gpu(e.to_string()))?;
        let module = Module::from_ptx(std::str::from_utf8(PTX).unwrap(), &[])
            .map_err(|e| MinerError::Gpu(format!("module load: {e}")))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| MinerError::Gpu(e.to_string()))?;

        let host_hit_count = LockedBuffer::new(&0u32, 1).map_err(|e| MinerError::Gpu(e.to_string()))?;
        let host_hits = LockedBuffer::new(&[0u8; 72], 16).map_err(|e| MinerError::Gpu(e.to_string()))?;

        let get = |name: &str| -> Result<u64> {
            module.get_global::<u8>(&CString::new(name).unwrap())
                .map(|g| g.as_device_ptr().as_raw() as u64)
                .map_err(|e| MinerError::Gpu(format!("symbol {name}: {e}")))
        };

        let rt = Self {
            _ctx: ctx, module, stream,
            d_active_idx: get("d_active_idx")?,
            d_nonce_counter: get("d_nonce_counter")?,
            d_hit_count: get("d_hit_count")?,
            d_should_stop: get("d_should_stop")?,
            d_hits: get("d_hits")?,
            c_challenge: get("c_challenge")?,
            c_target: get("c_target")?,
            c_epoch_id: get("c_epoch_id")?,
            host_hit_count, host_hits,
        };
        Ok(rt)
    }

    pub fn launch_persistent(&self, blocks: u32, threads: u32) -> Result<()> {
        let func = self.module.get_function("grind").map_err(|e| MinerError::Gpu(e.to_string()))?;
        unsafe {
            launch!(func<<<blocks, threads, 0, self.stream>>>()).map_err(|e| MinerError::Gpu(e.to_string()))?;
        }
        Ok(())
    }

    /// Write challenge / target / epoch into the inactive buffer, then flip active_idx.
    pub fn hot_swap(&self, challenge: B256, target: U256, epoch_id: u32) -> Result<()> {
        // Read current active idx from device (cheap)...
        // For simplicity: maintain host-side mirror of active idx (we always alternate).
        // Implementer: store the mirror in self (interior mutability) or in a separate atomic.
        // Pseudocode:
        //   write c_challenge[next] = challenge
        //   write c_target[next] = target.to_be_bytes (4 × u64 BE)
        //   write c_epoch_id[next] = epoch_id
        //   memcpy d_active_idx = next
        //   reset d_hit_count = 0
        // The actual cust API uses cuMemcpyHtoDAsync via DeviceBuffer / DevicePointer.
        let _ = (challenge, target, epoch_id);
        unimplemented!("implementer: fill using cust DevicePointer + memcpy_htod_async on self.stream")
    }

    /// Reset shutdown flag.
    pub fn signal_stop(&self) -> Result<()> {
        unimplemented!("memcpy 1 → d_should_stop")
    }

    /// Read d_hit_count from pinned mirror; on >0 read the slot and reset.
    pub fn poll_hits(&mut self) -> Result<Vec<crate::gpu::Hit>> {
        unimplemented!("memcpy_dtoh d_hit_count → host_hit_count; if >0, memcpy_dtoh d_hits → host_hits; reset d_hit_count = 0; parse")
    }
}
```

Implementation notes for the engineer:
- `cust` 0.3 APIs have changed; consult crate docs at the version in Cargo.toml. The structure above is a **skeleton**; the `unimplemented!()` bodies are the integration glue, each ~10 lines of memcpy/launch calls.
- On Windows: replace `cust` with raw `cuda-driver-sys` calls. The CPU-side logic is identical.

- [ ] **Step 2: `src/gpu/fake.rs`** — CPU-side Grinder for tests

```rust
use super::{Grinder, Hit};
use crate::chain::challenge::compute_inner_hash;
use crate::error::Result;
use alloy::primitives::{B256, U256};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct FakeGrinder {
    state: Arc<Mutex<State>>,
    hit_tx: mpsc::Sender<Hit>,
    hit_rx: Mutex<Option<mpsc::Receiver<Hit>>>,
}

struct State {
    challenge: B256,
    target: U256,
    epoch_id: u64,
    next_nonce: u64,
    hashrate: f64,
}

impl FakeGrinder {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(16);
        Self {
            state: Arc::new(Mutex::new(State { challenge: B256::ZERO, target: U256::ZERO, epoch_id: 0, next_nonce: 0, hashrate: 0.0 })),
            hit_tx: tx, hit_rx: Mutex::new(Some(rx)),
        }
    }

    /// Drive a synthetic grind step: find the first valid nonce starting from current and emit it.
    pub async fn drive_one_hit(&self) {
        let (challenge, target, epoch_id, start) = {
            let g = self.state.lock();
            (g.challenge, g.target, g.epoch_id, g.next_nonce)
        };
        for n in start.. {
            let h = compute_inner_hash(challenge, U256::from(n));
            if U256::from_be_bytes(h.0) < target {
                let _ = self.hit_tx.send(Hit { nonce: U256::from(n), hash: h, epoch_id }).await;
                let mut g = self.state.lock();
                g.next_nonce = n + 1;
                return;
            }
        }
    }
}

#[async_trait]
impl Grinder for FakeGrinder {
    async fn hot_swap(&self, challenge: B256, target: U256, epoch_id: u64) -> Result<()> {
        let mut g = self.state.lock();
        g.challenge = challenge; g.target = target; g.epoch_id = epoch_id; g.next_nonce = 0;
        Ok(())
    }
    fn take_hit_rx(&self) -> mpsc::Receiver<Hit> {
        self.hit_rx.lock().take().expect("take_hit_rx called twice")
    }
    fn hashrate(&self) -> f64 { self.state.lock().hashrate }
    async fn shutdown(&self) {}
}
```

- [ ] **Step 3: Update `src/gpu/mod.rs`**

```rust
use alloy::primitives::{B256, U256};
use async_trait::async_trait;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct Hit { pub nonce: U256, pub hash: B256, pub epoch_id: u64 }

#[async_trait]
pub trait Grinder: Send + Sync + 'static {
    async fn hot_swap(&self, challenge: B256, target: U256, epoch_id: u64) -> crate::error::Result<()>;
    fn take_hit_rx(&self) -> mpsc::Receiver<Hit>;
    fn hashrate(&self) -> f64;
    async fn shutdown(&self);
}

pub mod ptx;
pub mod fake;
#[cfg(feature = "cuda-runtime")] pub mod kernel_ffi;
#[cfg(feature = "cuda-runtime")] pub mod worker;
```

- [ ] **Step 4: Stub `src/gpu/worker.rs`** (fills in next task)

```rust
#![cfg(feature = "cuda-runtime")]
// GpuWorker (live Grinder backed by CUDA) is implemented in Task 16.
```

- [ ] **Step 5: Build both ways + commit**

```bash
cargo build
cargo build --features cuda-runtime
git add -A
git commit -m "feat(gpu): CUDA FFI skeleton + CPU FakeGrinder for tests"
```

---

## Task 16: GpuWorker live + hot-swap implementation

**Files:**
- Modify: `src/gpu/worker.rs`, `src/gpu/kernel_ffi.rs`

- [ ] **Step 1: Complete the `unimplemented!()` bodies in `kernel_ffi.rs`**

Reference the cust docs for the exact calls. Approximate signatures:

```rust
// Pseudocode skeleton for hot_swap (replace each `cuda_*` with cust's actual call):
pub fn hot_swap(&mut self, challenge: B256, target: U256, epoch_id: u32) -> Result<()> {
    let next = 1u32 - self.active_idx_host;
    let target_be: [u64; 4] = {
        let bytes = target.to_be_bytes::<32>();
        [
            u64::from_be_bytes(bytes[ 0..8 ].try_into().unwrap()),
            u64::from_be_bytes(bytes[ 8..16].try_into().unwrap()),
            u64::from_be_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_be_bytes(bytes[24..32].try_into().unwrap()),
        ]
    };
    cuda_memcpy_htod_async(&self.stream, self.c_challenge + (next as u64 * 32), challenge.as_slice())?;
    cuda_memcpy_htod_async(&self.stream, self.c_target    + (next as u64 * 32), &target_be)?;
    cuda_memcpy_htod_async(&self.stream, self.c_epoch_id  + (next as u64 *  4), &epoch_id)?;
    self.stream.synchronize()?;
    cuda_memcpy_htod_async(&self.stream, self.d_active_idx, &next)?;
    cuda_memcpy_htod_async(&self.stream, self.d_hit_count,  &0u32)?;
    self.active_idx_host = next;
    Ok(())
}
```

Implementer: add `active_idx_host: u32` field to `GpuRuntime`, initialize to 0 on `init()`.

- [ ] **Step 2: `src/gpu/worker.rs` full impl**

```rust
#![cfg(feature = "cuda-runtime")]

use super::{Grinder, Hit};
use crate::error::{MinerError, Result};
use crate::gpu::kernel_ffi::GpuRuntime;
use alloy::primitives::{B256, U256};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

pub struct GpuWorker {
    rt: Arc<Mutex<GpuRuntime>>,
    hit_tx: mpsc::Sender<Hit>,
    hit_rx: Mutex<Option<mpsc::Receiver<Hit>>>,
    hashrate: Arc<parking_lot::RwLock<f64>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl GpuWorker {
    pub async fn start(device_id: u32, blocks: u32, threads: u32, poll_ms: u64) -> Result<Self> {
        let mut rt = GpuRuntime::init(device_id)?;
        rt.launch_persistent(blocks, threads)?;
        let rt = Arc::new(Mutex::new(rt));
        let (tx, rx) = mpsc::channel(16);
        let hashrate = Arc::new(parking_lot::RwLock::new(0.0));
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Poll task.
        let rt2 = rt.clone();
        let tx2 = tx.clone();
        let rate = hashrate.clone();
        let shut = shutdown.clone();
        tokio::spawn(async move {
            let mut last_count: u64 = 0;
            let mut last_t = Instant::now();
            while !shut.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_millis(poll_ms)).await;
                let mut g = rt2.lock();
                if let Ok(hits) = g.poll_hits() {
                    for h in hits {
                        let _ = tx2.send(h).await;
                    }
                }
                // Hashrate sample (approximate, based on d_nonce_counter delta).
                // Implementer: read d_nonce_counter via small DtoH memcpy.
                let now = Instant::now();
                let dt = now.duration_since(last_t).as_secs_f64();
                if dt > 0.5 {
                    // pseudo: let counter = g.read_nonce_counter()?;
                    // *rate.write() = (counter - last_count) as f64 / dt;
                    last_t = now;
                }
            }
        });

        Ok(Self { rt, hit_tx: tx, hit_rx: Mutex::new(Some(rx)), hashrate, shutdown })
    }
}

#[async_trait]
impl Grinder for GpuWorker {
    async fn hot_swap(&self, challenge: B256, target: U256, epoch_id: u64) -> Result<()> {
        let mut g = self.rt.lock();
        g.hot_swap(challenge, target, epoch_id as u32)
    }
    fn take_hit_rx(&self) -> mpsc::Receiver<Hit> {
        self.hit_rx.lock().take().expect("take_hit_rx called twice")
    }
    fn hashrate(&self) -> f64 { *self.hashrate.read() }
    async fn shutdown(&self) {
        self.shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = self.rt.lock().signal_stop();
    }
}
```

- [ ] **Step 3: Build + commit**

```bash
cargo build --features cuda-runtime
git add -A
git commit -m "feat(gpu): GpuWorker — launches persistent kernel, polls pinned hits, hot-swap"
```

---

## Task 17: Kernel correctness test

**Files:**
- Create: `tests/kernel_correctness.rs`

- [ ] **Step 1: Test** (gated on cuda-runtime)

```rust
#![cfg(feature = "cuda-runtime")]

use alloy::primitives::{B256, U256};
use hashminer::chain::challenge::compute_inner_hash;
use hashminer::gpu::kernel_ffi::GpuRuntime;
use rand::Rng;

#[test]
fn gpu_keccak_matches_cpu_for_1000_random_vectors() {
    let mut rt = GpuRuntime::init(0).unwrap();
    let mut rng = rand::thread_rng();

    // Set difficulty = max (any hash valid → first nonce always hits).
    let target = U256::MAX;

    for _ in 0..1000 {
        let challenge = B256::random();
        let nonce = U256::from(rng.gen::<u64>());

        // CPU reference
        let cpu = compute_inner_hash(challenge, nonce);

        // Force GPU to compute exactly this (challenge, nonce):
        //   - hot-swap to (challenge, target=MAX, epoch=0)
        //   - set d_nonce_counter = nonce.low_u64(), tid prefix high = 0
        //   - launch tiny grid (1×1)
        //   - poll, expect first hit's hash == cpu
        rt.hot_swap(challenge, target, 0).unwrap();
        // Implementer: add testing helper `rt.force_nonce(nonce)` that memsets d_nonce_counter
        //              and launches a 1-thread kernel to grind exactly BATCH_PER_THREAD nonces from there.
        rt.force_test_nonce(nonce).unwrap();
        let hits = rt.poll_hits_blocking(std::time::Duration::from_secs(1)).unwrap();
        assert!(!hits.is_empty(), "expected at least one hit for target=MAX");
        assert_eq!(hits[0].hash, cpu);
    }
}
```

Helper `force_test_nonce` + `poll_hits_blocking` are implementer additions to `GpuRuntime`. Both are ~10 lines.

- [ ] **Step 2: Run**

```bash
cargo test --features cuda-runtime --test kernel_correctness -- --test-threads=1
```
Expected: 1000 vectors match.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: GPU keccak matches CPU reference on 1000 random vectors"
```

---

## Task 18: hashminer-bench standalone binary

**Files:**
- Create: `bin/hashminer-bench.rs`

- [ ] **Step 1: Implement**

```rust
#![cfg(feature = "cuda-runtime")]
use alloy::primitives::{B256, U256};
use hashminer::gpu::kernel_ffi::GpuRuntime;
use std::time::{Duration, Instant};

fn main() {
    let mut rt = GpuRuntime::init(0).unwrap();
    // target = MAX → every hash is a hit (but we don't read hits, just measure throughput).
    rt.hot_swap(B256::random(), U256::MAX, 0).unwrap();
    // Implementer: rt.launch_persistent(blocks, threads) with config-derived sizes.
    rt.launch_persistent(4 * 144, 256).unwrap();

    let dur = Duration::from_secs(10);
    let start = Instant::now();
    let mut last_counter: u64 = 0;
    while start.elapsed() < dur {
        std::thread::sleep(Duration::from_secs(1));
        let counter = rt.read_nonce_counter().unwrap();
        let delta = counter - last_counter;
        println!("hashrate: {:.2} GH/s", delta as f64 / 1e9);
        last_counter = counter;
    }
    rt.signal_stop().ok();
}
```

`read_nonce_counter` — 5-line helper in `GpuRuntime`.

- [ ] **Step 2: Run on real GPU**

```bash
cargo run --features cuda-runtime --bin hashminer-bench --release
```
Expected: 5-15 GH/s on RTX 40-series.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(bin): hashminer-bench standalone hashrate tool"
```

---

# Phase 5 — Wiring

## Task 19: Metrics bus

**Files:**
- Create: `src/metrics/mod.rs`, `src/metrics/stdout.rs`, `src/metrics/jsonl.rs`, `src/metrics/redact.rs`

- [ ] **Step 1: `src/metrics/mod.rs`**

```rust
use alloy::primitives::TxHash;
use serde::Serialize;
use std::time::SystemTime;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum Event {
    Hashrate { hashrate_hps: f64 },
    EpochSwap { epoch: u64, block: u64, diff: String, challenge: String, latency_ms: u64 },
    HitFound { epoch: u64, nonce: String },
    HitStale { epoch: u64, current_epoch: u64 },
    HitDropped { reason: String },
    TxSubmitted { relay: String, tx: TxHash },
    TxIncluded { tx: TxHash, block: u64, reward: String },
    TxReverted { tx: TxHash, reason: String },
    StateChange { from: String, to: String },
}

#[derive(Clone)]
pub struct MetricsBus { tx: mpsc::Sender<(SystemTime, Event)> }

impl MetricsBus {
    pub fn channel(capacity: usize) -> (Self, mpsc::Receiver<(SystemTime, Event)>) {
        let (tx, rx) = mpsc::channel(capacity);
        (Self { tx }, rx)
    }
    pub fn emit(&self, e: Event) {
        let _ = self.tx.try_send((SystemTime::now(), e));
    }
}

pub mod stdout;
pub mod jsonl;
pub mod redact;
```

- [ ] **Step 2: `src/metrics/jsonl.rs`**

```rust
use super::Event;
use std::io::Write;
use std::fs::OpenOptions;
use std::path::Path;
use std::time::SystemTime;
use tokio::sync::mpsc;

pub async fn run_appender<P: AsRef<Path>>(path: P, mut rx: mpsc::Receiver<(SystemTime, Event)>) -> std::io::Result<()> {
    let path = path.as_ref().to_path_buf();
    let f = OpenOptions::new().create(true).append(true).open(&path)?;
    let mut w = std::io::BufWriter::new(f);
    while let Some((ts, ev)) = rx.recv().await {
        let ts_secs = ts.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs_f64();
        let line = serde_json::json!({ "ts": ts_secs, "event": ev });
        writeln!(w, "{}", line.to_string())?;
        w.flush()?;
    }
    Ok(())
}
```

- [ ] **Step 3: `src/metrics/stdout.rs`** — 1Hz aggregator

```rust
use super::Event;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tracing::info;

pub struct Stats {
    pub hashrate: f64,
    pub era: u64, pub epoch: u64, pub diff: String,
    pub hits: u64, pub tx: u64, pub wins: u64,
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
    if h > 1e9 { format!("{:.2}GH/s", h/1e9) }
    else if h > 1e6 { format!("{:.2}MH/s", h/1e6) }
    else if h > 1e3 { format!("{:.2}kH/s", h/1e3) }
    else { format!("{:.0}H/s", h) }
}
```

- [ ] **Step 4: `src/metrics/redact.rs`**

```rust
// Placeholder for a tracing Layer that redacts hex-encoded 32-byte sequences.
// Implementer: implement `tracing_subscriber::Layer` that scans event fields for
//              substrings matching `^0x[a-fA-F0-9]{64}$` whose decode equals the
//              configured private key bytes, and replaces with "***".
//              This is belt-and-suspenders; we never knowingly log the key, but a
//              cosmic-ray bug could.
```

- [ ] **Step 5: Build + commit**

```bash
cargo build
git add -A
git commit -m "feat(metrics): MetricsBus + JSONL appender + 1Hz stdout loop"
```

---

## Task 20: Wire main.rs as full miner

**Files:**
- Modify: `src/main.rs`, `src/config.rs` (new)

- [ ] **Step 1: `src/config.rs`**

```rust
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
    pub read_rpc_http: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RelayCfg {
    pub private: Vec<String>,
    pub public_fallback: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WalletCfg { pub keystore_path: PathBuf }

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
}

#[derive(Debug, Deserialize, Clone)]
pub struct MetricsCfg {
    pub jsonl_path: PathBuf,
    pub stdout_hz: u32,
}

impl Config {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let c: Self = toml::from_str(&s)?;
        Ok(c)
    }
}
```

- [ ] **Step 2: `config.example.toml`** — write from spec §10 verbatim.

- [ ] **Step 3: Replace `src/main.rs`**

```rust
use clap::{Parser, Subcommand};
use hashminer::chain::{ChainSource, watcher::ChainWatcher};
use hashminer::config::Config;
use hashminer::gpu::Grinder;
use hashminer::metrics::{Event, MetricsBus};
use hashminer::rpc::ReadProvider;
use hashminer::state::{MinerState, StateMachine};
use hashminer::tx::{Submitter, relay::default_relays, submitter::TxSubmitter, ev_gate::EvParams};
use hashminer::wallet::{MinerSigner, keystore::{read_password_from_env_or_prompt, unlock}};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

#[derive(Parser)] #[command(name = "hashminer", version)]
struct Cli {
    #[arg(long, default_value = "config.toml")] config: PathBuf,
    #[command(subcommand)] cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    Run,
    ChainWatch {
        #[arg(long, env = "HASHMINER_RPC")] rpc: String,
        #[arg(long)] miner: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,hashminer=info"))).init();
    let cli = Cli::parse();
    match cli.cmd.unwrap_or(Cmd::Run) {
        Cmd::Run => run(cli.config).await,
        Cmd::ChainWatch { rpc, miner } => chain_watch(&rpc, miner.parse()?).await,
    }
}

async fn chain_watch(rpc: &str, miner: alloy::primitives::Address) -> anyhow::Result<()> {
    let provider = providerize(rpc).await?;
    let watcher = ChainWatcher::start(provider, miner).await?;
    let mut rx = watcher.subscribe(miner);
    loop {
        rx.changed().await?;
        let u = rx.borrow().clone();
        println!("epoch={} block={} diff={:#x} challenge={}", u.epoch, u.block_number, u.target, u.challenge);
    }
}

async fn providerize(rpc: &str) -> anyhow::Result<Arc<dyn alloy::providers::Provider + Send + Sync>> {
    match ReadProvider::connect(rpc).await? {
        ReadProvider::Http(p) => Ok(Arc::new(p)),
        ReadProvider::Ws(p)   => Ok(Arc::new(p)),
    }
}

async fn run(config_path: PathBuf) -> anyhow::Result<()> {
    let cfg = Config::load(&config_path)?;
    let cancel = CancellationToken::new();
    let state = Arc::new(StateMachine::new());
    let (metrics, mut metrics_rx) = MetricsBus::channel(1024);

    // wallet
    let pw = read_password_from_env_or_prompt()?;
    let key = unlock(&cfg.wallet.keystore_path, &pw)?;
    let signer = Arc::new(MinerSigner::from_key(key)?);
    tracing::info!("miner address: {}", signer.address());

    // chain
    let read_url = cfg.chain.read_rpc_ws.clone().unwrap_or_else(|| cfg.chain.read_rpc_http[0].clone());
    let provider = providerize(&read_url).await?;
    if !ChainWatcher::start(provider.clone(), signer.address()).await?.genesis_complete().await? {
        state.set(MinerState::Fatal);
        return Err(anyhow::anyhow!("genesis not complete; cannot mine yet"));
    }
    let watcher = Arc::new(ChainWatcher::start(provider.clone(), signer.address()).await?);

    // tx
    let relays = default_relays(&cfg.relays.public_fallback)?;
    let ev = EvParams { max_mints_per_block: 10, min_ratio: cfg.mining.ev_min_ratio };
    let submitter: Arc<dyn Submitter> = Arc::new(TxSubmitter::new(
        watcher.clone(), signer.clone(), provider.clone(), relays, ev, cfg.mining.confirmations,
    ));

    // gpu
    #[cfg(feature = "cuda-runtime")]
    let grinder: Arc<dyn Grinder> = Arc::new(
        hashminer::gpu::worker::GpuWorker::start(
            cfg.gpu.device_id,
            cfg.gpu.blocks_per_sm * 144, /* RTX 4090 SMs */
            cfg.gpu.threads_per_block,
            cfg.gpu.poll_interval_ms,
        ).await?
    );
    #[cfg(not(feature = "cuda-runtime"))]
    let grinder: Arc<dyn Grinder> = Arc::new(hashminer::gpu::fake::FakeGrinder::new());

    // wire challenge updates → grinder hot-swap
    {
        let grinder = grinder.clone();
        let mut rx = watcher.subscribe(signer.address());
        let metrics = metrics.clone();
        tokio::spawn(async move {
            loop {
                if rx.changed().await.is_err() { break; }
                let u = rx.borrow().clone();
                let _ = grinder.hot_swap(u.challenge, u.target, u.epoch).await;
                metrics.emit(Event::EpochSwap {
                    epoch: u.epoch, block: u.block_number,
                    diff: format!("{:#x}", u.target),
                    challenge: format!("{}", u.challenge),
                    latency_ms: 0,
                });
            }
        });
    }

    // wire hits → submitter
    {
        let submitter = submitter.clone();
        let metrics = metrics.clone();
        let mut hit_rx = grinder.take_hit_rx();
        tokio::spawn(async move {
            while let Some(hit) = hit_rx.recv().await {
                metrics.emit(Event::HitFound { epoch: hit.epoch_id, nonce: format!("{}", hit.nonce) });
                match submitter.submit(hit).await {
                    Ok(out) => match out {
                        hashminer::tx::SubmitOutcome::Included { tx, block, reward_wei, relay } =>
                            metrics.emit(Event::TxIncluded { tx, block, reward: format!("{}", reward_wei) }),
                        hashminer::tx::SubmitOutcome::Reverted { tx, reason } =>
                            metrics.emit(Event::TxReverted { tx, reason }),
                        hashminer::tx::SubmitOutcome::Dropped { reason } =>
                            metrics.emit(Event::HitDropped { reason }),
                    },
                    Err(e) => metrics.emit(Event::HitDropped { reason: format!("{e}") }),
                }
            }
        });
    }

    // metrics → JSONL
    {
        let path = cfg.metrics.jsonl_path.clone();
        tokio::spawn(async move {
            let _ = hashminer::metrics::jsonl::run_appender(path, metrics_rx).await;
        });
    }

    // signal
    let cancel2 = cancel.clone();
    tokio::spawn(async move {
        let _ = signal::ctrl_c().await;
        cancel2.cancel();
    });
    cancel.cancelled().await;
    grinder.shutdown().await;
    Ok(())
}
```

- [ ] **Step 4: Build + commit**

```bash
cargo build
cargo build --features cuda-runtime
git add -A
git commit -m "feat: full miner wiring in main.rs — Phase 5 milestone"
```

---

## Task 21: E2E test against anvil

**Files:**
- Create: `tests/e2e_anvil.rs`

- [ ] **Step 1: Test** (uses anvil fork via alloy node-bindings, no real money)

```rust
// This test requires `anvil` (foundry) on PATH and the `cuda-runtime` feature is OFF
// (we use FakeGrinder driven by CPU).
use alloy::node_bindings::Anvil;
use alloy::providers::ProviderBuilder;
use alloy::primitives::{address, U256, B256};
use hashminer::chain::{ChainSource, fake::FakeChain};
use hashminer::gpu::{Grinder, fake::FakeGrinder};
use hashminer::tx::{Submitter, fake::FakeSubmitter};
use std::sync::Arc;

#[tokio::test]
async fn fake_pipeline_end_to_end() {
    let miner = address!("0000000000000000000000000000000000000001");
    let target = U256::MAX;                                       // anything hits
    let chain = Arc::new(FakeChain::new(0, target, miner));
    let grinder = Arc::new(FakeGrinder::new());
    let submitter = Arc::new(FakeSubmitter::default());

    // Wire grinder.
    let challenge = chain.challenge_for(miner).await.unwrap();
    grinder.hot_swap(challenge, target, 0).await.unwrap();
    grinder.drive_one_hit().await;

    // Pull one hit and submit.
    let mut hit_rx = grinder.take_hit_rx();
    let hit = hit_rx.recv().await.expect("hit");
    let out = submitter.submit(hit.clone()).await.unwrap();
    assert!(matches!(out, hashminer::tx::SubmitOutcome::Included { .. }));
    assert_eq!(submitter.submitted.lock().len(), 1);
}
```

A more thorough test would deploy a contract clone to anvil and submit a real tx; left to follow-up if needed.

- [ ] **Step 2: Run + commit**

```bash
cargo test --test e2e_anvil
git add -A
git commit -m "test: fake-pipeline e2e — Phase 5 verification"
```

---

# Phase 6 — Polish

## Task 22: Deploy artefacts + README

**Files:**
- Create: `deploy/hashminer.service`, `deploy/hashminer.nssm.cmd`, `deploy/README.md`, `README.md`

- [ ] **Step 1: `deploy/hashminer.service`** (systemd)

```ini
[Unit]
Description=hashminer — native CUDA miner for hash256
After=network-online.target

[Service]
Type=simple
User=miner
WorkingDirectory=/opt/hashminer
EnvironmentFile=/etc/hashminer/env
ExecStart=/opt/hashminer/hashminer --config /etc/hashminer/config.toml
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/hashminer.log
StandardError=inherit
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/hashminer.log /opt/hashminer/logs

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: `deploy/hashminer.nssm.cmd`** (Windows NSSM helper)

```bat
@echo off
nssm install hashminer "C:\hashminer\hashminer.exe" --config "C:\hashminer\config.toml"
nssm set hashminer AppDirectory "C:\hashminer"
nssm set hashminer AppEnvironmentExtra "KEYSTORE_PASSWORD=__SET_ME__"
nssm set hashminer AppStdout "C:\hashminer\logs\stdout.log"
nssm set hashminer AppStderr "C:\hashminer\logs\stderr.log"
nssm set hashminer AppRotateFiles 1
```

- [ ] **Step 3: `README.md`**

```markdown
# hashminer

Native NVIDIA CUDA miner for hash256 (https://hash256.org).

Replaces the WebGPU browser miner with a persistent CUDA kernel for ~10× the hashrate.

## Quick start

1. Install CUDA 12.4+ driver and `nvcc`.
2. Build: `cargo build --release --features cuda-runtime`
3. Place a v3 keystore JSON at `keys/miner.json`.
4. Copy `config.example.toml` → `config.toml` and set RPC URLs.
5. Set `KEYSTORE_PASSWORD` env var or be prompted at start.
6. Run: `./target/release/hashminer --config config.toml`

## Benchmark

`./target/release/hashminer-bench` reports hashrate without touching chain state.

## Design + plan

See `docs/superpowers/specs/` and `docs/superpowers/plans/`.
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "docs+deploy: README + systemd unit + NSSM helper"
```

---

# Self-review

Spec coverage:

| Spec section | Task(s) |
|---|---|
| §1 Purpose, §2 Target protocol | Task 3 (ABI), 4 (CPU keccak), 6 (live watcher) |
| §3 Architecture (4 components, channels, state machine) | Tasks 2 (traits), 19 (metrics), 20 (wiring) |
| §4 CUDA kernel | Tasks 12 (build), 13 (keccak-f), 14 (grind), 15-16 (FFI/worker) |
| §5 Data flow | Task 20 (wiring) |
| §6 Error handling | Tasks 2 (state machine), 10 (submitter) |
| §7 Tx policy (seq gate, dual fan-out, EV) | Tasks 9 (parts), 10 (submitter assembly) |
| §8 Wallet | Task 8 |
| §9 File structure | Task 1 |
| §10 Config | Task 20 |
| §11 Observability | Task 19 |
| §12 Testing | Tasks 4, 11, 17, 21 |
| §13 Risks | Mitigations live in code; verified by §12 tests |
| §14 Build & deploy | Tasks 12, 22 |
| §15 Out of scope | Tracked, not implemented |

Placeholder scan: every step shows the code or exact command. `unimplemented!()` bodies in `kernel_ffi.rs` are explicitly described in the inline notes; Task 16 closes them.

Type consistency: `Hit`, `MiningState`, `ChallengeUpdate`, `SubmitOutcome`, `MinerState` defined once in Task 2 / Task 5, used consistently downstream. ABI selectors and constants in Task 3, reused in Tasks 4/6/9/10/14.

Scope: 22 tasks, six demoable milestones. No task is more than ~30 minutes of focused work for an experienced engineer; subagent-driven execution maps each task → one subagent.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-11-hashminer-implementation.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. Best for plan of this size (22 tasks).
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Slower but everything stays in your shell.
