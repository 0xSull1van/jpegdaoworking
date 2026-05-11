# hashminer — CLI GPU Miner for hash256

**Design doc**
**Date:** 2026-05-11
**Status:** v1 MVP design, approved by user for implementation

## 1. Purpose

Native NVIDIA CUDA replacement for the browser/WebGPU miner at `https://hash256.org/mine`. The browser implementation tops out at ~1.14 GH/s on RTX 40-series due to WebGPU + WASM overhead. Reference: `ArturBieniek4/sha3x_cudaminer` reaches **18.8 GH/s** on RTX 4090 for Keccak-256 with 41-byte input. Our input is 64 bytes (still one rate block of Keccak-256), so target throughput is **10–15 GH/s on RTX 4090** (~10× the WebGPU baseline).

Out of scope for v1: multi-GPU, non-NVIDIA hardware, pool/stratum mode, GUI, automatic key rotation, restart-from-checkpoint.

## 2. Target protocol

Resolved against verified Solidity source via Sourcify (`docs/superpowers/specs/_contract/Hash.sol`). All values below are quoted from the actual contract.

- **Network:** Ethereum mainnet, `chainId = 1`
- **Contract:** `0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc` (`contract Hash is ERC20, IHooks, ReentrancyGuard`, solc `^0.8.26`, immutable)

### 2.1 PoW rule (exact, verified)

```solidity
// _challenge(miner) — line 384
challenge = keccak256(abi.encode(block.chainid, address(this), miner, _epoch()))

// mine(nonce) — line 354
result    = keccak256(abi.encode(challenge, nonce))
require(uint256(result) < currentDifficulty, "InsufficientWork")

// _epoch() — line 393
epoch     = block.number / EPOCH_BLOCKS
```

**Encoding is `abi.encode` (NOT `encodePacked`).** Every value is padded to 32 bytes:
- `_challenge` input: `chainid` (uint256 BE, 32B) ‖ `address(this)` (left-padded to 32B) ‖ `miner` (left-padded to 32B) ‖ `epoch` (uint256 BE, 32B) = **128 bytes total**
- `mine` inner hash input: `challenge` (bytes32, 32B) ‖ `nonce` (uint256 BE, 32B) = **64 bytes total**

These are the exact byte-counts the GPU kernel will feed into the Keccak sponge. 64-byte input fits in one rate block of Keccak-256 (rate = 136B).

### 2.2 Contract constants

| Constant | Value |
|---|---|
| `TOTAL_SUPPLY` | 21 000 000 × 10¹⁸ |
| `MINING_SUPPLY` | 18 900 000 × 10¹⁸ |
| `ERA_MINTS` | 100 000 |
| `BASE_REWARD` | 100 × 10¹⁸ (first era) |
| `EPOCH_BLOCKS` | 100 |
| `ADJUSTMENT_INTERVAL` | 2 016 mints |
| `TARGET_BLOCKS_PER_MINT` | 5 |
| `MAX_MINTS_PER_BLOCK` | 10 |
| Initial difficulty | `type(uint256).max >> 32` = `0x00000000FFFF...FFFF` (32 leading zero bits) |
| Halving | `reward = era < 64 ? BASE_REWARD >> era : 0` where `era = totalMints / ERA_MINTS` |
| Difficulty retarget | ±4× max per `ADJUSTMENT_INTERVAL` mints, formula `next = old * taken_blocks / (ADJUSTMENT_INTERVAL * TARGET_BLOCKS_PER_MINT)` |

### 2.3 ABI we use

**Write:**
- `function mine(uint256 nonce) external nonReentrant` — single argument, only `nonce`

**Read (view):**
- `function currentDifficulty() public view returns (uint256)` — auto-generated, state var
- `function getChallenge(address miner) external view returns (bytes32)` — gives us challenge directly; useful for verification, but we compute locally to save RPC calls
- `function miningState() external view returns (uint256 era, uint256 reward, uint256 difficulty, uint256 minted, uint256 remaining, uint256 epoch, uint256 epochBlocksLeft_)` — **single-call** snapshot, primary read path for ChainWatcher
- `function currentReward() external view returns (uint256)`
- `function epochBlocksLeft() external view returns (uint256)`
- `function mintsInBlock(uint256 blockNumber) public view returns (uint256)` — auto-gen mapping; used by EV gate to check block cap pressure
- `function totalMints() public view returns (uint256)` — auto-gen
- `function genesisComplete() public view returns (bool)` — startup precondition

**Events:**
- `event Mined(address indexed miner, uint256 nonce, uint256 reward, uint256 era)` — listen for our `miner` address to confirm receipts and bookkeep balances
- `event Halving(uint256 era, uint256 reward)`
- `event DifficultyAdjusted(uint256 old, uint256 next, uint256 takenBlocks)`

**Reverts to handle:**
- `InsufficientWork()` — `result >= currentDifficulty` (stale challenge or competing miner won)
- `ProofAlreadyUsed()` — `(msg.sender, nonce, epoch)` triple was already used
- `BlockCapReached()` — current block already has 10 mints
- `SupplyExhausted()` — mining supply done (will not happen for ~years)
- `GenesisNotComplete()` — genesis sale still open; cannot mine yet

### 2.4 Startup precondition

`mine()` reverts with `GenesisNotComplete()` until the 1 050 000 HASH genesis sale closes. Miner must check `genesisComplete()` at startup; if false → `Fatal` exit with operator message.

## 3. Architecture

### 3.1 Components

Single Rust binary `hashminer`. Four cooperating subsystems behind explicit trait boundaries (`ChainSource`, `Grinder`, `Submitter`, `Metrics`). Traits enable in-memory fakes for testing the race-condition logic without CUDA or RPC.

```
┌────────────────────────────────────────────────────────────────────────┐
│                            hashminer                                   │
│                                                                        │
│   ┌──────────────┐   watch::      ┌──────────────┐   mpsc(16)         │
│   │ChainWatcher  │─ challenge ───▶│  GpuWorker   │─── hits ─────┐     │
│   │              │                │              │              │     │
│   │ - newHeads   │                │ - persistent │              ▼     │
│   │ - epoch poll │                │   kernel     │   ┌────────────┐   │
│   │ - diff reads │                │ - poll pinned│   │TxSubmitter │   │
│   │ - mint event │                │   buffer     │   │ - seq gate │   │
│   └──────┬───────┘                └──────────────┘   │ - dual fan │   │
│          │                                            └──────┬─────┘   │
│          ▼ receipts                                          │         │
│   ┌──────────────────────────────────────────────────────────▼──┐     │
│   │                      StateMachine                            │     │
│   │  Healthy / RpcDegraded / ChallengeStale /                    │     │
│   │  GpuFault / WalletLocked / Paused / Fatal                    │     │
│   └──────────────────────────┬──────────────────────────────────┘     │
│                              │                                         │
│                              ▼                                         │
│   ┌──────────────────────────────────────────────────────────────┐    │
│   │  Metrics (stdout 1Hz + JSONL append)                         │    │
│   └──────────────────────────────────────────────────────────────┘    │
│                                                                        │
│   RpcPool (read)             Wallet (Zeroizing<SecretKey>)             │
│   ├─ Alchemy/Infura          └─ eth-keystore v3 unlock at startup      │
│   └─ self-hosted (optional)                                            │
│                                                                        │
│   RelayPool (submit)                                                   │
│   ├─ MEV-Blocker  (rpc.mevblocker.io/fast)   primary                   │
│   ├─ Flashbots    (rpc.flashbots.net/fast)   primary                   │
│   └─ Public RPC                              fallback                  │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Channel topology

| Edge | Pattern | Capacity | Drop policy |
|---|---|---|---|
| ChainWatcher → GpuWorker (challenge) | `tokio::sync::watch` | latest-value | lagging consumer reads latest, never blocks |
| ChainWatcher → TxSubmitter (heads/receipts) | `tokio::sync::broadcast` | 64 | lossy ok |
| GpuWorker → TxSubmitter (hits) | `tokio::sync::mpsc` | 16 | drop **oldest** on full, bump `hits_dropped` |
| any → Metrics | `tokio::sync::mpsc` | 1024 | drop **oldest** on full |
| any → StateMachine | `tokio::sync::mpsc` | 256 | drop **oldest** on full |
| Shutdown | `tokio_util::sync::CancellationToken` | — | cooperative |

Rationale: grinding and submitting are independent failure domains. Hits channel filling = submitter is wedged (the real bug); newer hits are statistically equivalent to older ones, so drop-oldest surfaces the problem without poisoning the kernel.

### 3.3 State machine

Explicit enum, every subsystem reports transitions into it. Each state names the trigger and the recovery action.

| State | Trigger | Recovery |
|---|---|---|
| `Healthy` | all subsystems live | — |
| `RpcDegraded` | WS drop, repeated HTTP 5xx | fallback to public RPC; keep grinding on last known challenge; cap age at 5 blocks then → `ChallengeStale` |
| `ChallengeStale` | no new head > 2× expected block time, OR last challenge update > 5 blocks old | pause submission, keep grinding (cheap insurance); resume on healthy head |
| `GpuFault` | CUDA error, OOM, kernel watchdog hit | tear down context, re-init once; second fault → `Fatal` |
| `WalletLocked` | keystore decrypt fail | `Fatal` at startup only (no recovery path) |
| `Paused` | operator signal (SIGUSR1) | resume on operator signal |
| `Fatal` | unrecoverable | flush JSONL, exit non-zero |

Key invariant: **RPC loss never stops the kernel; GPU loss never poisons the wallet.**

## 4. CUDA kernel design

### 4.1 Memory layout (`kernel/keccak_grinder.cu`)

```cuda
// Double-buffered challenge: hot-swap without kernel restart
__constant__ uint8_t  c_challenge[2][32];
__constant__ uint64_t c_target[2][4];          // 256-bit, big-endian
__constant__ uint32_t c_epoch_id[2];           // monotonic
__device__   uint32_t d_active_idx;            // 0 or 1, host flips this

// Per-thread work distribution
__device__   uint64_t d_nonce_counter;         // global atomic

// Result return path (pinned host memory)
__device__   uint32_t d_hit_count;             // atomic write head
__device__   struct Hit {
    uint8_t  nonce[32];
    uint8_t  hash[32];
    uint32_t epoch_id;                          // captured at hit time
} d_hits[16];

// Control
__device__   uint32_t d_should_stop;
```

### 4.2 Persistent kernel

**Inner hash input layout — exactly 64 bytes (one Keccak rate block after padding):**

```
bytes  0..31:  challenge (bytes32, big-endian as-is)
bytes 32..63:  nonce (uint256, big-endian, MSB at byte 32)
```

After data, Keccak-256 pad rule (Ethereum/legacy SHA3, NOT FIPS-202): byte 64 = `0x01`, bytes 65..134 = `0x00`, byte 135 = `0x80`. Single rate-block absorb, then permutation, then squeeze first 32 bytes of state.

```cuda
extern "C" __global__ void grind() {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t base = atomicAdd(&d_nonce_counter, BATCH_PER_THREAD);

    while (!d_should_stop) {
        // Read active buffer index once per batch — cheap, avoids torn reads
        uint32_t idx = d_active_idx;
        uint32_t my_epoch = c_epoch_id[idx];

        for (uint32_t i = 0; i < BATCH_PER_THREAD; i++) {
            // Build 64-byte input: [challenge | nonce_BE]
            uint64_t state[25];
            #pragma unroll
            for (int w = 0; w < 25; w++) state[w] = 0;

            // Absorb challenge (4 × uint64_t) into rate
            state[0] ^= ((const uint64_t*)c_challenge[idx])[0];
            state[1] ^= ((const uint64_t*)c_challenge[idx])[1];
            state[2] ^= ((const uint64_t*)c_challenge[idx])[2];
            state[3] ^= ((const uint64_t*)c_challenge[idx])[3];

            // Absorb nonce as 32B big-endian uint256
            uint64_t nonce_lo = base + i;
            uint64_t nonce_hi = ((uint64_t)tid) << 32;        // upper bits use tid for disjoint ranges
            // Big-endian layout: bytes 32..47 = high 16 bytes (all zero), bytes 48..63 = low 16 bytes
            state[4] ^= 0;                                     // bytes 32..39
            state[5] ^= 0;                                     // bytes 40..47
            state[6] ^= __byte_perm_be(nonce_hi);              // bytes 48..55
            state[7] ^= __byte_perm_be(nonce_lo);              // bytes 56..63

            // Padding: 0x01 at byte 64 (state[8] low byte), 0x80 at byte 135 (state[16] high byte)
            state[8]  ^= 0x01ULL;
            state[16] ^= 0x8000000000000000ULL;

            keccak_f1600(state);                               // 24 rounds, fully unrolled

            // Result is first 32 bytes (state[0..3]); compare big-endian against target
            if (less_than_target_be(state, c_target[idx])) {
                uint32_t slot = atomicAdd(&d_hit_count, 1);
                if (slot < 16) {
                    store_nonce_be(d_hits[slot].nonce, nonce_hi, nonce_lo);
                    store_hash_be(d_hits[slot].hash, state);
                    d_hits[slot].epoch_id = my_epoch;          // captured atomically with hit
                }
            }
        }
        base = atomicAdd(&d_nonce_counter, BATCH_PER_THREAD);
    }
}
```

Key kernel details:
- Keccak state lives entirely in registers (25 × `uint64_t` = 200 bytes per thread). Ada Lovelace has 256 registers per thread which fits comfortably.
- `keccak_f1600` is fully unrolled (24 rounds × 5 steps = ~120 ops, compiler-friendly).
- Nonce upper 32 bits = `tid`, lower 64 bits = monotonic counter → each thread has a disjoint 2⁶⁴ nonce range, exhaustion impossible in practice.
- Big-endian byte order is preserved end-to-end to match Solidity's `abi.encode` layout.

### 4.3 Hot-swap protocol (host side)

```rust
// 1. Pick inactive buffer (1 - active)
// 2. Write new challenge + target + epoch_id to inactive slot
// 3. Fence + atomic write d_active_idx
// 4. Reset d_nonce_counter (optional; safe to leave for monotonic progress)
// 5. Reset d_hit_count

cuda_memcpy_to_symbol_async(c_challenge[next], &new_challenge, stream)?;
cuda_memcpy_to_symbol_async(c_target[next],    &new_target,    stream)?;
cuda_memcpy_to_symbol_async(c_epoch_id[next],  &new_epoch,     stream)?;
stream.synchronize()?;
cuda_atomic_store(d_active_idx, next, stream)?;
```

**Why double-buffered:** a kernel iteration that started reading `c_challenge[0]` continues with consistent `c_challenge[0]` + `c_target[0]` + `c_epoch_id[0]`. No torn reads. Stale hits get tagged with their epoch_id and filtered at the host boundary, never lost mid-batch.

### 4.4 Kernel tuning knobs (CLI/TOML config, not hardcoded)

- `threads_per_block` (default 256)
- `blocks_per_sm` (default 4)
- `batch_per_thread` (default 1024)
- `poll_interval_ms` (default 50)

Ada SM counts differ across RTX 4070/4080/4090; defaults assume 4090, user can tune via config.

### 4.5 Stand-alone benchmark binary

`hashminer-bench` runs the kernel against a synthetic target = max for N seconds, emits hashrate. Used for tuning and for verifying GPU health independent of chain state.

## 5. Data flow scenarios

### 5.1 Found valid nonce

1. Kernel writes `Hit { nonce, hash, epoch_id }` to `d_hits[slot]`, increments `d_hit_count`
2. Host poll thread (50ms tick) reads `d_hit_count` via mapped pinned memory
3. Reads new hits, resets count to 0
4. For each hit: verify `epoch_id == current_local_epoch` — drop stale
5. Sends `Hit` over mpsc to TxSubmitter
6. TxSubmitter: CPU-verify hash (paranoia check via `tiny-keccak`)
7. **EV gate:** check `expected_reward * P(inclusion) > tx_cost`; if not, drop hit and increment `hits_skipped_unprofitable`
8. **Final epoch re-check** via cached chain head: is `epoch` still current at the head we plan to land in?
9. Build tx: `to=contract, data=encode_mine_selector(nonce), gas=..., maxFee=..., maxPriorityFee=tip`
10. Sign with `Zeroizing<SecretKey>` from keystore
11. **Sequential gate:** only one in-flight tx at a time; wait for receipt or timeout before next submit
12. Fire-and-forget to both relays + (after delay) public RPC if neither relay confirms
13. First `eth_getTransactionReceipt` success wins; log `relay_win_rate`
14. Wait `K` confirmations (default 2) for reorg protection; only then mark mint as won in metrics
15. Release sequential gate

### 5.2 Epoch rotation

1. ChainWatcher receives newHead with `blockNumber`
2. New `epoch = blockNumber / 100` (Solidity integer division floor, per `_epoch()` line 393)
3. If `epoch != last_epoch`: compute `new_challenge = keccak256(chainId ‖ contract ‖ minerAddr ‖ new_epoch)` on CPU using `tiny-keccak`
4. Read `currentDifficulty` via `eth_call` (it may also have retargeted)
5. Publish `ChallengeUpdate { challenge, target, epoch }` to watch channel
6. GpuWorker: hot-swap protocol (§4.3), bumps `challenge_swap_latency_ms` metric
7. Kernel notices `d_active_idx` change at next batch boundary

### 5.3 Startup

1. Parse CLI > env > `config.toml` (later wins lower precedence)
2. Unlock keystore (`KEYSTORE_PASSWORD` env or interactive prompt)
3. Derive miner address; log it
4. Connect to read-RPC; verify chainId == 1, contract code hash matches expected
5. Read current epoch + difficulty + era state
6. Initialize CUDA: open device, load PTX, alloc constants/globals, pinned host buffer
7. Compute initial challenge + target, write to `c_*[0]`, set `d_active_idx = 0`
8. Subscribe to `newHeads` (WS preferred, polling fallback)
9. Launch persistent kernel (single CUDA launch for process lifetime)
10. Spawn tokio tasks: ChainWatcher, GpuWorker::poll_hits, TxSubmitter, Metrics, StateMachine
11. Install signal handlers: SIGINT/SIGTERM → set CancellationToken; SIGUSR1 → toggle Paused

## 6. Error handling

| Condition | Action |
|---|---|
| RPC connection lost (read) | exp backoff next endpoint; if all down → `RpcDegraded` |
| RPC connection lost (relay) | try next relay; if all down → public RPC fallback with `WARN` log |
| Tx revert (block cap, stale epoch) | log + `tx_reverted` counter; drop hit; no retry |
| Tx underpriced (relay reject) | bump tip 20%, retry up to 3 times; then drop |
| Tx stuck > 3 blocks no receipt | replacement tx (same nonce, +20% tip); mark as `tx_replaced` |
| CUDA recoverable error | tear down context, re-init once → state `GpuFault`; second error → `Fatal` |
| Kernel watchdog miss (no hits + no progress in 30s) | tear down + restart (likely driver hang) |
| Keystore unlock failed | exit code 3, immediate |
| Wrong chainId / contract code mismatch | exit code 4, immediate |
| Stale hit (epoch out of sync) | drop, `hits_stale` counter |
| Pinned buffer overflow (>16 hits) | warn; theoretically impossible at production difficulty |
| **Reorg detected** (receipt block hash changes) | revert "won" mark; resume from latest head |

## 7. Tx submission policy

- **Sequential gate**: only one in-flight tx per miner address. Track `(local_pending_nonce, submit_time)`. Release on receipt OR replacement timeout.
- **Dual fan-out**: same signed tx sent to MEV-Blocker AND Flashbots Protect simultaneously. No cancel of loser.
- **Public fallback**: if both private relays return error or no receipt within 2 blocks → send same tx to public RPC.
- **Replacement**: if no receipt in 3 blocks → resign with same nonce, +20% tip, refan-out.
- **EV gate**: pre-submit check `reward_now * P(win|10_per_block_cap) > gas_estimate * gasPrice`. P(win) is configurable, defaults to 0.7 conservatively. If failed, drop with `hits_skipped_unprofitable` counter.
- **Reorg protection**: don't count a mint as won until `K=2` confirmations.

## 8. Wallet & secrets

- Format: Ethereum keystore v3 JSON (geth-compatible)
- Decrypt: `eth-keystore` crate, password from interactive prompt or `KEYSTORE_PASSWORD` env var
- In-memory: `Zeroizing<[u8; 32]>` wrapping the secret scalar; explicit `Drop` clears
- Logging: `tracing` subscriber with a redaction layer at the formatter level — *not* at call sites — to prevent accidental log of the key
- Address derivation: standard secp256k1 → keccak256 → last 20 bytes

## 9. File structure

```
hashminer/
├── Cargo.toml
├── build.rs                          # nvcc compile kernel → PTX, embed at build time
├── config.example.toml
├── kernel/
│   ├── keccak_grinder.cu             # persistent kernel
│   ├── keccak_device.cuh             # keccak-f[1600] device fns
│   └── nonce_encode.cuh
├── src/
│   ├── main.rs                       # CLI, wiring, signal handling
│   ├── config.rs                     # toml + env + CLI overrides via clap
│   ├── state.rs                      # StateMachine enum + transitions
│   ├── chain/
│   │   ├── mod.rs                    # trait ChainSource
│   │   ├── watcher.rs                # ChainWatcher (live impl)
│   │   ├── fake.rs                   # in-mem fake for tests
│   │   ├── contract.rs               # alloy-sol! bindings
│   │   └── challenge.rs              # CPU keccak for challenge computation
│   ├── gpu/
│   │   ├── mod.rs                    # trait Grinder
│   │   ├── worker.rs                 # GpuWorker (live CUDA impl)
│   │   ├── fake.rs                   # CPU-side fake grinder for tests
│   │   ├── kernel_ffi.rs             # cust bindings, pinned alloc, hot-swap
│   │   └── ptx.rs                    # embedded PTX bytes
│   ├── tx/
│   │   ├── mod.rs                    # trait Submitter
│   │   ├── submitter.rs              # TxSubmitter (live)
│   │   ├── fake.rs                   # in-mem fake
│   │   ├── relay.rs                  # MEV-Blocker / Flashbots / public drivers
│   │   ├── nonce_manager.rs          # sequential gate, replacement
│   │   └── ev_gate.rs                # profitability check
│   ├── wallet/
│   │   ├── mod.rs
│   │   ├── keystore.rs               # v3 unlock
│   │   └── signer.rs                 # Zeroizing key, signing
│   ├── metrics/
│   │   ├── mod.rs
│   │   ├── stdout.rs                 # 1Hz human line
│   │   ├── jsonl.rs                  # event-by-event append
│   │   └── redact.rs                 # tracing redaction layer
│   └── rpc.rs                        # RpcPool, health checks, round-robin
├── tests/
│   ├── kernel_correctness.rs         # CPU vs GPU keccak vectors (CUDA feature)
│   ├── stale_hit_filter.rs           # fake Grinder + fake ChainSource
│   ├── epoch_hotswap.rs              # fake ChainSource, race semantics
│   ├── tx_nonce_gate.rs              # fake Submitter, sequential gate
│   └── e2e_anvil.rs                  # anvil fork + deployed contract
├── benches/
│   └── hashrate.rs                   # criterion harness
└── bin/
    └── hashminer-bench.rs            # standalone GPU benchmark
```

## 10. Configuration

`config.toml` example:
```toml
[chain]
read_rpc_ws  = "wss://eth-mainnet.g.alchemy.com/v2/<KEY>"
read_rpc_http = ["https://eth-mainnet.g.alchemy.com/v2/<KEY>", "https://mainnet.infura.io/v3/<KEY>"]
contract     = "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc"
chain_id     = 1

[relays]
private   = ["https://rpc.mevblocker.io/fast", "https://rpc.flashbots.net/fast"]
public_fallback = "https://eth-mainnet.g.alchemy.com/v2/<KEY>"
fallback_after_blocks = 2

[wallet]
keystore_path = "./keys/miner.json"
# password via KEYSTORE_PASSWORD env or interactive

[mining]
max_tip_gwei      = 3.0
ev_min_ratio      = 1.2     # only submit if expected_reward / tx_cost > 1.2
confirmations     = 2

[gpu]
device_id         = 0
threads_per_block = 256
blocks_per_sm     = 4
batch_per_thread  = 1024
poll_interval_ms  = 50

[metrics]
jsonl_path     = "./logs/metrics.jsonl"
stdout_hz      = 1
```

## 11. Observability

Metrics emitted as JSONL events and aggregated for 1Hz stdout line:

| Metric | Why |
|---|---|
| `hashrate_hps` | core sanity |
| `total_hashes` | for ETA |
| `era`, `epoch`, `difficulty` | chain state |
| `challenge_swap_latency_ms` | hot-swap correctness |
| `hits_total`, `hits_stale`, `hits_dropped`, `hits_skipped_unprofitable` | submission funnel |
| `tx_submitted`, `tx_reverted`, `tx_replaced` | tx layer health |
| `relay_win_rate{mev_blocker,flashbots,public}` | submission strategy |
| `hit_to_submit_ms`, `submit_to_receipt_ms` | latency budget |
| `state_transition{from,to}` | for incident review |
| `gpu_temp_c`, `gpu_power_w` | optional via NVML |

Stdout 1Hz line:
```
[14:32:01] hashrate=8.7GH/s diff=0x...ffffff era=1 epoch=250707 hits=2 tx=2 wins=2 (mev=1,fb=1) bal=200HASH
```

## 12. Testing strategy

**Layer 1 — kernel correctness** (`tests/kernel_correctness.rs`, requires `cuda-runtime` feature):
- 1000 random `(challenge, nonce)` pairs: GPU output byte-equal to `tiny-keccak::Keccak::v256`
- Force-hit: `target = max` → kernel must report hit on first nonce
- Hot-swap atomicity: swap challenge mid-grind, verify all post-swap hits carry new epoch_id

**Layer 2 — race & state logic** (`tests/stale_hit_filter.rs`, `tests/epoch_hotswap.rs`, no CUDA needed):
- Drive `Grinder` + `ChainSource` fakes; verify TxSubmitter drops stale hits when epoch advances mid-flight
- Verify watch::channel update propagates to GpuWorker poll in <100ms
- Property test: random sequence of (hit, epoch_advance, hit) → only current-epoch hits reach submitter

**Layer 3 — tx layer** (`tests/tx_nonce_gate.rs`):
- Sequential gate holds until receipt OR timeout
- Replacement tx uses same nonce with +20% tip
- Dual fan-out: first relay receipt wins; both relays acked but no inclusion → escalate to public

**Layer 4 — e2e** (`tests/e2e_anvil.rs`):
- Spin up `anvil` forked from mainnet
- Deploy contract clone with low difficulty
- Run miner with small grid; assert hit + tx + receipt + Mint event within 60s
- Test epoch rotation by mining 100 anvil blocks

**Layer 5 — benchmarks** (`benches/`, `bin/hashminer-bench.rs`):
- Criterion bench for kernel launch + first-batch latency
- Standalone benchmark binary for hashrate tuning (no chain interaction)

**CI matrix:**
- All Layer 2/3 tests on every Linux x86_64 runner
- Layer 1/4/5 gated on `cuda-runtime` feature, run on self-hosted CUDA runner if available; otherwise skip

## 13. Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| ~~Wrong `abi.encode` vs `encodePacked`~~ | RESOLVED | Verified source (Sourcify) confirms `abi.encode`; encoding fixed in §2.1 |
| Stale-hit race at epoch boundary | HIGH | Kernel tags hits with epoch_id (§4.2); submitter final re-check (§5.1) |
| GPU context recovery flaky | MED | Single re-init attempt then Fatal; rely on external supervisor (systemd/NSSM) for restart |
| Gas spike makes mining unprofitable | MED | EV gate (§7); user-tunable threshold |
| Reorg invalidates included mint | LOW | K-confirmation rule (§7) |
| Private relay both unavailable | LOW | Public RPC fallback; degrades to public-mempool speed |
| Block-cap collision (10 mints/block) → revert | LOW | EV gate uses `mintsInBlock(blockNumber)` view to skip submission if cap close |
| Difficulty retarget mid-grind | LOW | ChainWatcher polls `currentDifficulty` on each new head and triggers hot-swap (target buffer also double-buffered) |
| Genesis not yet closed at startup | LOW | Startup check `genesisComplete()` → Fatal exit with operator message |

## 14. Build & deploy

- Toolchain: Rust stable, `nvcc` 12.4+, CUDA driver 550+
- `build.rs` invokes `nvcc -arch=compute_89` (Ada) → PTX → `include_bytes!` into binary
- Cross-arch builds compile multiple `-gencode` flags (sm_80 Ampere, sm_89 Ada, sm_90 Hopper)
- Single static binary; only runtime dep is CUDA driver (already installed wherever NVIDIA GPU works)
- Supervisor: provide `systemd unit` and `NSSM service definition` in `deploy/`

## 15. Out-of-scope (explicitly deferred to v2+)

- Multi-GPU on one host (separate addresses per GPU)
- AMD/Intel via OpenCL/Vulkan
- Stratum-like pool mode with worker coordinator
- Auto-tuning of grid/block dimensions
- Hot keystore rotation
- WebUI / TUI dashboard
- Multi-network / testnet support beyond anvil
