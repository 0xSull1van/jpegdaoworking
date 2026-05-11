# hashminer

Native NVIDIA CUDA miner for [hash256](https://hash256.org) — replaces the browser/WebGPU miner with a persistent CUDA kernel and headless tx submission via MEV-Blocker + Flashbots Protect.

## Features

- Persistent CUDA kernel with **double-buffered `__constant__` challenge** for zero-downtime epoch hot-swap
- **Dual private-relay fan-out** (MEV-Blocker + Flashbots Protect) with public RPC fallback
- **Sequential nonce gate** + K-confirmation reorg protection
- **EV gate** — drops unprofitable hits (gas-aware)
- **v3 keystore** unlock with `Zeroizing` in-memory key
- **JSONL metrics** + 1Hz stdout
- Cross-platform: Linux (Vast.ai, dedicated rigs) and Windows
- Supports Ada Lovelace (4060 Ti / 4080 / 4090) and Blackwell (5090) out of the box

## Measured hashrate

| GPU | SMs | Hashrate (our kernel) | WebGPU comparison |
|---|---|---|---|
| RTX 4060 Ti 16GB | 34 | **~1.4 GH/s** | 1.14 GH/s |
| RTX 5090 (expected) | 170 | **~7 GH/s** | n/a |

---

# Quick start on Vast.ai (Linux + Jupyter)

Assuming you rented a Vast.ai instance with a CUDA 12.x or 13.x image and have Jupyter open.

## 1. Open terminal in Jupyter

`+` → `Terminal` (right side of Launcher). All commands below run there.

## 2. One-shot setup

```bash
# Install Rust (skip if already present — check `rustc --version`)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source $HOME/.cargo/env

# Clone repo
cd /workspace
git clone https://github.com/0xSull1van/jpegdaoworking.git hashminer
cd hashminer

# Verify CUDA toolkit
nvcc --version              # should print 12.x or 13.x
nvidia-smi                  # should list your GPU(s)

# Build (3-5 minutes first time)
cargo build --release --features cuda-runtime
```

If `cargo build` fails with `nvcc fatal: Unsupported gpu architecture 'compute_89'`, your CUDA toolkit is older than 11.8. Update to CUDA 12.x.

For RTX 5090 (Blackwell sm_120) — same binary works. PTX emitted as `compute_89` is forward-compatible via driver JIT. For optimal native code on Blackwell, rebuild with `CUDA_ARCH=compute_120` env var (see "Tuning for 5090" below).

## 3. Import private key

```bash
mkdir -p keys
./target/release/hashminer import-key --out keys --name miner.json
# Prompts (hidden input):
#   Private key (0x prefix optional): paste here
#   Encryption password: choose one
#   Confirm password: repeat
# Writes keys/miner.json and prints the derived 0x address.
```

Save the password somewhere safe. Without it `keys/miner.json` is unrecoverable.

## 4. Configure

```bash
cp config.example.toml config.toml
nano config.toml         # or vim
```

Verify `[gpu] sm_count` matches your card. Common values:

| GPU | sm_count |
|---|---|
| RTX 4060 Ti | 34 |
| RTX 4070 | 46 |
| RTX 4070 Ti | 60 |
| RTX 4080 | 76 |
| RTX 4090 | 128 |
| RTX 5070 Ti | 96 |
| RTX 5080 | 112 |
| RTX 5090 | 170 |

Public RPCs are pre-configured (publicnode.com + llamarpc + ankr). If you have an Alchemy/Infura key, swap them in — they're more reliable for 24/7.

## 5. Fund the address

Send **0.05-0.1 ETH** to the address `hashminer import-key` printed. This pays for `mine(nonce)` transaction gas. Each successful mint costs ~50k gas × current gas price.

Check current gas: https://etherscan.io/gastracker

## 6. Run miner

In Vast.ai Jupyter terminal, **use `tmux`** so the miner survives terminal disconnect:

```bash
tmux new -s miner
export KEYSTORE_PASSWORD='your_password_here'
export RUST_LOG='info,hashminer=info'
./target/release/hashminer --config config.toml run
```

Detach from tmux: `Ctrl+B`, then `D`. Reattach later: `tmux attach -t miner`.

## 7. Monitor

In another terminal tab:
```bash
tail -f logs/metrics.jsonl | head -200
```

Or pretty-print with `jq`:
```bash
apt-get install -y jq           # if not installed
tail -f logs/metrics.jsonl | jq -c '.event'
```

Watch for:
- `EpochSwap` — initial + every ~20 min when block_number crosses a 100-block boundary
- `HitFound` — GPU found valid nonce
- `TxSubmitted` — sent to relay
- `TxIncluded` — landed in a block, you got HASH

Check balance on etherscan:
```
https://etherscan.io/token/0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc?a=YOUR_ADDRESS
```

## Stop miner

```bash
tmux attach -t miner
# Then Ctrl+C inside tmux
```

Or kill from any shell:
```bash
pkill -INT hashminer
```

---

# Multi-GPU setup (Vast.ai 4× 5090, etc.)

Challenge in hash256 is **address-bound** — running multiple GPUs against the same address gives no benefit (kernel divides nonce-space, but only one tx can mint per block per address).

**Correct multi-GPU usage**: separate address per GPU.

```bash
# Create N keystores
for i in 1 2 3 4; do
  mkdir -p keys
  ./target/release/hashminer import-key --out keys --name "miner_$i.json"
done

# Per-GPU configs
for i in 1 2 3 4; do
  cp config.example.toml config_gpu$i.toml
  sed -i "s|miner.json|miner_$i.json|" config_gpu$i.toml
  sed -i "s|device_id         = 0|device_id         = $((i-1))|" config_gpu$i.toml
done

# Run each in its own tmux window
for i in 1 2 3 4; do
  tmux new-window -t miner: -n "gpu$i" \
    "KEYSTORE_PASSWORD=pw$i ./target/release/hashminer --config config_gpu$i.toml run"
done
```

Each GPU needs its own ETH balance for gas. Recommend funding 0.02 ETH per address.

---

# Tuning for RTX 5090 (Blackwell sm_120)

Optimal native PTX:

```bash
CUDA_ARCH=compute_120 cargo build --release --features cuda-runtime
```

If `compute_120` errors out (older nvcc), use `compute_90` (Hopper):

```bash
CUDA_ARCH=compute_90 cargo build --release --features cuda-runtime
```

Config tuning for 5090:

```toml
[gpu]
device_id         = 0
threads_per_block = 256
blocks_per_sm     = 4     # try 4-8
batch_per_thread  = 8192
poll_interval_ms  = 50
sm_count          = 170
```

Run `./target/release/hashminer-bench` first — should report 5-8 GH/s on a 5090. If lower, try `HASHMINER_BENCH_SMS=170` env var.

---

# Quick start on Windows

Requires:
- CUDA Toolkit 12.x (https://developer.nvidia.com/cuda-downloads)
- Visual Studio Build Tools 2022 with "Desktop development with C++" workload
- Rust stable (https://rustup.rs)

Open **"x64 Native Tools Command Prompt for VS 2022"** (NOT regular PowerShell — needs `cl.exe` in PATH):

```cmd
git clone https://github.com/0xSull1van/jpegdaoworking.git hashminer
cd hashminer
cargo build --release --features cuda-runtime
mkdir keys
.\target\release\hashminer.exe import-key --out keys --name miner.json
copy config.example.toml config.toml
notepad config.toml
set KEYSTORE_PASSWORD=your_password
.\target\release\hashminer.exe --config config.toml run
```

---

# Architecture

- `src/chain/` — `ChainWatcher` polls `miningState()` every 4s, publishes `ChallengeUpdate` via tokio watch
- `src/gpu/` — `GpuWorker` (cudarc 0.19) launches a persistent grind kernel on a dedicated compute stream; double-buffered constant memory enables atomic hot-swap
- `src/tx/` — `TxSubmitter` with sequential gate, dual-relay fan-out, receipt watch
- `src/wallet/` — eth-keystore v3 unlock + `Zeroizing<[u8;32]>` + alloy `PrivateKeySigner`
- `src/metrics/` — JSONL appender + 1Hz stdout
- `kernel/keccak_grinder.cu` — persistent Keccak-256 grind kernel

See `docs/superpowers/specs/2026-05-11-hash256-cli-miner-design.md` and `docs/superpowers/plans/2026-05-11-hashminer-implementation.md` for full design.

# Configuration reference

See `config.example.toml`. Key tunables:

| Setting | Purpose |
|---|---|
| `[chain] read_rpc_ws` / `read_rpc_http` | RPC endpoints (publicnode by default) |
| `[relays] private` | MEV-Blocker + Flashbots Protect submission endpoints |
| `[mining] max_tip_gwei` | gas priority fee tip |
| `[mining] ev_min_ratio` | refuse to mine if `reward × P(win) < ev_min_ratio × gas_cost` |
| `[mining] confirmations` | reorg protection — block depth before counting a mint as won |
| `[gpu] sm_count` | **must match your card**, see table above |

# Status & known limitations

- v0.1 — feature-complete MVP
- Mining economics depend on current HASH price + gas — check before running 24/7
- Single GPU per process (multi-GPU = multiple processes per address scheme)
- No automatic restart on crash — use `tmux` + supervisor (systemd unit included in `deploy/`)

# License

MIT
