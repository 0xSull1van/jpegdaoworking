# hashminer

Native NVIDIA CUDA miner for [hash256](https://hash256.org) — replaces the browser/WebGPU miner with a persistent CUDA kernel.

## Highlights

- **~10× faster than WebGPU** — target 10-15 GH/s on RTX 4090 (vs ~1.14 GH/s in the browser)
- Hot-swap challenge on epoch rotation without kernel restart
- Sequential nonce gate + dual private relay fan-out (MEV-Blocker + Flashbots Protect)
- v3 keystore unlock with Zeroizing in-memory key
- JSONL metrics + 1Hz stdout for monitoring
- Cross-platform: CUDA via [cudarc](https://github.com/coreylowman/cudarc) — works on Linux and Windows

## Quick start

```bash
# 1. Install CUDA 12.x driver + toolkit (nvcc in PATH)
# 2. Build
cargo build --release --features cuda-runtime

# 3. Create a v3 keystore
cp config.example.toml config.toml
# edit config.toml: RPC URLs, keystore path

# 4. Run
KEYSTORE_PASSWORD=... ./target/release/hashminer --config config.toml
```

## Benchmark only (no chain interaction)

```bash
./target/release/hashminer-bench
```

## Architecture

- `src/chain/` — `ChainWatcher` polls `miningState()` every 4s, publishes `ChallengeUpdate` on epoch rotation
- `src/gpu/` — `GpuWorker` (cudarc) launches a persistent grind kernel; double-buffered `__constant__` challenge for atomic hot-swap
- `src/tx/` — `TxSubmitter` with sequential gate, dual-relay fan-out, receipt watch with K-confirmation reorg protection
- `src/wallet/` — eth-keystore v3 unlock + `Zeroizing<[u8;32]>` key + alloy `PrivateKeySigner`
- `src/metrics/` — JSONL appender + 1Hz stdout

See `docs/superpowers/specs/` and `docs/superpowers/plans/` for full design.

## Configuration

See `config.example.toml`. Key tunables:

- `[gpu] threads_per_block / blocks_per_sm / batch_per_thread` — tune for your card. Defaults assume RTX 4090.
- `[mining] ev_min_ratio` — only submit if `reward × P(win) > ev_min_ratio × gas_cost`. Default 1.2.
- `[mining] confirmations` — wait this many blocks before marking a mint "won" (reorg protection).

## Status

v0.1 — feature-complete MVP. See `docs/superpowers/plans/2026-05-11-hashminer-implementation.md` for what was built.

## License

MIT
