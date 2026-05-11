#[cfg(feature = "cuda-runtime")]
mod cuda_bench {
    use alloy::primitives::{B256, U256};
    use hashminer::gpu::kernel_ffi::GpuRuntime;
    use std::time::{Duration, Instant};

    pub fn run() {
        let mut rt = GpuRuntime::init(0).expect("CUDA init");

        // target=0 → no hash ever passes; we measure pure keccak throughput.
        rt.hot_swap(B256::ZERO, U256::ZERO, 0).expect("hot_swap");

        // Default batch_size and block_size; user can override via env.
        // Defaults aim for ~200-500ms per launch on a 5-7 GH/s card.
        let batch_size: u64 = std::env::var("HASHMINER_BENCH_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1u64 << 30); // 1G = ~180ms at 5.6 GH/s
        let block_size: u32 = std::env::var("HASHMINER_BENCH_BLOCK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);

        eprintln!(
            "bench: batch_size={} ({:.0}M nonces/launch) block_size={}",
            batch_size,
            batch_size as f64 / 1e6,
            block_size
        );

        let dur = Duration::from_secs(10);
        let start = Instant::now();
        let mut nonce_start: u64 = 0;
        let mut total_hashes: u64 = 0;
        let mut sample_t = Instant::now();

        // Warm-up launch (compilation + cache fill).
        let _ = rt.launch_mine(nonce_start, batch_size, block_size).ok();
        nonce_start = nonce_start.wrapping_add(batch_size);

        sample_t = Instant::now();
        while start.elapsed() < dur {
            let _ = rt.launch_mine(nonce_start, batch_size, block_size).ok();
            total_hashes = total_hashes.wrapping_add(batch_size);
            nonce_start = nonce_start.wrapping_add(batch_size);

            let now = Instant::now();
            let dt = now.duration_since(sample_t).as_secs_f64();
            if dt >= 1.0 {
                let ghps = total_hashes as f64 / dt / 1e9;
                println!("hashrate: {:.2} GH/s", ghps);
                total_hashes = 0;
                sample_t = now;
            }
        }
    }
}

#[cfg(feature = "cuda-runtime")]
fn main() {
    cuda_bench::run();
}

#[cfg(not(feature = "cuda-runtime"))]
fn main() {
    eprintln!("hashminer-bench requires --features cuda-runtime");
    std::process::exit(1);
}
