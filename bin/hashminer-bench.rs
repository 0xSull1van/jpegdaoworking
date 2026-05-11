#[cfg(feature = "cuda-runtime")]
mod cuda_bench {
    use alloy::primitives::{B256, U256};
    use hashminer::gpu::kernel_ffi::GpuRuntime;
    use std::time::{Duration, Instant};

    pub fn run() {
        let mut rt = GpuRuntime::init(0).expect("CUDA init");

        // Use target = 0 so NO hash ever passes — we measure pure keccak throughput
        // without atomicAdd contention on d_hit_count. (Earlier target=MAX caused
        // every thread to hit the global hit-count atomic, serializing the kernel.)
        rt.hot_swap(B256::ZERO, U256::ZERO, 0).expect("hot_swap");

        // Per-SM tuning. RTX 4060 Ti = 34 SMs (Ada). 4060/3060 = 28, 3080 = 68, 4090 = 128.
        // Override with env HASHMINER_BENCH_SMS for other cards.
        let sm_count: u32 = std::env::var("HASHMINER_BENCH_SMS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(34);
        let blocks = 4 * sm_count;
        let threads = 256;
        eprintln!("launching {} blocks × {} threads ({} SMs assumed)", blocks, threads, sm_count);
        rt.launch_persistent(blocks, threads).expect("launch");

        let dur = Duration::from_secs(10);
        let start = Instant::now();
        // First sample after 2 seconds — let the kernel ramp up.
        std::thread::sleep(Duration::from_secs(2));
        let mut last_counter: u64 = rt.read_nonce_counter().expect("read counter");
        let mut last_t = Instant::now();
        while start.elapsed() < dur {
            std::thread::sleep(Duration::from_secs(1));
            let counter = rt.read_nonce_counter().expect("read counter");
            let now = Instant::now();
            let dt = now.duration_since(last_t).as_secs_f64();
            let delta = counter.saturating_sub(last_counter) as f64;
            println!("hashrate: {:.2} GH/s", delta / dt / 1e9);
            last_counter = counter;
            last_t = now;
        }
        rt.signal_stop().ok();
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
