#[cfg(feature = "cuda-runtime")]
mod cuda_bench {
    use alloy::primitives::{B256, U256};
    use hashminer::gpu::kernel_ffi::GpuRuntime;
    use std::time::{Duration, Instant};

    pub fn run() {
        let mut rt = GpuRuntime::init(0).expect("CUDA init");
        // target = MAX → every hash is a hit (but we don't drain hits, just measure throughput).
        // Use B256::ZERO as challenge — value doesn't affect throughput.
        rt.hot_swap(B256::ZERO, U256::MAX, 0).expect("hot_swap");
        // RTX 4090 has 144 SMs; default config 4 × 144 blocks × 256 threads.
        rt.launch_persistent(4 * 144, 256).expect("launch");

        let dur = Duration::from_secs(10);
        let start = Instant::now();
        let mut last_counter: u64 = 0;
        while start.elapsed() < dur {
            std::thread::sleep(Duration::from_secs(1));
            let counter = rt.read_nonce_counter().expect("read counter");
            let delta = counter - last_counter;
            println!("hashrate: {:.2} GH/s", delta as f64 / 1e9);
            last_counter = counter;
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
