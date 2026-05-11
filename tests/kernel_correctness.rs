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
        let challenge = B256::from(rand::random::<[u8; 32]>());
        let nonce = U256::from(rng.gen::<u64>());

        // CPU reference
        let cpu = compute_inner_hash(challenge, nonce);

        // GPU: hot-swap (challenge, target=MAX, epoch=0), force exactly this nonce, launch 1×1, poll.
        rt.hot_swap(challenge, target, 0).unwrap();
        rt.force_test_nonce(nonce).unwrap();
        let hits = rt
            .poll_hits_blocking(std::time::Duration::from_secs(1))
            .unwrap();
        assert!(!hits.is_empty(), "expected at least one hit for target=MAX");
        assert_eq!(hits[0].hash, cpu);
    }
}
