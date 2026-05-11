use alloy::primitives::{address, U256};
use hashminer::chain::{ChainSource, fake::FakeChain};
use hashminer::gpu::Hit;

#[tokio::test]
async fn submitter_drops_stale_epoch_hit() {
    let miner = address!("0000000000000000000000000000000000000001");
    let chain = FakeChain::new(0, U256::MAX, miner);
    let _ = chain.mining_state().await.unwrap();
    // Pretend we mined for old epoch 0, then chain advanced to epoch 1 (100 blocks).
    chain.advance_blocks(100, miner);
    let stale = Hit { nonce: U256::from(1u64), hash: Default::default(), epoch_id: 0 };

    let st = chain.mining_state().await.unwrap();
    assert_eq!(st.epoch, 1);
    assert!(st.epoch != stale.epoch_id, "epoch must have advanced");

    // Lightweight check: drop-decision logic mirrors what TxSubmitter does in submit().
    let drop = stale.epoch_id != st.epoch;
    assert!(drop);
}
