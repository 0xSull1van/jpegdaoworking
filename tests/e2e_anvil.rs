use alloy::primitives::{address, U256};
use hashminer::chain::{fake::FakeChain, ChainSource};
use hashminer::gpu::{fake::FakeGrinder, Grinder};
use hashminer::tx::{fake::FakeSubmitter, Submitter};
use std::sync::Arc;

#[tokio::test]
async fn fake_pipeline_end_to_end() {
    let miner = address!("0000000000000000000000000000000000000001");
    let target = U256::MAX; // anything hits
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
