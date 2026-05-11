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
        if let Some(out) = self.force_outcome.lock().clone() {
            return Ok(out);
        }
        Ok(SubmitOutcome::Included {
            tx: TxHash::ZERO,
            block: 0,
            reward_wei: U256::from(100u64),
            relay: "fake".into(),
        })
    }
}
