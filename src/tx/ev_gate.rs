use crate::chain::ChainSource;
use crate::error::Result;
use alloy::primitives::U256;

pub struct EvParams {
    /// Maximum number of mint transactions allowed per block.
    pub max_mints_per_block: u64,
    /// Minimum required ratio of (expected_reward / cost).
    /// e.g. 1.2 means "only submit if expected value is at least 1.2× the gas cost".
    pub min_ratio: f64,
}

pub struct EvGate<'a, C: ChainSource> {
    pub chain: &'a C,
    pub params: EvParams,
}

impl<'a, C: ChainSource> EvGate<'a, C> {
    /// Returns `Ok(true)` if submitting is profitable at current head/state.
    pub async fn allow(&self, reward_wei: U256, gas_cost_wei: U256) -> Result<bool> {
        // Refuse if current block is already saturated.
        let head = self.chain.head();
        let mints = self.chain.mints_in_block(head).await?;
        if mints >= self.params.max_mints_per_block {
            return Ok(false);
        }
        // Naive win-probability heuristic: P(win) = (cap - mints) / cap.
        let p_win = (self.params.max_mints_per_block.saturating_sub(mints) as f64)
            / self.params.max_mints_per_block as f64;
        let reward = reward_wei.to_string().parse::<f64>().unwrap_or(0.0);
        let cost = gas_cost_wei.to_string().parse::<f64>().unwrap_or(f64::INFINITY);
        if cost == 0.0 {
            return Ok(true);
        }
        Ok((reward * p_win) / cost > self.params.min_ratio)
    }
}
