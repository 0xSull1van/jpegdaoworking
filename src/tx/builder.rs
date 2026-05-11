use alloy::network::TransactionBuilder;
use alloy::primitives::{Address, Bytes, U256};
use alloy::rpc::types::TransactionRequest;
use alloy::sol_types::SolCall;
use crate::chain::contract::{Hash, CONTRACT};
use crate::error::Result;

pub struct MineTxParams {
    pub from: Address,
    /// Ethereum tx nonce (NOT mining nonce).
    pub nonce: u64,
    /// The PoW nonce passed to `mine(uint256 nonce)`.
    pub mine_nonce: U256,
    pub max_fee_per_gas: u128,
    pub max_priority_fee_per_gas: u128,
    pub gas_limit: u64,
    pub chain_id: u64,
}

pub fn build_mine_tx(p: MineTxParams) -> Result<TransactionRequest> {
    let calldata: Bytes = Hash::mineCall { nonce: p.mine_nonce }.abi_encode().into();
    Ok(TransactionRequest::default()
        .with_from(p.from)
        .with_to(CONTRACT)
        .with_nonce(p.nonce)
        .with_chain_id(p.chain_id)
        .with_input(calldata)
        .with_gas_limit(p.gas_limit)
        .with_max_fee_per_gas(p.max_fee_per_gas)
        .with_max_priority_fee_per_gas(p.max_priority_fee_per_gas))
}
