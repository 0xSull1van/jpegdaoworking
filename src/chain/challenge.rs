use alloy::primitives::{Address, B256, U256};
use tiny_keccak::{Hasher, Keccak};

/// keccak256(abi.encode(chainid, address(this), miner, epoch)).
/// Output bytes match Solidity exactly: 4 fields × 32 bytes = 128 bytes input.
pub fn compute_challenge(chain_id: u64, contract: Address, miner: Address, epoch: u64) -> B256 {
    let mut buf = [0u8; 128];
    buf[24..32].copy_from_slice(&chain_id.to_be_bytes());           // chainid in last 8 bytes of slot 0
    buf[44..64].copy_from_slice(contract.as_slice());               // address right-aligned in slot 1
    buf[76..96].copy_from_slice(miner.as_slice());                  // miner right-aligned in slot 2
    buf[120..128].copy_from_slice(&epoch.to_be_bytes());            // epoch in last 8 bytes of slot 3
    keccak(&buf)
}

/// keccak256(abi.encode(challenge, nonce)): 32B challenge ‖ 32B nonce BE.
pub fn compute_inner_hash(challenge: B256, nonce: U256) -> B256 {
    let mut buf = [0u8; 64];
    buf[0..32].copy_from_slice(challenge.as_slice());
    buf[32..64].copy_from_slice(&nonce.to_be_bytes::<32>());
    keccak(&buf)
}

fn keccak(input: &[u8]) -> B256 {
    let mut k = Keccak::v256();
    k.update(input);
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    B256::from(out)
}
