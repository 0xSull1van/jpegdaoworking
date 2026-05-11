use alloy::primitives::{address, b256, U256};
use hashminer::chain::challenge::{compute_challenge, compute_inner_hash};

#[test]
fn challenge_matches_solidity_layout() {
    // _challenge inputs (all padded to 32B by abi.encode):
    //   chainid = 1
    //   contract = 0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc
    //   miner = 0x000...001 (test)
    //   epoch = 250707
    let miner = address!("0000000000000000000000000000000000000001");
    let epoch = 250707u64;
    let got = compute_challenge(1, hashminer::chain::contract::CONTRACT, miner, epoch);

    // Independently computed reference (tiny-keccak with explicit 128-byte buffer).
    let mut buf = [0u8; 128];
    buf[31] = 1;                                                    // chainid uint256 BE
    buf[44..64].copy_from_slice(hashminer::chain::contract::CONTRACT.as_slice()); // contract left-padded to 32B
    buf[76..96].copy_from_slice(miner.as_slice());                  // miner left-padded
    buf[96..128].copy_from_slice(&U256::from(epoch).to_be_bytes::<32>()); // epoch uint256 BE

    use tiny_keccak::{Hasher, Keccak};
    let mut k = Keccak::v256();
    k.update(&buf);
    let mut out = [0u8; 32];
    k.finalize(&mut out);

    assert_eq!(got.as_slice(), &out);
}

#[test]
fn inner_hash_matches_64byte_encoding() {
    let challenge = b256!("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff");
    let nonce = U256::from(0x4242u64);
    let got = compute_inner_hash(challenge, nonce);

    let mut buf = [0u8; 64];
    buf[0..32].copy_from_slice(challenge.as_slice());
    buf[32..64].copy_from_slice(&nonce.to_be_bytes::<32>());

    use tiny_keccak::{Hasher, Keccak};
    let mut k = Keccak::v256();
    k.update(&buf);
    let mut out = [0u8; 32];
    k.finalize(&mut out);

    assert_eq!(got.as_slice(), &out);
}
