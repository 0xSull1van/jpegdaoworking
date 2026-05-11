use hashminer::wallet::{keystore::unlock, MinerSigner};
use tempfile::tempdir;

#[test]
fn roundtrip_v3_keystore() {
    let dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let key = [0x42u8; 32];
    let filename =
        eth_keystore::encrypt_key(dir.path(), &mut rng, key, "pw", None).unwrap();
    let abs = dir.path().join(filename);

    let unlocked = unlock(&abs, "pw").unwrap();
    assert_eq!(unlocked.as_slice(), &key);

    let signer = MinerSigner::from_key(unlocked).unwrap();
    // Address derivation: deterministic for this key.
    println!("derived: {}", signer.address());
}
