use crate::error::{MinerError, Result};
use rand::rngs::OsRng;
use std::path::Path;
use zeroize::Zeroizing;

/// Decrypt v3 keystore JSON, return raw 32-byte private key wrapped in Zeroizing.
pub fn unlock<P: AsRef<Path>>(path: P, password: &str) -> Result<Zeroizing<[u8; 32]>> {
    let bytes = eth_keystore::decrypt_key(path.as_ref(), password)
        .map_err(|e| MinerError::Keystore(e.to_string()))?;
    if bytes.len() != 32 {
        return Err(MinerError::Keystore(format!(
            "decrypted key wrong size: {}",
            bytes.len()
        )));
    }
    let mut arr = Zeroizing::new([0u8; 32]);
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

/// Create a v3 keystore JSON in `dir` with an optional explicit `name`.
/// Returns the absolute path to the written keystore file.
pub fn create_keystore(
    dir: &Path,
    key_bytes: &Zeroizing<[u8; 32]>,
    password: &str,
    name: Option<&str>,
) -> Result<std::path::PathBuf> {
    std::fs::create_dir_all(dir).map_err(MinerError::Io)?;
    let mut rng = OsRng;
    let filename = eth_keystore::encrypt_key(dir, &mut rng, key_bytes.as_slice(), password, name)
        .map_err(|e| MinerError::Keystore(e.to_string()))?;
    Ok(dir.join(filename))
}

/// Prompt for password if `KEYSTORE_PASSWORD` env not set.
pub fn read_password_from_env_or_prompt() -> Result<Zeroizing<String>> {
    if let Ok(p) = std::env::var("KEYSTORE_PASSWORD") {
        return Ok(Zeroizing::new(p));
    }
    let p = rpassword::prompt_password("keystore password: ")
        .map_err(|e| MinerError::Keystore(e.to_string()))?;
    Ok(Zeroizing::new(p))
}
