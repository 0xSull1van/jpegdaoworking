/// Embedded PTX bytes from the nvcc build step. Empty when `cuda-runtime` is off.
#[cfg(feature = "cuda-runtime")]
pub const PTX: &[u8] = include_bytes!(env!("PTX_PATH"));

#[cfg(not(feature = "cuda-runtime"))]
pub const PTX: &[u8] = b"";
