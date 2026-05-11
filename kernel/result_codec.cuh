#pragma once
#include <stdint.h>

// state[0..3] holds the first 32 bytes of the Keccak output.
// Each state word is a little-endian uint64; bytes 0..7 of the hash are
// the LS byte..MS byte of state[0]. For big-endian uint256 comparison
// (Solidity treats hash as bytes32, "less than target" big-endian),
// we byte-swap each word.

// Byte-swap a 64-bit value via PRMT (1-cycle byte permute on Ada Lovelace).
// __byte_perm(a, b, ctrl): result byte k = byte (ctrl>>(4k)&0xF) of {a, b}
// where a contributes bytes 0..3 and b contributes 4..7.
// ctrl = 0x0123 → result = bytes [3,2,1,0] of `a` = byte-reversed a.
__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    uint32_t lo = static_cast<uint32_t>(x);
    uint32_t hi = static_cast<uint32_t>(x >> 32);
    uint32_t new_lo = __byte_perm(hi, 0u, 0x0123u);  // reverse high half → low
    uint32_t new_hi = __byte_perm(lo, 0u, 0x0123u);  // reverse low half  → high
    return (static_cast<uint64_t>(new_hi) << 32) | static_cast<uint64_t>(new_lo);
}

__device__ __forceinline__ bool less_than_target_be(const uint64_t state[25], const uint64_t target[4]) {
    // Compare 32-byte big-endian values: target[0] is the MS 8 bytes of the 256-bit target.
    uint64_t h0 = bswap64(state[0]);   // bytes 0..7  (MSB)
    uint64_t h1 = bswap64(state[1]);
    uint64_t h2 = bswap64(state[2]);
    uint64_t h3 = bswap64(state[3]);   // bytes 24..31 (LSB)
    if (h0 != target[0]) return h0 < target[0];
    if (h1 != target[1]) return h1 < target[1];
    if (h2 != target[2]) return h2 < target[2];
    return h3 < target[3];
}

__device__ __forceinline__ void store_hash_be(uint8_t out[32], const uint64_t state[25]) {
    uint64_t h0 = bswap64(state[0]);
    uint64_t h1 = bswap64(state[1]);
    uint64_t h2 = bswap64(state[2]);
    uint64_t h3 = bswap64(state[3]);
    *reinterpret_cast<uint64_t*>(out +  0) = h0;
    *reinterpret_cast<uint64_t*>(out +  8) = h1;
    *reinterpret_cast<uint64_t*>(out + 16) = h2;
    *reinterpret_cast<uint64_t*>(out + 24) = h3;
}
