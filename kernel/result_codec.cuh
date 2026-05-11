#pragma once
#include <stdint.h>

// state[0..3] holds the first 32 bytes of the Keccak output.
// Each state word is a little-endian uint64; bytes 0..7 of the hash are
// the LS byte..MS byte of state[0]. For big-endian uint256 comparison
// (Solidity treats hash as bytes32, "less than target" big-endian),
// we byte-swap each word.

__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    return ((x & 0x00000000000000FFULL) << 56)
         | ((x & 0x000000000000FF00ULL) << 40)
         | ((x & 0x0000000000FF0000ULL) << 24)
         | ((x & 0x00000000FF000000ULL) <<  8)
         | ((x & 0x000000FF00000000ULL) >>  8)
         | ((x & 0x0000FF0000000000ULL) >> 24)
         | ((x & 0x00FF000000000000ULL) >> 40)
         | ((x & 0xFF00000000000000ULL) >> 56);
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
