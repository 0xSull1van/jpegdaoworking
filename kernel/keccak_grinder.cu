#include "keccak_device.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// One-shot Keccak-256 mining kernel.
//
// Each thread tries exactly ONE nonce = nonce_start + globalThreadId, then exits.
// Host launches kernels in a loop, incrementing nonce_start by grid_size between
// launches. This architecture matches proven high-perf miners (sha3x_cudaminer,
// friend's reference) and avoids:
//   • atomicAdd on a global nonce counter (every thread had to do this in the
//     old persistent design — eliminates contention)
//   • long-lived register state across iterations (compiler can fully optimize
//     a one-shot thread vs an infinite outer loop)
//   • driver scheduler depriorities on persistent kernels (observed on Blackwell)
//
// Input layout (matches Solidity `keccak256(abi.encode(challenge, nonce))`):
//   bytes  0..31 — challenge (32B)        → state lanes 0..3 (LE host pre-load)
//   bytes 32..63 — nonce as uint256 BE    → state lanes 4..7
//     (for nonces < 2^64: lanes 4,5,6 = 0; lane 7 = bswap64(nonce))
// Padding (Keccak-256, not FIPS-202):
//   byte 64  = 0x01  → state[8]  low byte
//   byte 135 = 0x80  → state[16] high byte (rate = 136 bytes)
// ─────────────────────────────────────────────────────────────────────────────

// 24 round constants of Keccak-f[1600], stored in CUDA constant memory so the
// L1 constant cache broadcasts them to all threads in a warp in one cycle.
__constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __forceinline__ uint64_t bswap64_kernel(uint64_t x) {
    uint32_t lo = __byte_perm((uint32_t)x,         0u, 0x0123u);
    uint32_t hi = __byte_perm((uint32_t)(x >> 32), 0u, 0x0123u);
    return ((uint64_t)lo << 32) | (uint64_t)hi;
}

__device__ __forceinline__ void keccak_f1600_arr(uint64_t* st) {
    uint64_t bc0, bc1, bc2, bc3, bc4, t, tmp;
    #pragma unroll
    for (int r = 0; r < 24; r++) {
        // Theta
        bc0 = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc1 = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc2 = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc3 = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc4 = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        t = bc4 ^ ROL64(bc1, 1);
        st[ 0] ^= t; st[ 5] ^= t; st[10] ^= t; st[15] ^= t; st[20] ^= t;
        t = bc0 ^ ROL64(bc2, 1);
        st[ 1] ^= t; st[ 6] ^= t; st[11] ^= t; st[16] ^= t; st[21] ^= t;
        t = bc1 ^ ROL64(bc3, 1);
        st[ 2] ^= t; st[ 7] ^= t; st[12] ^= t; st[17] ^= t; st[22] ^= t;
        t = bc2 ^ ROL64(bc4, 1);
        st[ 3] ^= t; st[ 8] ^= t; st[13] ^= t; st[18] ^= t; st[23] ^= t;
        t = bc3 ^ ROL64(bc0, 1);
        st[ 4] ^= t; st[ 9] ^= t; st[14] ^= t; st[19] ^= t; st[24] ^= t;

        // Rho + Pi (standard chain 1→10→7→11→…→6→1, 24 lane rotations)
        t = st[1];
        tmp = st[10]; st[10] = ROL64(t,  1); t = tmp;
        tmp = st[ 7]; st[ 7] = ROL64(t,  3); t = tmp;
        tmp = st[11]; st[11] = ROL64(t,  6); t = tmp;
        tmp = st[17]; st[17] = ROL64(t, 10); t = tmp;
        tmp = st[18]; st[18] = ROL64(t, 15); t = tmp;
        tmp = st[ 3]; st[ 3] = ROL64(t, 21); t = tmp;
        tmp = st[ 5]; st[ 5] = ROL64(t, 28); t = tmp;
        tmp = st[16]; st[16] = ROL64(t, 36); t = tmp;
        tmp = st[ 8]; st[ 8] = ROL64(t, 45); t = tmp;
        tmp = st[21]; st[21] = ROL64(t, 55); t = tmp;
        tmp = st[24]; st[24] = ROL64(t,  2); t = tmp;
        tmp = st[ 4]; st[ 4] = ROL64(t, 14); t = tmp;
        tmp = st[15]; st[15] = ROL64(t, 27); t = tmp;
        tmp = st[23]; st[23] = ROL64(t, 41); t = tmp;
        tmp = st[19]; st[19] = ROL64(t, 56); t = tmp;
        tmp = st[13]; st[13] = ROL64(t,  8); t = tmp;
        tmp = st[12]; st[12] = ROL64(t, 25); t = tmp;
        tmp = st[ 2]; st[ 2] = ROL64(t, 43); t = tmp;
        tmp = st[20]; st[20] = ROL64(t, 62); t = tmp;
        tmp = st[14]; st[14] = ROL64(t, 18); t = tmp;
        tmp = st[22]; st[22] = ROL64(t, 39); t = tmp;
        tmp = st[ 9]; st[ 9] = ROL64(t, 61); t = tmp;
        tmp = st[ 6]; st[ 6] = ROL64(t, 20); t = tmp;
        st[1] = ROL64(t, 44);

        // Chi (5 rows of 5 lanes each)
        bc0 = st[ 0]; bc1 = st[ 1]; bc2 = st[ 2]; bc3 = st[ 3]; bc4 = st[ 4];
        st[ 0] = bc0 ^ ((~bc1) & bc2); st[ 1] = bc1 ^ ((~bc2) & bc3); st[ 2] = bc2 ^ ((~bc3) & bc4);
        st[ 3] = bc3 ^ ((~bc4) & bc0); st[ 4] = bc4 ^ ((~bc0) & bc1);

        bc0 = st[ 5]; bc1 = st[ 6]; bc2 = st[ 7]; bc3 = st[ 8]; bc4 = st[ 9];
        st[ 5] = bc0 ^ ((~bc1) & bc2); st[ 6] = bc1 ^ ((~bc2) & bc3); st[ 7] = bc2 ^ ((~bc3) & bc4);
        st[ 8] = bc3 ^ ((~bc4) & bc0); st[ 9] = bc4 ^ ((~bc0) & bc1);

        bc0 = st[10]; bc1 = st[11]; bc2 = st[12]; bc3 = st[13]; bc4 = st[14];
        st[10] = bc0 ^ ((~bc1) & bc2); st[11] = bc1 ^ ((~bc2) & bc3); st[12] = bc2 ^ ((~bc3) & bc4);
        st[13] = bc3 ^ ((~bc4) & bc0); st[14] = bc4 ^ ((~bc0) & bc1);

        bc0 = st[15]; bc1 = st[16]; bc2 = st[17]; bc3 = st[18]; bc4 = st[19];
        st[15] = bc0 ^ ((~bc1) & bc2); st[16] = bc1 ^ ((~bc2) & bc3); st[17] = bc2 ^ ((~bc3) & bc4);
        st[18] = bc3 ^ ((~bc4) & bc0); st[19] = bc4 ^ ((~bc0) & bc1);

        bc0 = st[20]; bc1 = st[21]; bc2 = st[22]; bc3 = st[23]; bc4 = st[24];
        st[20] = bc0 ^ ((~bc1) & bc2); st[21] = bc1 ^ ((~bc2) & bc3); st[22] = bc2 ^ ((~bc3) & bc4);
        st[23] = bc3 ^ ((~bc4) & bc0); st[24] = bc4 ^ ((~bc0) & bc1);

        // Iota
        st[0] ^= RC[r];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel entry point.
//
// Args:
//   challenge[4]  — keccak state lanes 0..3 (LE uint64, pre-loaded by host)
//   nonce_start   — base nonce for this launch (host increments by grid_size)
//   target[4]     — difficulty as 4 BE uint64 (target[0] = MSB)
//   result        — winning nonce written here (atomic, first winner wins)
//   found         — atomic flag: 0 = not found, 1 = found
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void mine(
    const uint64_t* __restrict__ challenge,
    uint64_t                     nonce_start,
    const uint64_t* __restrict__ target,
    uint64_t*                    result,
    int*                         found
) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Early-out if another thread already found a solution in this launch.
    // *found is a regular global load (cached in L1); cheap.
    if (*found) return;

    uint64_t nonce = nonce_start + gid;

    // Build initial sponge state for this nonce.
    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0ULL;

    // Absorb 32-byte challenge into lanes 0..3 (host pre-converted to LE uint64).
    st[0] = challenge[0];
    st[1] = challenge[1];
    st[2] = challenge[2];
    st[3] = challenge[3];

    // Absorb 32-byte big-endian nonce into lanes 4..7.
    // For nonces < 2^64: lanes 4, 5, 6 = 0; lane 7 = nonce as LE-interpreted BE bytes.
    st[7] = bswap64_kernel(nonce);

    // Keccak-256 padding (rate = 136 bytes after 64-byte input):
    //   byte 64  = 0x01  → low byte of state[8]
    //   byte 135 = 0x80  → high byte of state[16]
    st[8]  = 0x0000000000000001ULL;
    st[16] = 0x8000000000000000ULL;

    keccak_f1600_arr(st);

    // Output: first 32 bytes of state, compared big-endian against target.
    uint64_t h0 = bswap64_kernel(st[0]);
    uint64_t h1 = bswap64_kernel(st[1]);
    uint64_t h2 = bswap64_kernel(st[2]);
    uint64_t h3 = bswap64_kernel(st[3]);

    uint64_t d0 = target[0];
    uint64_t d1 = target[1];
    uint64_t d2 = target[2];
    uint64_t d3 = target[3];

    bool less = (h0 <  d0) ||
                (h0 == d0 && h1 <  d1) ||
                (h0 == d0 && h1 == d1 && h2 <  d2) ||
                (h0 == d0 && h1 == d1 && h2 == d2 && h3 <  d3);

    if (less && atomicCAS(found, 0, 1) == 0) {
        *result = nonce;
    }
}
