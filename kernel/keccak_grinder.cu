#include "keccak_device.cuh"
#include "result_codec.cuh"

// Double-buffered challenge + target (hot-swap without kernel restart).
__constant__ uint8_t  c_challenge[2][32];
__constant__ uint64_t c_target[2][4];          // big-endian
__constant__ uint32_t c_epoch_id[2];

__device__ uint32_t d_active_idx;
__device__ uint64_t d_nonce_counter;
__device__ uint32_t d_hit_count;
__device__ uint32_t d_should_stop;

struct Hit { uint8_t nonce[32]; uint8_t hash[32]; uint32_t epoch_id; uint8_t _pad[4]; };
__device__ Hit d_hits[16];

#ifndef BATCH_PER_THREAD
#define BATCH_PER_THREAD 8192
#endif

#ifndef STOP_CHECK_EVERY
#define STOP_CHECK_EVERY 16
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Fully-inlined Keccak-f[1600] with named state variables.
//
// Critical optimization: by operating on 25 distinct uint64_t locals (s0..s24)
// rather than an array `uint64_t s[25]`, we guarantee no `&s` pointer is ever
// materialised, so the compiler MUST keep state in registers (no local-memory
// spill). Each round below expands to ~90 pure SSA operations the compiler
// schedules freely with high ILP.
//
// The 24 round constants are baked into 24 ROUND() invocations to avoid any
// runtime constant lookup or loop branch.
// ─────────────────────────────────────────────────────────────────────────────

#define THETA() do {                                                          \
    uint64_t C0 = s0 ^ s5  ^ s10 ^ s15 ^ s20;                                 \
    uint64_t C1 = s1 ^ s6  ^ s11 ^ s16 ^ s21;                                 \
    uint64_t C2 = s2 ^ s7  ^ s12 ^ s17 ^ s22;                                 \
    uint64_t C3 = s3 ^ s8  ^ s13 ^ s18 ^ s23;                                 \
    uint64_t C4 = s4 ^ s9  ^ s14 ^ s19 ^ s24;                                 \
    uint64_t D0 = C4 ^ ROL64(C1, 1);                                          \
    uint64_t D1 = C0 ^ ROL64(C2, 1);                                          \
    uint64_t D2 = C1 ^ ROL64(C3, 1);                                          \
    uint64_t D3 = C2 ^ ROL64(C4, 1);                                          \
    uint64_t D4 = C3 ^ ROL64(C0, 1);                                          \
    s0 ^= D0; s5  ^= D0; s10 ^= D0; s15 ^= D0; s20 ^= D0;                     \
    s1 ^= D1; s6  ^= D1; s11 ^= D1; s16 ^= D1; s21 ^= D1;                     \
    s2 ^= D2; s7  ^= D2; s12 ^= D2; s17 ^= D2; s22 ^= D2;                     \
    s3 ^= D3; s8  ^= D3; s13 ^= D3; s18 ^= D3; s23 ^= D3;                     \
    s4 ^= D4; s9  ^= D4; s14 ^= D4; s19 ^= D4; s24 ^= D4;                     \
} while (0)

#define RHO_PI() do {                                                         \
    uint64_t t = s1, b;                                                       \
    b = s10; s10 = ROL64(t,  1); t = b;                                       \
    b = s7;  s7  = ROL64(t,  3); t = b;                                       \
    b = s11; s11 = ROL64(t,  6); t = b;                                       \
    b = s17; s17 = ROL64(t, 10); t = b;                                       \
    b = s18; s18 = ROL64(t, 15); t = b;                                       \
    b = s3;  s3  = ROL64(t, 21); t = b;                                       \
    b = s5;  s5  = ROL64(t, 28); t = b;                                       \
    b = s16; s16 = ROL64(t, 36); t = b;                                       \
    b = s8;  s8  = ROL64(t, 45); t = b;                                       \
    b = s21; s21 = ROL64(t, 55); t = b;                                       \
    b = s24; s24 = ROL64(t,  2); t = b;                                       \
    b = s4;  s4  = ROL64(t, 14); t = b;                                       \
    b = s15; s15 = ROL64(t, 27); t = b;                                       \
    b = s23; s23 = ROL64(t, 41); t = b;                                       \
    b = s19; s19 = ROL64(t, 56); t = b;                                       \
    b = s13; s13 = ROL64(t,  8); t = b;                                       \
    b = s12; s12 = ROL64(t, 25); t = b;                                       \
    b = s2;  s2  = ROL64(t, 43); t = b;                                       \
    b = s20; s20 = ROL64(t, 62); t = b;                                       \
    b = s14; s14 = ROL64(t, 18); t = b;                                       \
    b = s22; s22 = ROL64(t, 39); t = b;                                       \
    b = s9;  s9  = ROL64(t, 61); t = b;                                       \
    b = s6;  s6  = ROL64(t, 20); t = b;                                       \
                s1 = ROL64(t, 44);                                            \
} while (0)

#define CHI_ROW(a, b, c, d, e) do {                                           \
    uint64_t t0 = a, t1 = b, t2 = c, t3 = d, t4 = e;                          \
    a = t0 ^ ((~t1) & t2);                                                    \
    b = t1 ^ ((~t2) & t3);                                                    \
    c = t2 ^ ((~t3) & t4);                                                    \
    d = t3 ^ ((~t4) & t0);                                                    \
    e = t4 ^ ((~t0) & t1);                                                    \
} while (0)

#define CHI() do {                                                            \
    CHI_ROW(s0,  s1,  s2,  s3,  s4);                                          \
    CHI_ROW(s5,  s6,  s7,  s8,  s9);                                          \
    CHI_ROW(s10, s11, s12, s13, s14);                                         \
    CHI_ROW(s15, s16, s17, s18, s19);                                         \
    CHI_ROW(s20, s21, s22, s23, s24);                                         \
} while (0)

#define ROUND(rc) do { THETA(); RHO_PI(); CHI(); s0 ^= (rc); } while (0)

extern "C" __global__ void grind() {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t base = atomicAdd((unsigned long long*)&d_nonce_counter, (unsigned long long)BATCH_PER_THREAD);

    // tid is constant per thread → bswap once.
    const uint64_t nonce_hi    = static_cast<uint64_t>(tid);
    const uint64_t nonce_hi_bs = bswap64(nonce_hi);

    uint32_t outer = 0;
    while (true) {
        if ((outer++ % STOP_CHECK_EVERY) == 0) {
            if (atomicAdd(&d_should_stop, 0) != 0) break;
        }
        const uint32_t idx = d_active_idx;
        const uint32_t my_epoch = c_epoch_id[idx];

        // Pre-compute bswapped challenge words for this batch.
        const uint64_t* challenge_w = reinterpret_cast<const uint64_t*>(c_challenge[idx]);
        const uint64_t ch0 = bswap64(challenge_w[0]);
        const uint64_t ch1 = bswap64(challenge_w[1]);
        const uint64_t ch2 = bswap64(challenge_w[2]);
        const uint64_t ch3 = bswap64(challenge_w[3]);

        // Target words for comparison (already native u64 BE in c_target).
        const uint64_t t0 = c_target[idx][0];
        const uint64_t t1 = c_target[idx][1];
        const uint64_t t2 = c_target[idx][2];
        const uint64_t t3 = c_target[idx][3];

        #pragma unroll 1
        for (uint32_t i = 0; i < BATCH_PER_THREAD; i++) {
            // ─── Initialize sponge state directly in registers ──────────────
            // Absorb 64-byte input: [challenge(32B) | nonce_BE(32B)]
            //   bytes  0..31 = challenge (s0..s3 after bswap)
            //   bytes 32..47 = 0          (s4=s5=0)
            //   bytes 48..55 = nonce_hi BE (s6)
            //   bytes 56..63 = nonce_lo BE (s7)
            // Padding (Keccak-256, not FIPS-202):
            //   byte  64 = 0x01 → LSB of s8
            //   byte 135 = 0x80 → MSB of s16
            uint64_t s0 = ch0;
            uint64_t s1 = ch1;
            uint64_t s2 = ch2;
            uint64_t s3 = ch3;
            uint64_t s4 = 0;
            uint64_t s5 = 0;
            uint64_t s6 = nonce_hi_bs;
            const uint64_t nonce_lo = base + i;
            uint64_t s7 = bswap64(nonce_lo);
            uint64_t s8 = 0x0000000000000001ULL;
            uint64_t s9 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0, s14 = 0, s15 = 0;
            uint64_t s16 = 0x8000000000000000ULL;
            uint64_t s17 = 0, s18 = 0, s19 = 0, s20 = 0, s21 = 0, s22 = 0, s23 = 0, s24 = 0;

            // ─── 24 rounds of Keccak-f[1600], fully unrolled ────────────────
            ROUND(0x0000000000000001ULL);
            ROUND(0x0000000000008082ULL);
            ROUND(0x800000000000808AULL);
            ROUND(0x8000000080008000ULL);
            ROUND(0x000000000000808BULL);
            ROUND(0x0000000080000001ULL);
            ROUND(0x8000000080008081ULL);
            ROUND(0x8000000000008009ULL);
            ROUND(0x000000000000008AULL);
            ROUND(0x0000000000000088ULL);
            ROUND(0x0000000080008009ULL);
            ROUND(0x000000008000000AULL);
            ROUND(0x000000008000808BULL);
            ROUND(0x800000000000008BULL);
            ROUND(0x8000000000008089ULL);
            ROUND(0x8000000000008003ULL);
            ROUND(0x8000000000008002ULL);
            ROUND(0x8000000000000080ULL);
            ROUND(0x000000000000800AULL);
            ROUND(0x800000008000000AULL);
            ROUND(0x8000000080008081ULL);
            ROUND(0x8000000000008080ULL);
            ROUND(0x0000000080000001ULL);
            ROUND(0x8000000080008008ULL);

            // ─── Output: first 32 bytes of state, big-endian comparison ─────
            const uint64_t h0 = bswap64(s0);
            const uint64_t h1 = bswap64(s1);
            const uint64_t h2 = bswap64(s2);
            const uint64_t h3 = bswap64(s3);
            bool hit;
            if      (h0 != t0) hit = h0 < t0;
            else if (h1 != t1) hit = h1 < t1;
            else if (h2 != t2) hit = h2 < t2;
            else               hit = h3 < t3;

            if (hit) {
                const uint32_t slot = atomicAdd(&d_hit_count, 1);
                if (slot < 16) {
                    uint64_t* out = reinterpret_cast<uint64_t*>(d_hits[slot].nonce);
                    out[0] = 0;
                    out[1] = 0;
                    out[2] = nonce_hi_bs;
                    out[3] = bswap64(nonce_lo);
                    uint64_t* hash_out = reinterpret_cast<uint64_t*>(d_hits[slot].hash);
                    hash_out[0] = h0;
                    hash_out[1] = h1;
                    hash_out[2] = h2;
                    hash_out[3] = h3;
                    d_hits[slot].epoch_id = my_epoch;
                }
            }
        }
        base = atomicAdd((unsigned long long*)&d_nonce_counter, (unsigned long long)BATCH_PER_THREAD);
    }
}
