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

// How often to check d_should_stop. Higher = lower atomic overhead, slower shutdown.
// At ~2 outer iters/sec/thread with BATCH=8192, check every 16 outers ≈ 8 sec to react.
#ifndef STOP_CHECK_EVERY
#define STOP_CHECK_EVERY 16
#endif

extern "C" __global__ void grind() {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t base = atomicAdd((unsigned long long*)&d_nonce_counter, (unsigned long long)BATCH_PER_THREAD);

    // tid is constant per thread → its bswapped form can be precomputed once.
    const uint64_t nonce_hi    = static_cast<uint64_t>(tid);
    const uint64_t nonce_hi_bs = bswap64(nonce_hi);   // bytes 48..55 of the absorb block (state word 6)

    uint32_t outer = 0;
    while (true) {
        if ((outer++ % STOP_CHECK_EVERY) == 0) {
            if (atomicAdd(&d_should_stop, 0) != 0) break;
        }
        uint32_t idx = d_active_idx;
        uint32_t my_epoch = c_epoch_id[idx];

        // Hoist challenge absorption out of the inner loop. Within one outer iteration
        // c_challenge[idx] is fixed (we only flip via hot_swap between batches).
        // Precomputing 4 bswapped words saves 4 bswap64s per inner iteration = ~4096
        // bswaps saved per outer iteration per thread.
        const uint64_t* challenge_w = reinterpret_cast<const uint64_t*>(c_challenge[idx]);
        const uint64_t ch0 = bswap64(challenge_w[0]);
        const uint64_t ch1 = bswap64(challenge_w[1]);
        const uint64_t ch2 = bswap64(challenge_w[2]);
        const uint64_t ch3 = bswap64(challenge_w[3]);

        // Target words for less-than comparison (already big-endian native u64 in c_target).
        const uint64_t t0 = c_target[idx][0];
        const uint64_t t1 = c_target[idx][1];
        const uint64_t t2 = c_target[idx][2];
        const uint64_t t3 = c_target[idx][3];

        #pragma unroll 1
        for (uint32_t i = 0; i < BATCH_PER_THREAD; i++) {
            // Build initial sponge state. Only the nonce varies per iteration.
            // All zero-XORs and padding XORs are folded by the compiler.
            uint64_t s0 = ch0;
            uint64_t s1 = ch1;
            uint64_t s2 = ch2;
            uint64_t s3 = ch3;
            // bytes 32..47 of input = 0 → s[4], s[5] = 0
            uint64_t s4 = 0;
            uint64_t s5 = 0;
            // bytes 48..55 = bswap(nonce_hi), bytes 56..63 = bswap(nonce_lo)
            uint64_t s6 = nonce_hi_bs;
            uint64_t nonce_lo = base + i;
            uint64_t s7 = bswap64(nonce_lo);
            // Padding: byte 64 = 0x01 (LSB of state[8]), byte 135 = 0x80 (MSB of state[16]).
            uint64_t s8 = 0x01ULL;
            uint64_t s9 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0, s14 = 0, s15 = 0;
            uint64_t s16 = 0x8000000000000000ULL;
            uint64_t s17 = 0, s18 = 0, s19 = 0, s20 = 0, s21 = 0, s22 = 0, s23 = 0, s24 = 0;

            // Pack into the array form expected by keccak_f1600.
            uint64_t st[25] = { s0, s1, s2, s3, s4, s5, s6, s7, s8, s9,
                                s10, s11, s12, s13, s14, s15, s16, s17, s18, s19,
                                s20, s21, s22, s23, s24 };

            keccak_f1600(st);

            // Inline target comparison (avoids passing c_target[idx] every iter; uses local registers).
            uint64_t h0 = bswap64(st[0]);
            uint64_t h1 = bswap64(st[1]);
            uint64_t h2 = bswap64(st[2]);
            uint64_t h3 = bswap64(st[3]);
            bool hit;
            if (h0 != t0)      hit = h0 < t0;
            else if (h1 != t1) hit = h1 < t1;
            else if (h2 != t2) hit = h2 < t2;
            else               hit = h3 < t3;

            if (hit) {
                uint32_t slot = atomicAdd(&d_hit_count, 1);
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
