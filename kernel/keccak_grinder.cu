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
#define BATCH_PER_THREAD 1024
#endif

extern "C" __global__ void grind() {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t base = atomicAdd((unsigned long long*)&d_nonce_counter, (unsigned long long)BATCH_PER_THREAD);

    while (atomicAdd(&d_should_stop, 0) == 0) {
        uint32_t idx = d_active_idx;
        uint32_t my_epoch = c_epoch_id[idx];
        const uint64_t* challenge_w = reinterpret_cast<const uint64_t*>(c_challenge[idx]);

        #pragma unroll 1
        for (uint32_t i = 0; i < BATCH_PER_THREAD; i++) {
            uint64_t s[25];
            #pragma unroll
            for (int w = 0; w < 25; w++) s[w] = 0;

            // Absorb 32-byte challenge into rate words 0..3.
            // Big-endian byte stream → state words are little-endian uint64, so bswap each.
            s[0] ^= bswap64(challenge_w[0]);
            s[1] ^= bswap64(challenge_w[1]);
            s[2] ^= bswap64(challenge_w[2]);
            s[3] ^= bswap64(challenge_w[3]);

            // Absorb 32-byte big-endian nonce uint256 into words 4..7.
            // We use thread id as the high bits of the nonce so threads have disjoint ranges:
            //   nonce_be[0..15]  = 0
            //   nonce_be[16..23] = nonce_hi big-endian (we set nonce_hi = tid here)
            //   nonce_be[24..31] = nonce_lo big-endian (counter base+i)
            // In state words: s[4] = bytes 32..39 = 0, s[5] = bytes 40..47 = 0,
            //                 s[6] = bytes 48..55 = bswap(nonce_hi), s[7] = bytes 56..63 = bswap(nonce_lo).
            uint64_t nonce_lo = base + i;
            uint64_t nonce_hi = (uint64_t)tid;
            s[4] ^= 0;
            s[5] ^= 0;
            s[6] ^= bswap64(nonce_hi);
            s[7] ^= bswap64(nonce_lo);

            // Padding: byte 64 = 0x01 (LSB of state[8]),
            //          byte 135 = 0x80 (MSB of state[16]).
            s[8]  ^= 0x01ULL;
            s[16] ^= 0x8000000000000000ULL;

            keccak_f1600(s);

            if (less_than_target_be(s, c_target[idx])) {
                uint32_t slot = atomicAdd(&d_hit_count, 1);
                if (slot < 16) {
                    // Store nonce big-endian: bytes 0..15 zero, 16..23 = nonce_hi BE, 24..31 = nonce_lo BE.
                    uint64_t* out = reinterpret_cast<uint64_t*>(d_hits[slot].nonce);
                    out[0] = 0;
                    out[1] = 0;
                    out[2] = bswap64(nonce_hi);
                    out[3] = bswap64(nonce_lo);
                    store_hash_be(d_hits[slot].hash, s);
                    d_hits[slot].epoch_id = my_epoch;
                }
            }
        }
        base = atomicAdd((unsigned long long*)&d_nonce_counter, (unsigned long long)BATCH_PER_THREAD);
    }
}
