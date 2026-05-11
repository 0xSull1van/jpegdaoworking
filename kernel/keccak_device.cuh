#pragma once
#include <stdint.h>

__constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __forceinline__ uint64_t ROL64(uint64_t x, uint32_t n) {
    return (x << n) | (x >> (64u - n));
}

__device__ __forceinline__ void keccak_f1600(uint64_t s[25]) {
    #pragma unroll
    for (int r = 0; r < 24; r++) {
        // Theta
        uint64_t C0 = s[0]^s[5]^s[10]^s[15]^s[20];
        uint64_t C1 = s[1]^s[6]^s[11]^s[16]^s[21];
        uint64_t C2 = s[2]^s[7]^s[12]^s[17]^s[22];
        uint64_t C3 = s[3]^s[8]^s[13]^s[18]^s[23];
        uint64_t C4 = s[4]^s[9]^s[14]^s[19]^s[24];
        uint64_t D0 = C4 ^ ROL64(C1, 1);
        uint64_t D1 = C0 ^ ROL64(C2, 1);
        uint64_t D2 = C1 ^ ROL64(C3, 1);
        uint64_t D3 = C2 ^ ROL64(C4, 1);
        uint64_t D4 = C3 ^ ROL64(C0, 1);
        s[0]^=D0; s[5]^=D0; s[10]^=D0; s[15]^=D0; s[20]^=D0;
        s[1]^=D1; s[6]^=D1; s[11]^=D1; s[16]^=D1; s[21]^=D1;
        s[2]^=D2; s[7]^=D2; s[12]^=D2; s[17]^=D2; s[22]^=D2;
        s[3]^=D3; s[8]^=D3; s[13]^=D3; s[18]^=D3; s[23]^=D3;
        s[4]^=D4; s[9]^=D4; s[14]^=D4; s[19]^=D4; s[24]^=D4;

        // Rho + Pi
        uint64_t t = s[1];
        uint64_t b;
        b = s[10]; s[10] = ROL64(t,  1);  t = b;
        b = s[ 7]; s[ 7] = ROL64(t,  3);  t = b;
        b = s[11]; s[11] = ROL64(t,  6);  t = b;
        b = s[17]; s[17] = ROL64(t, 10);  t = b;
        b = s[18]; s[18] = ROL64(t, 15);  t = b;
        b = s[ 3]; s[ 3] = ROL64(t, 21);  t = b;
        b = s[ 5]; s[ 5] = ROL64(t, 28);  t = b;
        b = s[16]; s[16] = ROL64(t, 36);  t = b;
        b = s[ 8]; s[ 8] = ROL64(t, 45);  t = b;
        b = s[21]; s[21] = ROL64(t, 55);  t = b;
        b = s[24]; s[24] = ROL64(t,  2);  t = b;
        b = s[ 4]; s[ 4] = ROL64(t, 14);  t = b;
        b = s[15]; s[15] = ROL64(t, 27);  t = b;
        b = s[23]; s[23] = ROL64(t, 41);  t = b;
        b = s[19]; s[19] = ROL64(t, 56);  t = b;
        b = s[13]; s[13] = ROL64(t,  8);  t = b;
        b = s[12]; s[12] = ROL64(t, 25);  t = b;
        b = s[ 2]; s[ 2] = ROL64(t, 43);  t = b;
        b = s[20]; s[20] = ROL64(t, 62);  t = b;
        b = s[14]; s[14] = ROL64(t, 18);  t = b;
        b = s[22]; s[22] = ROL64(t, 39);  t = b;
        b = s[ 9]; s[ 9] = ROL64(t, 61);  t = b;
        b = s[ 6]; s[ 6] = ROL64(t, 20);  t = b;
                   s[ 1] = ROL64(t, 44);

        // Chi
        #pragma unroll
        for (int y = 0; y < 25; y += 5) {
            uint64_t a0 = s[y+0], a1 = s[y+1], a2 = s[y+2], a3 = s[y+3], a4 = s[y+4];
            s[y+0] = a0 ^ ((~a1) & a2);
            s[y+1] = a1 ^ ((~a2) & a3);
            s[y+2] = a2 ^ ((~a3) & a4);
            s[y+3] = a3 ^ ((~a4) & a0);
            s[y+4] = a4 ^ ((~a0) & a1);
        }

        // Iota
        s[0] ^= RC[r];
    }
}
