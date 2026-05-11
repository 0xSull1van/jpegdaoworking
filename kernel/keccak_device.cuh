#pragma once
#include <stdint.h>

// ROL64: rotate-left 64 bits using native 32-bit funnel shift.
// On Ada Lovelace SHF.L is 2-cycle vs ~9 cycles for emulated 64-bit shifts.
// All call sites pass a compile-time constant `n` so branches fold to one path.
__device__ __forceinline__ uint64_t ROL64(uint64_t x, uint32_t n) {
    uint32_t lo = static_cast<uint32_t>(x);
    uint32_t hi = static_cast<uint32_t>(x >> 32);
    uint32_t new_hi, new_lo;
    if (n >= 32u) {
        uint32_t m = n - 32u;
        new_hi = __funnelshift_l(hi, lo, m);
        new_lo = __funnelshift_l(lo, hi, m);
    } else {
        new_hi = __funnelshift_l(lo, hi, n);
        new_lo = __funnelshift_l(hi, lo, n);
    }
    return (static_cast<uint64_t>(new_hi) << 32) | static_cast<uint64_t>(new_lo);
}

// Note: keccak_f1600 and RC[] are no longer here. The grind kernel now expands
// 24 rounds inline via macros operating on 25 named local variables (s0..s24),
// guaranteeing the state lives in registers instead of local memory.
