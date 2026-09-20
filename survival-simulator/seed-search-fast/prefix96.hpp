// Single-seed terrain prefix with a 96-word output window, shared by the CUDA kernel and
// its host-side correctness test. The math is identical to mt_prefix.hpp's fast::Batch,
// minus the lane/group vectorisation the GPU does not need: one seed per thread.
//
// PFX_FN is the function qualifier: `__device__ __forceinline__` when compiled by nvcc,
// plain `inline` for the host test. Keeping the math in one header is what lets the host
// test vouch for the GPU kernel's arithmetic even on a machine that cannot build CUDA.
#pragma once
#include <cstdint>

#ifndef PFX_FN
#define PFX_FN inline
#endif

namespace pfx96 {

// Output words produced per seed. 96 covers 99.99995% of seeds; a full-domain GPU scan
// measured 83 overflows in 4,294,967,296 seeds (1 in 51.7M). An overflow is NOT a
// rejection -- the caller must re-check those seeds exactly, or the true seed could be
// missed. Set PFX_WINDOW=128 to make overflow impossible (max observed need: 97).
#ifndef PFX_WINDOW
#define PFX_WINDOW 96
#endif
constexpr int WINDOW = PFX_WINDOW;
constexpr int LOW = WINDOW + 1; // mt[0..96] needed by the twist for words 0..95
constexpr int SITES = 10;

PFX_FN uint32_t temper(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

// `init` is CPython's init_genrand(19650218) state, the constant every seeding starts
// from. Returns false if this seed's prefix needs more than WINDOW words; the caller must
// then fall back to an exact scalar path rather than truncate the rejection sampling.
PFX_FN bool prefix(uint32_t seed, const uint32_t* init, int* xs, int* ys, int* types) {
    uint32_t low[LOW];
    uint32_t out[WINDOW];

    // pass A: loop-1 chain only, to obtain mt[1] after the first seeding loop
    uint32_t prev = init[0];
    prev = (init[1] ^ ((prev ^ (prev >> 30)) * 1664525u)) + seed;
    const uint32_t a1_1_first = prev;
    for (int i = 2; i < 624; i++)
        prev = (init[i] ^ ((prev ^ (prev >> 30)) * 1664525u)) + seed;
    const uint32_t a1_0 = prev;                                   // wrap: mt[0] = mt[623]
    const uint32_t a1_1_final = (a1_1_first ^ ((a1_0 ^ (a1_0 >> 30)) * 1664525u)) + seed;

    // pass B: loop-1 recomputed, fused with loop-2
    uint32_t p1 = a1_1_first, p2 = a1_1_final, h0 = 0, h1 = 0;
    for (int i = 2; i < LOW; i++) {
        p1 = (init[i] ^ ((p1 ^ (p1 >> 30)) * 1664525u)) + seed;
        p2 = (p1 ^ ((p2 ^ (p2 >> 30)) * 1566083941u)) - (uint32_t)i;
        low[i] = p2;
    }
    for (int i = LOW; i < 624; i++) {
        p1 = (init[i] ^ ((p1 ^ (p1 >> 30)) * 1664525u)) + seed;
        p2 = (p1 ^ ((p2 ^ (p2 >> 30)) * 1566083941u)) - (uint32_t)i;
        const int k = i - 397;
        if (i == 397) h0 = p2;
        else if (i == 398) h1 = p2;
        else if (k >= 2 && k < WINDOW) {
            const uint32_t y = (low[k] & 0x80000000u) | (low[k + 1] & 0x7fffffffu);
            out[k] = temper(p2 ^ (y >> 1) ^ ((0u - (y & 1u)) & 0x9908b0dfu));
        }
    }
    // close loop 2, then words 0 and 1 (they depend on mt[1], written last)
    const uint32_t a2_0 = p2;
    const uint32_t a2_1 = (a1_1_final ^ ((a2_0 ^ (a2_0 >> 30)) * 1566083941u)) - 1u;
    low[0] = 0x80000000u;
    low[1] = a2_1;
    {
        uint32_t y = (low[0] & 0x80000000u) | (low[1] & 0x7fffffffu);
        out[0] = temper(h0 ^ (y >> 1) ^ ((0u - (y & 1u)) & 0x9908b0dfu));
        y = (low[1] & 0x80000000u) | (low[2] & 0x7fffffffu);
        out[1] = temper(h1 ^ (y >> 1) ^ ((0u - (y & 1u)) & 0x9908b0dfu));
    }

    // rejection sampling exactly as CPython's randbelow: top 11 bits, then top 3 bits
    int pos = 0;
    for (int i = 0; i < SITES; i++) {
        uint32_t r;
        do { if (pos >= WINDOW) return false; r = out[pos++] >> 21; } while (r >= 1600u);
        xs[i] = (int)r;
        do { if (pos >= WINDOW) return false; r = out[pos++] >> 21; } while (r >= 1200u);
        ys[i] = (int)r;
    }
    for (int i = 0; i < SITES; i++) {
        uint32_t r;
        do { if (pos >= WINDOW) return false; r = out[pos++] >> 29; } while (r >= 4u);
        types[i] = (int)r;
    }
    return true;
}

}  // namespace pfx96
