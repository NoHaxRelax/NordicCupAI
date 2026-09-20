// Host-side correctness test for prefix96.hpp, the single-seed kernel the CUDA port uses.
//
// This machine cannot build CUDA (no MSVC toolchain), so this is what vouches for the GPU
// kernel's arithmetic: the same header, compiled by g++, must reproduce CPython's terrain
// prefix for every pinned vector and for contiguous sweeps at the domain edges. Only the
// CUDA plumbing (launch, atomics, memory) remains unverified until it is built on Linux.
//
//   build: g++ -O2 -std=c++17 -I.. -o test_prefix96 test_prefix96.cpp
//   run:   ./test_prefix96 vectors.txt
#include "../prefix96.hpp"
#include "../mt_prefix.hpp"      // scalar CPython reference and Prefix
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

uint32_t g_initial[624];

static bool same(const Prefix& a, const int* xs, const int* ys, const int* ty) {
    for (int i = 0; i < 10; i++)
        if (a.xs[i] != xs[i] || a.ys[i] != ys[i] || a.types[i] != ty[i]) return false;
    return true;
}

int main(int argc, char** argv) {
    build_initial();
    int failures = 0; long long checked = 0, overflow = 0;

    // 1. CPython vectors (one-word keys only: prefix96 is the single-word GPU path)
    if (argc > 1) {
        std::ifstream in(argv[1]);
        std::string line; int n = 0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string keytok; ss >> keytok;
            if (keytok.find(',') != std::string::npos) continue;   // multi-word: CPU-only path
            const uint32_t seed = (uint32_t)std::stoull(keytok);
            Prefix want;
            for (int i = 0; i < 10; i++) ss >> want.xs[i] >> want.ys[i];
            for (int i = 0; i < 10; i++) ss >> want.types[i];
            int xs[10], ys[10], ty[10];
            if (!pfx96::prefix(seed, g_initial, xs, ys, ty)) { overflow++; reference_prefix(seed, want); continue; }
            if (!same(want, xs, ys, ty) && failures++ < 5) printf("  MISMATCH vs CPython at seed %u\n", seed);
            n++; checked++;
        }
        printf("cpython vectors  : %d checked\n", n);
    }
    // 2. contiguous sweeps against the exact scalar reference
    const uint64_t ranges[][2] = {{0ull, 16384ull}, {614466944ull - 8192, 16384ull}, {(1ull << 32) - 16384, 16384ull}};
    for (auto& r : ranges) {
        int bad = 0;
        for (uint64_t s = r[0]; s < r[0] + r[1]; s++) {
            Prefix ref; reference_prefix((uint32_t)s, ref);
            int xs[10], ys[10], ty[10];
            if (!pfx96::prefix((uint32_t)s, g_initial, xs, ys, ty)) { overflow++; continue; }
            checked++;
            if (!same(ref, xs, ys, ty) && bad++ < 3) { failures++; printf("  SWEEP mismatch at seed %llu\n", (unsigned long long)s); }
        }
        printf("sweep from %-12llu: %s\n", (unsigned long long)r[0], bad ? "FAIL" : "ok");
    }
    printf("\nwindow=%d  seeds checked: %lld  window overflows (exact fallback): %lld\n",
           pfx96::WINDOW, checked, overflow);
    printf("%s\n", failures ? "FAILED" : "ALL PREFIX96 CHECKS PASSED");
    return failures ? 1 : 0;
}
