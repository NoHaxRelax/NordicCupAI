// Correctness gate for the fast seed-prefix kernel.
//
//   1. Every CPython vector from gen_vectors.py must reproduce exactly, including the
//      multi-word keys that CPython builds for seeds >= 2^32.
//   2. Contiguous sweeps (domain start, the live-recovered seed, domain end) must agree
//      with the exact scalar reference for every seed, with batch alignment varied so
//      each seed is exercised in several lane/group positions.
//   3. The output-window fallback rate is reported; it must stay rare and must never
//      change a result, since the fallback path is the exact scalar reference.
//
//   build: g++ -O2 -march=native -std=c++17 -I.. -o test_prefix test_prefix.cpp
//   run:   python gen_vectors.py > vectors.txt && ./test_prefix vectors.txt
#include "../mt_prefix.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

uint32_t g_initial[624];

static bool same(const Prefix& a, const Prefix& b) {
    for (int i = 0; i < 10; i++)
        if (a.xs[i] != b.xs[i] || a.ys[i] != b.ys[i] || a.types[i] != b.types[i]) return false;
    return true;
}

static void dump(const char* tag, uint32_t seed, const Prefix& p) {
    printf("    %s seed=%u xs=", tag, seed);
    for (int i = 0; i < 10; i++) printf("%d,", p.xs[i]);
    printf(" ys=");
    for (int i = 0; i < 10; i++) printf("%d,", p.ys[i]);
    printf(" t=");
    for (int i = 0; i < 10; i++) printf("%d,", p.types[i]);
    printf("\n");
}

int main(int argc, char** argv) {
    build_initial();
    int failures = 0, multiword = 0;
    long long fallbacks = 0, checked = 0;

    // ---- 1. CPython ground-truth vectors ----
    if (argc > 1) {
        std::ifstream in(argv[1]);
        if (!in) { printf("cannot open %s\n", argv[1]); return 2; }
        std::string line;
        int n = 0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string keytok;
            ss >> keytok;
            // "w0" or "w0,w1,..." : CPython's little-endian 32-bit key words
            std::vector<uint32_t> key;
            {
                std::istringstream ks(keytok);
                std::string w;
                while (std::getline(ks, w, ',')) key.push_back((uint32_t)std::stoull(w));
            }
            const uint32_t seed = key[0];
            Prefix want;
            for (int i = 0; i < 10; i++) ss >> want.xs[i] >> want.ys[i];
            for (int i = 0; i < 10; i++) ss >> want.types[i];

            Prefix ref;
            reference_prefix(key, ref);
            if (!same(want, ref) && failures++ < 5) {
                printf("  CPYTHON vs scalar-reference mismatch (%zu-word key):\n", key.size());
                dump("cpython  ", seed, want);
                dump("reference", seed, ref);
            }
            // The batch kernel is the one-word hot path only; multi-word keys go through
            // the scalar reference in production too (see terrain_filter_fast.cpp), so the
            // check above is the whole check for them.
            if (key.size() == 1) {
                static thread_local fast::Batch b;
                b.run(seed);
                Prefix got;
                if (!b.generate(0, 0, got)) { fallbacks++; reference_prefix(key, got); }
                if (!same(want, got) && failures++ < 5) {
                    printf("  CPYTHON vs fast-kernel mismatch:\n");
                    dump("cpython", seed, want);
                    dump("fast   ", seed, got);
                }
            } else {
                multiword++;
            }
            n++;
            checked++;
        }
        printf("cpython vectors  : %d checked (%d with multi-word keys)\n", n, multiword);
    } else {
        printf("cpython vectors  : SKIPPED (no vectors file given)\n");
    }

    // ---- 2. contiguous sweeps at several batch alignments ----
    const uint64_t BATCH = (uint64_t)LANES * GROUPS;
    struct Range { const char* name; uint64_t start, count; };
    const Range ranges[] = {
        {"domain start", 0ull, 8192ull},
        {"live seed 614466944", 614466944ull - 4096, 8192ull},
        {"domain end", (1ull << 32) - 8192, 8192ull},
    };
    for (const auto& r : ranges) {
        int bad = 0;
        for (uint64_t shift = 0; shift < BATCH; shift += (BATCH > 4 ? BATCH / 4 : 1)) {
            uint64_t begin = r.start >= shift ? r.start - shift : 0;
            for (uint64_t s = begin; s + BATCH <= r.start + r.count; s += BATCH) {
                static thread_local fast::Batch b;
                b.run(s);
                for (int g = 0; g < GROUPS; g++) for (int l = 0; l < LANES; l++) {
                    uint64_t seed = s + (uint64_t)g * LANES + l;
                    if (seed >= (1ull << 32)) continue;
                    Prefix ref, got;
                    reference_prefix((uint32_t)seed, ref);
                    if (!b.generate(g, l, got)) { fallbacks++; reference_prefix((uint32_t)seed, got); }
                    checked++;
                    if (!same(ref, got) && bad++ < 3) {
                        failures++;
                        printf("  SWEEP mismatch in %s:\n", r.name);
                        dump("reference", (uint32_t)seed, ref);
                        dump("fast     ", (uint32_t)seed, got);
                    }
                }
            }
        }
        printf("sweep %-22s: %s\n", r.name, bad ? "FAIL" : "ok");
    }

    printf("\nLANES=%d GROUPS=%d window=%d\n", LANES, GROUPS, fast::WINDOW);
    printf("seeds checked    : %lld\n", checked);
    printf("window fallbacks : %lld (%.6f%%, exact scalar path)\n",
           fallbacks, checked ? 100.0 * fallbacks / checked : 0.0);
    printf("%s\n", failures ? "FAILED" : "ALL CHECKS PASSED");
    return failures ? 1 : 0;
}
