// Drop-in replacement for seed-recovery-v4's terrain_filter.
//
// Identical CLI, identical stdout contract, identical candidate output format, so the
// frozen V4 streaming coordinator can call this binary with no changes:
//
//   terrain_filter_fast SAMPLES START COUNT THREADS CANDIDATES_OUT [STOP_FILE]
//
// SAMPLES is a comma-separated list of files, each holding "x y label" lines.
// Candidates are written one per line ("dataset seed" when several datasets are given).
//
// Optional environment, kept out of the positional CLI so the V4 contract is untouched:
//   MAX_CANDIDATES=N      stop once N hits have been written (exit 4, receipt says capped)
//   SEED_HIGH_WORDS=a,b   search multi-word CPython keys: START..START+COUNT sweeps word 0
//                         while words 1.. are fixed to a,b (seed = w0 + a*2^32 + b*2^64)
//
// The seeding kernel is mt_prefix.hpp; see README.md for what changed and the measurements.
#include "mt_prefix.hpp"
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

uint32_t g_initial[624];

struct Sample { int x, y, label; double radius; };

// Two tests, deliberately kept separate.
//
// EXACT (all radii zero): the observed label must equal the nearest site's type. Integer
// squared distances, no rounding, and it is what the reliability harness uses because its
// sample positions come straight from the engine.
//
// RELAXED (any radius > 0): the localisation-tolerant test that the Rust filter this
// replaces implements (scripts/seed_search_rust/src/lib.rs:240-254) and that the Python
// reference mirrors (scripts/seed_survey_probe.py:100-117). An earlier version of this
// file dropped the radius entirely, which is a false-negative channel: a live agent's
// estimated position carries r = 1.5 (geometric registration) to 12 (shared estimator),
// and the exact test then rejects the TRUE seed on samples whose int() lookup flipped.
//
// The relaxed test is sound. If p_hat is within r of the pixel q whose label was read,
// then d(p_hat,S_lab) <= d(q,S_lab)+r = d(q,S_any)+r <= d(p_hat,S_any)+2r. The 2r slack
// is exactly tight, so a correctly declared radius can never reject the true seed -- but
// an UNDERSTATED radius silently can, and the candidate yield barely moves when it does.
// Scan at a generous radius and refine downward rather than guessing it tightly.
static inline bool map_matches_exact(const int* xs, const int* ys, const int* types,
                                     const std::vector<Sample>& samples) {
    for (const auto& s : samples) {
        int best = 0, dist = 2147483647;
        for (int i = 0; i < 10; i++) {
            int dx = xs[i] - s.x, dy = ys[i] - s.y, d = dx * dx + dy * dy;
            if (d < dist) { dist = d; best = i; }
        }
        if (types[best] != s.label) return false;
    }
    return true;
}

static inline bool map_matches_relaxed(const int* xs, const int* ys, const int* types,
                                       const std::vector<Sample>& samples) {
    for (const auto& s : samples) {
        double closest = 1e300, desired = 1e300;
        for (int i = 0; i < 10; i++) {
            const double dx = double(xs[i] - s.x), dy = double(ys[i] - s.y);
            const double sq = dx * dx + dy * dy;
            if (sq < closest) closest = sq;
            if (types[i] == s.label && sq < desired) desired = sq;
        }
        const double bound = std::sqrt(closest) + 2.0 * s.radius;
        if (desired > bound * bound + 1e-9) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "terrain_filter_fast SAMPLES START COUNT THREADS CANDIDATES_OUT [STOP_FILE]\n";
        return 2;
    }

    // Sample parsing is strict on purpose. The lenient version had two silent failure
    // modes: a river label (4) can never match any seed, so one stray river sample turned
    // a good scan into "0 hits, exit 0", indistinguishable from "seed not in range"; and a
    // malformed line stopped the read loop early, keeping a truncated sample set that then
    // matched a huge fraction of the domain. Both now fail loudly before any scanning.
    std::vector<std::vector<Sample>> datasets;
    std::vector<int> distinct_labels;
    std::vector<double> max_radius;
    {
        std::stringstream paths(argv[1]);
        std::string path;
        while (std::getline(paths, path, ',')) {
            std::ifstream file(path);
            if (!file) { std::cerr << "Cannot open " << path << "\n"; return 2; }
            std::vector<Sample> samples; Sample s; int seen[4] = {0, 0, 0, 0};
            std::string line;
            while (std::getline(file, line)) {
                if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
                std::istringstream ls(line);
                s.radius = 0.0;                       // 4th column optional: "x y label [r]"
                if (!(ls >> s.x >> s.y >> s.label)) {
                    std::cerr << path << ": parse error after " << samples.size()
                              << " samples; refusing to scan with a truncated sample set\n";
                    return 2;
                }
                ls >> s.radius;                        // absent -> 0 -> exact test
                if (!(s.radius >= 0.0) || s.radius > 500.0) {
                    std::cerr << path << ": radius " << s.radius << " out of range [0,500]\n";
                    return 2;
                }
                if (s.label < 0 || s.label > 3) {
                    std::cerr << path << ": label " << s.label << " at (" << s.x << "," << s.y
                              << ") is not a land biome in [0,3]. River samples (4) carry no "
                                 "information about the base map; drop them upstream.\n";
                    return 2;
                }
                if (s.x < 0 || s.x >= 1600 || s.y < 0 || s.y >= 1200) {
                    std::cerr << path << ": sample (" << s.x << "," << s.y
                              << ") lies outside the 1600x1200 map\n";
                    return 2;
                }
                seen[s.label] = 1;
                samples.push_back(s);
            }
            if (!file.eof()) {
                std::cerr << path << ": parse error after " << samples.size()
                          << " samples; refusing to scan with a truncated sample set\n";
                return 2;
            }
            if (samples.empty()) { std::cerr << path << ": no terrain samples\n"; return 2; }
            distinct_labels.push_back(seen[0] + seen[1] + seen[2] + seen[3]);
            double rmax = 0.0;
            for (const auto& q : samples) if (q.radius > rmax) rmax = q.radius;
            max_radius.push_back(rmax);
            datasets.push_back(std::move(samples));
        }
    }
    // Diversity, not count, is what narrows the domain: a single-label sample set is
    // consistent with roughly a third of all seeds. Warn loudly; the caller decides.
    for (size_t d = 0; d < datasets.size(); d++)
        if (distinct_labels[d] < 2)
            std::cerr << "warning: dataset " << d << " has " << datasets[d].size()
                      << " samples but a single biome label; expect a candidate explosion. "
                         "Collect samples across more biomes before a full-domain scan.\n";

    uint64_t start = std::stoull(argv[2]), count = std::stoull(argv[3]), end = start + count;
    if (end > (1ULL << 32)) { std::cerr << "Range outside uint32\n"; return 2; }
    int workers = std::stoi(argv[4]);
    if (workers < 1 || workers > 256) { std::cerr << "Bad thread count\n"; return 2; }

    uint64_t max_candidates = 0;   // 0 = unlimited
    if (const char* e = std::getenv("MAX_CANDIDATES")) max_candidates = std::strtoull(e, nullptr, 10);

    // Multi-word keys: the verifier (fastsim/_engine.cpp) already handles them; this is
    // the only component that was uint32-bound. Word 0 is swept, words 1.. are fixed.
    std::vector<uint32_t> high_words;
    if (const char* e = std::getenv("SEED_HIGH_WORDS")) {
        std::stringstream ss(e); std::string w;
        while (std::getline(ss, w, ',')) if (!w.empty()) high_words.push_back((uint32_t)std::stoull(w));
    }
    const fast::KeyPlan plan(high_words);
    const bool multiword = plan.words() > 1;

    build_initial();

    std::atomic<uint64_t> next(start), done(0), hits(0);
    std::mutex output_lock;
    std::ofstream output(argv[5]);
    auto begin = std::chrono::steady_clock::now();

    // Chunk is a multiple of the batch so a worker never straddles a partial batch.
    constexpr uint64_t BATCH = (uint64_t)LANES * GROUPS;
    const uint64_t CHUNK = ((4096 + BATCH - 1) / BATCH) * BATCH;

    std::vector<std::thread> threads;
    for (int w = 0; w < workers; w++) threads.emplace_back([&] {
        static thread_local fast::Batch batch;
        std::string pending;   // per-thread hit buffer: one lock per chunk, not per hit
        auto drain = [&] {
            if (pending.empty()) return;
            std::lock_guard<std::mutex> lock(output_lock);
            output << pending; output.flush(); pending.clear();
        };
        while (true) {
            if (argc == 7 && std::filesystem::exists(argv[6])) break;
            if (max_candidates && hits.load() >= max_candidates) break;
            uint64_t a = next.fetch_add(CHUNK);
            if (a >= end) break;
            uint64_t b = std::min(a + CHUNK, end);
            for (uint64_t seed = a; seed < b; seed += BATCH) {
                // The SIMD batch kernel covers the one-word key, which is the whole uint32
                // domain and therefore the hot path. Multi-word keys (an evaluation seed at
                // or above 2^32) go through the exact scalar path instead: ~2000x slower per
                // seed, but those searches are a bounded window around a clock value, not a
                // full-domain sweep, so simplicity and obvious correctness win there.
                if (multiword) {
                    for (uint64_t s = seed; s < std::min(seed + BATCH, b); s++) {
                        Prefix p;
                        reference_prefix(plan.key_for(uint32_t(s)), p);
                        for (size_t d = 0; d < datasets.size(); d++) {
                            const bool ok = max_radius[d] > 0.0
                                ? map_matches_relaxed(p.xs, p.ys, p.types, datasets[d])
                                : map_matches_exact(p.xs, p.ys, p.types, datasets[d]);
                            if (ok) {
                                if (datasets.size() > 1) { pending += std::to_string(d); pending += ' '; }
                                pending += std::to_string(s); pending += '\n';
                                hits++;
                            }
                        }
                    }
                    continue;
                }
                batch.run(seed);
                for (int g = 0; g < GROUPS; g++) {
                    for (int lane = 0; lane < LANES; lane++) {
                        uint64_t s = seed + (uint64_t)g * LANES + lane;
                        if (s >= b) break;
                        Prefix p;
                        // Rare (<1 in 2e6) overflow of the output window: fall back to the
                        // exact scalar CPython path rather than truncating rejection sampling.
                        if (!batch.generate(g, lane, p)) reference_prefix(plan.key_for(uint32_t(s)), p);
                        for (size_t d = 0; d < datasets.size(); d++) {
                            const bool ok = max_radius[d] > 0.0
                                ? map_matches_relaxed(p.xs, p.ys, p.types, datasets[d])
                                : map_matches_exact(p.xs, p.ys, p.types, datasets[d]);
                            if (ok) {
                                if (datasets.size() > 1) { pending += std::to_string(d); pending += ' '; }
                                pending += std::to_string(s); pending += '\n';
                                hits++;
                            }
                        }
                    }
                }
            }
            drain();
            uint64_t previous = done.fetch_add(b - a);
            if ((previous >> 26) != ((previous + b - a) >> 26)) {
                double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - begin).count();
                std::lock_guard<std::mutex> lock(output_lock);
                std::cout << "{\"progress\":true,\"tested\":" << done
                          << ",\"hits\":" << hits << ",\"seconds\":" << elapsed << "}" << std::endl;
            }
        }
        drain();
    });
    for (auto& t : threads) t.join();

    const bool capped = max_candidates && hits.load() >= max_candidates;
    const bool complete = (done.load() == count) && !capped;
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::cout << "{\"tested\":" << done << ",\"hits\":" << hits << ",\"seconds\":" << elapsed
              << ",\"seeds_per_second\":" << double(done) / elapsed
              << ",\"threads\":" << workers
              << ",\"complete\":" << (complete ? "true" : "false")
              << ",\"capped\":" << (capped ? "true" : "false")
              << ",\"key_words\":" << plan.words()
              << ",\"datasets\":[";
    for (size_t d = 0; d < datasets.size(); d++)
        std::cout << (d ? "," : "") << "{\"samples\":" << datasets[d].size()
                  << ",\"distinct_labels\":" << distinct_labels[d]
                  << ",\"max_radius\":" << max_radius[d]
                  << ",\"test\":\"" << (max_radius[d] > 0.0 ? "relaxed" : "exact") << "\"}";
    std::cout << "]}\n";
    return capped ? 4 : 0;
}
