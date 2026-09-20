// Re-filter a retained candidate list against newly acquired terrain samples.
//
//   refine_candidates CANDIDATES_IN SAMPLES THREADS CANDIDATES_OUT
//
// This is the second half of "scan once early, then refine for free". A full 2^32 sweep
// costs minutes of CPU; re-testing a few thousand retained seeds costs milliseconds, so
// once an early scan has produced a candidate list there is no reason to ever sweep the
// domain again as better observations arrive.
//
// CANDIDATES_IN is one seed per line, optionally "dataset seed" as terrain_filter_fast
// writes it for multi-dataset runs; the dataset column is preserved.
#include "mt_prefix.hpp"
#include <atomic>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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
    if (argc != 5) {
        std::cerr << "refine_candidates CANDIDATES_IN SAMPLES THREADS CANDIDATES_OUT\n";
        return 2;
    }
    std::vector<Sample> samples;
    {
        std::ifstream f(argv[2]); Sample s; std::string line;
        while (std::getline(f, line)) {                 // "x y label [radius]"
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            std::istringstream ls(line); s.radius = 0.0;
            if (!(ls >> s.x >> s.y >> s.label)) { std::cerr << "bad sample line\n"; return 2; }
            ls >> s.radius;
            samples.push_back(s);
        }
        if (samples.empty()) { std::cerr << "No terrain samples\n"; return 2; }
    }
    std::vector<std::pair<int, uint32_t>> cands;   // (dataset, seed)
    {
        std::ifstream f(argv[1]); std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            unsigned long long a, b;
            if (!(ss >> a)) continue;
            if (ss >> b) cands.push_back({(int)a, (uint32_t)b});
            else         cands.push_back({-1, (uint32_t)a});
        }
    }
    int workers = std::stoi(argv[3]);
    if (workers < 1 || workers > 256) { std::cerr << "Bad thread count\n"; return 2; }

    double rmax = 0.0;
    for (const auto& q : samples) if (q.radius > rmax) rmax = q.radius;

    build_initial();
    std::vector<char> keep(cands.size(), 0);
    std::atomic<size_t> next(0);
    auto begin = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    for (int w = 0; w < workers; w++) threads.emplace_back([&] {
        while (true) {
            size_t i = next.fetch_add(256);
            if (i >= cands.size()) break;
            size_t j = std::min(i + 256, cands.size());
            for (; i < j; i++) {
                Prefix p;
                reference_prefix(cands[i].second, p);   // exact scalar path; the list is small
                keep[i] = (rmax > 0.0 ? map_matches_relaxed(p.xs, p.ys, p.types, samples)
                                     : map_matches_exact(p.xs, p.ys, p.types, samples)) ? 1 : 0;
            }
        }
    });
    for (auto& t : threads) t.join();

    size_t kept = 0;
    {
        std::ofstream out(argv[4]);
        for (size_t i = 0; i < cands.size(); i++) if (keep[i]) {
            if (cands[i].first >= 0) out << cands[i].first << ' ';
            out << cands[i].second << '\n';
            kept++;
        }
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::cout << "{\"input\":" << cands.size() << ",\"kept\":" << kept
              << ",\"samples\":" << samples.size()
              << ",\"max_radius\":" << rmax
              << ",\"test\":\"" << (rmax > 0.0 ? "relaxed" : "exact") << "\"" << ",\"seconds\":" << elapsed
              << ",\"threads\":" << workers << "}\n";
    return 0;
}
