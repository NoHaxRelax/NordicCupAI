// Streaming candidate filter: scan the domain once, early, then narrow as evidence arrives.
//
//   stream_filter CANDIDATES_IN SAMPLES THREADS [SURVIVORS_OUT]
//
// The pipeline this belongs to is "scan early, then refine", and the reason it exists is
// that `refine_candidates` re-derives every candidate's terrain prefix on every call. That
// is ~1,247 Mersenne Twister seeding steps per candidate per refinement -- fine for a few
// thousand candidates, ruinous for the ~10^5-10^6 you retain if you dispatch the scan at
// t=2s instead of t=15s.
//
// So: derive each candidate's 10 Voronoi sites and 10 biome types ONCE, keep them, and
// test each new sample against the cached prefixes. The seeding cost is paid a single
// time; thereafter a new observation costs 10 integer distance computations per surviving
// candidate, and the surviving set shrinks fast, so later samples are cheaper still.
//
// This turns seed recovery from "wait until the evidence is conclusive, then search" into
// "search immediately on weak evidence, then let the world narrow it for free" -- which
// matters because acquisition, not search, is what dominates time-to-seed.
#include "mt_prefix.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

uint32_t g_initial[624];

struct Sample { int x, y, label; double radius; };

// One candidate's cached terrain prefix: 10 sites + 10 types, 30 ints = 120 B.
// At 10^6 candidates that is 120 MB, which is the whole memory budget of this approach.
struct Cached {
    uint32_t seed;
    int16_t xs[10], ys[10];
    int8_t types[10];
};

static inline bool sample_ok_exact(const Cached& c, const Sample& s) {
    int best = 0, dist = 2147483647;
    for (int i = 0; i < 10; i++) {
        int dx = c.xs[i] - s.x, dy = c.ys[i] - s.y, d = dx * dx + dy * dy;
        if (d < dist) { dist = d; best = i; }
    }
    return c.types[best] == s.label;
}

// Localisation-tolerant test; see README.md. A correctly declared radius can never reject
// the true seed, an understated one silently can.
static inline bool sample_ok_relaxed(const Cached& c, const Sample& s) {
    double closest = 1e300, desired = 1e300;
    for (int i = 0; i < 10; i++) {
        const double dx = double(c.xs[i] - s.x), dy = double(c.ys[i] - s.y);
        const double sq = dx * dx + dy * dy;
        if (sq < closest) closest = sq;
        if (c.types[i] == s.label && sq < desired) desired = sq;
    }
    const double bound = std::sqrt(closest) + 2.0 * s.radius;
    return !(desired > bound * bound + 1e-9);
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "stream_filter CANDIDATES_IN SAMPLES THREADS [SURVIVORS_OUT]\n";
        return 2;
    }
    const int workers = std::stoi(argv[3]);
    if (workers < 1 || workers > 256) { std::cerr << "Bad thread count\n"; return 2; }

    // ---- samples, in arrival order: "x y label [radius]" -------------------------------
    std::vector<Sample> samples;
    {
        std::ifstream f(argv[2]);
        if (!f) { std::cerr << "Cannot open " << argv[2] << "\n"; return 2; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            std::istringstream ls(line);
            Sample s; s.radius = 0.0;
            if (!(ls >> s.x >> s.y >> s.label)) {
                std::cerr << argv[2] << ": parse error\n"; return 2;
            }
            ls >> s.radius;
            if (s.label < 0 || s.label > 3) {
                std::cerr << argv[2] << ": label " << s.label
                          << " is not a land biome in [0,3]\n";
                return 2;
            }
            samples.push_back(s);
        }
        if (samples.empty()) { std::cerr << "No terrain samples\n"; return 2; }
    }

    // ---- candidate seeds, "seed" or "dataset seed" -------------------------------------
    std::vector<uint32_t> seeds;
    {
        std::ifstream f(argv[1]);
        if (!f) { std::cerr << "Cannot open " << argv[1] << "\n"; return 2; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            unsigned long long a, b;
            if (!(ss >> a)) continue;
            seeds.push_back((ss >> b) ? (uint32_t)b : (uint32_t)a);
        }
        if (seeds.empty()) { std::cerr << "No candidates\n"; return 2; }
    }

    build_initial();
    const auto t_start = std::chrono::steady_clock::now();
    auto since = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    };

    // ---- pay the seeding cost once -----------------------------------------------------
    std::vector<Cached> live(seeds.size());
    {
        std::atomic<size_t> next(0);
        std::vector<std::thread> ts;
        for (int w = 0; w < workers; w++) ts.emplace_back([&] {
            while (true) {
                size_t i = next.fetch_add(1024);
                if (i >= seeds.size()) break;
                size_t j = std::min(i + 1024, seeds.size());
                for (; i < j; i++) {
                    Prefix p;
                    reference_prefix(seeds[i], p);
                    Cached& c = live[i];
                    c.seed = seeds[i];
                    for (int k = 0; k < 10; k++) {
                        c.xs[k] = (int16_t)p.xs[k];
                        c.ys[k] = (int16_t)p.ys[k];
                        c.types[k] = (int8_t)p.types[k];
                    }
                }
            }
        });
        for (auto& t : ts) t.join();
    }
    const double t_cache = since();
    std::cout << "{\"stage\":\"cache\",\"candidates\":" << live.size()
              << ",\"seconds\":" << t_cache
              << ",\"bytes\":" << live.size() * sizeof(Cached) << "}\n";

    // ---- narrow, one observation at a time ---------------------------------------------
    for (size_t si = 0; si < samples.size(); si++) {
        const Sample& s = samples[si];
        const bool relaxed = s.radius > 0.0;
        const double t0 = since();
        const size_t before = live.size();

        if (live.size() < 4096 || workers == 1) {
            live.erase(std::remove_if(live.begin(), live.end(), [&](const Cached& c) {
                           return !(relaxed ? sample_ok_relaxed(c, s) : sample_ok_exact(c, s));
                       }), live.end());
        } else {
            // Partition in parallel, then compact in order so the survivor list stays
            // deterministic regardless of thread count.
            std::vector<char> keep(live.size(), 0);
            std::atomic<size_t> next(0);
            std::vector<std::thread> ts;
            for (int w = 0; w < workers; w++) ts.emplace_back([&] {
                while (true) {
                    size_t i = next.fetch_add(4096);
                    if (i >= live.size()) break;
                    size_t j = std::min(i + 4096, live.size());
                    for (; i < j; i++)
                        keep[i] = (relaxed ? sample_ok_relaxed(live[i], s)
                                           : sample_ok_exact(live[i], s)) ? 1 : 0;
                }
            });
            for (auto& t : ts) t.join();
            size_t w2 = 0;
            for (size_t i = 0; i < live.size(); i++) if (keep[i]) live[w2++] = live[i];
            live.resize(w2);
        }

        const double dt = since() - t0;
        std::cout << "{\"sample\":" << (si + 1)
                  << ",\"test\":\"" << (relaxed ? "relaxed" : "exact") << "\""
                  << ",\"before\":" << before << ",\"after\":" << live.size()
                  << ",\"seconds\":" << dt
                  << ",\"elapsed\":" << since() << "}" << std::endl;
        if (live.size() <= 1) {
            std::cout << "{\"stage\":\"unique\",\"samples_used\":" << (si + 1)
                      << ",\"remaining\":" << live.size()
                      << ",\"elapsed\":" << since() << "}\n";
            break;
        }
    }

    if (argc == 5) {
        std::ofstream out(argv[4]);
        for (const auto& c : live) out << c.seed << '\n';
    }
    std::cout << "{\"stage\":\"done\",\"survivors\":" << live.size()
              << ",\"cache_seconds\":" << t_cache
              << ",\"total_seconds\":" << since() << "}\n";
    return 0;
}
