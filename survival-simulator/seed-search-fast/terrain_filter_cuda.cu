// CUDA port of terrain_filter_fast: the same terrain prefix scan on the GPU.
//
// Same CLI, same stdout contract, same candidate file format as terrain_filter_fast and
// as seed-recovery-v4's terrain_filter, so the frozen coordinator can call it unchanged:
//
//   terrain_filter_cuda SAMPLES START COUNT THREADS CANDIDATES_OUT [STOP_FILE]
//
// THREADS is kept for CLI compatibility and ignored; use CUDA_BLOCK/CUDA_CHUNK env vars.
//
// Why the GPU shape differs from the CPU one: on the CPU the MT seeding chain is
// latency-bound, so the kernel interleaves GROUPS independent chains to fill the pipeline.
// A GPU already has thousands of independent warps in flight, so one seed per thread
// hides that latency for free and the GROUPS machinery is dropped. What replaces it as
// the limiting factor is per-thread state: `low` and `out` are ~770 B/thread of local
// memory, so occupancy, not arithmetic, sets the ceiling.
//
// The output window defaults to 128 words (PFX_WINDOW), which CANNOT overflow. Measured over 2,000,000 seeds,
// 96 words covers 99.99995% of the prefix; the rare overrun is reported so the caller can
// re-check those seeds exactly on the CPU rather than silently truncating the rejection
// sampling.
//
// STATUS: NOT BUILT OR RUN ON THIS MACHINE (no MSVC toolchain for nvcc on Windows). The
// arithmetic lives in prefix96.hpp and is verified on the host by tests/test_prefix96.cpp
// against CPython vectors; only the CUDA plumbing here is unverified. Build on Linux:
//   nvcc -O3 -arch=sm_89 -o terrain_filter_cuda terrain_filter_cuda.cu
// and require tests/test_prefix96 to pass AND a small-range scan to reproduce the CPU
// filter's candidate list before trusting any GPU result.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_OK(call)                                                                  \
    do {                                                                               \
        cudaError_t _e = (call);                                                       \
        if (_e != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                           \
                         cudaGetErrorString(_e), __FILE__, __LINE__);                  \
            return 3;                                                                  \
        }                                                                              \
    } while (0)

#define PFX_FN __device__ __forceinline__
#include "prefix96.hpp"
#undef PFX_FN

static const int MAX_SITES = pfx96::SITES;

__constant__ uint32_t c_initial[624];

struct Sample { int x, y, label; };

__global__ void scan_kernel(uint64_t start, uint64_t count,
                            const Sample* __restrict__ samples,
                            const int* __restrict__ ds_begin,
                            const int* __restrict__ ds_end,
                            int ndatasets,
                            uint2* __restrict__ hits, unsigned int* __restrict__ nhits,
                            unsigned int max_hits, unsigned int* __restrict__ noverflow) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t seed = (uint32_t)(start + idx);

    int xs[MAX_SITES], ys[MAX_SITES], types[MAX_SITES];
    if (!pfx96::prefix(seed, c_initial, xs, ys, types)) {
        // NOT a rejection. The window was too short for this seed, so it is unresolved and
        // must be re-checked exactly on the CPU. Emitting it with dataset id 0xFFFFFFFF
        // keeps it in the candidate stream rather than silently dropping it.
        atomicAdd(noverflow, 1u);
        unsigned int slot = atomicAdd(nhits, 1u);
        if (slot < max_hits) hits[slot] = make_uint2(0xFFFFFFFFu, seed);
        return;
    }
    for (int d = 0; d < ndatasets; d++) {
        bool ok = true;
        for (int s = ds_begin[d]; s < ds_end[d] && ok; s++) {
            const Sample sm = samples[s];
            int best = 0, dist = 2147483647;
#pragma unroll
            for (int i = 0; i < MAX_SITES; i++) {
                int dx = xs[i] - sm.x, dy = ys[i] - sm.y, dd = dx * dx + dy * dy;
                if (dd < dist) { dist = dd; best = i; }
            }
            if (types[best] != sm.label) ok = false;
        }
        if (ok) {
            unsigned int slot = atomicAdd(nhits, 1u);
            if (slot < max_hits) hits[slot] = make_uint2((unsigned int)d, seed);
        }
    }
}

// CPython's init_genrand(19650218), the constant state every seeding starts from.
static void build_initial(uint32_t* mt) {
    mt[0] = 19650218u;
    for (int i = 1; i < 624; i++)
        mt[i] = (1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (uint32_t)i);
}

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::fprintf(stderr,
                     "terrain_filter_cuda SAMPLES START COUNT THREADS CANDIDATES_OUT [STOP_FILE]\n");
        return 2;
    }
    std::vector<Sample> flat;
    std::vector<int> ds_begin, ds_end;
    {
        std::stringstream paths(argv[1]);
        std::string path;
        while (std::getline(paths, path, ',')) {
            std::ifstream f(path);
            Sample s;
            int b = (int)flat.size();
            while (f >> s.x >> s.y >> s.label) flat.push_back(s);
            if ((int)flat.size() == b) { std::fprintf(stderr, "No terrain samples\n"); return 2; }
            ds_begin.push_back(b);
            ds_end.push_back((int)flat.size());
        }
    }
    const uint64_t start = strtoull(argv[2], nullptr, 10);
    const uint64_t count = strtoull(argv[3], nullptr, 10);
    if (start + count > (1ull << 32)) { std::fprintf(stderr, "Range outside uint32\n"); return 2; }
    const int ndatasets = (int)ds_begin.size();

    int block = 128, chunk_mb = 64;
    if (const char* e = std::getenv("CUDA_BLOCK")) block = atoi(e);
    if (const char* e = std::getenv("CUDA_CHUNK")) chunk_mb = atoi(e);
    const uint64_t chunk = (uint64_t)chunk_mb << 20;
    const unsigned int MAX_HITS = 1u << 22;

    uint32_t initial[624];
    build_initial(initial);
    CUDA_OK(cudaMemcpyToSymbol(c_initial, initial, sizeof(initial)));

    Sample* d_samples = nullptr;
    int *d_begin = nullptr, *d_end = nullptr;
    uint2* d_hits = nullptr;
    unsigned int *d_nhits = nullptr, *d_noverflow = nullptr;
    CUDA_OK(cudaMalloc(&d_samples, flat.size() * sizeof(Sample)));
    CUDA_OK(cudaMemcpy(d_samples, flat.data(), flat.size() * sizeof(Sample), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_begin, ndatasets * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_end, ndatasets * sizeof(int)));
    CUDA_OK(cudaMemcpy(d_begin, ds_begin.data(), ndatasets * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_end, ds_end.data(), ndatasets * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_hits, (size_t)MAX_HITS * sizeof(uint2)));
    CUDA_OK(cudaMalloc(&d_nhits, sizeof(unsigned int)));
    CUDA_OK(cudaMalloc(&d_noverflow, sizeof(unsigned int)));
    CUDA_OK(cudaMemset(d_noverflow, 0, sizeof(unsigned int)));

    std::ofstream output(argv[5]);
    std::vector<uint2> host_hits(MAX_HITS);
    std::vector<uint32_t> overflow_seeds;   // window too short: unresolved, re-check on CPU
    uint64_t tested = 0, total_hits = 0;
    bool truncated = false;
    auto begin = std::chrono::steady_clock::now();

    for (uint64_t off = 0; off < count; off += chunk) {
        if (argc == 7) { std::ifstream stop(argv[6]); if (stop.good()) break; }
        const uint64_t n = (off + chunk < count) ? chunk : (count - off);
        CUDA_OK(cudaMemset(d_nhits, 0, sizeof(unsigned int)));
        const uint64_t grid = (n + block - 1) / block;
        scan_kernel<<<(unsigned int)grid, block>>>(start + off, n, d_samples, d_begin, d_end,
                                                   ndatasets, d_hits, d_nhits, MAX_HITS,
                                                   d_noverflow);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());

        unsigned int h = 0;
        CUDA_OK(cudaMemcpy(&h, d_nhits, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        const unsigned int keep = h > MAX_HITS ? MAX_HITS : h;
        if (h > MAX_HITS) truncated = true;
        if (keep) {
            CUDA_OK(cudaMemcpy(host_hits.data(), d_hits, (size_t)keep * sizeof(uint2),
                               cudaMemcpyDeviceToHost));
            for (unsigned int i = 0; i < keep; i++) {
                // dataset id 0xFFFFFFFF marks a window overflow: unresolved, not a hit
                if (host_hits[i].x == 0xFFFFFFFFu) { overflow_seeds.push_back(host_hits[i].y); continue; }
                if (ndatasets > 1) output << host_hits[i].x << ' ';
                output << host_hits[i].y << '\n';
            }
            output.flush();
        }
        tested += n;
        total_hits += h;
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::printf("{\"progress\":true,\"tested\":%llu,\"hits\":%llu,\"seconds\":%g}\n",
                    (unsigned long long)tested, (unsigned long long)total_hits, el);
        std::fflush(stdout);
    }

    unsigned int overflow = 0;
    CUDA_OK(cudaMemcpy(&overflow, d_noverflow, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (!overflow_seeds.empty()) {
        std::ofstream ov(std::string(argv[5]) + ".overflow");
        for (uint32_t s : overflow_seeds) ov << s << '\n';
    }
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::printf("{\"tested\":%llu,\"hits\":%llu,\"seconds\":%g,\"seeds_per_second\":%g,"
                "\"window_overflow\":%u,\"truncated\":%s,\"device\":\"cuda\"}\n",
                (unsigned long long)tested, (unsigned long long)total_hits, el,
                (double)tested / el, overflow, truncated ? "true" : "false",
                overflow_seeds.empty() ? "" : (std::string(argv[5]) + ".overflow").c_str());
    cudaFree(d_samples); cudaFree(d_begin); cudaFree(d_end);
    cudaFree(d_hits); cudaFree(d_nhits); cudaFree(d_noverflow);
    return 0;
}
