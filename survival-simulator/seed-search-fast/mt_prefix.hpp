// Seed-prefix kernels: baseline (V4 terrain_filter shape) vs optimized.
// Both produce the 10 Voronoi site coords + 10 biome types the terrain filter needs.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

// ---------------- exact scalar CPython reference (ground truth + rare fallback) -------------
struct PyRandom {
    uint32_t mt[624]; int mti = 625;
    void init_genrand(uint32_t s){ mt[0]=s; for(mti=1;mti<624;mti++) mt[mti]=(1812433253U*(mt[mti-1]^(mt[mti-1]>>30))+(uint32_t)mti); }
    // CPython's general form. random.Random(int) splits |n| into little-endian 32-bit
    // words; a seed below 2^32 is the one-word case. Copied from fastsim/_engine.cpp.
    void init_by_array(const std::vector<uint32_t>& key){
        size_t len=key.size(); init_genrand(19650218U);
        size_t i=1,j=0,k=(624>len?624:len);
        for(;k;k--){ mt[i]=(mt[i]^((mt[i-1]^(mt[i-1]>>30))*1664525U))+key[j]+(uint32_t)j; i++;j++;
            if(i>=624){mt[0]=mt[623];i=1;} if(j>=len)j=0; }
        for(k=623;k;k--){ mt[i]=(mt[i]^((mt[i-1]^(mt[i-1]>>30))*1566083941U))-(uint32_t)i; i++;
            if(i>=624){mt[0]=mt[623];i=1;} }
        mt[0]=0x80000000U; mti=624;
    }
    void init_by_array(uint32_t seed){ init_by_array(std::vector<uint32_t>{seed}); }
    uint32_t genrand(){ static const uint32_t mag01[2]={0x0U,0x9908b0dfU}; uint32_t y;
        if(mti>=624){ int kk;
            for(kk=0;kk<624-397;kk++){ y=(mt[kk]&0x80000000U)|(mt[kk+1]&0x7fffffffU); mt[kk]=mt[kk+397]^(y>>1)^mag01[y&1U]; }
            for(;kk<623;kk++){ y=(mt[kk]&0x80000000U)|(mt[kk+1]&0x7fffffffU); mt[kk]=mt[kk+(397-624)]^(y>>1)^mag01[y&1U]; }
            y=(mt[623]&0x80000000U)|(mt[0]&0x7fffffffU); mt[623]=mt[396]^(y>>1)^mag01[y&1U]; mti=0; }
        y=mt[mti++]; y^=(y>>11); y^=(y<<7)&0x9d2c5680U; y^=(y<<15)&0xefc60000U; y^=(y>>18); return y; }
};

struct Prefix { int xs[10], ys[10], types[10]; };

inline void reference_prefix(const std::vector<uint32_t>& key, Prefix& p){
    PyRandom r; r.init_by_array(key);
    auto below=[&](int limit,int bits){ uint32_t v; do{ v=r.genrand()>>(32-bits); }while(v>=(uint32_t)limit); return (int)v; };
    for(int i=0;i<10;i++){ p.xs[i]=below(1600,11); p.ys[i]=below(1200,11); }
    for(int i=0;i<10;i++) p.types[i]=below(4,3);
}
inline void reference_prefix(uint32_t seed, Prefix& p){ reference_prefix(std::vector<uint32_t>{seed}, p); }

extern uint32_t g_initial[624];
inline void build_initial(){ PyRandom b; b.init_genrand(19650218U); std::memcpy(g_initial,b.mt,sizeof(g_initial)); }

#ifndef LANES
#define LANES 16
#endif
#ifndef GROUPS
#define GROUPS 4
#endif

// Alignment: native everywhere. On Windows this REQUIRES -mincoming-stack-boundary=4
// (see the build lines in README.md). MinGW's thread entry only guarantees 16-byte stack
// alignment, but GCC assumes 32 and emits `vmovdqa` to 16-byte-aligned slots, which
// faults in worker threads while the main thread happens to survive. Verified under gdb:
// `vmovdqa %ymm0,0x70(%rsp)` with rsp 32-aligned, so the slot itself was 16-aligned.
// The flag tells GCC the truth so it emits a realignment prologue where it needs one.
typedef uint32_t U __attribute__((vector_size(LANES*sizeof(uint32_t))));

static inline U splat(uint32_t v){ U r; for(int i=0;i<LANES;i++) r[i]=v; return r; }
static inline U temper(U y){ y^=y>>11; y^=(y<<7)&splat(0x9d2c5680U); y^=(y<<15)&splat(0xefc60000U); y^=y>>18; return y; }

// The V4 kernel, kept for A/B regression only. Define SEEDCRACK_BASELINE to compile it;
// the TU must then also define baseline::vector_initial.
#ifdef SEEDCRACK_BASELINE
namespace baseline {
extern U vector_initial[624];
inline void setup(){ for(int i=0;i<624;i++) vector_initial[i]=splat(g_initial[i]); }

struct SeedGroup {
    U mt[624]; uint32_t cache[256][LANES]; uint32_t seed; int generated;

    void init(uint32_t first){
        seed=first; generated=0; std::memcpy(mt,vector_initial,sizeof(mt));
        U keys; for(int lane=0;lane<LANES;lane++) keys[lane]=first+lane;
        int i=1;
        for(int k=0;k<624;k++){ mt[i]=(mt[i]^((mt[i-1]^(mt[i-1]>>30))*splat(1664525U)))+keys;
            if(++i>=624){mt[0]=mt[623];i=1;} }
        for(int k=0;k<623;k++){ mt[i]=(mt[i]^((mt[i-1]^(mt[i-1]>>30))*splat(1566083941U)))-splat((uint32_t)i);
            if(++i>=624){mt[0]=mt[623];i=1;} }
        mt[0]=splat(0x80000000U);
    }
    uint32_t word(int lane,int index){
        if(index>=256){ PyRandom rng; rng.init_by_array(seed+lane); uint32_t v=0;
            for(int i=0;i<=index;i++) v=rng.genrand(); return v; }
        while(generated<=index){ int i=generated;
            U y=(mt[i]&splat(0x80000000U))|(mt[i+1]&splat(0x7fffffffU));
            y=mt[(i+397)%624]^(y>>1)^((splat(0U)-(y&splat(1U)))&splat(0x9908b0dfU));
            mt[i]=y; y=temper(y);
            std::memcpy(cache[i],&y,sizeof(y)); generated++; }
        return cache[index][lane];
    }
    void generate(int lane, Prefix& p){
        int pos=0;
        auto below=[&](int limit,int bits){ uint32_t r; do{ r=word(lane,pos++)>>(32-bits); }while(r>=(uint32_t)limit); return (int)r; };
        for(int i=0;i<10;i++){ p.xs[i]=below(1600,11); p.ys[i]=below(1200,11); }
        for(int i=0;i<10;i++) p.types[i]=below(4,3);
    }
};
} // namespace baseline
#endif

// ================= FAST: register-chain seeding, windowed output, G-way interleave =========
// 1. initial[] is read as scalars (2.5 KB, L1-resident) and broadcast, not a 40 KB vector array.
// 2. The 624-word seeded state is never materialised. Pass A carries the loop-1 chain in a
//    register to obtain mt[1]_final; pass B re-runs loop 1 fused with loop 2 so the two
//    dependency chains overlap for free, storing only the 129 low words the twist needs.
// 3. Output words 0..WINDOW-1 are twisted+tempered on the fly (>=99.99995% of seeds need <=96).
// 4. GROUPS independent chains interleave to hide the ~8-cycle per-step dependency latency.
namespace fast {
constexpr int WINDOW = 128;
constexpr int LOW    = WINDOW + 1;   // mt[0..128] needed by the twist for words 0..127

// Which CPython key the scan sweeps. Word 0 varies per lane; words 1.. are fixed.
// The verifier (fastsim/_engine.cpp) already handles multi-word keys; this made the
// scanner match it, so an evaluation seed above 2^32 becomes a bounded window to scan
// rather than an invalidated claim.
//
// In loop 1 the addend at iteration kk is key[kk % K] + (kk % K). With K == 1 that is the
// seed itself every step, which is the fast path below. For K > 1 the table decides per
// step whether the addend is the per-lane word 0 or a broadcast constant.
struct KeyPlan {
    std::vector<uint32_t> high;   // words 1..K-1
    uint32_t addc[624];           // broadcast addend when !lane_slot[kk]
    bool lane_slot[624];          // true when iteration kk adds word 0 (+0)
    explicit KeyPlan(const std::vector<uint32_t>& high_words = {}) : high(high_words) {
        const int K = words();
        for (int kk = 0; kk < 624; kk++) {
            const int j = kk % K;
            lane_slot[kk] = (j == 0);
            addc[kk] = j ? high[j - 1] + (uint32_t)j : 0u;
        }
    }
    int words() const { return 1 + (int)high.size(); }
    std::vector<uint32_t> key_for(uint32_t w0) const {
        std::vector<uint32_t> k; k.reserve(words()); k.push_back(w0);
        k.insert(k.end(), high.begin(), high.end()); return k;
    }
};

struct Batch {
    uint32_t out[GROUPS][WINDOW][LANES];
    uint32_t base[GROUPS];

    // ONE-WORD HOT PATH. Deliberately a plain non-template method with the loops written
    // out: an earlier refactor that routed both key widths through a function template
    // taking a lambda made GCC spill `ymm` registers to the stack, and MinGW's thread
    // entry only guarantees 16-byte stack alignment, so `vmovdqa %ymm0,0x70(%rsp)` faulted
    // in worker threads non-deterministically (confirmed under gdb; no -m*stack* flag
    // fixed it reliably). Keeping this body register-resident is what makes it correct as
    // well as fast, so do not "tidy" it into a shared template.
    inline void run(uint64_t first_seed){
        U p1[GROUPS], p2[GROUPS], keys[GROUPS];
        U a1_1_first[GROUPS], a1_1_final[GROUPS];
        static thread_local U low[GROUPS][LOW];
        const U C1 = splat(1664525U), C2 = splat(1566083941U);
        const U TOP = splat(0x80000000U), BOT = splat(0x7fffffffU), MAG = splat(0x9908b0dfU), ONE = splat(1U);

        for(int g=0; g<GROUPS; ++g){
            base[g] = uint32_t(first_seed + (uint64_t)g*LANES);
            U k; for(int l=0;l<LANES;l++) k[l] = base[g] + l;
            keys[g] = k;
        }
        // ---- pass A: loop-1 chain only, to obtain mt[1] after the first seeding loop ----
        for(int g=0; g<GROUPS; ++g){
            U prev = splat(g_initial[0]);
            prev = (splat(g_initial[1]) ^ ((prev^(prev>>30))*C1)) + keys[g];
            a1_1_first[g] = prev; p1[g] = prev;
        }
        for(int i=2;i<624;i++){
            U ci = splat(g_initial[i]);
            for(int g=0; g<GROUPS; ++g) p1[g] = (ci ^ ((p1[g]^(p1[g]>>30))*C1)) + keys[g];
        }
        for(int g=0; g<GROUPS; ++g){
            U a1_0 = p1[g];                                   // loop-1 wrap: mt[0] = mt[623]
            a1_1_final[g] = (a1_1_first[g] ^ ((a1_0^(a1_0>>30))*C1)) + keys[g];
            p1[g] = a1_1_first[g];
            p2[g] = a1_1_final[g];
        }
        // ---- pass B: loop-1 recompute fused with loop-2 (two chains, one latency) ----
        for(int i=2;i<LOW;i++){
            U ci = splat(g_initial[i]); U si = splat((uint32_t)i);
            for(int g=0; g<GROUPS; ++g){
                p1[g] = (ci ^ ((p1[g]^(p1[g]>>30))*C1)) + keys[g];
                p2[g] = (p1[g] ^ ((p2[g]^(p2[g]>>30))*C2)) - si;
                low[g][i] = p2[g];
            }
        }
        // words 0 and 1 depend on mt[1], which loop 2 writes last, so hold A2[397..398] back
        U h0[GROUPS], h1[GROUPS];
        for(int i=LOW;i<624;i++){
            U ci = splat(g_initial[i]); U si = splat((uint32_t)i);
            const int k = i - 397;
            for(int g=0; g<GROUPS; ++g){
                p1[g] = (ci ^ ((p1[g]^(p1[g]>>30))*C1)) + keys[g];
                p2[g] = (p1[g] ^ ((p2[g]^(p2[g]>>30))*C2)) - si;
                if(i == 397)      h0[g] = p2[g];
                else if(i == 398) h1[g] = p2[g];
                else if(k >= 2 && k < WINDOW){
                    U y = (low[g][k]&TOP)|(low[g][k+1]&BOT);
                    U w = temper(p2[g] ^ (y>>1) ^ ((splat(0U)-(y&ONE))&MAG));
                    std::memcpy(out[g][k], &w, sizeof(w));
                }
            }
        }
        // ---- close loop 2, then finish words 0 and 1 ----
        for(int g=0; g<GROUPS; ++g){
            U a2_0 = p2[g];
            U a2_1 = (a1_1_final[g] ^ ((a2_0^(a2_0>>30))*C2)) - ONE;
            low[g][0] = TOP;          // post-seeding override mt[0] = 0x80000000
            low[g][1] = a2_1;
            U y = (low[g][0]&TOP)|(low[g][1]&BOT);
            U w = temper(h0[g] ^ (y>>1) ^ ((splat(0U)-(y&ONE))&MAG));
            std::memcpy(out[g][0], &w, sizeof(w));
            y = (low[g][1]&TOP)|(low[g][2]&BOT);
            w = temper(h1[g] ^ (y>>1) ^ ((splat(0U)-(y&ONE))&MAG));
            std::memcpy(out[g][1], &w, sizeof(w));
        }
    }

    inline bool generate(int g,int lane, Prefix& p) const {
        int pos=0; bool ok=true;
        auto below=[&](int limit,int bits)->int{ uint32_t r;
            do{ if(pos>=WINDOW){ ok=false; return 0; } r=out[g][pos++][lane]>>(32-bits); }while(r>=(uint32_t)limit);
            return (int)r; };
        for(int i=0;i<10;i++){ p.xs[i]=below(1600,11); if(!ok)return false; p.ys[i]=below(1200,11); if(!ok)return false; }
        for(int i=0;i<10;i++){ p.types[i]=below(4,3); if(!ok)return false; }
        return true;
    }
};
} // namespace fast
