// Exact terrain-prefix RNG: interleave independent AVX2 chains to hide multiply
// latency, and skip MT state/output work never used by the terrain filter.
// Long rejection sequences are retried with the full scalar generator by caller.
#include <immintrin.h>
#include <array>
#ifndef SEED_BATCH_LANES
#define SEED_BATCH_LANES 16
#endif
struct SeedBatchFast {
    static constexpr int lanes=SEED_BATCH_LANES, groups=lanes/8, prefix=96;
    static_assert(lanes>=8 && lanes%8==0);
    static_assert(prefix<=227, "Longer prefixes need in-place twist feedback");
    alignas(32) uint32_t words[prefix][lanes];
    void init(uint32_t seed) {
        static const std::array<uint32_t,624> base=[] {
            PyRandom r;r.init_genrand(19650218U);std::array<uint32_t,624> a;
            std::copy(r.mt,r.mt+624,a.begin());return a;
        }();
        __m256i mt[624][groups],keys[groups];
        for(int g=0;g<groups;g++)keys[g]=_mm256_setr_epi32(seed+8*g,seed+8*g+1,seed+8*g+2,seed+8*g+3,seed+8*g+4,seed+8*g+5,seed+8*g+6,seed+8*g+7);
        for(int i=0;i<624;i++)for(int g=0;g<groups;g++)mt[i][g]=_mm256_set1_epi32(base[i]);
        auto mul1=_mm256_set1_epi32(1664525U),mul2=_mm256_set1_epi32(1566083941U);
        int i=1;
        for(int k=624;k;k--){
            for(int g=0;g<groups;g++){
                auto prev=_mm256_xor_si256(mt[i-1][g],_mm256_srli_epi32(mt[i-1][g],30));
                mt[i][g]=_mm256_add_epi32(_mm256_xor_si256(mt[i][g],_mm256_mullo_epi32(prev,mul1)),keys[g]);
            }
            if(++i>=624){for(int g=0;g<groups;g++)mt[0][g]=mt[623][g];i=1;}
        }
        // Complete seed mixing: its wraparound updates word 1 last, and that
        // update affects even the first output. Only output generation is cut.
        for(int k=623;k;k--){
            for(int g=0;g<groups;g++){
                auto prev=_mm256_xor_si256(mt[i-1][g],_mm256_srli_epi32(mt[i-1][g],30));
                mt[i][g]=_mm256_sub_epi32(_mm256_xor_si256(mt[i][g],_mm256_mullo_epi32(prev,mul2)),_mm256_set1_epi32(i));
            }
            if(++i>=624){for(int g=0;g<groups;g++)mt[0][g]=mt[623][g];i=1;}
        }
        auto upper=_mm256_set1_epi32(0x80000000U),lower=_mm256_set1_epi32(0x7fffffffU);
        auto one=_mm256_set1_epi32(1),matrix=_mm256_set1_epi32(0x9908b0dfU),zero=_mm256_setzero_si256();
        for(int g=0;g<groups;g++)mt[0][g]=upper;
        for(int j=0;j<prefix;j++)for(int g=0;g<groups;g++){
            auto y=_mm256_or_si256(_mm256_and_si256(mt[j][g],upper),_mm256_and_si256(mt[j+1][g],lower));
            auto odd=_mm256_sub_epi32(zero,_mm256_and_si256(y,one));
            y=_mm256_xor_si256(mt[j+397][g],_mm256_xor_si256(_mm256_srli_epi32(y,1),_mm256_and_si256(odd,matrix)));
            y=_mm256_xor_si256(y,_mm256_srli_epi32(y,11));
            y=_mm256_xor_si256(y,_mm256_and_si256(_mm256_slli_epi32(y,7),_mm256_set1_epi32(0x9d2c5680U)));
            y=_mm256_xor_si256(y,_mm256_and_si256(_mm256_slli_epi32(y,15),_mm256_set1_epi32(0xefc60000U)));
            y=_mm256_xor_si256(y,_mm256_srli_epi32(y,18));
            _mm256_store_si256(reinterpret_cast<__m256i*>(&words[j][8*g]),y);
        }
    }
    struct Lane {
        const SeedBatchFast& batch;int lane,index=0;
        uint32_t getrandbits(int k){if(index==prefix)throw std::overflow_error("terrain prefix exhausted");return batch.words[index++][lane]>>(32-k);}
        int64_t randbelow(int64_t n){int k=0;for(auto m=n;m;m>>=1)k++;auto v=getrandbits(k);while(v>=n)v=getrandbits(k);return v;}
    };
};
