// Eight independent CPython seed initializations per AVX2 instruction.
// Same integer operations as PyRandom. No changes to the seed domain or filter.
#include <immintrin.h>
#include <array>
struct SeedBatch8 {
    alignas(32) uint32_t words[624][8];
    void init(uint32_t seed) {
        static const std::array<uint32_t,624> base=[] {
            PyRandom r;r.init_genrand(19650218U);std::array<uint32_t,624> b;
            std::copy(r.mt,r.mt+624,b.begin());return b;
        }();
        __m256i mt[624];
        for(int i=0;i<624;i++)mt[i]=_mm256_set1_epi32(base[i]);
        auto keys=_mm256_setr_epi32(seed,seed+1,seed+2,seed+3,seed+4,seed+5,seed+6,seed+7);
        auto mul1=_mm256_set1_epi32(1664525U),mul2=_mm256_set1_epi32(1566083941U);
        int i=1;
        for(int k=624;k;k--){
            auto prev=_mm256_xor_si256(mt[i-1],_mm256_srli_epi32(mt[i-1],30));
            mt[i]=_mm256_add_epi32(_mm256_xor_si256(mt[i],_mm256_mullo_epi32(prev,mul1)),keys);
            if(++i>=624){mt[0]=mt[623];i=1;}
        }
        for(int k=623;k;k--){
            auto prev=_mm256_xor_si256(mt[i-1],_mm256_srli_epi32(mt[i-1],30));
            mt[i]=_mm256_sub_epi32(_mm256_xor_si256(mt[i],_mm256_mullo_epi32(prev,mul2)),_mm256_set1_epi32(i));
            if(++i>=624){mt[0]=mt[623];i=1;}
        }
        mt[0]=_mm256_set1_epi32(0x80000000U);
        auto upper=_mm256_set1_epi32(0x80000000U),lower=_mm256_set1_epi32(0x7fffffffU);
        auto one=_mm256_set1_epi32(1),matrix=_mm256_set1_epi32(0x9908b0dfU),zero=_mm256_setzero_si256();
        for(int j=0;j<624;j++){
            auto y=_mm256_or_si256(_mm256_and_si256(mt[j],upper),_mm256_and_si256(mt[(j+1)%624],lower));
            auto odd=_mm256_sub_epi32(zero,_mm256_and_si256(y,one));
            mt[j]=_mm256_xor_si256(mt[(j+397)%624],_mm256_xor_si256(_mm256_srli_epi32(y,1),_mm256_and_si256(odd,matrix)));
            // Do not store until all twists finish: later iterations use earlier mt.
        }
        for(int j=0;j<624;j++)_mm256_store_si256(reinterpret_cast<__m256i*>(words[j]),mt[j]);
    }
    struct Lane {
        const SeedBatch8& batch;int lane,index=0;
        uint32_t getrandbits(int k){
            if(index==624)throw std::overflow_error("unusually long rejection sequence");
            uint32_t y=batch.words[index++][lane];
            y^=y>>11;y^=(y<<7)&0x9d2c5680U;y^=(y<<15)&0xefc60000U;y^=y>>18;
            return y>>(32-k);
        }
        int64_t randbelow(int64_t n){
            int k=0;for(auto m=n;m;m>>=1)k++;
            auto value=getrandbits(k);while(value>=n)value=getrandbits(k);return value;
        }
    };
};
