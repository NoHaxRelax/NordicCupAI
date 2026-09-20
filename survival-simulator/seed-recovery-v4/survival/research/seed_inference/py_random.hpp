// Extracted without changes from survival/fastsim/_engine.cpp for the seed-filter experiment.
#include <cstdint>
#include <vector>
struct PyRandom {
    uint32_t mt[624];
    int mti = 625;

    void init_genrand(uint32_t s) {
        mt[0] = s;
        for (mti = 1; mti < 624; mti++)
            mt[mti] = (1812433253U * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + (uint32_t)mti);
    }
    void init_by_array(const std::vector<uint32_t>& key) {
        size_t len = key.size();
        init_genrand(19650218U);
        size_t i = 1, j = 0, k = (624 > len ? 624 : len);
        for (; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) + key[j] + (uint32_t)j;
            i++; j++;
            if (i >= 624) { mt[0] = mt[623]; i = 1; }
            if (j >= len) j = 0;
        }
        for (k = 623; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - (uint32_t)i;
            i++;
            if (i >= 624) { mt[0] = mt[623]; i = 1; }
        }
        mt[0] = 0x80000000U;
        mti = 624;
    }
    uint32_t genrand() {
        static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
        uint32_t y;
        if (mti >= 624) {
            int kk;
            for (kk = 0; kk < 624 - 397; kk++) {
                y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
                mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            for (; kk < 623; kk++) {
                y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
                mt[kk] = mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            y = (mt[623] & 0x80000000U) | (mt[0] & 0x7fffffffU);
            mt[623] = mt[396] ^ (y >> 1) ^ mag01[y & 0x1U];
            mti = 0;
        }
        y = mt[mti++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }
    double random() {
        uint32_t a = genrand() >> 5, b = genrand() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }
    double uniform(double a, double b) { return a + (b - a) * random(); }
    uint32_t getrandbits(int k) { return genrand() >> (32 - k); }  // 1 <= k <= 32
    int64_t randbelow(int64_t n) {
        int k = 0;
        for (int64_t m = n; m; m >>= 1) k++;
        uint32_t r = getrandbits(k);
        while ((int64_t)r >= n) r = getrandbits(k);
        return r;
    }
    int64_t randint(int64_t a, int64_t b) { return a + randbelow(b - a + 1); }
};
