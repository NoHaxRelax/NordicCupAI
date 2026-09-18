//! Bit-exact port of CPython's `random.Random` (Mersenne Twister MT19937 plus the Python-level
//! algorithms for random(), uniform(), getrandbits(), _randbelow(), randint() and choice()).
//! Reference: CPython Modules/_randommodule.c and Lib/random.py (3.12).

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

#[derive(Clone)]
pub struct PyRandom {
    mt: [u32; N],
    mti: usize,
}

impl PyRandom {
    /// `random.Random(seed)` for a non-negative integer seed.
    pub fn new(seed: u64) -> Self {
        let mut r = PyRandom { mt: [0; N], mti: N + 1 };
        // CPython: key = abs(seed) as little-endian 32-bit words, at least one word
        let mut key: Vec<u32> = Vec::new();
        let mut s = seed;
        if s == 0 { key.push(0); }
        while s > 0 { key.push((s & 0xffff_ffff) as u32); s >>= 32; }
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1_812_433_253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        self.mti = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        let klen = key.len();
        let mut k = if N > klen { N } else { klen };
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1_664_525)))
                .wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1; j += 1;
            if i >= N { self.mt[0] = self.mt[N - 1]; i = 1; }
            if j >= klen { j = 0; }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1_566_083_941))).wrapping_sub(i as u32);
            i += 1;
            if i >= N { self.mt[0] = self.mt[N - 1]; i = 1; }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
    }

    pub fn genrand_u32(&mut self) -> u32 {
        if self.mti >= N {
            let mag01 = [0u32, MATRIX_A];
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// random(): 53-bit float in [0, 1)
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// uniform(a, b) = a + (b-a) * random()
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.random() }

    /// getrandbits(k) for 0 < k <= 64
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        if k <= 32 { return (self.genrand_u32() >> (32 - k)) as u64; }
        let words = ((k - 1) / 32 + 1) as usize;
        let mut out: u64 = 0; let mut kk = k;
        for i in 0..words {
            let mut r = self.genrand_u32();
            if kk < 32 { r >>= 32 - kk; }
            out |= (r as u64) << (32 * i);
            kk = kk.wrapping_sub(32);
        }
        out
    }

    /// _randbelow_with_getrandbits(n): uniform integer in [0, n)
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 { return 0; }
        let k = 64 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n { return r; }
        }
    }

    /// randint(a, b) = randrange(a, b+1)
    pub fn randint(&mut self, a: i64, b: i64) -> i64 { a + self.randbelow((b + 1 - a) as u64) as i64 }

    /// choice(seq) -> index into a sequence of length n
    pub fn choice_index(&mut self, n: usize) -> usize { self.randbelow(n as u64) as usize }
}
