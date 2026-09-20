const double PI = 3.141592653589793;
const double TWO_PI = 2.0 * 3.141592653589793;
const double INF = std::numeric_limits<double>::infinity();

// ----------------------------------------------------------------------------
// CPython random.Random (MT19937)
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// Python / numpy float semantics
// ----------------------------------------------------------------------------
inline double py_mod(double a, double b) {  // float % and np.remainder
    // For |a| < b and a != 0 fmod(a, b) == a exactly, so these match the general path.
    if (b > 0) {
        if (a > 0 && a < b) return a;
        if (a < 0 && a > -b) return a + b;
    }
    double mod = std::fmod(a, b);
    if (!b) return mod;
    if (mod) {
        if ((b < 0) != (mod < 0)) mod += b;
    } else {
        mod = std::copysign(0.0, b);
    }
    return mod;
}
inline double wrap_angle(double a) { return py_mod(a + PI, TWO_PI) - PI; }  // (a + pi) % 2pi - pi

inline double py_floordiv(double vx, double wx) {  // float // and np.floor_divide
    double mod = std::fmod(vx, wx);
    double div = (vx - mod) / wx;
    if (mod) {
        if ((wx < 0) != (mod < 0)) { mod += wx; div -= 1.0; }
    }
    double floordiv;
    if (div) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, vx / wx);
    }
    return floordiv;
}

// CPython hash of a float (equals the int hash for integral values).
inline int64_t py_hash_double(double v) {
    const int BITS = 61;
    const uint64_t MOD = (((uint64_t)1) << BITS) - 1;
    if (!std::isfinite(v)) return v > 0 ? 314159 : -314159;
    int e;
    double m = std::frexp(v, &e);
    int sign = 1;
    if (m < 0) { sign = -1; m = -m; }
    uint64_t x = 0, y;
    while (m) {
        x = ((x << 28) & MOD) | x >> (BITS - 28);
        m *= 268435456.0;
        e -= 28;
        y = (uint64_t)m;
        m -= y;
        x += y;
        if (x >= MOD) x -= MOD;
    }
    e = e >= 0 ? e % BITS : BITS - 1 - ((-1 - e) % BITS);
    x = ((x << e) & MOD) | x >> (BITS - e);
    x = x * (uint64_t)(int64_t)sign;
    if (x == (uint64_t)-1) x = (uint64_t)-2;
    return (int64_t)x;
}
inline int64_t py_hash_int(int64_t v) {  // non-negative small ints only
    const uint64_t MOD = (((uint64_t)1) << 61) - 1;
    int64_t h = (int64_t)((uint64_t)v % MOD);
    return h == -1 ? -2 : h;
}
inline int64_t py_tuple_hash(const int64_t* lanes, size_t len) {
    const uint64_t P1 = 11400714785074694791ULL, P2 = 14029467366897019727ULL, P5 = 2870177450012600261ULL;
    uint64_t acc = P5;
    for (size_t i = 0; i < len; i++) {
        acc += (uint64_t)lanes[i] * P2;
        acc = (acc << 31) | (acc >> 33);
        acc *= P1;
    }
    acc += len ^ (P5 ^ 3527539UL);
    if (acc == (uint64_t)-1) return 1546275796;
    return (int64_t)acc;
}
inline int64_t edge_hash(double x1, double y1, double x2, double y2) {
    int64_t a[2] = {py_hash_double(x1), py_hash_double(y1)};
    int64_t b[2] = {py_hash_double(x2), py_hash_double(y2)};
    int64_t t[2] = {py_tuple_hash(a, 2), py_tuple_hash(b, 2)};
    return py_tuple_hash(t, 2);
}

// ----------------------------------------------------------------------------
// CPython 3.12 set table (Objects/setobject.c), keys are small integer ids.
// Two keys are equal iff their ids are equal.
// ----------------------------------------------------------------------------
struct PySetEmu {
    struct Entry { int64_t hash; int32_t key; };  // key -1 = NULL, -2 = dummy
    static const int LINEAR_PROBES = 9;
    static const int PERTURB_SHIFT = 5;
    std::vector<Entry> table;
    size_t mask = 7;
    int64_t fill = 0, used = 0;

    PySetEmu() { clear(); }
    void clear() {
        table.assign(8, Entry{0, -1});
        mask = 7; fill = 0; used = 0;
    }
    static void insert_clean(std::vector<Entry>& tab, size_t mask, int32_t key, int64_t hash) {
        size_t perturb = (size_t)hash;
        size_t i = (size_t)hash & mask;
        while (true) {
            Entry* e = &tab[i];
            if (e->key == -1) { e->key = key; e->hash = hash; return; }
            if (i + LINEAR_PROBES <= mask) {
                for (int j = 0; j < LINEAR_PROBES; j++) {
                    e++;
                    if (e->key == -1) { e->key = key; e->hash = hash; return; }
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }
    void resize(int64_t minused) {
        size_t newsize = 8;
        while (newsize <= (size_t)minused) newsize <<= 1;
        if (newsize == 8 && mask == 7 && fill == used) return;
        std::vector<Entry> old;
        old.swap(table);
        table.assign(newsize, Entry{0, -1});
        mask = newsize - 1;
        if (fill != used) fill = used;
        for (const Entry& e : old)
            if (e.key >= 0) insert_clean(table, mask, e.key, e.hash);
    }
    void add(int32_t key, int64_t hash) {
        size_t m = mask;
        size_t i = (size_t)hash & m;
        Entry* freeslot = nullptr;
        size_t perturb = (size_t)hash;
        Entry* entry;
        while (true) {
            entry = &table[i];
            int probes = (i + LINEAR_PROBES <= m) ? LINEAR_PROBES : 0;
            do {
                if (entry->hash == 0 && entry->key == -1) goto found_unused_or_dummy;
                if (entry->hash == hash) {
                    if (entry->key == key) return;
                } else if (entry->hash == -1) {
                    freeslot = entry;
                }
                entry++;
            } while (probes--);
            perturb >>= PERTURB_SHIFT;
            i = (i * 5 + 1 + perturb) & m;
        }
    found_unused_or_dummy:
        if (freeslot) {
            used++;
            freeslot->key = key; freeslot->hash = hash;
            return;
        }
        fill++; used++;
        entry->key = key; entry->hash = hash;
        if ((size_t)fill * 5 < m * 3) return;
        resize(used > 50000 ? used * 2 : used * 4);
    }
    long lookup(int32_t key, int64_t hash) const {
        size_t perturb = (size_t)hash;
        size_t i = (size_t)hash & mask;
        while (true) {
            size_t idx = i;
            int probes = (i + LINEAR_PROBES <= mask) ? LINEAR_PROBES : 0;
            do {
                const Entry& e = table[idx];
                if (e.hash == 0 && e.key == -1) return -1;
                if (e.hash == hash && e.key == key) return (long)idx;
                idx++;
            } while (probes--);
            perturb >>= PERTURB_SHIFT;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }
    bool discard(int32_t key, int64_t hash) {
        long idx = lookup(key, hash);
        if (idx < 0) return false;
        table[idx].key = -2; table[idx].hash = -1;
        used--;
        return true;
    }
    void merge(const PySetEmu& other) {
        if (&other == this || other.used == 0) return;
        if ((fill + other.used) * 5 >= (int64_t)mask * 3) resize((used + other.used) * 2);
        if (fill == 0 && mask == other.mask && other.fill == other.used) {
            for (size_t i = 0; i <= other.mask; i++)
                if (other.table[i].key != -1) table[i] = other.table[i];
            fill = other.fill; used = other.used;
            return;
        }
        if (fill == 0) {
            fill = other.used; used = other.used;
            for (size_t i = 0; i <= other.mask; i++) {
                const Entry& e = other.table[i];
                if (e.key >= 0) insert_clean(table, mask, e.key, e.hash);
            }
            return;
        }
        for (size_t i = 0; i <= other.mask; i++) {
            const Entry e = other.table[i];
            if (e.key >= 0) add(e.key, e.hash);
        }
    }
    template <class F> void for_each(F f) const {
        for (const Entry& e : table)
            if (e.key >= 0) f(e.key);
    }
    void to_vector(std::vector<int32_t>& out) const {
        out.clear();
        for (const Entry& e : table)
            if (e.key >= 0) out.push_back(e.key);
    }
};
