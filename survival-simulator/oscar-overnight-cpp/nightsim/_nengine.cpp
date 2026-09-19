// Native port of the survival simulator engine (vendor/survival-simulator, source
// commit acfc31a). It reproduces the Python engine operation by operation:
//
// * CPython's Mersenne Twister and the exact sequence of random calls, including
//   the 1.9M draws spent colouring the biome surface at start-up;
// * numpy/libm float maths in the same evaluation order (build with
//   -ffp-contract=off so no multiply-add is fused);
// * CPython set iteration order for every set the engine iterates (grid chunks,
//   local neighbourhoods, obstacle edges), with object hashes supplied by a
//   creation counter (see README: the Python engine hashes by memory address,
//   which is the only thing that makes its own runs non-reproducible);
// * GEOS point-in-polygon semantics used by shapely's Polygon.contains;
// * list mutation during iteration (a death skips the next agent, a rotten fruit
//   skips the next fruit, a dead tree skips the next tree).
//
// Rendering (pygame surfaces and colours) is omitted: it never feeds back into
// the simulation apart from the random draws, which are replayed.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <unordered_set>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

namespace {

// ----------------------------------------------------------------------------
// numpy's own float64 loops for sin, cos, arctan2 and hypot. numpy may use SIMD
// kernels (e.g. SVML arctan2 on AVX-512) that round differently from libm, so
// the engine calls exactly the loop numpy would, taken from the ufunc objects.
// ----------------------------------------------------------------------------
struct NpLoop { PyUFuncGenericFunction fn = nullptr; void* data = nullptr; };
NpLoop np_sin_l, np_cos_l, np_atan2_l, np_hypot_l;

inline double call1(const NpLoop& l, double x) {
    double out;
    char* args[2] = {(char*)&x, (char*)&out};
    npy_intp n = 1, steps[2] = {sizeof(double), sizeof(double)};
    l.fn(args, &n, steps, l.data);
    return out;
}
inline double call2(const NpLoop& l, double a, double b) {
    double out;
    char* args[3] = {(char*)&a, (char*)&b, (char*)&out};
    npy_intp n = 1, steps[3] = {sizeof(double), sizeof(double), sizeof(double)};
    l.fn(args, &n, steps, l.data);
    return out;
}
// Batched calls: numpy runs the same kernel for arrays as for one element.
inline void vec1(const NpLoop& l, double (*fb)(double), const double* in, double* out, size_t n) {
    if (!n) return;
    if (!l.fn) { for (size_t i = 0; i < n; i++) out[i] = fb(in[i]); return; }
    char* args[2] = {(char*)in, (char*)out};
    npy_intp nn = (npy_intp)n, steps[2] = {sizeof(double), sizeof(double)};
    l.fn(args, &nn, steps, l.data);
}
inline void vec2(const NpLoop& l, double (*fb)(double, double), const double* a, const double* b, double* out, size_t n) {
    if (!n) return;
    if (!l.fn) { for (size_t i = 0; i < n; i++) out[i] = fb(a[i], b[i]); return; }
    char* args[3] = {(char*)a, (char*)b, (char*)out};
    npy_intp nn = (npy_intp)n, steps[3] = {sizeof(double), sizeof(double), sizeof(double)};
    l.fn(args, &nn, steps, l.data);
}
inline double libm_sin(double x) { return std::sin(x); }
inline double libm_cos(double x) { return std::cos(x); }
inline double libm_atan2(double y, double x) { return std::atan2(y, x); }
inline double libm_hypot(double x, double y) { return std::hypot(x, y); }
inline void vsin(const double* in, double* out, size_t n) { vec1(np_sin_l, libm_sin, in, out, n); }
inline void vcos(const double* in, double* out, size_t n) { vec1(np_cos_l, libm_cos, in, out, n); }
inline void vatan2(const double* y, const double* x, double* out, size_t n) { vec2(np_atan2_l, libm_atan2, y, x, out, n); }
inline void vhypot(const double* x, const double* y, double* out, size_t n) { vec2(np_hypot_l, libm_hypot, x, y, out, n); }

inline double np_sin(double x) { return np_sin_l.fn ? call1(np_sin_l, x) : std::sin(x); }
inline double np_cos(double x) { return np_cos_l.fn ? call1(np_cos_l, x) : std::cos(x); }
inline double np_atan2(double y, double x) { return np_atan2_l.fn ? call2(np_atan2_l, y, x) : std::atan2(y, x); }
inline double np_hypot(double x, double y) { return np_hypot_l.fn ? call2(np_hypot_l, x, y) : std::hypot(x, y); }

bool find_loop(PyObject* ufunc, int nin, NpLoop& out) {
    PyUFuncObject* uf = (PyUFuncObject*)ufunc;
    if (uf->nin != nin || uf->nout != 1) return false;
    int nargs = nin + 1;
    for (int i = 0; i < uf->ntypes; i++) {
        bool ok = true;
        for (int j = 0; j < nargs; j++)
            if (uf->types[i * nargs + j] != NPY_DOUBLE) ok = false;
        if (ok && uf->functions[i]) {
            out.fn = uf->functions[i];
            out.data = uf->data ? uf->data[i] : nullptr;
            return true;
        }
    }
    return false;
}

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
int64_t py_hash_double(double v) {
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
int64_t py_hash_int(int64_t v) {  // non-negative small ints only
    const uint64_t MOD = (((uint64_t)1) << 61) - 1;
    int64_t h = (int64_t)((uint64_t)v % MOD);
    return h == -1 ? -2 : h;
}
int64_t py_tuple_hash(const int64_t* lanes, size_t len) {
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
int64_t edge_hash(double x1, double y1, double x2, double y2) {
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

// ----------------------------------------------------------------------------
// GEOS robust orientation (CGAlgorithmsDD) and ray-crossing point location
// ----------------------------------------------------------------------------
struct DD {
    double hi, lo;
    DD(double h, double l) : hi(h), lo(l) {}
    explicit DD(double h) : hi(h), lo(0.0) {}
    void selfAdd(double yhi, double ylo) {
        double H, h, T, t, S, s, e, f;
        S = hi + yhi; T = lo + ylo; e = S - hi; f = T - lo;
        s = S - e; t = T - f;
        s = (yhi - e) + (hi - s); t = (ylo - f) + (lo - t);
        e = s + T; H = S + e; h = e + (S - H); e = t + h;
        double zhi = H + e;
        double zlo = e + (H - zhi);
        hi = zhi; lo = zlo;
    }
    void selfMultiply(double yhi, double ylo) {
        const double SPLIT = 134217729.0;
        double hx, tx, hy, ty, C, c;
        C = SPLIT * hi; hx = C - hi; c = SPLIT * yhi;
        hx = C - hx; tx = hi - hx; hy = c - yhi;
        C = hi * yhi; hy = c - hy; ty = yhi - hy;
        c = ((((hx * hy - C) + hx * ty) + tx * hy) + tx * ty) + (hi * ylo + lo * yhi);
        double zhi = C + c; hx = C - zhi;
        double zlo = c + hx;
        hi = zhi; lo = zlo;
    }
};
inline DD dd_add(DD a, const DD& b) { a.selfAdd(b.hi, b.lo); return a; }
inline DD dd_sub(DD a, const DD& b) { a.selfAdd(-1 * b.hi, -1 * b.lo); return a; }
inline DD dd_mul(DD a, const DD& b) { a.selfMultiply(b.hi, b.lo); return a; }

inline int orientation_sign(double d) { return d > 0 ? 1 : (d < 0 ? -1 : 0); }

int orientation_index(double pax, double pay, double pbx, double pby, double pcx, double pcy) {
    // orientationIndexFilter
    const double DP_SAFE_EPSILON = 1e-15;
    double detsum;
    const double detleft = (pax - pcx) * (pby - pcy);
    const double detright = (pay - pcy) * (pbx - pcx);
    const double det = detleft - detright;
    bool decided = false;
    int res = 0;
    if (detleft > 0.0) {
        if (detright <= 0.0) { res = orientation_sign(det); decided = true; }
        else detsum = detleft + detright;
    } else if (detleft < 0.0) {
        if (detright >= 0.0) { res = orientation_sign(det); decided = true; }
        else detsum = -detleft - detright;
    } else {
        res = orientation_sign(det); decided = true;
    }
    if (decided) return res;
    const double errbound = DP_SAFE_EPSILON * detsum;
    if ((det >= errbound) || (-det >= errbound)) return orientation_sign(det);
    DD dx1 = dd_add(DD(pbx), DD(-pax));
    DD dy1 = dd_add(DD(pby), DD(-pay));
    DD dx2 = dd_add(DD(pcx), DD(-pbx));
    DD dy2 = dd_add(DD(pcy), DD(-pby));
    DD d = dd_sub(dd_mul(dx1, dy2), dd_mul(dy1, dx2));
    if (d.hi < 0.0 || (d.hi == 0.0 && d.lo < 0.0)) return -1;
    if (d.hi > 0.0 || (d.hi == 0.0 && d.lo > 0.0)) return 1;
    return 0;
}

// shapely Polygon(ring).contains(Point(px, py)); ring is closed (first == last).
bool polygon_contains(const std::vector<double>& rx, const std::vector<double>& ry,
                      double minx, double maxx, double miny, double maxy, double px, double py) {
    if (px < minx || px > maxx || py < miny || py > maxy) return false;
    int crossings = 0;
    size_t n = rx.size();
    for (size_t i = 1; i < n; i++) {
        double p1x = rx[i - 1], p1y = ry[i - 1], p2x = rx[i], p2y = ry[i];
        if (p1x < px && p2x < px) continue;
        if (px == p2x && py == p2y) return false;
        if (p1y == py && p2y == py) {
            double mn = p1x, mx = p2x;
            if (mn > mx) { mn = p2x; mx = p1x; }
            if (px >= mn && px <= mx) return false;
            continue;
        }
        if ((p1y > py && p2y <= py) || (p2y > py && p1y <= py)) {
            int sign = orientation_index(p1x, p1y, p2x, p2y, px, py);
            if (sign == 0) return false;
            if (p2y < p1y) sign = -sign;
            if (sign > 0) crossings++;
        }
    }
    return (crossings % 2) == 1;
}

// ----------------------------------------------------------------------------
// Biomes
// ----------------------------------------------------------------------------
enum Biome : uint8_t { FOREST = 0, SWAMP = 1, DESERT = 2, GRASSLAND = 3, RIVER = 4 };
struct BiomeInfo { const char* name; double move_penalty, tree_spawn_rate, fruit_spawn_rate, energy_drain_rate; int palette; };
const BiomeInfo BIOMES[5] = {
    {"forest", 1.0, 1.0, 0.1, 1.0, 3},
    {"swamp", 0.5, 0.9, 0.08, 1.0, 4},
    {"desert", 0.8, 0.1, 0.05, 1.0, 4},
    {"grassland", 1.0, 0.5, 0.1, 1.0, 3},
    {"river", 0.3, 0.0, 0.0, 1.0, 4},
};

// ----------------------------------------------------------------------------
// Entities
// ----------------------------------------------------------------------------
struct Creature {
    double x, y, size, speed, sprint_speed, age, energy, max_energy, direction;
    double hearing_radius, vision_radius, cone_angle, max_age;
    int64_t id = -1;       // agent_id (agents only)
    int64_t hash = 0;      // stand-in for Python's id()-based hash
    int32_t key = 0;       // set key (== creation serial)
    bool resting = true;   // predators only
    double dbg_tdist = -1., dbg_look = 0., dbg_ang = 0.; int dbg_mode = 0;   // nightsim debug: predator's last target
    // Python int-ness of traits (affects only how values print; all maths is double)
    bool speed_int = false, sprint_int = false, hearing_int = false, vision_int = false, max_energy_int = false;
};

struct Fruit { double x, y, radius, energy, age; int64_t fruit_id, hash; int32_t key; bool removed = false; };
struct Tree { double x, y, radius, age; int64_t hash; int32_t key; };
struct Obstacle { double x, y, w, h; };
struct Edge { double x1, y1, x2, y2; int64_t hash; };

struct Obs {
    // type: 0 Fruit, 1 Agent, 2 Predator, 3 Tree, 4 Edge
    int type;
    double distance, angle, rel_dir;
    bool has_rel_dir;
    int64_t id;
    bool has_id;
    double c[4];
};

struct Event { int kind; double t; int64_t id; double age, energy; };  // kind 0 starvation, 1 predator, 2 fruit eaten

// ----------------------------------------------------------------------------
// Engine
// ----------------------------------------------------------------------------
class Engine {
public:
    int W, H, CS;
    double dt;
    bool predators_enabled;
    PyRandom rng;

    std::vector<uint8_t> biome;  // index x * H + y

    std::vector<Creature> agents;
    std::vector<Creature> predators;
    std::vector<Fruit> fruits;
    std::vector<Tree> trees;
    std::vector<Obstacle> obstacles;
    std::vector<Edge> edges;  // unique by value, index == set key
    std::vector<int32_t> edge_set_order;

    int64_t next_agent_id = 0, next_fruit_id = 0, next_serial = 1;
    double score = 0.0, time = 0.0;

    std::unordered_map<int64_t, std::vector<Obs>> agent_observations;
    std::vector<Event> events;

    // chunk grids (dense, offset by 1 so chunk -1 is valid)
    int NCX, NCY;
    std::vector<PySetEmu> grid_agents, grid_fruits, grid_trees, grid_predators;
    bool agents_dirty = true, fruits_dirty = true, trees_dirty = true, predators_dirty = true;
    std::vector<std::vector<int32_t>> grid_obstacles;  // obstacle indices per chunk
    std::vector<PySetEmu> grid_edges;
    std::vector<std::vector<int32_t>> local_edges_cache;      // ordered union per centre chunk
    std::vector<std::vector<int32_t>> local_obstacles_cache;  // obstacles per centre chunk
    std::vector<bool> local_cache_ready;
    std::vector<std::vector<int32_t>> local_edge_rank;  // per centre chunk: position in local order, -1 if absent

    // Static 100 px index of obstacle corners and edges (vision only). Rays depend on
    // the set of corners in range, not their order, and candidate edges are re-sorted
    // into the local set order, so results equal the full scan.
    static const int GCELL = 100;
    int GNX = 0, GNY = 0;
    struct CornerRef { double x, y; int32_t edge; };
    std::vector<std::vector<CornerRef>> corner_cells;
    std::vector<std::vector<int32_t>> edge_cells;
    // Per (cell, centre chunk, vision radius): local edges that can lie within
    // reach of any point of the cell, sorted by local set order. Each call then
    // applies the exact per-position filter to this short list.
    std::unordered_map<int64_t, std::vector<int32_t>> cell_edge_cand;
    std::vector<double> radius_ids;
    std::vector<std::pair<int32_t, int32_t>> cand;
    const std::vector<int32_t>& cell_candidates(int gx, int gy, int ci, const std::vector<int32_t>& rank, double vr) {
        int64_t rid = -1;
        for (size_t i = 0; i < radius_ids.size(); i++)
            if (radius_ids[i] == vr) { rid = (int64_t)i; break; }
        if (rid < 0) { radius_ids.push_back(vr); rid = (int64_t)radius_ids.size() - 1; }
        int64_t key = (((int64_t)gx * GNY + gy) * (int64_t)(NCX * NCY) + ci) * 4096 + rid;
        auto it = cell_edge_cand.find(key);
        if (it != cell_edge_cand.end()) return it->second;
        const double R = vr + 3.0;
        const double cx0 = (double)gx * GCELL, cx1 = cx0 + GCELL, cy0 = (double)gy * GCELL, cy1 = cy0 + GCELL;
        cand.clear();
        for (size_t k = 0; k < edges.size(); k++) {
            if (rank[k] < 0) continue;
            const Edge& e = edges[k];
            double bx0 = std::min(e.x1, e.x2), bx1 = std::max(e.x1, e.x2);
            double by0 = std::min(e.y1, e.y2), by1 = std::max(e.y1, e.y2);
            double ddx = bx1 < cx0 ? cx0 - bx1 : (bx0 > cx1 ? bx0 - cx1 : 0.0);
            double ddy = by1 < cy0 ? cy0 - by1 : (by0 > cy1 ? by0 - cy1 : 0.0);
            if (ddx * ddx + ddy * ddy > R * R) continue;
            cand.push_back({rank[k], (int32_t)k});
        }
        std::sort(cand.begin(), cand.end());
        std::vector<int32_t> out;
        out.reserve(cand.size());
        for (auto& rk : cand) out.push_back(rk.second);
        return cell_edge_cand.emplace(key, std::move(out)).first->second;
    }

    // scratch
    std::vector<double> rays, ring_x, ring_y, poly_key, e_vx, e_vy, e_a1, e_b1, e_nt;
    std::vector<double> cdx, cdy, cbase, rcos, rsin, pdx, pdy, pang;
    std::vector<double> odx, ody, ohyp, oang;
    std::vector<size_t> oidx;
    std::vector<double> vpx, vpy, pdist, pang2;
    std::vector<char> pnear, pvis;
    std::vector<int32_t> tmp_keys;
    std::vector<int> poly_order;

    Engine(int width, int height, int chunk_size, int starting_agents, int starting_predators,
           int starting_fruits, int starting_trees, const std::vector<uint32_t>& seed_key, double dt_,
           bool predators_enabled_)
        : W(width), H(height), CS(chunk_size), dt(dt_), predators_enabled(predators_enabled_) {
        rng.init_by_array(seed_key);
        NCX = W / CS + 4;
        NCY = H / CS + 4;
        grid_agents.resize(NCX * NCY);
        grid_fruits.resize(NCX * NCY);
        grid_trees.resize(NCX * NCY);
        grid_predators.resize(NCX * NCY);
        grid_edges.resize(NCX * NCY);
        grid_obstacles.resize(NCX * NCY);
        local_edges_cache.resize(NCX * NCY);
        local_obstacles_cache.resize(NCX * NCY);
        local_cache_ready.assign(NCX * NCY, false);
        local_edge_rank.resize(NCX * NCY);

        generate_map(10, 1);
        render_biome_surface();
        // boundaries (thickness 30)
        add_obstacle(0, 0, W, 30);
        add_obstacle(0, H - 30, W, 30);
        add_obstacle(0, 0, 30, H);
        add_obstacle(W - 30, 0, 30, H);

        int n_obstacles = W / 20;
        for (int i = 0; i < n_obstacles; i++) {
            double w = rng.uniform(30, 100);
            double h = rng.uniform(30, 100);
            double x = rng.uniform(0, W - w);
            double y = rng.uniform(0, H - h);
            add_obstacle(x, y, w, h);
        }
        build_edges();
        for (int i = 0; i < starting_agents; i++) spawn_agent_random();
        for (int i = 0; i < starting_predators; i++) spawn_predator();
        for (int i = 0; i < starting_fruits; i++) spawn_fruit_random();
        for (int i = 0; i < starting_trees; i++) spawn_tree();
        for (Tree& t : trees) tree_grow(t, rng.uniform(20, 80));
    }

    // ------------------------------------------------------------ map
    void generate_map(int num_biomes, int num_rivers) {
        std::vector<int64_t> px(num_biomes), py(num_biomes);
        for (int i = 0; i < num_biomes; i++) {
            px[i] = rng.randint(0, W - 1);
            py[i] = rng.randint(0, H - 1);
        }
        std::vector<uint8_t> types(num_biomes);
        for (int i = 0; i < num_biomes; i++) types[i] = (uint8_t)rng.randbelow(4);
        biome.assign((size_t)W * H, 0);
        for (int x = 0; x < W; x++)
            for (int y = 0; y < H; y++) {
                int best = 0;
                int64_t bd = (x - px[0]) * (x - px[0]) + (y - py[0]) * (y - py[0]);
                for (int i = 1; i < num_biomes; i++) {
                    int64_t d = (x - px[i]) * (x - px[i]) + (y - py[i]) * (y - py[i]);
                    if (d < bd) { bd = d; best = i; }
                }
                biome[(size_t)x * H + y] = types[best];
            }
        for (int r = 0; r < num_rivers; r++) {
            int start_edge = (int)rng.randbelow(4);
            int end_edge = (int)rng.randbelow(4);
            int64_t sx, sy, ex, ey;
            edge_point(start_edge, sx, sy);
            edge_point(end_edge, ex, ey);
            std::vector<std::pair<int64_t, int64_t>> path;
            river_path(sx, sy, ex, ey, path);
            int64_t radius = rng.randint(20, 100);
            int64_t r2 = radius * radius;
            std::vector<uint8_t> mask((size_t)W * H, 0);
            if (path.empty()) path.push_back({-1, 0});  // scipy EDT with no background pixel
            for (auto& p : path) {
                int64_t x0 = std::max<int64_t>(0, p.first - radius), x1 = std::min<int64_t>(W - 1, p.first + radius);
                for (int64_t x = x0; x <= x1; x++) {
                    int64_t dx2 = (x - p.first) * (x - p.first);
                    int64_t rem = r2 - dx2;
                    if (rem < 0) continue;
                    int64_t dy = (int64_t)std::sqrt((double)rem);
                    while (dy * dy > rem) dy--;
                    while ((dy + 1) * (dy + 1) <= rem) dy++;
                    int64_t y0 = std::max<int64_t>(0, p.second - dy), y1 = std::min<int64_t>(H - 1, p.second + dy);
                    for (int64_t y = y0; y <= y1; y++) mask[(size_t)x * H + y] = 1;
                }
            }
            for (size_t i = 0; i < mask.size(); i++)
                if (mask[i]) biome[i] = RIVER;
        }
    }
    void edge_point(int edge, int64_t& x, int64_t& y) {
        if (edge == 0) { x = rng.randint(0, W - 1); y = 0; }
        else if (edge == 1) { x = rng.randint(0, W - 1); y = H - 1; }
        else if (edge == 2) { x = 0; y = rng.randint(0, H - 1); }
        else { x = W - 1; y = rng.randint(0, H - 1); }
    }
    void river_path(int64_t sx, int64_t sy, int64_t ex, int64_t ey, std::vector<std::pair<int64_t, int64_t>>& path) {
        const double max_turn = 10.0 * (PI / 180.0);
        double direction = np_atan2((double)(ey - sy), (double)(ex - sx));
        int64_t cx = sx, cy = sy;
        int counter = 0;
        while (np_hypot((double)(cx - ex), (double)(cy - ey)) > 5 && counter < 10000) {
            path.push_back({cx, cy});
            direction += rng.uniform(-max_turn, max_turn);
            if (rng.random() < 0.9) {
                double target = np_atan2((double)(ey - cy), (double)(ex - cx));
                direction += 0.3 * (target - direction + PI) / (2 * PI) - PI;
            }
            int64_t step = rng.randint(3, 7);
            int64_t nx = (int64_t)((double)cx + (double)step * np_cos(direction));
            int64_t ny = (int64_t)((double)cy + (double)step * np_sin(direction));
            nx = std::max<int64_t>(0, std::min<int64_t>(W - 1, nx));
            ny = std::max<int64_t>(0, std::min<int64_t>(H - 1, ny));
            if (nx == cx && ny == cy) {
                int64_t dx = ex > cx ? 1 : (ex < cx ? -1 : 0);
                int64_t dy = ey > cy ? 1 : (ey < cy ? -1 : 0);
                nx = std::max<int64_t>(0, std::min<int64_t>(W - 1, cx + dx));
                ny = std::max<int64_t>(0, std::min<int64_t>(H - 1, cy + dy));
            }
            cx = nx; cy = ny;
            counter++;
        }
    }
    void render_biome_surface() {
        // Environment._render_biome_surface draws one colour per pixel with rng.choice.
        size_t n = (size_t)W * H;
        for (size_t i = 0; i < n; i++) {
            int pal = BIOMES[biome[i]].palette;
            if (pal == 3) { while (rng.getrandbits(2) >= 3) {} }
            else { while (rng.getrandbits(3) >= 4) {} }
        }
    }
    inline uint8_t biome_at(double x, double y) const {
        // min(max(int(x), 0), W - 1)
        int64_t ix = (int64_t)x, iy = (int64_t)y;
        ix = std::min<int64_t>(std::max<int64_t>(ix, 0), W - 1);
        iy = std::min<int64_t>(std::max<int64_t>(iy, 0), H - 1);
        return biome[(size_t)ix * H + iy];
    }

    // ------------------------------------------------------------ grid
    inline int chunk_index(long cx, long cy) const { return (int)((cx + 1) * NCY + (cy + 1)); }
    inline long to_chunk(double v) const { return (long)py_floordiv(v, (double)CS); }
    inline bool chunk_valid(long cx, long cy) const { return cx >= -1 && cy >= -1 && cx + 1 < NCX && cy + 1 < NCY; }

    void add_obstacle(double x, double y, double w, double h) { obstacles.push_back({x, y, w, h}); }

    void build_edges() {
        // self.edges = set(); for obs in obstacles: self.edges.update([top, right, bottom, left])
        PySetEmu es;
        std::vector<Edge> uniq;
        auto key_of = [&](const Edge& e) -> int32_t {
            for (size_t i = 0; i < uniq.size(); i++)
                if (uniq[i].x1 == e.x1 && uniq[i].y1 == e.y1 && uniq[i].x2 == e.x2 && uniq[i].y2 == e.y2) return (int32_t)i;
            uniq.push_back(e);
            return (int32_t)(uniq.size() - 1);
        };
        for (const Obstacle& o : obstacles) {
            Edge four[4] = {
                {o.x, o.y, o.x + o.w, o.y, 0},
                {o.x + o.w, o.y, o.x + o.w, o.y + o.h, 0},
                {o.x, o.y + o.h, o.x + o.w, o.y + o.h, 0},
                {o.x, o.y, o.x, o.y + o.h, 0},
            };
            for (Edge& e : four) {
                e.hash = edge_hash(e.x1, e.y1, e.x2, e.y2);
                int32_t k = key_of(e);
                es.add(k, e.hash);
            }
        }
        edges = uniq;
        es.to_vector(edge_set_order);
        // grid_edges
        for (int32_t k : edge_set_order) {
            const Edge& e = edges[k];
            long c0x = to_chunk(e.x1), c0y = to_chunk(e.y1), c1x = to_chunk(e.x2), c1y = to_chunk(e.y2);
            for (long cx = c0x; cx <= c1x; cx++)
                for (long cy = c0y; cy <= c1y; cy++)
                    if (chunk_valid(cx, cy)) grid_edges[chunk_index(cx, cy)].add(k, e.hash);
        }
        GNX = W / GCELL + 1; GNY = H / GCELL + 1;
        corner_cells.assign((size_t)GNX * GNY, {});
        edge_cells.assign((size_t)GNX * GNY, {});
        for (size_t k = 0; k < edges.size(); k++) {
            const Edge& e = edges[k];
            corner_cells[cell_of(e.x1, e.y1)].push_back({e.x1, e.y1, (int32_t)k});
            corner_cells[cell_of(e.x2, e.y2)].push_back({e.x2, e.y2, (int32_t)k});
            int x0 = cell_x(std::min(e.x1, e.x2)), x1 = cell_x(std::max(e.x1, e.x2));
            int y0 = cell_y(std::min(e.y1, e.y2)), y1 = cell_y(std::max(e.y1, e.y2));
            for (int cx = x0; cx <= x1; cx++)
                for (int cy = y0; cy <= y1; cy++) edge_cells[(size_t)cx * GNY + cy].push_back((int32_t)k);
        }
        // grid_obstacles (order is irrelevant: only used by np.any)
        for (size_t i = 0; i < obstacles.size(); i++) {
            const Obstacle& o = obstacles[i];
            long c0x = to_chunk(o.x), c0y = to_chunk(o.y), c1x = to_chunk(o.x + o.w), c1y = to_chunk(o.y + o.h);
            for (long cx = c0x; cx <= c1x; cx++)
                for (long cy = c0y; cy <= c1y; cy++)
                    if (chunk_valid(cx, cy)) grid_obstacles[chunk_index(cx, cy)].push_back((int32_t)i);
        }
    }

    inline int cell_x(double v) const { long c = (long)std::floor(v / GCELL); return (int)std::min<long>(std::max<long>(c, 0), GNX - 1); }
    inline int cell_y(double v) const { long c = (long)std::floor(v / GCELL); return (int)std::min<long>(std::max<long>(c, 0), GNY - 1); }
    inline size_t cell_of(double x, double y) const { return (size_t)cell_x(x) * GNY + cell_y(y); }

    void ensure_local_static(long cx, long cy) {
        int ci = chunk_index(cx, cy);
        if (local_cache_ready[ci]) return;
        PySetEmu local;
        std::vector<int32_t> obs;
        std::vector<bool> seen(obstacles.size(), false);
        for (long dx = -1; dx <= 1; dx++)
            for (long dy = -1; dy <= 1; dy++) {
                long nx = cx + dx, ny = cy + dy;
                if (!chunk_valid(nx, ny)) continue;
                int ni = chunk_index(nx, ny);
                local.merge(grid_edges[ni]);
                for (int32_t o : grid_obstacles[ni])
                    if (!seen[o]) { seen[o] = true; obs.push_back(o); }
            }
        local.to_vector(local_edges_cache[ci]);
        local_edge_rank[ci].assign(edges.size(), -1);
        for (size_t r = 0; r < local_edges_cache[ci].size(); r++) local_edge_rank[ci][local_edges_cache[ci][r]] = (int32_t)r;
        local_obstacles_cache[ci] = obs;
        local_cache_ready[ci] = true;
    }
    const std::vector<int32_t>& local_edges(double x, double y) {
        long cx = to_chunk(x), cy = to_chunk(y);
        ensure_local_static(cx, cy);
        return local_edges_cache[chunk_index(cx, cy)];
    }
    const std::vector<int32_t>& local_obstacles(double x, double y) {
        long cx = to_chunk(x), cy = to_chunk(y);
        ensure_local_static(cx, cy);
        return local_obstacles_cache[chunk_index(cx, cy)];
    }

    template <class T>
    void rebuild_grid(std::vector<PySetEmu>& grid, const std::vector<T>& items) {
        for (PySetEmu& s : grid) if (s.used || s.fill) s.clear();
        for (const T& it : items) {
            long cx = to_chunk(it.x), cy = to_chunk(it.y);
            if (chunk_valid(cx, cy)) grid[chunk_index(cx, cy)].add(it.key, it.hash);
        }
    }
    void refresh_grids() {
        if (agents_dirty) { rebuild_grid(grid_agents, agents); agents_dirty = false; invalidate(uc_agents); index_keys(agents, key_to_index_agent); }
        if (fruits_dirty) { rebuild_grid(grid_fruits, fruits); fruits_dirty = false; invalidate(uc_fruits); index_keys(fruits, key_to_index_fruit); }
        if (trees_dirty) { rebuild_grid(grid_trees, trees); trees_dirty = false; invalidate(uc_trees); index_keys(trees, key_to_index_tree); }
        if (predators_dirty) { rebuild_grid(grid_predators, predators); predators_dirty = false; invalidate(uc_preds); index_keys(predators, key_to_index_pred); }
    }
    // Neighbourhood unions depend only on the grid and the centre chunk, so they are
    // cached per chunk and dropped whenever that grid is rebuilt.
    struct UnionCache { std::vector<PySetEmu> sets; std::vector<char> ok; };
    UnionCache uc_agents, uc_fruits, uc_trees, uc_preds;
    const PySetEmu& cached_union(std::vector<PySetEmu>& grid, UnionCache& uc, double x, double y) {
        if (uc.sets.empty()) { uc.sets.resize(grid.size()); uc.ok.assign(grid.size(), 0); }
        long cx = to_chunk(x), cy = to_chunk(y);
        int ci = chunk_index(cx, cy);
        if (!uc.ok[ci]) { local_union(grid, x, y, uc.sets[ci]); uc.ok[ci] = 1; }
        return uc.sets[ci];
    }
    static void invalidate(UnionCache& uc) { std::fill(uc.ok.begin(), uc.ok.end(), 0); }
    void local_union(std::vector<PySetEmu>& grid, double x, double y, PySetEmu& out) {
        out.clear();
        long cx = to_chunk(x), cy = to_chunk(y);
        for (long dx = -1; dx <= 1; dx++)
            for (long dy = -1; dy <= 1; dy++) {
                long nx = cx + dx, ny = cy + dy;
                if (!chunk_valid(nx, ny)) continue;
                out.merge(grid[chunk_index(nx, ny)]);
            }
    }

    // ------------------------------------------------------------ spawning
    bool is_position_free(double x, double y, double w, double h) const {
        if (x < 0 || y < 0 || x + w > W || y + h > H) return false;
        for (const Obstacle& o : obstacles)
            if (o.x < x + w && o.x + o.w > x && o.y < y + h && o.y + o.h > y) return false;
        return true;
    }

    Creature make_agent(double x, double y, double speed, double sprint, double energy, double max_energy,
                        double hearing, double vision, double cone) {
        Creature a;
        a.key = (int32_t)next_serial; a.hash = py_hash_int(next_serial); next_serial++;
        a.x = x; a.y = y; a.size = 5; a.speed = speed; a.sprint_speed = sprint;
        a.age = 0.0; a.energy = energy; a.max_energy = max_energy;
        a.direction = rng.uniform(0, 2 * PI);
        a.hearing_radius = hearing; a.vision_radius = vision; a.cone_angle = cone;
        a.max_age = 60 + rng.uniform(0, 60);
        return a;
    }
    void add_agent(Creature a) {
        a.id = next_agent_id++;
        agents.push_back(a);
        agents_dirty = true;
    }
    void spawn_agent_random() {
        while (true) {
            double x = rng.uniform(20, W - 20);
            double y = rng.uniform(20, H - 20);
            if (is_position_free(x, y, 20, 20)) {
                Creature a = make_agent(x, y, 10, 20, 150, 500.0, 50, 200, PI / 3);
                a.speed_int = a.sprint_int = a.hearing_int = a.vision_int = true;
                add_agent(a);
                return;
            }
        }
    }
    void spawn_child(const Creature& parent) {
        double angle = rng.uniform(0, 2 * PI);
        double dist = rng.uniform(10, 30);
        double x = parent.x + dist * np_cos(angle);
        double y = parent.y + dist * np_sin(angle);
        x = std::min(std::max(x, 20.0), (double)(W - 20));
        y = std::min(std::max(y, 20.0), (double)(H - 20));
        const double lo = 1 - 0.5, hi = 1 + 0.5;
        double speed = parent.speed, sprint = parent.sprint_speed, max_energy = parent.max_energy;
        double hearing = parent.hearing_radius, vision = parent.vision_radius, cone = parent.cone_angle;
        bool speed_int = parent.speed_int, sprint_int = parent.sprint_int, max_energy_int = parent.max_energy_int;
        bool hearing_int = parent.hearing_int, vision_int = parent.vision_int;
        if (rng.random() < 0.1) { speed = parent.speed * rng.uniform(lo, hi); speed_int = false; }
        if (rng.random() < 0.1) { sprint = parent.sprint_speed * rng.uniform(lo, hi); sprint_int = false; }
        if (rng.random() < 0.1) { max_energy = parent.max_energy * rng.uniform(lo, hi); max_energy_int = false; }
        if (rng.random() < 0.1) { hearing = parent.hearing_radius * rng.uniform(lo, hi); hearing_int = false; }
        if (rng.random() < 0.1) { vision = parent.vision_radius * rng.uniform(lo, hi); vision_int = false; }
        if (rng.random() < 0.1) { cone = parent.cone_angle * rng.uniform(lo, hi); }
        // min(value, bound): Python returns the bound object when value is larger
        if (20 < speed) { speed = 20; speed_int = true; }
        if (40 < sprint) { sprint = 40; sprint_int = true; }
        if (1000 < max_energy) { max_energy = 1000; max_energy_int = true; }
        double max_hearing = CS / 4.0;
        if (max_hearing < hearing) { hearing = max_hearing; hearing_int = false; }
        if ((double)CS < vision) { vision = CS; vision_int = true; }
        if (PI / 2 < cone) cone = PI / 2;
        // Agent(x, y, rng) built only to read baseline traits (consumes two draws)
        make_agent(x, y, 10, 20, 75.0, 500.0, 50, 200, PI / 3);
        Creature a = is_position_free(x, y, 20, 20)
                         ? make_agent(x, y, speed, sprint, 75.0, max_energy, hearing, vision, cone)
                         : make_agent(parent.x, parent.y, speed, sprint, 75.0, max_energy, hearing, vision, cone);
        a.speed_int = speed_int; a.sprint_int = sprint_int; a.max_energy_int = max_energy_int;
        a.hearing_int = hearing_int; a.vision_int = vision_int;
        add_agent(a);
    }
    std::function<void(const Creature&, int)> on_kill;   // nightsim diagnostics: called before an agent is removed
    struct DeathRec { int cause; double t, age, e, maxe, speed, sprint, x, y; int npred150; double dpred; int prest, nearwall, pop, npred, evading, old, haspost; };
    std::vector<DeathRec> death_log;
    void kill_agent_at(size_t idx, int cause) {
        const Creature& a = agents[idx];
        if (on_kill) on_kill(a, cause);
        events.push_back({cause, time, a.id, a.age, a.energy});
        agent_observations.erase(a.id);
        agents.erase(agents.begin() + idx);
        agents_dirty = true;
    }
    bool try_add_fruit(double x, double y, double radius) {
        if (!is_position_free(x, y, radius, radius)) return false;
        Fruit f;
        f.key = (int32_t)next_serial; f.hash = py_hash_int(next_serial); next_serial++;
        f.x = x; f.y = y; f.radius = radius; f.energy = 20; f.age = 0;
        f.fruit_id = next_fruit_id++;
        fruits.push_back(f);
        fruits_dirty = true;
        return true;
    }
    void spawn_fruit_random() {
        const double radius = 5;
        double x = rng.uniform(25, W - 25);
        double y = rng.uniform(25, H - 25);
        for (int i = 0; i < 50; i++) {
            if (is_position_free(x, y, radius, radius)) break;
            x = rng.uniform(0, W - radius);
            y = rng.uniform(0, H - radius);
        }
        try_add_fruit(x, y, radius);
    }
    void spawn_fruit_around_tree(const Tree& t) {
        double angle = rng.uniform(0, 2 * PI);
        double dist = rng.uniform(t.radius, t.radius * 3);
        double x = t.x + dist * np_cos(angle);
        double y = t.y + dist * np_sin(angle);
        try_add_fruit(x, y, 5);
    }
    void spawn_tree() {
        double x = rng.uniform(25, W - 25);
        double y = rng.uniform(25, H - 25);
        for (int i = 0; i < 50; i++) {
            if (is_position_free(x, y, 25, 25)) break;
            x = rng.uniform(0, W - 25);
            y = rng.uniform(0, H - 25);
        }
        double rate = BIOMES[biome[(size_t)(int64_t)x * H + (int64_t)y]].tree_spawn_rate;
        if (rate > rng.random()) {
            Tree t;
            t.key = (int32_t)next_serial; t.hash = py_hash_int(next_serial); next_serial++;
            t.x = x; t.y = y; t.radius = 10; t.age = 0;
            trees.push_back(t);
            trees_dirty = true;
        }
    }
    static void tree_grow(Tree& t, double amount) {
        t.radius = std::min(20.0, t.radius + amount);
        t.age += amount;
    }
    void spawn_predator() {
        if (!predators_enabled) return;
        double x = rng.uniform(20, W - 20);
        double y = rng.uniform(20, H - 20);
        double size = 10;
        if (!is_position_free(x, y, size, size)) return;
        Creature p;
        p.key = (int32_t)next_serial; p.hash = py_hash_int(next_serial); next_serial++;
        p.x = x; p.y = y; p.size = size; p.speed = 11; p.sprint_speed = 15;
        p.age = 0.0; p.energy = 0.0; p.max_energy = 200.0;
        p.direction = rng.uniform(0, 2 * PI);
        p.hearing_radius = 60; p.vision_radius = 250; p.cone_angle = PI / 3; p.max_age = 0;
        p.resting = true;
        predators.push_back(p);
        predators_dirty = true;
    }

    // ------------------------------------------------------------ movement
    bool in_obstacle(double px, double py, double radius, const std::vector<int32_t>& obs) const {
        for (int32_t i : obs) {
            const Obstacle& o = obstacles[i];
            double left = o.x - radius, right = o.x + o.w + radius;
            double top = o.y - radius, bottom = o.y + o.h + radius;
            if ((left < px) && (px < right) && (top < py) && (py < bottom)) return true;
        }
        return false;
    }
    void keep_in_bounds(Creature& c) const {
        c.x = std::max(c.size, std::min(W - c.size, c.x));
        c.y = std::max(c.size, std::min(H - c.size, c.y));
    }
    void update_entity_position(Creature& c, double distance, bool has_dir, double direction,
                                const std::vector<int32_t>& local_obs) {
        const double walking_cost = 0.05, sprinting_cost = 0.5;
        if (distance < 0) distance = 0;
        if (distance > c.sprint_speed) distance = c.sprint_speed;
        if (c.energy < c.max_energy / 5 && distance > c.speed) distance = c.speed;
        if (distance <= c.speed) c.energy -= distance * walking_cost;
        else c.energy -= c.speed * walking_cost + (distance - c.speed) * sprinting_cost;
        direction = has_dir ? c.direction + direction : c.direction;
        distance *= BIOMES[biome_at(c.x, c.y)].move_penalty;
        double prev_x = c.x, prev_y = c.y;
        double new_x = prev_x + distance * np_cos(direction);
        double new_y = prev_y + distance * np_sin(direction);
        if (in_obstacle(new_x, new_y, c.size, local_obs)) {
            const double angle_step = PI / 18;
            const int max_attempts = (int)(2 * PI / angle_step);
            for (int i = 0; i < max_attempts; i++) {
                double k = (double)((i + 1) / 2);
                double test_angle = direction + angle_step * k * ((i % 2) ? -1.0 : 1.0);
                double tx = prev_x + distance * np_cos(test_angle);
                double ty = prev_y + distance * np_sin(test_angle);
                if (!in_obstacle(tx, ty, c.size, local_obs)) { c.x = tx; c.y = ty; break; }
            }
        } else {
            c.x = new_x; c.y = new_y;
        }
        keep_in_bounds(c);
    }
    static void update_entity_direction(Creature& c, double turn) {
        c.direction += turn;
        c.energy -= std::min(PI, std::fabs(turn)) / (2 * PI);
    }

    // ------------------------------------------------------------ perception
    // compute_visibility(): fills ring (agent position + sorted polygon, closed) and hit edges.
    void compute_visibility(double x, double y, double dir, double cone, double vr,
                            const std::vector<int32_t>& rank, int ci, std::vector<int32_t>& hit_edges,
                            double& minx, double& maxx, double& miny, double& maxy) {
        const double eps = 1e-3;
        rays.clear();
        double vr2 = std::pow(vr, 2.0);
        double half = cone / 2;
        // Corners whose direction is > half + 0.01 rad off the view axis cannot yield a
        // ray (offsets are only 0.001), so the exact test below is skipped for them.
        const double cd = np_cos(dir), sd = np_sin(dir);
        const bool use_cone_filter = half + 0.01 < PI / 2;
        const double cos_lim = use_cone_filter ? np_cos(half + 0.01) : -2.0;
        cdx.clear(); cdy.clear();
        const int gx0 = cell_x(x - vr - 1), gx1 = cell_x(x + vr + 1);
        const int gy0 = cell_y(y - vr - 1), gy1 = cell_y(y + vr + 1);
        for (int gx = gx0; gx <= gx1; gx++) {
            for (int gy = gy0; gy <= gy1; gy++) {
                for (const CornerRef& cr : corner_cells[(size_t)gx * GNY + gy]) {
                    if (rank[cr.edge] < 0) continue;  // corner of an edge outside the 3x3 chunks
                    double dx = cr.x - x, dy = cr.y - y;
                    double d2 = dx * dx + dy * dy;
                    if (!(d2 <= vr2)) continue;
                    if (use_cone_filter) {
                        double dot = dx * cd + dy * sd;
                        if (dot < cos_lim * std::sqrt(d2) - 1e-9 * (std::fabs(dx) + std::fabs(dy))) continue;
                    }
                    cdx.push_back(dx); cdy.push_back(dy);
                }
            }
        }
        cbase.resize(cdx.size());
        vatan2(cdy.data(), cdx.data(), cbase.data(), cdx.size());
        for (size_t i = 0; i < cbase.size(); i++) {
            const double offs[3] = {-eps, 0.0, eps};
            for (double off : offs) {
                double a = cbase[i] + off;
                double rel = py_mod(a - dir + PI, TWO_PI) - PI;
                if (std::fabs(rel) <= half) rays.push_back(a);
            }
        }
        rays.push_back(dir - cone / 2);
        rays.push_back(dir + cone / 2);
        rays.push_back(dir - cone / 4);
        rays.push_back(dir + cone / 4);
        rays.push_back(dir);
        std::sort(rays.begin(), rays.end());
        rays.erase(std::unique(rays.begin(), rays.end()), rays.end());

        // Edges that cannot be hit within the vision radius never change the result.
        tmp_keys.clear();
        e_vx.clear(); e_vy.clear(); e_a1.clear(); e_b1.clear(); e_nt.clear();
        const double reach = vr + 1.0;
        for (int32_t k : cell_candidates(cell_x(x), cell_y(y), ci, rank, vr)) {
            const Edge& e = edges[k];
            double bx0 = std::min(e.x1, e.x2), bx1 = std::max(e.x1, e.x2);
            double by0 = std::min(e.y1, e.y2), by1 = std::max(e.y1, e.y2);
            double ddx = x < bx0 ? bx0 - x : (x > bx1 ? x - bx1 : 0.0);
            double ddy = y < by0 ? by0 - y : (y > by1 ? y - by1 : 0.0);
            if (ddx > reach || ddy > reach || ddx * ddx + ddy * ddy > reach * reach) continue;
            tmp_keys.push_back(k);
            double vx = e.x2 - e.x1, vy = e.y2 - e.y1, a1 = e.x1 - x, b1 = e.y1 - y;
            e_vx.push_back(vx); e_vy.push_back(vy); e_a1.push_back(a1); e_b1.push_back(b1);
            e_nt.push_back(-vy * a1 + vx * b1);  // numerator of t, same for every ray
        }

        size_t nr = rays.size();
        ring_x.resize(nr + 2); ring_y.resize(nr + 2);
        poly_key.resize(nr);
        poly_order.resize(nr);
        vpx.resize(nr); vpy.resize(nr);
        double* px = vpx.data();
        double* py = vpy.data();
        rcos.resize(nr); rsin.resize(nr);
        vcos(rays.data(), rcos.data(), nr);
        vsin(rays.data(), rsin.data(), nr);
        hit_edges.clear();
        for (size_t r = 0; r < nr; r++) {
            double rdx = rcos[r], rdy = rsin[r];
            double best = INF;
            int32_t best_k = -1;
            const size_t ne = tmp_keys.size();
            // Divisions run only when cheap bounds cannot already rule the edge out:
            // |nt| > best*|det|*(1+2^-45) implies fl(nt/det) >= best, and
            // |nu| > |det|*(1+2^-45) implies fl(nu/det) > 1 (rounding error is ~2^-52).
            const double SLACK = 1.0 + 0x1p-45;
            for (size_t q = 0; q < ne; q++) {
                double det = -rdx * e_vy[q] + rdy * e_vx[q];
                if (!(std::fabs(det) >= 1e-8)) continue;
                const bool dneg = det < 0;
                const double adet = std::fabs(det);
                double nt = e_nt[q];
                if (nt != 0 && ((nt < 0) != dneg)) continue;  // t < 0
                if (std::fabs(nt) > best * adet * SLACK) continue;
                double nu = -rdy * e_a1[q] + rdx * e_b1[q];
                if (nu != 0 && ((nu < 0) != dneg)) continue;  // u < 0
                if (std::fabs(nu) > adet * SLACK) continue;   // u > 1
                double t = nt / det;
                if (!(t < best)) continue;  // an equal or larger t never replaces the first minimum
                double u = nu / det;
                if (!((t >= 0) && (u >= 0) && (u <= 1))) continue;
                best = t; best_k = tmp_keys[q];
            }
            double min_t = best < vr ? best : vr;  // np.minimum
            px[r] = x + rdx * min_t;
            py[r] = y + rdy * min_t;
            if (min_t < vr) hit_edges.push_back(best_k);
        }
        pdy.resize(nr); pdx.resize(nr); pang.resize(nr);
        for (size_t r = 0; r < nr; r++) { pdy[r] = py[r] - y; pdx[r] = px[r] - x; }
        vatan2(pdy.data(), pdx.data(), pang.data(), nr);
        for (size_t r = 0; r < nr; r++) {
            double a = pang[r] - dir;
            poly_key[r] = py_mod(a + PI, TWO_PI) - PI;
            poly_order[r] = (int)r;
        }
        // stable insertion sort (list.sort is stable; n is ~15)
        for (size_t i = 1; i < nr; i++) {
            int v = poly_order[i];
            size_t j = i;
            while (j > 0 && poly_key[v] < poly_key[poly_order[j - 1]]) { poly_order[j] = poly_order[j - 1]; j--; }
            poly_order[j] = v;
        }
        ring_x[0] = x; ring_y[0] = y;
        minx = maxx = x; miny = maxy = y;
        for (size_t r = 0; r < nr; r++) {
            double vx = px[poly_order[r]], vy = py[poly_order[r]];
            ring_x[r + 1] = vx; ring_y[r + 1] = vy;
            minx = std::min(minx, vx); maxx = std::max(maxx, vx);
            miny = std::min(miny, vy); maxy = std::max(maxy, vy);
        }
        ring_x[nr + 1] = x; ring_y[nr + 1] = y;
    }

    struct Target { double x, y, direction; int64_t id; };

    void process_objects(const Creature& c, const std::vector<Target>& objs, int tag, bool include_direction,
                         bool include_id, double minx, double maxx, double miny, double maxy,
                         std::vector<Obs>& out) {
        if (objs.empty()) return;
        size_t n = objs.size();
        pdist.resize(n); pang2.resize(n); pnear.resize(n); pvis.resize(n);
        double* dist = pdist.data();
        double* ang = pang2.data();
        char* nearby = pnear.data();
        char* visible = pvis.data();
        double half = c.cone_angle / 2.0;
        // Objects clearly beyond both radii can be neither heard nor seen; the
        // margin keeps this exact despite rounding in hypot.
        double reach = std::max(c.hearing_radius, c.vision_radius) + 1.0;
        double reach2 = reach * reach;
        const double cd = np_cos(c.direction), sd = np_sin(c.direction);
        const bool use_cone_filter = half + 0.01 < PI / 2;
        const double cos_lim = use_cone_filter ? np_cos(half + 0.01) : -2.0;
        oidx.clear(); odx.clear(); ody.clear();
        for (size_t i = 0; i < n; i++) {
            nearby[i] = 0; visible[i] = 0;
            double dx = objs[i].x - c.x, dy = objs[i].y - c.y;
            if (dx * dx + dy * dy > reach2) continue;
            oidx.push_back(i); odx.push_back(dx); ody.push_back(dy);
        }
        size_t m = oidx.size();
        ohyp.resize(m);
        vhypot(odx.data(), ody.data(), ohyp.data(), m);
        size_t k = 0;  // compact to the objects whose angle is needed
        for (size_t j = 0; j < m; j++) {
            size_t i = oidx[j];
            double dx = odx[j], dy = ody[j];
            dist[i] = ohyp[j];
            if (!(dist[i] <= c.hearing_radius) && use_cone_filter &&
                dx * cd + dy * sd < cos_lim * std::sqrt(dx * dx + dy * dy) - 1e-9 * (std::fabs(dx) + std::fabs(dy)))
                continue;  // outside the cone and not heard
            oidx[k] = i; odx[k] = dx; ody[k] = dy; k++;
        }
        oang.resize(k);
        vatan2(ody.data(), odx.data(), oang.data(), k);
        for (size_t j = 0; j < k; j++) {
            size_t i = oidx[j];
            double a = oang[j] - c.direction;
            ang[i] = py_mod(a + PI, TWO_PI) - PI;
            nearby[i] = dist[i] <= c.hearing_radius;
            visible[i] = !nearby[i] && (dist[i] <= c.vision_radius) && (std::fabs(ang[i]) <= half);
        }
        auto emit = [&](size_t i) {
            Obs o;
            o.type = tag; o.distance = dist[i]; o.angle = ang[i];
            o.has_rel_dir = include_direction; o.has_id = include_id;
            o.rel_dir = 0; o.id = 0;
            if (include_direction) {
                double a = np_atan2(c.y - objs[i].y, c.x - objs[i].x) - objs[i].direction;
                o.rel_dir = py_mod(a + PI, TWO_PI) - PI;
            }
            if (include_id) o.id = objs[i].id;
            out.push_back(o);
        };
        for (size_t i = 0; i < n; i++)
            if (nearby[i]) emit(i);
        for (size_t i = 0; i < n; i++)
            if (visible[i] && polygon_contains(ring_x, ring_y, minx, maxx, miny, maxy, objs[i].x, objs[i].y)) emit(i);
    }

    std::vector<Target> t_fruits, t_agents, t_preds, t_trees;
    std::vector<int32_t> hit_edges, keys_buf;
    PySetEmu s_agents, s_fruits, s_trees, s_preds;
    std::vector<int32_t> key_to_index_agent, key_to_index_fruit, key_to_index_tree, key_to_index_pred;

    template <class T>
    void index_keys(const std::vector<T>& items, std::vector<int32_t>& map) {
        if (map.size() < (size_t)next_serial) map.resize((size_t)next_serial + 1024, -1);
        for (size_t i = 0; i < items.size(); i++) map[items[i].key] = (int32_t)i;
    }

    // Creature.observe(); agents/fruits/trees/predators given as ordered sets.
    void observe(const Creature& c, const PySetEmu* agents_set, const PySetEmu* fruits_set,
                 const PySetEmu* trees_set, const PySetEmu* preds_set, std::vector<Obs>& out) {
        out.clear();
        double cos_dir = np_cos(-c.direction), sin_dir = np_sin(-c.direction);
        double minx, maxx, miny, maxy;
        long ccx = to_chunk(c.x), ccy = to_chunk(c.y);
        ensure_local_static(ccx, ccy);
        const std::vector<int32_t>& rank = local_edge_rank[chunk_index(ccx, ccy)];
        compute_visibility(c.x, c.y, c.direction, c.cone_angle, c.vision_radius, rank, chunk_index(ccx, ccy), hit_edges, minx, maxx, miny, maxy);

        if (fruits_set && fruits_set->used) {
            t_fruits.clear();
            fruits_set->for_each([&](int32_t k) { const Fruit& f = fruits[key_to_index_fruit[k]]; t_fruits.push_back({f.x, f.y, 0, 0}); });
            process_objects(c, t_fruits, 0, false, false, minx, maxx, miny, maxy, out);
        }
        if (agents_set && agents_set->used) {
            t_agents.clear();
            agents_set->for_each([&](int32_t k) {
                if (k == c.key) return;
                const Creature& a = agents[key_to_index_agent[k]];
                t_agents.push_back({a.x, a.y, a.direction, a.id});
            });
            process_objects(c, t_agents, 1, true, true, minx, maxx, miny, maxy, out);
        }
        if (preds_set && preds_set->used) {
            t_preds.clear();
            preds_set->for_each([&](int32_t k) {
                if (k == c.key) return;
                const Creature& p = predators[key_to_index_pred[k]];
                t_preds.push_back({p.x, p.y, p.direction, 0});
            });
            process_objects(c, t_preds, 2, true, false, minx, maxx, miny, maxy, out);
        }
        if (trees_set && trees_set->used) {
            t_trees.clear();
            trees_set->for_each([&](int32_t k) { const Tree& t = trees[key_to_index_tree[k]]; t_trees.push_back({t.x, t.y, 0, 0}); });
            process_objects(c, t_trees, 3, false, false, minx, maxx, miny, maxy, out);
        }
        for (int32_t k : hit_edges) {
            const Edge& e = edges[k];
            double dxs = e.x1 - c.x, dys = e.y1 - c.y, dxe = e.x2 - c.x, dye = e.y2 - c.y;
            Obs o;
            o.type = 4; o.has_rel_dir = false; o.has_id = false; o.distance = o.angle = o.rel_dir = 0; o.id = 0;
            o.c[0] = dxs * cos_dir - dys * sin_dir;
            o.c[1] = dxs * sin_dir + dys * cos_dir;
            o.c[2] = dxe * cos_dir - dye * sin_dir;
            o.c[3] = dxe * sin_dir + dye * cos_dir;
            out.push_back(o);
        }
    }

    // ------------------------------------------------------------ step
    struct Action { int64_t agent_id; double move_distance; bool has_dir; double move_direction; double turn_angle; bool spawn; };

    long find_agent(int64_t id) const {
        for (size_t i = 0; i < agents.size(); i++)
            if (agents[i].id == id) return (long)i;
        return -1;
    }

    void agent_step(const Action& act) {
        long idx = find_agent(act.agent_id);
        if (idx < 0) return;
        {
            Creature& a = agents[idx];
            const std::vector<int32_t>& lobs = local_obstacles(a.x, a.y);
            update_entity_position(a, act.move_distance, act.has_dir, act.move_direction, lobs);
            agents_dirty = true;
            update_entity_direction(a, act.turn_angle);
        }
        if (act.spawn && agents[idx].energy > 100) {
            Creature parent = agents[idx];
            spawn_child(parent);
            agents[idx].energy -= 100;
        }
    }

    std::vector<Obs> pred_obs;
    std::vector<Obs> obs_buf;

    void non_agent_step() {
        // ---- agents
        size_t i = 0;
        while (i < agents.size()) {
            Creature& a = agents[i];
            a.age += dt;
            a.energy -= dt * BIOMES[biome_at(a.x, a.y)].energy_drain_rate;
            if (a.energy <= 0) { kill_agent_at(i, 0); i++; continue; }
            if (a.age > a.max_age) a.energy -= 0.01 * a.age;

            refresh_grids();
            s_agents = cached_union(grid_agents, uc_agents, a.x, a.y);
            const PySetEmu& fr_set = cached_union(grid_fruits, uc_fruits, a.x, a.y);
            const PySetEmu& tr_set = cached_union(grid_trees, uc_trees, a.x, a.y);
            // an agent is never in the predator set, so that set is used as cached
            const PySetEmu& pr_set = cached_union(grid_predators, uc_preds, a.x, a.y);
            if (s_agents.lookup(a.key, a.hash) >= 0) s_agents.discard(a.key, a.hash);

            std::vector<Obs>& obs = agent_observations[a.id];
            observe(a, &s_agents, &fr_set, &tr_set, &pr_set, obs);

            if (fr_set.used) {
                // Touching does not depend on earlier meals, so collect in the engine's
                // reversed set order, apply meals in that order, then drop them.
                fr_set.to_vector(keys_buf);
                Creature& ag = agents[i];
                bool ate = false;
                for (size_t r = keys_buf.size(); r-- > 0;) {
                    Fruit& fr = fruits[key_to_index_fruit[keys_buf[r]]];
                    double dx = ag.x - fr.x, dy = ag.y - fr.y;
                    double lim = ag.size + fr.radius + 1.0;
                    if (dx * dx + dy * dy > lim * lim) continue;
                    double distance = np_hypot(dx, dy);
                    if (distance < ag.size + fr.radius) {
                        double e = ag.energy + fr.energy;
                        ag.energy = (e < ag.max_energy) ? e : ag.max_energy;
                        score += fr.energy / 1000;
                        events.push_back({2, time, ag.id, fr.age, fr.energy});
                        fr.removed = true;
                        ate = true;
                    }
                }
                if (ate) {
                    fruits.erase(std::remove_if(fruits.begin(), fruits.end(), [](const Fruit& q) { return q.removed; }), fruits.end());
                    fruits_dirty = true;
                }
            }
            i++;
        }

        // ---- predators
        for (size_t p = 0; p < predators.size(); p++) {
            {
                Creature& pr = predators[p];
                if (pr.resting) {
                    if (pr.energy > pr.max_energy * 0.5) pr.resting = false;
                    else { pr.energy += dt * 30; continue; }
                }
            }
            refresh_grids();
            s_agents = cached_union(grid_agents, uc_agents, predators[p].x, predators[p].y);
            const std::vector<int32_t>& lobs = local_obstacles(predators[p].x, predators[p].y);
            observe(predators[p], &s_agents, nullptr, nullptr, nullptr, pred_obs);
            predator_act(predators[p], pred_obs, lobs);

            refresh_grids();
            Creature& pr = predators[p];
            s_agents = cached_union(grid_agents, uc_agents, pr.x, pr.y);
            s_agents.to_vector(keys_buf);
            if (!keys_buf.empty()) {
                std::vector<int32_t> touching;
                for (size_t q = 0; q < keys_buf.size(); q++) {
                    const Creature& ag = agents[key_to_index_agent[keys_buf[q]]];
                    double dx = ag.x - pr.x, dy = ag.y - pr.y;
                    if (np_hypot(dx, dy) < pr.size + ag.size) touching.push_back(keys_buf[q]);
                }
                for (size_t q = touching.size(); q-- > 0;) {
                    long ai = -1;
                    for (size_t z = 0; z < agents.size(); z++)
                        if (agents[z].key == touching[q]) { ai = (long)z; break; }
                    if (ai < 0) continue;
                    double ae = agents[ai].energy;
                    double e = pr.energy + ae;
                    pr.energy = (e < pr.max_energy) ? e : pr.max_energy;
                    score -= ae / 100;
                    kill_agent_at((size_t)ai, 1);
                }
            }
            if (pr.energy <= 0) pr.resting = true;
        }

        // ---- fruits
        size_t f = 0;
        while (f < fruits.size()) {
            Fruit& fr = fruits[f];
            if (fr.age > 100) {
                fruits.erase(fruits.begin() + f);
                fruits_dirty = true;
                f++;
                continue;
            }
            double amount = 2 * dt;
            fr.age += amount;
            if (fr.energy < 60) { fr.energy += amount; fr.radius += amount * 0.1; }
            f++;
        }

        // ---- trees
        double half_trees = trees.size() / 2.0;
        double chance = (100.0 / (half_trees > 1 ? half_trees : 1.0)) * dt;
        chance *= std::pow(0.5, time / 300);
        if (rng.random() < chance) spawn_tree();

        size_t t = 0;
        while (t < trees.size()) {
            Tree& tr = trees[t];
            tree_grow(tr, 1 * dt);
            if (tr.age > 50 + (100 - 50) * std::pow(rng.random(), 0.5)) {
                trees.erase(trees.begin() + t);
                trees_dirty = true;
                t++;
                continue;
            } else if (tr.age >= 20) {
                if (rng.random() < dt * BIOMES[biome_at(tr.x, tr.y)].fruit_spawn_rate) {
                    Tree copy = tr;
                    spawn_fruit_around_tree(copy);
                }
            }
            t++;
        }

        time += dt;
        score += dt;

        double np = (double)std::max<size_t>(1, predators.size());
        double pchance = (1 / np) * dt * time * 0.0001;
        if (pchance > rng.random()) spawn_predator();
    }

    void predator_act(Creature& pr, const std::vector<Obs>& obs, const std::vector<int32_t>& lobs) {
        bool do_move = false, do_turn = false, move_has_dir = false;
        double move_dist = 0, move_dir = 0, turn = 0;
        const Obs* closest = nullptr;
        for (const Obs& o : obs)
            if (o.type == 1 && (!closest || o.distance < closest->distance)) closest = &o;
        bool edges_seen = false;
        for (const Obs& o : obs)
            if (o.type == 4) { edges_seen = true; break; }
        pr.dbg_tdist = closest ? closest->distance : -1.; pr.dbg_look = closest ? closest->rel_dir : 0.; pr.dbg_ang = closest ? closest->angle : 0.;
        pr.dbg_mode = closest ? ((std::fabs(closest->rel_dir) > PI / 2 || closest->distance < pr.hearing_radius * 1.5) ? 1 : 2) : (edges_seen ? 3 : 4);
        if (closest) {
            double d = closest->distance, ang = closest->angle, look = closest->rel_dir;
            if (std::fabs(look) > PI * 1 / 2 || d < pr.hearing_radius * 1.5) {
                double ts = std::max(-0.3, std::min(0.3, ang * 0.5));
                if (std::fabs(ang) > 0.05) {
                    do_turn = true; turn = ts;
                    do_move = true; move_dist = std::min(pr.sprint_speed, d); move_has_dir = true; move_dir = ts;
                } else {
                    do_move = true; move_dist = std::min(pr.sprint_speed, d); move_has_dir = true; move_dir = ang;
                }
            } else {
                double sgn = look > 0 ? 1.0 : (look < 0 ? -1.0 : 0.0);
                double pivot = -sgn;
                double mdir = ang + pivot * PI * 1 / 4;
                do_move = true; move_dist = pr.sprint_speed; move_has_dir = true; move_dir = mdir;
                double dx_move = pr.sprint_speed * np_cos(mdir);
                double dy_move = pr.sprint_speed * np_sin(mdir);
                double x_agent = d * np_cos(ang);
                double y_agent = d * np_sin(ang);
                double x_new = x_agent - dx_move;
                double y_new = y_agent - dy_move;
                do_turn = true; turn = np_atan2(y_new, x_new);
            }
        } else if (edges_seen) {
            double best_x = 0, best_y = 0, best_d = INF;
            bool first = true;
            for (const Obs& o : obs) {
                if (o.type != 4) continue;
                double x1 = o.c[0], y1 = o.c[1], x2 = o.c[2], y2 = o.c[3];
                double dx = x2 - x1, dy = y2 - y1;
                double tt = (-(x1 * dx + y1 * dy)) / (dx * dx + dy * dy);
                tt = std::max(0.0, std::min(1.0, tt));
                double cx = x1 + tt * dx, cy = y1 + tt * dy;
                double h = np_hypot(cx, cy);
                if (first || h < best_d) { best_d = h; best_x = cx; best_y = cy; first = false; }
            }
            double dist = std::max(np_hypot(best_x, best_y) - pr.size, 2.0);
            double ang = np_atan2(best_y, best_x);
            ang = py_mod(ang + PI, TWO_PI) - PI;
            double ta = ang > 0 ? -PI / dist : PI / dist;
            do_turn = true; turn = ta;
            do_move = true; move_dist = pr.speed; move_has_dir = true; move_dir = ta;
        } else {
            do_turn = true; turn = rng.uniform(-0.1, 0.1);
            do_move = true; move_dist = pr.speed; move_has_dir = false;
        }
        if (do_move) {
            update_entity_position(pr, move_dist, move_has_dir, move_dir, lobs);
            predators_dirty = true;
        }
        if (do_turn) update_entity_direction(pr, turn);
    }
};

#include "_npolicy.hpp"

// ----------------------------------------------------------------------------
// Python bindings
// ----------------------------------------------------------------------------
double g_pred_life = 0.;   // nightsim test: a predator older than this is 'trapped' (parked asleep in a corner)
struct EngineObject {
    PyObject_HEAD
    Engine* eng;
    orchard::Policy* pol;
};

// Policy input exactly as the state dict the Python policy receives.
std::vector<orchard::AState> policy_states(Engine* e) {
    static const std::vector<Obs> empty;
    std::vector<orchard::AState> out;
    out.reserve(e->agents.size());
    for (const Creature& a : e->agents) {
        auto it = e->agent_observations.find(a.id);
        orchard::AState s;
        s.aid = a.id; s.obs = it == e->agent_observations.end() ? &empty : &it->second;
        s.energy = a.energy; s.biome = e->biome_at(a.x, a.y); s.age = a.age; s.speed = a.speed;
        s.sprint = a.sprint_speed; s.hear = a.hearing_radius; s.cone = a.cone_angle; s.vr = a.vision_radius;
        s.max_energy = a.max_energy;
        out.push_back(s);
    }
    return out;
}

PyObject *s_type, *s_distance, *s_angle, *s_rel_dir, *s_id, *s_coords, *s_Fruit, *s_Agent, *s_Predator, *s_Tree, *s_Edge;
PyObject *s_agent_id, *s_observations, *s_energy, *s_biome, *s_age, *s_speed, *s_sprint_speed, *s_hearing_radius;
PyObject *s_vision_angle, *s_vision_range, *s_max_energy, *s_score, *s_sim_time, *s_num_agents;
PyObject *s_move_distance, *s_move_direction, *s_turn_angle, *s_spawn_agent;
PyObject* s_biome_names[5];

PyObject* num(double v, bool as_int) {
    if (as_int) return PyLong_FromLongLong((long long)v);
    return PyFloat_FromDouble(v);
}

PyObject* build_obs_list(const std::vector<Obs>& obs) {
    PyObject* list = PyList_New((Py_ssize_t)obs.size());
    if (!list) return nullptr;
    PyObject* tags[5] = {s_Fruit, s_Agent, s_Predator, s_Tree, s_Edge};
    for (size_t i = 0; i < obs.size(); i++) {
        const Obs& o = obs[i];
        PyObject* d = PyDict_New();
        PyDict_SetItem(d, s_type, tags[o.type]);
        if (o.type == 4) {
            PyObject* p0 = PyTuple_New(2);
            PyTuple_SET_ITEM(p0, 0, PyFloat_FromDouble(o.c[0])); PyTuple_SET_ITEM(p0, 1, PyFloat_FromDouble(o.c[1]));
            PyObject* p1 = PyTuple_New(2);
            PyTuple_SET_ITEM(p1, 0, PyFloat_FromDouble(o.c[2])); PyTuple_SET_ITEM(p1, 1, PyFloat_FromDouble(o.c[3]));
            PyObject* c = PyTuple_New(2);
            PyTuple_SET_ITEM(c, 0, p0); PyTuple_SET_ITEM(c, 1, p1);
            PyDict_SetItem(d, s_coords, c);
            Py_DECREF(c);
        } else {
            PyObject* v = PyFloat_FromDouble(o.distance); PyDict_SetItem(d, s_distance, v); Py_DECREF(v);
            v = PyFloat_FromDouble(o.angle); PyDict_SetItem(d, s_angle, v); Py_DECREF(v);
            if (o.has_rel_dir) { v = PyFloat_FromDouble(o.rel_dir); PyDict_SetItem(d, s_rel_dir, v); Py_DECREF(v); }
            if (o.has_id) { v = PyLong_FromLongLong(o.id); PyDict_SetItem(d, s_id, v); Py_DECREF(v); }
        }
        PyList_SET_ITEM(list, (Py_ssize_t)i, d);
    }
    return list;
}

PyObject* build_state(Engine* e) {
    PyObject* list = PyList_New((Py_ssize_t)e->agents.size());
    static const std::vector<Obs> empty;
    for (size_t i = 0; i < e->agents.size(); i++) {
        const Creature& a = e->agents[i];
        auto it = e->agent_observations.find(a.id);
        PyObject* ol = build_obs_list(it == e->agent_observations.end() ? empty : it->second);
        PyObject* d = PyDict_New();
        PyObject* v;
        v = PyLong_FromLongLong(a.id); PyDict_SetItem(d, s_agent_id, v); Py_DECREF(v);
        PyDict_SetItem(d, s_observations, ol); Py_DECREF(ol);
        v = PyFloat_FromDouble(a.energy); PyDict_SetItem(d, s_energy, v); Py_DECREF(v);
        PyDict_SetItem(d, s_biome, s_biome_names[e->biome_at(a.x, a.y)]);
        v = PyFloat_FromDouble(a.age); PyDict_SetItem(d, s_age, v); Py_DECREF(v);
        v = num(a.speed, a.speed_int); PyDict_SetItem(d, s_speed, v); Py_DECREF(v);
        v = num(a.sprint_speed, a.sprint_int); PyDict_SetItem(d, s_sprint_speed, v); Py_DECREF(v);
        v = num(a.hearing_radius, a.hearing_int); PyDict_SetItem(d, s_hearing_radius, v); Py_DECREF(v);
        v = PyFloat_FromDouble(a.cone_angle); PyDict_SetItem(d, s_vision_angle, v); Py_DECREF(v);
        v = num(a.vision_radius, a.vision_int); PyDict_SetItem(d, s_vision_range, v); Py_DECREF(v);
        v = num(a.max_energy, a.max_energy_int); PyDict_SetItem(d, s_max_energy, v); Py_DECREF(v);
        PyList_SET_ITEM(list, (Py_ssize_t)i, d);
    }
    PyObject* res = PyDict_New();
    PyObject* v = PyFloat_FromDouble(e->score); PyDict_SetItem(res, s_score, v); Py_DECREF(v);
    v = PyFloat_FromDouble(e->time); PyDict_SetItem(res, s_sim_time, v); Py_DECREF(v);
    v = PyLong_FromSize_t(e->agents.size()); PyDict_SetItem(res, s_num_agents, v); Py_DECREF(v);
    PyDict_SetItem(res, s_observations, list); Py_DECREF(list);
    return res;
}

bool get_double(PyObject* obj, PyObject* name, bool is_dict, double& out, bool* is_none = nullptr) {
    PyObject* v = is_dict ? PyDict_GetItem(obj, name) : PyObject_GetAttr(obj, name);
    if (!v) {
        if (!PyErr_Occurred()) PyErr_Format(PyExc_KeyError, "action is missing %S", name);
        return false;
    }
    if (is_none && v == Py_None) {
        *is_none = true;
        if (!is_dict) Py_DECREF(v);
        return true;
    }
    if (is_none) *is_none = false;
    out = PyFloat_AsDouble(v);
    if (!is_dict) Py_DECREF(v);
    return !(out == -1.0 && PyErr_Occurred());
}

PyObject* Engine_step(EngineObject* self, PyObject* args) {
    PyObject* actions;
    if (!PyArg_ParseTuple(args, "O", &actions)) return nullptr;
    PyObject* seq = PySequence_Fast(actions, "actions must be a sequence");
    if (!seq) return nullptr;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    std::vector<Engine::Action> acts;
    acts.reserve((size_t)n);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        PyObject *aid_obj, *act;
        if (!PyArg_ParseTuple(item, "OO", &aid_obj, &act)) { Py_DECREF(seq); return nullptr; }
        Engine::Action a;
        a.agent_id = PyLong_AsLongLong(aid_obj);
        if (a.agent_id == -1 && PyErr_Occurred()) { Py_DECREF(seq); return nullptr; }
        bool is_dict = PyDict_Check(act);
        bool dir_none = false;
        if (!get_double(act, s_move_distance, is_dict, a.move_distance) ||
            !get_double(act, s_move_direction, is_dict, a.move_direction, &dir_none) ||
            !get_double(act, s_turn_angle, is_dict, a.turn_angle)) { Py_DECREF(seq); return nullptr; }
        a.has_dir = !dir_none;
        PyObject* sp = is_dict ? PyDict_GetItem(act, s_spawn_agent) : PyObject_GetAttr(act, s_spawn_agent);
        if (!sp) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_KeyError, "spawn_agent"); Py_DECREF(seq); return nullptr; }
        int t = PyObject_IsTrue(sp);
        if (!is_dict) Py_DECREF(sp);
        if (t < 0) { Py_DECREF(seq); return nullptr; }
        a.spawn = t;
        acts.push_back(a);
    }
    Py_DECREF(seq);
    Engine* e = self->eng;
    for (const auto& a : acts) e->agent_step(a);
    e->non_agent_step();
    return build_state(e);
}

PyObject* Engine_state(EngineObject* self, PyObject*) { return build_state(self->eng); }

bool parse_params(PyObject* d, orchard::Params& P) {
    if (!d || d == Py_None) return true;
    if (!PyDict_Check(d)) { PyErr_SetString(PyExc_TypeError, "config must be a dict"); return false; }
    struct F { const char* k; double* v; };
    F fs[] = {{"cap_mult", &P.cap_mult}, {"cap_min", &P.cap_min}, {"cap_max", &P.cap_max}, {"n0", &P.n0},
              {"tree_half", &P.tree_half}, {"tree_slots", &P.tree_slots}, {"breed_reserve", &P.breed_reserve},
              {"emergency_reserve", &P.emergency_reserve}, {"ripen_wait", &P.ripen_wait}, {"sweep_rate", &P.sweep_rate},
              {"explore_radius", &P.explore_radius}, {"fit_vision", &P.fit_vision}, {"fit_hear", &P.fit_hear},
              {"fit_energy", &P.fit_energy}, {"births_per_tick", &P.births_per_tick}, {"fruit_reach", &P.fruit_reach},
              {"tree_reach", &P.tree_reach}, {"site_min", &P.site_min}, {"breed_reserve_late", &P.breed_reserve_late},
              {"reserve_t0", &P.reserve_t0}, {"reserve_t1", &P.reserve_t1}, {"dist_pen", &P.dist_pen},
              {"vo_win_fruit", &P.vo_win_fruit}, {"vo_win_far", &P.vo_win_far}, {"vo_cap", &P.vo_cap},
              {"heir_age", &P.heir_age}, {"heir_reserve", &P.heir_reserve}, {"explore_min", &P.explore_min},
              {"travel_turn", &P.travel_turn}, {"heir_slack", &P.heir_slack}, {"repost_every", &P.repost_every},
              {"switch_gain", &P.switch_gain}, {"fruit_min_wait", &P.fruit_min_wait}, {"no_eat_age", &P.no_eat_age},
              {"late_still_t", &P.late_still_t}, {"post_radius", &P.post_radius}, {"min_stay", &P.min_stay},
              {"hungry_margin", &P.hungry_margin}, {"fit_speed", &P.fit_speed}, {"explore_energy", &P.explore_energy},
              {"watch_patience", &P.watch_patience}, {"watch_reach", &P.watch_reach}, {"watch_refresh", &P.watch_refresh},
              {"select_min_young", &P.select_min_young}, {"dump_food_site", &P.dump_food_site}, {"dump_mult", &P.dump_mult},
              {"cluster_radius", &P.cluster_radius}, {"spread_weight", &P.spread_weight}, {"low_pop_reserve", &P.low_pop_reserve},
              {"lone_reach_mult", &P.lone_reach_mult}, {"old_reach", &P.old_reach}, {"rot_margin", &P.rot_margin},
              {"dump_after_t", &P.dump_after_t}, {"cap_tree_slack", &P.cap_tree_slack}, {"cap_hard_min", &P.cap_hard_min},
              {"nursery_bonus", &P.nursery_bonus}, {"late_t", &P.late_t}, {"pred_mode", &P.pred_mode}, {"merge_anchored", &P.merge_anchored}, {"hide_mode", &P.hide_mode}, {"hide_r", &P.hide_r}, {"hide_trigger", &P.hide_trigger}, {"trap_post_w", &P.trap_post_w}, {"trap_post_r", &P.trap_post_r}, {"decoy_old", &P.decoy_old}, {"decoy_e", &P.decoy_e}, {"decoy_r", &P.decoy_r}, {"evade_closest", &P.evade_closest}, {"spawn_pred_r", &P.spawn_pred_r}, {"keeper_mode", &P.keeper_mode}, {"keeper_r", &P.keeper_r}, {"keeper_reserve", &P.keeper_reserve}, {"rep_timeout", &P.rep_timeout}, {"keeper_post_w", &P.keeper_post_w}, {"keeper_post_r", &P.keeper_post_r}, {"site_dist_w", &P.site_dist_w}, {"no_spawn", &P.no_spawn}, {"fit_speed_cap", &P.fit_speed_cap}, {"oracle_trees", &P.oracle_trees}, {"oracle_r", &P.oracle_r}, {"age_infer", &P.age_infer}, {"age_fruit", &P.age_fruit}, {"dead_misses", &P.dead_misses}, {"fruit_misses", &P.fruit_misses}, {"occ_walls", &P.occ_walls}, {"vis_margin_tree", &P.vis_margin_tree}, {"vis_margin_fruit", &P.vis_margin_fruit}, {"trap_mode", &P.trap_mode}, {"wall_min_n", &P.wall_min_n}, {"trap_depth", &P.trap_depth}, {"wall_tol", &P.wall_tol}, {"wall_min_obs", &P.wall_min_obs}, {"trap_start", &P.trap_start}, {"bait_margin", &P.bait_margin}, {"bait_min_life", &P.bait_min_life}, {"bait_young_pen", &P.bait_young_pen}, {"trap_keepout", &P.trap_keepout}, {"trap_bait_fixed", &P.trap_bait_fixed}, {"guide_near", &P.guide_near}, {"guide_far", &P.guide_far}, {"guide_acq", &P.guide_acq}, {"guide_min_e", &P.guide_min_e}, {"guide_lost", &P.guide_lost}, {"guide_hand", &P.guide_hand}, {"guide_acq_sprint", &P.guide_acq_sprint}, {"guide_block_ang", &P.guide_block_ang}, {"guide_slow", &P.guide_slow}, {"guide_fastclose", &P.guide_fastclose}, {"guide_side_pen", &P.guide_side_pen}, {"bait_on_sight", &P.bait_on_sight}, {"guide_sprint_until", &P.guide_sprint_until}, {"guide_max_dist", &P.guide_max_dist}, {"guide_lane_w", &P.guide_lane_w}, {"guide_pred_lane_max", &P.guide_pred_lane_max}, {"guide_wait_max", &P.guide_wait_max}, {"guide_relay", &P.guide_relay}, {"guide_relay_min", &P.guide_relay_min}, {"guide_relay_ahead", &P.guide_relay_ahead}, {"guide_relay_r", &P.guide_relay_r}, {"guide_wallclear", &P.guide_wallclear}, {"pred_wallclear", &P.pred_wallclear}, {"refuge_mode", &P.refuge_mode}, {"refuge_r", &P.refuge_r}, {"refuge_trigger", &P.refuge_trigger}, {"refuge_leave", &P.refuge_leave}, {"refuge_slow_only", &P.refuge_slow_only}, {"refuge_post_w", &P.refuge_post_w}, {"refuge_post_r", &P.refuge_post_r}, {"refuge_clear", &P.refuge_clear}, {"refuge_sprint", &P.refuge_sprint}, {"site_safe", &P.site_safe}, {"refuge_verify", &P.refuge_verify}, {"wall_conflict", &P.wall_conflict}, {"guide_clear", &P.guide_clear}, {"pred_avoid_w", &P.pred_avoid_w}, {"pred_avoid_r", &P.pred_avoid_r}, {"pred_avoid_t", &P.pred_avoid_t}, {"child_prio", &P.child_prio}, {"sprint_floor", &P.sprint_floor}, {"sprint_floor_breed", &P.sprint_floor_breed}, {"sprint_floor_unripe", &P.sprint_floor_unripe}, {"guide_route", &P.guide_route}, {"guide_mapclear", &P.guide_mapclear}, {"guide_ctrl", &P.guide_ctrl}, {"guide_gap", &P.guide_gap}, {"guide_ctrl_acq", &P.guide_ctrl_acq}, {"guide_lag", &P.guide_lag}, {"guide_chase_cos", &P.guide_chase_cos}, {"guide_pv", &P.guide_pv}, {"guide_pv_near", &P.guide_pv_near}, {"guide_pv_far", &P.guide_pv_far}, {"guide_pv_dT", &P.guide_pv_dT}, {"guide_plan", &P.guide_plan}, {"guide_safe", &P.guide_safe}, {"guide_keep", &P.guide_keep}, {"guide_sprint_pen", &P.guide_sprint_pen}, {"guide_chased", &P.guide_chased}, {"guide_chase_r", &P.guide_chase_r}, {"guide_release", &P.guide_release}, {"guide_lead_sprint", &P.guide_lead_sprint}, {"test_freeze", &P.test_freeze}, {"pred_r", &P.pred_r}, {"pred_sprint_r", &P.pred_sprint_r}, {"pred_face", &P.pred_face}, {"pred_face_r", &P.pred_face_r}, {"pred_share", &P.pred_share}, {"pred_dodge_r", &P.pred_dodge_r}, {"pred_dodge_ang", &P.pred_dodge_ang}, {"pred_dodge_hold", &P.pred_dodge_hold}, {"pred_dodge_hold_face", &P.pred_dodge_hold_face}, {"l_fruit_reach", &P.l_fruit_reach}, {"l_tree_reach", &P.l_tree_reach}, {"l_watch_reach", &P.l_watch_reach}, {"l_explore_energy", &P.l_explore_energy}, {"l_cap_min", &P.l_cap_min}, {"l_cap_mult", &P.l_cap_mult}, {"l_cap_tree_slack", &P.l_cap_tree_slack}, {"l_cap_hard_min", &P.l_cap_hard_min}, {"l_sweep_rate", &P.l_sweep_rate}, {"l_watch_patience", &P.l_watch_patience}, {"l_explore_radius", &P.l_explore_radius}, {"l_old_reach", &P.l_old_reach}, {"l_dist_pen", &P.l_dist_pen}, {"l_births_per_tick", &P.l_births_per_tick}, {"l_emergency_reserve", &P.l_emergency_reserve}, {"l_low_pop_reserve", &P.l_low_pop_reserve}};
    for (F& f : fs) {
        PyObject* v = PyDict_GetItemString(d, f.k);
        if (!v) continue;
        double x = PyFloat_AsDouble(v);
        if (x == -1.0 && PyErr_Occurred()) return false;
        *f.v = x;
    }
    struct B { const char* k; bool* v; };
    B bs[] = {{"idle_sweep", &P.idle_sweep}, {"extra_old", &P.extra_old}, {"cull", &P.cull}, {"heir_select", &P.heir_select},
              {"heir_at_food", &P.heir_at_food}, {"old_eat_last", &P.old_eat_last}, {"heir_needs_site", &P.heir_needs_site}};
    for (B& b : bs) {
        PyObject* v = PyDict_GetItemString(d, b.k);
        if (!v) continue;
        int t = PyObject_IsTrue(v);
        if (t < 0) return false;
        *b.v = t;
    }
    PyObject* fm = PyDict_GetItemString(d, "feed_mode");
    if (fm) {
        const char* sv = PyUnicode_AsUTF8(fm);
        if (!sv) return false;
        P.feed_breed = std::string(sv) == "breed";
    }
    return true;
}

PyObject* Engine_policy_init(EngineObject* self, PyObject* args) {
    PyObject *key_obj, *cfg = nullptr;
    if (!PyArg_ParseTuple(args, "O|O", &key_obj, &cfg)) return nullptr;
    PyObject* seq = PySequence_Fast(key_obj, "seed_key must be a sequence");
    if (!seq) return nullptr;
    std::vector<uint32_t> key;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++) key.push_back((uint32_t)PyLong_AsUnsignedLong(PySequence_Fast_GET_ITEM(seq, i)));
    Py_DECREF(seq);
    if (PyErr_Occurred()) return nullptr;
    if (key.empty()) key.push_back(0);
    orchard::Params P;
    if (!parse_params(cfg, P)) return nullptr;
    delete self->pol;
    self->pol = new orchard::Policy(key, P);
    if (getenv("NIGHT_POLLOG")) self->pol->dbg_log = true;
    if (getenv("NIGHT_REFLOG")) {   // log every predator kill of an agent that was heading to / holding in a refuge: belief vs truth
        orchard::Policy* pol = self->pol; Engine* eng = self->eng;
        eng->on_kill = [pol, eng](const Creature& a, int cause) {
            if (cause != 1 || !pol->minds.has(a.id)) return;
            auto& m = *pol->minds.at(a.id);
            if (m.hide_idx < 0 || !pol->groups.has(m.group)) return;
            auto& g = *pol->groups.at(m.group);
            if (m.hide_idx >= (int)g.sites.size()) return;
            const auto& st = g.sites[m.hide_idx];
            double bx = m.pose->p.x, by = m.pose->p.y;
            double err = std::hypot(bx - a.x, by - a.y);
            double dtg = std::hypot(a.x - st.goal.x, a.y - st.goal.y), dbg = std::hypot(bx - st.goal.x, by - st.goal.y);
            double dtm = std::hypot(a.x - st.mouth.x, a.y - st.mouth.y);
            // nearest predator (truth)
            double dp = 1e9; for (auto& p : eng->predators) dp = std::min(dp, std::hypot(p.x - a.x, p.y - a.y));
            fprintf(stderr, "REFKILL t=%.1f id=%lld in=%d err=%.1f dtrue_goal=%.1f dbelief_goal=%.1f dtrue_mouth=%.1f gap=%.1f rear_ok=%d dpred=%.1f npred=%zu\n",
                    eng->time, (long long)a.id, m.refuge_in ? 1 : 0, err, dtg, dbg, dtm, st.gap, st.rear_ok ? 1 : 0, dp, eng->predators.size());
        };
    } else if (getenv("NIGHT_DEATHS")) {   // per-death record for the failure diagnosis (all causes)
        orchard::Policy* pol = self->pol; Engine* eng = self->eng;
        eng->on_kill = [pol, eng](const Creature& a, int cause) {
            Engine::DeathRec r{}; r.cause = cause; r.t = eng->time; r.age = a.age; r.e = a.energy; r.maxe = a.max_energy; r.speed = a.speed; r.sprint = a.sprint_speed; r.x = a.x; r.y = a.y;
            r.dpred = 1e9; r.npred150 = 0; r.prest = 0;
            for (auto& p : eng->predators) { double d = std::hypot(p.x - a.x, p.y - a.y); if (d < 150.) r.npred150++; if (d < r.dpred) { r.dpred = d; r.prest = p.resting ? 1 : 0; } }
            r.nearwall = eng->is_position_free(a.x - 15., a.y - 15., 30., 30.) ? 0 : 1;
            r.pop = (int)eng->agents.size(); r.npred = (int)eng->predators.size();
            r.evading = 0; r.old = 0; r.haspost = 0;
            if (pol->minds.has(a.id)) { auto& m = *pol->minds.at(a.id); r.evading = (eng->time - m.evade_t < 1.0) ? 1 : 0; r.old = m.old ? 1 : 0; r.haspost = m.has_post ? 1 : 0; }
            if (eng->death_log.size() < 20000) eng->death_log.push_back(r);
        };
    } else self->eng->on_kill = nullptr;
    if (cfg && PyDict_Check(cfg) && PyDict_GetItemString(cfg, "_debug_merge")) self->pol->debug_merge = true;
    Py_RETURN_NONE;
}

PyObject* acts_to_py(const std::vector<orchard::Act>& acts) {
    PyObject* list = PyList_New((Py_ssize_t)acts.size());
    for (size_t i = 0; i < acts.size(); i++) {
        const orchard::Act& a = acts[i];
        PyList_SET_ITEM(list, (Py_ssize_t)i, Py_BuildValue("(LdddN)", (long long)a.aid, a.dist, a.direction, a.turn, PyBool_FromLong(a.spawn)));
    }
    return list;
}

// Decisions of the native policy for the current state (does not step).
PyObject* Engine_policy_act(EngineObject* self, PyObject*) {
    if (!self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    Engine* e = self->eng;
    auto acts = self->pol->call(policy_states(e), e->time);
    return acts_to_py(acts);
}

// Debug view of the native policy's per-agent memory (minds in dict order).
PyObject* Engine_policy_minds(EngineObject* self, PyObject*) {
    if (!self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    PyObject* list = PyList_New(0);
    self->pol->minds.each([&](const int64_t& aid, orchard::MindP& m) {
        PyObject* t = Py_BuildValue("(LLdddOOOOOLddLLd)", (long long)aid, (long long)m->group, m->pose->p.x, m->pose->p.y, m->pose->theta,
                                    m->has_post ? PyLong_FromLongLong(m->post) : (Py_INCREF(Py_None), Py_None),
                                    m->has_fruit ? PyLong_FromLongLong(m->fruit) : (Py_INCREF(Py_None), Py_None),
                                    m->old ? Py_True : Py_False,
                                    m->has_explore ? Py_BuildValue("(dd)", m->explore_p.x, m->explore_p.y) : (Py_INCREF(Py_None), Py_None),
                                    m->has_watch ? Py_BuildValue("(dd)", m->watch_p.x, m->watch_p.y) : (Py_INCREF(Py_None), Py_None),
                                    (long long)m->edges.size(), m->best_d, m->energy_prev, (long long)m->prev_marks.size(),
                                    (long long)m->hear_hist.size(), m->prev_pose ? m->prev_pose->theta : 0.0);
        PyList_Append(list, t); Py_DECREF(t);
    });
    return list;
}
PyObject* Engine_policy_groups(EngineObject* self, PyObject*) {
    if (!self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    PyObject* list = PyList_New(0);
    self->pol->groups.each([&](const int64_t& gid, orchard::GroupP& g) {
        PyObject* trees = PyList_New(0);
        g->trees.each([&](const int64_t& tid, orchard::TreeP& t) {
            PyObject* x = Py_BuildValue("(LddddON)", (long long)tid, t->p.x, t->p.y, t->first, t->last, t->dead ? Py_True : Py_False,
                                        PyList_New(0));
            PyList_Append(trees, x); Py_DECREF(x);
        });
        PyObject* fruits = PyList_New(0);
        g->fruits.each([&](const int64_t& fid, orchard::FruitP& f) {
            PyObject* x = Py_BuildValue("(Ldddd)", (long long)fid, f->p.x, f->p.y, f->born_lo, f->born_hi);
            PyList_Append(fruits, x); Py_DECREF(x);
        });
        PyObject* x = Py_BuildValue("(LONNLLL)", (long long)gid, g->anchored ? Py_True : Py_False, trees, fruits,
                                    (long long)g->cells.size(), (long long)g->next_tree, (long long)g->next_fruit);
        PyList_Append(list, x); Py_DECREF(x);
    });
    return list;
}

// Native loop: while agents live and time < horizon, act + step; returns after the
// step at which sim_time >= stop_at - 1e-6 (for sampling) or when the run ends.
PyObject* Engine_run_policy(EngineObject* self, PyObject* args) {
    double horizon, stop_at;
    if (!PyArg_ParseTuple(args, "dd", &horizon, &stop_at)) return nullptr;
    if (!self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    Engine* e = self->eng;
    long steps = 0; size_t peak = e->agents.size();
    Py_BEGIN_ALLOW_THREADS
    static std::unordered_map<int64_t, std::pair<double,double>> last_pose;   // jump log (debug only)
    while (!e->agents.empty() && e->time < horizon) {
        if (self->pol->dbg_log) {
            // log pose jumps > 20 units between consecutive policy calls, with the true pose
            for (const Creature& a : e->agents) {
                if (!self->pol->minds.has(a.id)) continue;
                auto& mp = self->pol->minds.at(a.id)->pose;
                auto it = last_pose.find(a.id);
                if (it != last_pose.end()) {
                    double jx = mp->p.x - it->second.first, jy = mp->p.y - it->second.second;
                    if (std::sqrt(jx*jx + jy*jy) > 20.)
                        fprintf(stderr, "[t=%.1f] JUMP agent %lld pose (%.0f,%.0f)->(%.0f,%.0f) true (%.0f,%.0f) err after %.0f\n", e->time, (long long)a.id,
                                it->second.first, it->second.second, mp->p.x, mp->p.y, a.x, a.y, std::sqrt((mp->p.x-a.x)*(mp->p.x-a.x)+(mp->p.y-a.y)*(mp->p.y-a.y)));
                }
                last_pose[a.id] = {mp->p.x, mp->p.y};
            }
        }
        if (self->pol->P.oracle_trees > 0.) {
            self->pol->oracle_p.clear(); self->pol->oracle_age.clear();
            for (const Tree& t : e->trees) { self->pol->oracle_p.push_back(orchard::P2{t.x, t.y}); self->pol->oracle_age.push_back(t.age); }
        }
        auto acts = self->pol->call(policy_states(e), e->time);
        for (const auto& a : acts) {
            Engine::Action ea{a.aid, a.dist, true, a.direction, a.turn, a.spawn};
            e->agent_step(ea);
        }
        e->non_agent_step();
        if (g_pred_life > 0.) {
            for (auto& pr : e->predators) {
                pr.age += e->dt;
                if (pr.age > g_pred_life && pr.energy > -1e11) { pr.resting = true; pr.energy = -1e12; pr.x = 12.; pr.y = 12.; e->predators_dirty = true; }
            }
        }
        steps++;
        if (e->agents.size() > peak) peak = e->agents.size();
        if (e->time >= stop_at - 1e-6) break;
    }
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(ln)", steps, (Py_ssize_t)peak);
}

PyObject* Engine_agents(EngineObject* self, PyObject*) {
    Engine* e = self->eng;
    PyObject* list = PyList_New((Py_ssize_t)e->agents.size());
    for (size_t i = 0; i < e->agents.size(); i++) {
        const Creature& a = e->agents[i];
        PyList_SET_ITEM(list, (Py_ssize_t)i, Py_BuildValue("(LdddddNNNNdNdd)", (long long)a.id, a.x, a.y, a.direction, a.age, a.energy,
                                                           num(a.max_energy, a.max_energy_int), num(a.speed, a.speed_int),
                                                           num(a.sprint_speed, a.sprint_int), num(a.hearing_radius, a.hearing_int),
                                                           a.vision_radius, PyBool_FromLong(a.vision_int), a.cone_angle, a.max_age));
    }
    return list;
}
PyObject* Engine_predators(EngineObject* self, PyObject*) {
    Engine* e = self->eng;
    PyObject* list = PyList_New((Py_ssize_t)e->predators.size());
    for (size_t i = 0; i < e->predators.size(); i++) {
        const Creature& p = e->predators[i];
        PyList_SET_ITEM(list, (Py_ssize_t)i, Py_BuildValue("(ddddN)", p.x, p.y, p.direction, p.energy, PyBool_FromLong(p.resting)));
    }
    return list;
}
PyObject* Engine_fruits(EngineObject* self, PyObject*) {
    Engine* e = self->eng;
    PyObject* list = PyList_New((Py_ssize_t)e->fruits.size());
    for (size_t i = 0; i < e->fruits.size(); i++) {
        const Fruit& f = e->fruits[i];
        PyList_SET_ITEM(list, (Py_ssize_t)i, Py_BuildValue("(Ldddddd)", (long long)f.fruit_id, f.x, f.y, f.energy, f.age, f.radius, 0.0));
    }
    return list;
}
PyObject* Engine_trees(EngineObject* self, PyObject*) {
    Engine* e = self->eng;
    PyObject* list = PyList_New((Py_ssize_t)e->trees.size());
    for (size_t i = 0; i < e->trees.size(); i++) {
        const Tree& t = e->trees[i];
        PyList_SET_ITEM(list, (Py_ssize_t)i, Py_BuildValue("(dddd)", t.x, t.y, t.radius, t.age));
    }
    return list;
}
PyObject* Engine_obstacles(EngineObject* self, PyObject*) {
    Engine* e = self->eng;
    PyObject* list = PyList_New((Py_ssize_t)e->obstacles.size());
    for (size_t i = 0; i < e->obstacles.size(); i++) {
        const Obstacle& o = e->obstacles[i];
        PyList_SET_ITEM(list, (Py_ssize_t)i, Py_BuildValue("(dddd)", o.x, o.y, o.w, o.h));
    }
    return list;
}
PyObject* Engine_pop_events(EngineObject* self, PyObject*) {
    Engine* e = self->eng;
    PyObject* list = PyList_New((Py_ssize_t)e->events.size());
    static const char* kinds[3] = {"starvation", "predator", "fruit"};
    for (size_t i = 0; i < e->events.size(); i++) {
        const Event& ev = e->events[i];
        PyList_SET_ITEM(list, (Py_ssize_t)i, Py_BuildValue("(sdLdd)", kinds[ev.kind], ev.t, (long long)ev.id, ev.age, ev.energy));
    }
    e->events.clear();
    return list;
}
PyObject* Engine_biome_map(EngineObject* self, PyObject*) {
    Engine* e = self->eng;
    return PyBytes_FromStringAndSize((const char*)e->biome.data(), (Py_ssize_t)e->biome.size());
}
PyObject* Engine_rng_state(EngineObject* self, PyObject*) {
    // (version, tuple of 625 ints, None) like random.Random.getstate()
    Engine* e = self->eng;
    PyObject* t = PyTuple_New(625);
    for (int i = 0; i < 624; i++) PyTuple_SET_ITEM(t, i, PyLong_FromUnsignedLong(e->rng.mt[i]));
    PyTuple_SET_ITEM(t, 624, PyLong_FromLong(e->rng.mti));
    return Py_BuildValue("(iNO)", 3, t, Py_None);
}
// ---- nightsim scenario hooks (tests only; never used by the policy)
PyObject* Engine_dbg_keep_agent(EngineObject* self, PyObject* args) {
    // keep only the agent with this id (or the first agent if id < 0); returns its (id, x, y)
    long long keep; if (!PyArg_ParseTuple(args, "L", &keep)) return nullptr;
    Engine* e = self->eng; if (e->agents.empty()) Py_RETURN_NONE;
    size_t k = 0; bool found = false;
    for (size_t i = 0; i < e->agents.size(); i++) if (keep < 0 || e->agents[i].id == keep) { k = i; found = true; break; }
    if (!found) Py_RETURN_NONE;
    Creature a = e->agents[k]; e->agents.clear(); e->agents.push_back(a); e->agents_dirty = true;
    return Py_BuildValue("(Ldd)", (long long)a.id, a.x, a.y);
}
PyObject* Engine_dbg_set_agent(EngineObject* self, PyObject* args) {
    // set the traits/state of an existing agent: (id, x, y, direction, energy, speed, sprint, max_energy, hearing, vision, cone, max_age)
    long long id; double x, y, d, en, sp, spr, me, he, vi, co, ma;
    if (!PyArg_ParseTuple(args, "Lddddddddddd", &id, &x, &y, &d, &en, &sp, &spr, &me, &he, &vi, &co, &ma)) return nullptr;
    Engine* e = self->eng;
    for (auto& a : e->agents) if (a.id == id) {
        a.x = x; a.y = y; a.direction = d; a.energy = en; a.speed = sp; a.sprint_speed = spr; a.max_energy = me;
        a.hearing_radius = he; a.vision_radius = vi; a.cone_angle = co; a.max_age = ma;
        e->agents_dirty = true; Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}
PyObject* Engine_dbg_add_predator(EngineObject* self, PyObject* args) {
    // (x, y, direction, energy, resting) -> True if the position is free
    double x, y, d, en; int rest;
    if (!PyArg_ParseTuple(args, "ddddp", &x, &y, &d, &en, &rest)) return nullptr;
    Engine* e = self->eng;
    if (!e->is_position_free(x, y, 10, 10)) Py_RETURN_FALSE;
    Creature p;
    p.key = (int32_t)e->next_serial; p.hash = py_hash_int(e->next_serial); e->next_serial++;
    p.x = x; p.y = y; p.size = 10; p.speed = 11; p.sprint_speed = 15;
    p.age = 0.0; p.energy = en; p.max_energy = 200.0; p.direction = d;
    p.hearing_radius = 60; p.vision_radius = 250; p.cone_angle = PI / 3; p.max_age = 0;
    p.resting = rest != 0;
    e->predators.push_back(p); e->predators_dirty = true;
    Py_RETURN_TRUE;
}
PyObject* Engine_dbg_free(EngineObject* self, PyObject* args) {
    double x, y, sz; if (!PyArg_ParseTuple(args, "ddd", &x, &y, &sz)) return nullptr;
    if (self->eng->is_position_free(x, y, sz, sz)) Py_RETURN_TRUE; Py_RETURN_FALSE;
}
PyObject* Engine_dbg_pred_life(EngineObject* self, PyObject* args) {
    double v; if (!PyArg_ParseTuple(args, "d", &v)) return nullptr; g_pred_life = v; Py_RETURN_NONE;
}
PyObject* Engine_dbg_deaths(EngineObject* self, PyObject*) {
    // diagnostics: drain the per-death log (NIGHT_DEATHS=1): (cause, t, age, e, maxe, speed, sprint, x, y, npred150, dpred, prest, nearwall, pop, npred, evading, old, haspost)
    PyObject* L = PyList_New(0);
    for (auto& r : self->eng->death_log) {
        PyObject* t = Py_BuildValue("(iddddddddidiiiiiii)", r.cause, r.t, r.age, r.e, r.maxe, r.speed, r.sprint, r.x, r.y, r.npred150, r.dpred, r.prest, r.nearwall, r.pop, r.npred, r.evading, r.old, r.haspost);
        PyList_Append(L, t); Py_DECREF(t);
    }
    self->eng->death_log.clear();
    return L;
}
PyObject* Engine_dbg_biome(EngineObject* self, PyObject* args) {
    double x, y; if (!PyArg_ParseTuple(args, "dd", &x, &y)) return nullptr;
    return PyLong_FromLong(self->eng->biome_at(x, y));
}
PyObject* Engine_dbg_walls(EngineObject* self, PyObject*) {
    // policy wall faces of anchored groups: [(gid, horiz, c, lo, hi, solid, n, n_obs, confirmed)]
    PyObject* L = PyList_New(0);
    if (!self->pol) return L;
    self->pol->groups.each([&](const int64_t& gid, orchard::GroupP& g) {
        if (!g->anchored) return;
        for (auto& w : g->walls) {
            PyObject* t = Py_BuildValue("(LidddiLLiL)", (long long)gid, w.horiz ? 1 : 0, w.c, w.lo, w.hi, (int)w.solid, (long long)w.n, (long long)w.n_obs, self->pol->confirmed(w) ? 1 : 0, (long long)w.n_conf);
            PyList_Append(L, t); Py_DECREF(t);
        }
    });
    return L;
}
PyObject* Engine_dbg_sites(EngineObject* self, PyObject*) {
    // policy trap sites of anchored groups: [(gid, goal_x, goal_y, mouth_x, mouth_y, out_x, out_y, overlap, gap, rear_ok, n_walls)]
    PyObject* L = PyList_New(0);
    if (!self->pol) return L;
    self->pol->groups.each([&](const int64_t& gid, orchard::GroupP& g) {
        for (auto& st : g->sites) {
            int64_t conf = 0; for (auto& w : g->walls) if (self->pol->confirmed(w)) conf++;
            PyObject* t = Py_BuildValue("(LdddddddddL)", (long long)gid, st.goal.x, st.goal.y, st.mouth.x, st.mouth.y, st.out.x, st.out.y,
                                        st.overlap, st.gap, st.rear_ok ? 1.0 : 0.0, (long long)conf);
            PyList_Append(L, t); Py_DECREF(t);
        }
    });
    return L;
}
PyObject* Engine_dbg_pred_blocked(EngineObject* self, PyObject* args) {
    // True if a predator (r=10) cannot stand at (x, y)
    double x, y; if (!PyArg_ParseTuple(args, "dd", &x, &y)) return nullptr;
    Engine* e = self->eng;
    for (auto& o : e->obstacles) {
        if ((o.x - 10 < x) && (x < o.x + o.w + 10) && (o.y - 10 < y) && (y < o.y + o.h + 10)) Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}
PyObject* Engine_dbg_keep_agents(EngineObject* self, PyObject* args) {
    // keep the first n agents; returns [(id, x, y)]
    long long n; if (!PyArg_ParseTuple(args, "L", &n)) return nullptr;
    Engine* e = self->eng;
    if ((long long)e->agents.size() > n) e->agents.resize((size_t)n);
    e->agents_dirty = true;
    PyObject* L = PyList_New(0);
    for (auto& a : e->agents) { PyObject* t = Py_BuildValue("(Ldd)", (long long)a.id, a.x, a.y); PyList_Append(L, t); Py_DECREF(t); }
    return L;
}
PyObject* Engine_dbg_load_walls(EngineObject* self, PyObject*) {
    // tests: give every anchored policy group the true obstacle faces as confirmed walls
    if (!self->pol) Py_RETURN_NONE;
    Engine* e = self->eng; long long n = 0;
    self->pol->groups.each([&](const int64_t&, orchard::GroupP& g) {
        if (!g->anchored) return;
        g->walls.clear();
        for (auto& o : e->obstacles) {
            auto mk = [&](bool horiz, double c, double lo, double hi, double solid) {
                orchard::Group::Wall w{horiz, c, lo, hi, solid, e->time, 100};
                w.cs = {c}; w.los = {lo}; w.his = {hi}; w.obs1 = 0; w.obs2 = 1; w.n_obs = 2;
                g->walls.push_back(w);
            };
            mk(true, o.y, o.x, o.x + o.w, +1.);          // top face: solid below (+y)
            mk(true, o.y + o.h, o.x, o.x + o.w, -1.);    // bottom face: solid above (-y)
            mk(false, o.x, o.y, o.y + o.h, +1.);         // left face: solid to +x
            mk(false, o.x + o.w, o.y, o.y + o.h, -1.);   // right face: solid to -x
        }
        g->sites_t = -1e9; n++;
    });
    return PyLong_FromLongLong(n);
}
PyObject* Engine_dbg_freeze(EngineObject* self, PyObject* args) {
    // tests: freeze these agent ids (policy plans zero for them)
    PyObject* seq; if (!PyArg_ParseTuple(args, "O", &seq)) return nullptr;
    if (!self->pol) Py_RETURN_NONE;
    self->pol->frozen.clear();
    PyObject* it = PySequence_Fast(seq, "ids"); if (!it) return nullptr;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(it); i++) self->pol->frozen.insert(PyLong_AsLongLong(PySequence_Fast_GET_ITEM(it, i)));
    Py_DECREF(it); Py_RETURN_NONE;
}
PyObject* Engine_dbg_roles(EngineObject* self, PyObject*) {
    // tests: per anchored group: (gid, has_trap, bait, rep, guide, guide_state, n_retired)
    PyObject* L = PyList_New(0);
    if (!self->pol) return L;
    self->pol->groups.each([&](const int64_t& gid, orchard::GroupP& g) {
        double gx = 0, gy = 0; if (g->guide >= 0 && self->pol->minds.has(g->guide)) { auto& mp = self->pol->minds.at(g->guide)->pose; gx = mp->p.x; gy = mp->p.y; }
        double bx = 0, by = 0; int64_t bb = g->bait >= 0 ? g->bait : g->rep;
        if (bb >= 0 && self->pol->minds.has(bb)) { auto& mp = self->pol->minds.at(bb)->pose; bx = mp->p.x; by = mp->p.y; }
        PyObject* t = Py_BuildValue("(LiLLLiLdddddddd)", (long long)gid, g->has_trap ? 1 : 0, (long long)g->bait, (long long)g->rep, (long long)g->guide, g->guide_state, (long long)g->retired.size(),
                                    g->guide_dprev, self->pol->time - g->guide_seen, gx, gy, g->guide_pred.x, g->guide_pred.y, bx, by);
        PyList_Append(L, t); Py_DECREF(t);
    });
    return L;
}
PyObject* Engine_dbg_true_poses(EngineObject* self, PyObject*) {
    // tests: set every mind's pose to the true pose and anchor its group
    if (!self->pol) Py_RETURN_NONE;
    Engine* e = self->eng; long long n = 0;
    for (auto& a : e->agents) {
        if (!self->pol->minds.has(a.id)) continue;
        orchard::Mind& m = *self->pol->minds.at(a.id);
        m.pose = orchard::mkpose(orchard::P2{a.x, a.y}, a.direction);
        self->pol->groups.at(m.group)->anchored = true; n++;
    }
    return PyLong_FromLongLong(n);
}
PyObject* Engine_dbg_pseen(EngineObject* self, PyObject*) {
    // tests: [(gid, [(x, y)]...)] shared predator sightings, plus per-agent count of predator observations
    PyObject* L = PyList_New(0);
    if (!self->pol) return L;
    self->pol->groups.each([&](const int64_t& gid, orchard::GroupP& g) {
        PyObject* pts = PyList_New(0);
        for (auto& q : g->pseen) { PyObject* t = Py_BuildValue("(dd)", q.p.x, q.p.y); PyList_Append(pts, t); Py_DECREF(t); }
        PyObject* row = Py_BuildValue("(LN)", (long long)gid, pts); PyList_Append(L, row); Py_DECREF(row);
    });
    for (auto& st : self->pol->states) {
        long long n2 = 0; for (auto& o : *st.obs) if (o.type == 2) n2++;
        PyObject* row = Py_BuildValue("(LL)", (long long)st.aid, n2); PyList_Append(L, row); Py_DECREF(row);
    }
    return L;
}
PyObject* Engine_dbg_pred_info(EngineObject* self, PyObject*) {
    // tests: per predator (target distance or -1, look=agent rel_dir, angle, mode 1 direct/2 pivot/3 edge-avoid/4 wander)
    Engine* e = self->eng; PyObject* L = PyList_New(0);
    for (auto& p : e->predators) { PyObject* t = Py_BuildValue("(dddi)", p.dbg_tdist, p.dbg_look, p.dbg_ang, p.dbg_mode); PyList_Append(L, t); Py_DECREF(t); }
    return L;
}
PyObject* Engine_dbg_trap(EngineObject* self, PyObject*) {
    // tests: (mouth_x, mouth_y, goal_x, goal_y, bait, n_retired, guide_done, n_walls) of the largest anchored group with a trap, else None
    if (!self->pol) Py_RETURN_NONE;
    orchard::GroupP best; size_t bn = 0;
    self->pol->groups.each([&](const int64_t&, orchard::GroupP& g) { if (g->has_trap && g->agents.size() >= bn) { best = g; bn = g->agents.size(); } });
    if (!best) Py_RETURN_NONE;
    long long conf = 0; for (auto& w : best->walls) if (self->pol->confirmed(w)) conf++;
    return Py_BuildValue("(ddddLLLL(LLLLLLLLLLLL))", best->trap.mouth.x, best->trap.mouth.y, best->trap.goal.x, best->trap.goal.y, (long long)best->bait, (long long)best->retired.size(), (long long)best->guide_done, conf,
                         (long long)best->ep_start, (long long)best->ep_chase, (long long)best->ep_state3, (long long)best->ep_hand, (long long)best->ep_died, (long long)best->ep_lost,
                         (long long)best->d_far, (long long)best->d_multi, (long long)best->d_slow, (long long)best->d_stuck, (long long)best->d_early, (long long)best->baits_born);
}
PyObject* Engine_dbg_pose(EngineObject* self, PyObject* args) {
    // tests: policy pose (x, y, theta, group, anchored) of an agent, or None
    long long aid; if (!PyArg_ParseTuple(args, "L", &aid)) return nullptr;
    if (!self->pol || !self->pol->minds.has(aid)) Py_RETURN_NONE;
    auto& m = *self->pol->minds.at(aid); auto g = self->pol->groups.at(m.group);
    return Py_BuildValue("(dddLi)", m.pose->p.x, m.pose->p.y, m.pose->theta, (long long)m.group, g->anchored ? 1 : 0);
}
PyObject* Engine_dbg_keeper(EngineObject* self, PyObject*) {
    if (!self->pol) Py_RETURN_NONE;
    orchard::GroupP best; size_t bn = 0;
    self->pol->groups.each([&](const int64_t&, orchard::GroupP& g) { if (g->has_trap && g->agents.size() >= bn) { best = g; bn = g->agents.size(); } });
    if (!best) Py_RETURN_NONE;
    return Py_BuildValue("(Lidd)", (long long)best->keeper, best->keeper_spawn ? 1 : 0, best->trap.rear.x, best->trap.rear.y);
}
PyObject* Engine_dbg_eval(EngineObject* self, PyObject*) {
    // counters from the policy's predator layer
    if (!self->pol) Py_RETURN_NONE;
    PyObject* gs = PyList_New(12); for (int i = 0; i < 12; i++) PyList_SetItem(gs, i, PyLong_FromLongLong(self->pol->gstat[i]));
    return Py_BuildValue("{s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:N}", "n_evading", (long long)self->pol->n_evading, "refuge_events", (long long)self->pol->refuge_events, "refuge_holds", (long long)self->pol->refuge_holds, "died_route", (long long)self->pol->refuge_died_route, "died_hold", (long long)self->pol->refuge_died_hold, "died_exit", (long long)self->pol->refuge_died_exit, "exits", (long long)self->pol->refuge_exits, "aborts", (long long)self->pol->refuge_aborts, "gstat", gs);
}

PyObject* Engine_get_info(EngineObject* self, PyObject*) {
    Engine* e = self->eng;
    return Py_BuildValue("{s:d,s:d,s:L,s:L,s:i,s:i,s:i}", "time", e->time, "score", e->score,
                         "next_agent_id", (long long)e->next_agent_id, "next_fruit_id", (long long)e->next_fruit_id,
                         "width", e->W, "height", e->H, "chunk_size", e->CS);
}

PyObject* mod_set_numpy_loops(PyObject*, PyObject* args) {
    PyObject *sin_u, *cos_u, *atan2_u, *hypot_u;
    if (!PyArg_ParseTuple(args, "OOOO", &sin_u, &cos_u, &atan2_u, &hypot_u)) return nullptr;
    NpLoop a, b, c, d;
    if (!find_loop(sin_u, 1, a) || !find_loop(cos_u, 1, b) || !find_loop(atan2_u, 2, c) || !find_loop(hypot_u, 2, d)) {
        PyErr_SetString(PyExc_RuntimeError, "float64 loop not found in numpy ufunc");
        return nullptr;
    }
    np_sin_l = a; np_cos_l = b; np_atan2_l = c; np_hypot_l = d;
    Py_RETURN_NONE;
}
PyObject* mod_math(PyObject*, PyObject* args) {  // for tests: engine's view of the four functions
    const char* name; double a, b = 0;
    if (!PyArg_ParseTuple(args, "sd|d", &name, &a, &b)) return nullptr;
    std::string n(name);
    if (n == "sin") return PyFloat_FromDouble(np_sin(a));
    if (n == "cos") return PyFloat_FromDouble(np_cos(a));
    if (n == "arctan2") return PyFloat_FromDouble(np_atan2(a, b));
    if (n == "hypot") return PyFloat_FromDouble(np_hypot(a, b));
    PyErr_SetString(PyExc_ValueError, name);
    return nullptr;
}
PyMethodDef module_methods[] = {
    {"set_numpy_loops", mod_set_numpy_loops, METH_VARARGS, "set_numpy_loops(np.sin, np.cos, np.arctan2, np.hypot)"},
    {"math", mod_math, METH_VARARGS, "math(name, a[, b]) through the engine's loops"},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef Engine_methods[] = {
    {"step", (PyCFunction)Engine_step, METH_VARARGS, "step(actions) -> state dict (same as step_environment)"},
    {"state", (PyCFunction)Engine_state, METH_NOARGS, "current state dict without stepping"},
    {"agents", (PyCFunction)Engine_agents, METH_NOARGS, "list of (id,x,y,direction,age,energy,max_energy,speed,sprint,hearing,vision,vision_is_int,cone,max_age)"},
    {"predators", (PyCFunction)Engine_predators, METH_NOARGS, "list of (x,y,direction,energy,resting)"},
    {"fruits", (PyCFunction)Engine_fruits, METH_NOARGS, "list of (fruit_id,x,y,energy,age,radius,0)"},
    {"trees", (PyCFunction)Engine_trees, METH_NOARGS, "list of (x,y,radius,age)"},
    {"obstacles", (PyCFunction)Engine_obstacles, METH_NOARGS, "list of (x,y,w,h)"},
    {"pop_events", (PyCFunction)Engine_pop_events, METH_NOARGS, "deaths and eaten fruit since last call"},
    {"biome_map", (PyCFunction)Engine_biome_map, METH_NOARGS, "bytes, index x*height+y, 0 forest 1 swamp 2 desert 3 grassland 4 river"},
    {"rng_state", (PyCFunction)Engine_rng_state, METH_NOARGS, "random.Random.getstate() equivalent"},
    {"info", (PyCFunction)Engine_get_info, METH_NOARGS, "time, score, counters"},
    {"dbg_keep_agent", (PyCFunction)Engine_dbg_keep_agent, METH_VARARGS, "tests: keep one agent"},
    {"dbg_set_agent", (PyCFunction)Engine_dbg_set_agent, METH_VARARGS, "tests: set agent state"},
    {"dbg_add_predator", (PyCFunction)Engine_dbg_add_predator, METH_VARARGS, "tests: add predator"},
    {"dbg_free", (PyCFunction)Engine_dbg_free, METH_VARARGS, "tests: is position free"},
    {"dbg_eval", (PyCFunction)Engine_dbg_eval, METH_NOARGS, "tests: policy predator counters"},
    {"dbg_pred_life", (PyCFunction)Engine_dbg_pred_life, METH_VARARGS, "tests: park predators older than T (perfect-trap model)"},
    {"dbg_walls", (PyCFunction)Engine_dbg_walls, METH_NOARGS, "tests: policy wall faces"},
    {"dbg_biome", (PyCFunction)Engine_dbg_biome, METH_VARARGS, "tests: biome index at (x, y): 0 forest 1 swamp 2 desert 3 grassland 4 river"},
    {"dbg_deaths", (PyCFunction)Engine_dbg_deaths, METH_NOARGS, "diagnostics: per-death records"},
    {"dbg_sites", (PyCFunction)Engine_dbg_sites, METH_NOARGS, "tests: policy trap sites"},
    {"dbg_keep_agents", (PyCFunction)Engine_dbg_keep_agents, METH_VARARGS, "tests: keep first n agents"},
    {"dbg_load_walls", (PyCFunction)Engine_dbg_load_walls, METH_NOARGS, "tests: true walls into the policy map"},
    {"dbg_freeze", (PyCFunction)Engine_dbg_freeze, METH_VARARGS, "tests: freeze agent ids"},
    {"dbg_roles", (PyCFunction)Engine_dbg_roles, METH_NOARGS, "tests: trap roles per group"},
    {"dbg_true_poses", (PyCFunction)Engine_dbg_true_poses, METH_NOARGS, "tests: true poses + anchor"},
    {"dbg_pseen", (PyCFunction)Engine_dbg_pseen, METH_NOARGS, "tests: shared predator sightings"},
    {"dbg_pred_info", (PyCFunction)Engine_dbg_pred_info, METH_NOARGS, "tests: predator target/mode"},
    {"dbg_trap", (PyCFunction)Engine_dbg_trap, METH_NOARGS, "tests: current trap of the main group"},
    {"dbg_pose", (PyCFunction)Engine_dbg_pose, METH_VARARGS, "tests: policy pose of an agent"},
    {"dbg_keeper", (PyCFunction)Engine_dbg_keeper, METH_NOARGS, "tests: keeper id, spawn flag, rear point"},
    {"dbg_pred_blocked", (PyCFunction)Engine_dbg_pred_blocked, METH_VARARGS, "tests: predator cannot stand here"},
    {"policy_init", (PyCFunction)Engine_policy_init, METH_VARARGS, "policy_init(seed_key, config_dict): native orchard policy"},
    {"policy_act", (PyCFunction)Engine_policy_act, METH_NOARGS, "native orchard decisions for the current state: [(aid, dist, dir, turn, spawn)]"},
    {"policy_minds", (PyCFunction)Engine_policy_minds, METH_NOARGS, "debug: native minds"},
    {"policy_groups", (PyCFunction)Engine_policy_groups, METH_NOARGS, "debug: native groups"},
    {"run_policy", (PyCFunction)Engine_run_policy, METH_VARARGS, "run_policy(horizon, stop_at) -> (steps, peak_agents); native policy + engine loop"},
    {nullptr, nullptr, 0, nullptr}};

void Engine_dealloc(EngineObject* self) {
    delete self->eng;
    delete self->pol;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

int Engine_init(EngineObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"seed_key", "env_width", "env_height", "chunk_size", "starting_agents", "starting_predators",
                                   "starting_fruits", "starting_trees", "dt", "predators", nullptr};
    PyObject* key_obj;
    int w = 1600, h = 1200, cs = 400, na = 5, np_ = 0, nf = 32, nt = 50, preds = 1;
    double dt = 0.1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iiiiiiidp", (char**)kwlist, &key_obj, &w, &h, &cs, &na, &np_, &nf, &nt, &dt, &preds))
        return -1;
    PyObject* seq = PySequence_Fast(key_obj, "seed_key must be a sequence of uint32");
    if (!seq) return -1;
    std::vector<uint32_t> key;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++) {
        unsigned long v = PyLong_AsUnsignedLong(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred()) { Py_DECREF(seq); return -1; }
        key.push_back((uint32_t)v);
    }
    Py_DECREF(seq);
    if (key.empty()) key.push_back(0);
    delete self->eng;
    delete self->pol;
    self->pol = nullptr;
    Py_BEGIN_ALLOW_THREADS
    self->eng = new Engine(w, h, cs, na, np_, nf, nt, key, dt, preds != 0);
    Py_END_ALLOW_THREADS
    return 0;
}

PyTypeObject EngineType = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "_nengine", "Native survival simulator engine", -1, module_methods};

}  // namespace

PyMODINIT_FUNC PyInit__nengine(void) {
    EngineType.tp_name = "_engine.Engine";
    EngineType.tp_basicsize = sizeof(EngineObject);
    EngineType.tp_flags = Py_TPFLAGS_DEFAULT;
    EngineType.tp_new = PyType_GenericNew;
    EngineType.tp_init = (initproc)Engine_init;
    EngineType.tp_dealloc = (destructor)Engine_dealloc;
    EngineType.tp_methods = Engine_methods;
    if (PyType_Ready(&EngineType) < 0) return nullptr;
#define INTERN(var, s) var = PyUnicode_InternFromString(s)
    INTERN(s_type, "type"); INTERN(s_distance, "distance"); INTERN(s_angle, "angle"); INTERN(s_rel_dir, "rel_dir");
    INTERN(s_id, "id"); INTERN(s_coords, "coords"); INTERN(s_Fruit, "Fruit"); INTERN(s_Agent, "Agent");
    INTERN(s_Predator, "Predator"); INTERN(s_Tree, "Tree"); INTERN(s_Edge, "Edge");
    INTERN(s_agent_id, "agent_id"); INTERN(s_observations, "observations"); INTERN(s_energy, "energy");
    INTERN(s_biome, "biome"); INTERN(s_age, "age"); INTERN(s_speed, "speed"); INTERN(s_sprint_speed, "sprint_speed");
    INTERN(s_hearing_radius, "hearing_radius"); INTERN(s_vision_angle, "vision_angle"); INTERN(s_vision_range, "vision_range");
    INTERN(s_max_energy, "max_energy"); INTERN(s_score, "score"); INTERN(s_sim_time, "sim_time"); INTERN(s_num_agents, "num_agents");
    INTERN(s_move_distance, "move_distance"); INTERN(s_move_direction, "move_direction"); INTERN(s_turn_angle, "turn_angle");
    INTERN(s_spawn_agent, "spawn_agent");
    for (int i = 0; i < 5; i++) s_biome_names[i] = PyUnicode_InternFromString(BIOMES[i].name);
#undef INTERN
    PyObject* m = PyModule_Create(&moduledef);
    if (!m) return nullptr;
    Py_INCREF(&EngineType);
    PyModule_AddObject(m, "Engine", (PyObject*)&EngineType);
    return m;
}
