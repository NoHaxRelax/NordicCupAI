// Native port of the orchard policy (fastsim/policy/orchard_ref.py + best-config.json).
// Included by _engine.cpp after the Engine class. Decision-identical to the Python
// policy: CPython's math.hypot/math.dist algorithm, round(), float % and //, dict
// insertion order, int-set iteration order (PySetEmu) and random.Random are all
// reproduced, and every expression keeps Python's evaluation order.
//
// Python semantics notes used throughout:
//   min(a, b) returns a unless b < a; max(a, b) returns a unless b > a.
//   Pose objects are shared (Python references): _transform_group replaces m.pose
//   with a new object while callers may still hold the old one (see _observe).

namespace orchard {

const double TAU = 2 * 3.141592653589793;
const double OPI = 3.141592653589793;
const double OINF = std::numeric_limits<double>::infinity();
const double W = 1600., H = 1200., CELL = 100.;

inline double pmin(double a, double b) { return b < a ? b : a; }
inline double pmax(double a, double b) { return b > a ? b : a; }

// ---- CPython 3.12 math.hypot / math.dist (vector_norm with fma-based dl_mul)
inline double vector_norm2(double a, double b) {
    // a, b are already fabs'd; max computed as CPython does
    double max = 0.0;
    if (a > max) max = a;
    if (b > max) max = b;
    if (std::isinf(max)) return max;
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    if (max == 0.0) return max;
    int max_e;
    uint64_t mb; std::memcpy(&mb, &max, 8);
    int bexp = (int)((mb >> 52) & 0x7ff);
    if (bexp != 0) max_e = bexp - 1022;  // frexp exponent of a normal number
    else std::frexp(max, &max_e);
    if (max_e < -1023) {
        // subnormal path (never hit by this policy's magnitudes); mirror CPython
        const double DMIN = std::numeric_limits<double>::min();
        return DMIN * vector_norm2(a / DMIN, b / DMIN);
    }
    double scale;
    if (-max_e >= -1022 && -max_e <= 1023) { uint64_t sb = (uint64_t)(-max_e + 1023) << 52; std::memcpy(&scale, &sb, 8); }
    else scale = std::ldexp(1.0, -max_e);
    double csum = 1.0, frac1 = 0.0, frac2 = 0.0;
    double vec[2] = {a, b};
    for (int i = 0; i < 2; i++) {
        double x = vec[i] * scale;
        double pr_hi = x * x;
        double pr_lo = std::fma(x, x, -pr_hi);
        double sm_hi = csum + pr_hi;
        double sm_lo = (csum - sm_hi) + pr_hi;
        csum = sm_hi;
        frac1 += pr_lo;
        frac2 += sm_lo;
    }
    double h = std::sqrt(csum - 1.0 + (frac1 + frac2));
    double pr_hi = -h * h;
    double pr_lo = std::fma(-h, h, -pr_hi);
    double sm_hi = csum + pr_hi;
    double sm_lo = (csum - sm_hi) + pr_hi;
    csum = sm_hi;
    frac1 += pr_lo;
    frac2 += sm_lo;
    double x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    return h / scale;
}
inline double hypot2(double x, double y) { return vector_norm2(std::fabs(x), std::fabs(y)); }

struct P2 { double x, y; };
inline bool operator==(const P2& a, const P2& b) { return a.x == b.x && a.y == b.y; }
inline double dist(const P2& p, const P2& q) { return vector_norm2(std::fabs(p.x - q.x), std::fabs(p.y - q.y)); }
// Comparisons of dist(p, q) with a threshold. The squared distance decides clear
// cases (relative margin 1e-12 >> rounding error); borderline cases use the exact
// CPython value, so results equal `math.dist(p, q) < thr` etc.
inline int dist_cmp(const P2& p, const P2& q, double thr) {  // -1: surely < thr, 1: surely > thr, 0: unsure
    if (!(thr > 0) || std::isinf(thr)) return 0;
    double dx = p.x - q.x, dy = p.y - q.y;
    double s = dx * dx + dy * dy, t2 = thr * thr;
    if (s < t2 * (1 - 1e-12)) return -1;
    if (s > t2 * (1 + 1e-12)) return 1;
    return 0;
}
inline bool dist_lt(const P2& p, const P2& q, double thr) { int c = dist_cmp(p, q, thr); return c ? c < 0 : dist(p, q) < thr; }
inline bool dist_le(const P2& p, const P2& q, double thr) { int c = dist_cmp(p, q, thr); return c ? c < 0 : dist(p, q) <= thr; }
inline bool dist_gt(const P2& p, const P2& q, double thr) { int c = dist_cmp(p, q, thr); return c ? c > 0 : dist(p, q) > thr; }
inline P2 add(P2 a, P2 b) { return {a.x + b.x, a.y + b.y}; }
inline P2 sub(P2 a, P2 b) { return {a.x - b.x, a.y - b.y}; }
inline P2 mul(P2 a, double s) { return {a.x * s, a.y * s}; }
inline double norm(P2 a) { return hypot2(a.x, a.y); }
inline P2 rot(P2 a, double t) {
    double c = std::cos(t), s = std::sin(t);
    return {a.x * c - a.y * s, a.x * s + a.y * c};
}
inline P2 unit(double t) { return {std::cos(t), std::sin(t)}; }
inline double wrap(double a) { return py_mod(a + OPI, TAU) - OPI; }
inline bool segments_cross(P2 a, P2 b, P2 c, P2 d) {
    auto orient = [](P2 p, P2 q, P2 r) { return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x); };
    double o1 = orient(a, b, c), o2 = orient(a, b, d), o3 = orient(c, d, a), o4 = orient(c, d, b);
    return ((o1 > 0) != (o2 > 0)) && ((o3 > 0) != (o4 > 0));
}
inline double point_segment(P2 p, P2 a, P2 b) {
    P2 v = sub(b, a), w = sub(p, a);
    double t = pmax(0., pmin(1., (v.x * w.x + v.y * w.y) / pmax(v.x * v.x + v.y * v.y, 1e-9)));
    return dist(p, add(a, mul(v, t)));
}
struct CellK { int64_t x, y; };
inline bool operator==(const CellK& a, const CellK& b) { return a.x == b.x && a.y == b.y; }
struct CellHash { size_t operator()(const CellK& c) const { return std::hash<int64_t>()(c.x * 1000003LL ^ c.y); } };
inline CellK cell_of(P2 p) { return {(int64_t)std::floor(p.x / CELL), (int64_t)std::floor(p.y / CELL)}; }
inline int64_t py_round(double x) {  // round(float) -> int, half to even
    double r = std::round(x);
    if (std::fabs(x - r) == 0.5) r = 2.0 * std::round(x / 2.0);
    return (int64_t)r;
}
inline int64_t int_floordiv(double a, double b) { return (int64_t)py_floordiv(a, b); }

// ---- Python containers
struct IntSet {
    PySetEmu s;
    void add(int64_t v) { s.add((int32_t)v, py_hash_int(v)); }
    void discard(int64_t v) { s.discard((int32_t)v, py_hash_int(v)); }
    bool has(int64_t v) const { return s.lookup((int32_t)v, py_hash_int(v)) >= 0; }
    size_t size() const { return (size_t)s.used; }
    bool empty() const { return s.used == 0; }
    void clear() { s.clear(); }
    template <class F> void each(F f) const { s.for_each([&](int32_t k) { f((int64_t)k); }); }
    std::vector<int64_t> list() const { std::vector<int64_t> v; each([&](int64_t k) { v.push_back(k); }); return v; }
};

template <class K, class V, class Hs = std::hash<K>>
struct ODict {  // Python dict: insertion order, delete + reinsert moves to the end
    std::vector<K> keys;
    std::vector<V> vals;
    std::vector<char> alive;
    std::unordered_map<K, size_t, Hs> idx;
    size_t n = 0;
    V* get(const K& k) {
        auto it = idx.find(k);
        return it == idx.end() ? nullptr : &vals[it->second];
    }
    const V* get(const K& k) const {
        auto it = idx.find(k);
        return it == idx.end() ? nullptr : &vals[it->second];
    }
    bool has(const K& k) const { return idx.count(k) > 0; }
    V& at(const K& k) { return vals[idx.at(k)]; }
    void set(const K& k, V v) {
        auto it = idx.find(k);
        if (it != idx.end()) { vals[it->second] = std::move(v); return; }
        idx[k] = keys.size();
        keys.push_back(k); vals.push_back(std::move(v)); alive.push_back(1); n++;
    }
    void erase(const K& k) {
        auto it = idx.find(k);
        if (it == idx.end()) return;
        alive[it->second] = 0; vals[it->second] = V(); idx.erase(it); n--;
        if (keys.size() > 64 && n * 2 < keys.size()) compact();
    }
    void compact() {
        std::vector<K> k2; std::vector<V> v2; std::vector<char> a2;
        idx.clear();
        for (size_t i = 0; i < keys.size(); i++)
            if (alive[i]) { idx[keys[i]] = k2.size(); k2.push_back(keys[i]); v2.push_back(std::move(vals[i])); a2.push_back(1); }
        keys.swap(k2); vals.swap(v2); alive.swap(a2);
    }
    size_t size() const { return n; }
    bool empty() const { return n == 0; }
    std::vector<K> key_list() const {
        std::vector<K> out;
        for (size_t i = 0; i < keys.size(); i++) if (alive[i]) out.push_back(keys[i]);
        return out;
    }
    template <class F> void each(F f) {
        for (size_t i = 0; i < keys.size(); i++) if (alive[i]) f(keys[i], vals[i]);
    }
    void clear() { keys.clear(); vals.clear(); alive.clear(); idx.clear(); n = 0; }
};

// ---- policy data
struct PoseObj { P2 p; double theta; };
using Pose = std::shared_ptr<PoseObj>;
inline Pose mkpose(P2 p, double th) { return std::make_shared<PoseObj>(PoseObj{p, th}); }
inline P2 transform(const PoseObj& ps, P2 local) { return add(ps.p, rot(local, ps.theta)); }
inline P2 polar(const PoseObj& ps, const Obs& o) {
    return transform(ps, P2{std::cos(o.angle) * o.distance, std::sin(o.angle) * o.distance});
}
inline void local_of(const PoseObj& ps, P2 p, double& d, double& ang) {
    P2 dd = sub(p, ps.p);
    d = norm(dd);
    ang = wrap(std::atan2(dd.y, dd.x) - ps.theta);
}

struct TreeM {
    int64_t id; P2 p; double first, last; bool fresh;
    double fruit_seen = -OINF; bool dead = false; IntSet assigned; int64_t fruit_here = 0, fruit_free = 0;
};
struct FruitM { int64_t id; P2 p; double born_lo, born_hi, last; bool has_claim = false; int64_t claimed = 0; };
using TreeP = std::shared_ptr<TreeM>;
using FruitP = std::shared_ptr<FruitM>;

struct EdgeMem { P2 a, b; double t; };
struct Mark { P2 q; double win; };
struct Hear { double t; P2 p; double r; };
struct Blocked { P2 p; double until; };
struct LastAction { double dist, direction, turn; int biome; double energy, speed, sprint, max_e; };
struct CellV { double last; int biome; };  // biome -1 = None

struct Mind {
    int64_t aid; int64_t group; Pose pose; double born;
    bool has_last = false; LastAction last_action{};
    bool spawned_ok = false;
    Pose prev_pose;  // null = None
    double prev_hear = 50., prev_cone = OPI / 3, prev_vis = 200.;
    std::vector<EdgeMem> edges;
    bool old = false; double old_since = OINF;
    bool has_eprev = false; double energy_prev = 0;
    bool has_post = false; int64_t post = 0;
    bool has_fruit = false; int64_t fruit = 0;
    bool has_explore = false; P2 explore_p{}; double explore_until = 0;
    int64_t sweep_left = 0;
    bool has_tkey = false; int64_t tkey_x = 0, tkey_y = 0;
    double best_d = OINF; int64_t no_progress = 0;
    bool has_detour = false; double detour_dir = 0, detour_until = -1.;
    std::vector<Blocked> blocked;
    double sweep_sign = 1.;
    std::vector<Mark> prev_marks;
    std::vector<Hear> hear_hist;
    bool heir_done = false;
    double repost_at = 0., post_since = -OINF, last_site = 0.;
    bool has_watch = false; P2 watch_p{}; double watch_t = 0;
};
using MindP = std::shared_ptr<Mind>;

struct Group {
    int64_t id;
    IntSet agents;
    ODict<int64_t, TreeP> trees;
    ODict<int64_t, FruitP> fruits;
    // current orchard.py: cell -> list of objects in insertion order (append / remove first match)
    std::unordered_map<CellK, std::vector<TreeP>, CellHash> tgrid;
    std::unordered_map<CellK, std::vector<FruitP>, CellHash> fgrid;
    ODict<CellK, CellV, CellHash> cells;
    bool anchored = false;
    int64_t next_tree = 0, next_fruit = 0;
    std::unordered_set<int64_t> seen_trees, seen_fruits;

    template <class T>
    static void list_remove(std::unordered_map<CellK, std::vector<T>, CellHash>& grid, const CellK& c, const T& obj) {
        auto it = grid.find(c);
        if (it == grid.end()) return;
        auto& v = it->second;
        for (size_t i = 0; i < v.size(); i++)
            if (v[i] == obj) { v.erase(v.begin() + i); return; }
    }
    void add_tree(const TreeP& t) { trees.set(t->id, t); tgrid[cell_of(t->p)].push_back(t); }
    void del_tree(int64_t tid) {
        TreeP t = trees.at(tid); trees.erase(tid);
        list_remove(tgrid, cell_of(t->p), t);
    }
    void move_tree(const TreeP& t, P2 p) {
        list_remove(tgrid, cell_of(t->p), t);
        t->p = p;
        tgrid[cell_of(p)].push_back(t);
    }
    void add_fruit(const FruitP& f) { fruits.set(f->id, f); fgrid[cell_of(f->p)].push_back(f); }
    void del_fruit(int64_t fid) {
        FruitP f = fruits.at(fid); fruits.erase(fid);
        list_remove(fgrid, cell_of(f->p), f);
    }
    void rebuild_grids() {
        tgrid.clear(); fgrid.clear();
        trees.each([&](const int64_t&, TreeP& t) { tgrid[cell_of(t->p)].push_back(t); });
        fruits.each([&](const int64_t&, FruitP& f) { fgrid[cell_of(f->p)].push_back(f); });
    }
    template <class T>
    void near(std::unordered_map<CellK, std::vector<T>, CellHash>& grid, P2 p, double r, std::vector<T>& out) {
        out.clear();
        CellK c = cell_of(p);
        int64_t n = int_floordiv(r, CELL) + 1;
        for (int64_t dx = -n; dx <= n; dx++)
            for (int64_t dy = -n; dy <= n; dy++) {
                auto it = grid.find(CellK{c.x + dx, c.y + dy});
                if (it == grid.end() || it->second.empty()) continue;
                for (const T& obj : it->second)
                    if (dist_le(obj->p, p, r)) out.push_back(obj);
            }
    }
    std::vector<TreeP> near_trees(P2 p, double r) { std::vector<TreeP> o; near(tgrid, p, r, o); return o; }
    std::vector<FruitP> near_fruits(P2 p, double r) { std::vector<FruitP> o; near(fgrid, p, r, o); return o; }
};
using GroupP = std::shared_ptr<Group>;

// biome indices follow the engine: 0 forest 1 swamp 2 desert 3 grassland 4 river
const double MOVE_PENALTY[5] = {1.0, 0.5, 0.8, 1.0, 0.3};
const double TREE_RATE[5] = {1.0, 0.9, 0.1, 0.5, 0.0};
const double FRUIT_RATE[5] = {0.1, 0.08, 0.05, 0.1, 0.0};
inline double tree_rate_or(int b, double dflt) { return b >= 0 ? TREE_RATE[b] : dflt; }
inline double fruit_rate_or(int b, double dflt) { return b >= 0 ? FRUIT_RATE[b] : dflt; }

struct AState {
    int64_t aid; const std::vector<Obs>* obs;
    double energy; int biome; double age, speed, sprint, hear, cone, vr, max_energy;
};

struct Params {
    double cap_mult = 0.3, cap_min = 4, cap_max = 20, n0 = 80., tree_half = 600., tree_slots = 1,
           breed_reserve = 200., emergency_reserve = 105., ripen_wait = 20., sweep_rate = 0.03,
           explore_radius = 450., fit_vision = 1.0, fit_hear = 0.3, fit_energy = 0.2, births_per_tick = 3,
           fruit_reach = 200., tree_reach = 420., site_min = 5., breed_reserve_late = 200., reserve_t0 = 600.,
           reserve_t1 = 1800., dist_pen = 0.1, vo_win_fruit = 4.5, vo_win_far = 4.5, vo_cap = 12.,
           heir_age = 55., heir_reserve = 250., explore_min = 60., travel_turn = 0.25, heir_slack = 0.05,
           repost_every = 10., switch_gain = 100., fruit_min_wait = 0., no_eat_age = OINF, late_still_t = OINF,
           post_radius = 30., min_stay = 15., hungry_margin = 5., fit_speed = 0.3, explore_energy = 200.,
           watch_patience = 30., watch_reach = 500., watch_refresh = 60., select_min_young = 0,
           dump_food_site = 2, dump_mult = 1.0, cluster_radius = 0., spread_weight = 0., low_pop_reserve = 200.,
           lone_reach_mult = 1.0, old_reach = 60., rot_margin = 47., dump_after_t = OINF, cap_tree_slack = 1,
           cap_hard_min = 2, nursery_bonus = 0.;
    // late-game schedule (nightsim): from time late_t on, each l_* that is not NaN replaces its parameter
    // predator layer (nightsim): pred_mode 0 off, 1 evade (face nearest threat, back away; sprint when close)
    double merge_anchored = 0.;
    double pred_mode = 0., pred_r = 200., pred_sprint_r = 90., pred_face = 1., pred_face_r = 260.;
    double late_t = OINF, l_fruit_reach = NAN, l_tree_reach = NAN, l_watch_reach = NAN, l_explore_energy = NAN, l_cap_min = NAN, l_cap_mult = NAN, l_cap_tree_slack = NAN, l_cap_hard_min = NAN, l_sweep_rate = NAN, l_watch_patience = NAN, l_explore_radius = NAN, l_old_reach = NAN, l_dist_pen = NAN;
    bool idle_sweep = true, extra_old = true, cull = false, heir_select = true, heir_at_food = false,
         old_eat_last = true, heir_needs_site = true;
    bool feed_breed = false;  // feed_mode == 'breed' (else 'hungry')
};

struct Act { int64_t aid; double dist, direction, turn; bool spawn; };

class Policy {
public:
    Params P;
    PyRandom rng;
    double time = 0.;
    ODict<int64_t, MindP> minds;
    ODict<int64_t, GroupP> groups;
    int64_t next_group = 0;
    std::vector<int64_t> last_spawners;
    std::unordered_set<int64_t> culled;
    bool debug_merge = false;

    // per-call state
    std::vector<AState> states;
    std::unordered_map<int64_t, size_t> sidx;
    const AState& st(int64_t aid) const { return states[sidx.at(aid)]; }
    bool in_states(int64_t aid) const { return sidx.count(aid) > 0; }
    Mind& M(int64_t aid) { return *minds.at(aid); }
    Group& G(int64_t gid) { return *groups.at(gid); }

    Policy(const std::vector<uint32_t>& seed_key, const Params& p) : P(p) { rng.init_by_array(seed_key); }

    // ------------------------------------------------------------ groups
    GroupP new_group() {
        auto g = std::make_shared<Group>(); g->id = next_group;
        groups.set(g->id, g); next_group++;
        return g;
    }
    void transform_group(Group& g, double dth, P2 shift) {
        auto T = [&](P2 q) { return add(rot(q, dth), shift); };
        g.agents.each([&](int64_t aid) {
            Mind& m = M(aid);
            m.pose = mkpose(T(m.pose->p), wrap(m.pose->theta + dth));
            if (m.prev_pose) m.prev_pose = mkpose(T(m.prev_pose->p), wrap(m.prev_pose->theta + dth));
            for (auto& e : m.edges) { e.a = T(e.a); e.b = T(e.b); }
            for (auto& b : m.blocked) b.p = T(b.p);
            for (auto& h : m.hear_hist) h.p = T(h.p);
            if (m.has_explore) m.explore_p = T(m.explore_p);
            if (m.has_watch) m.watch_p = T(m.watch_p);
            if (m.has_detour) m.detour_dir = wrap(m.detour_dir + dth);
            m.has_tkey = false;
        });
        g.trees.each([&](const int64_t&, TreeP& t) { t->p = T(t->p); });
        g.fruits.each([&](const int64_t&, FruitP& f) { f->p = T(f->p); });
        g.rebuild_grids();
        ODict<CellK, CellV, CellHash> cells;
        g.cells.each([&](const CellK& c, CellV& v) {
            P2 q = T(P2{((double)c.x + .5) * CELL, ((double)c.y + .5) * CELL});
            CellK nc = cell_of(q);
            CellV* e = cells.get(nc);
            if (e) { e->last = pmax(e->last, v.last); if (e->biome < 0) e->biome = v.biome; }
            else cells.set(nc, v);
        });
        g.cells = std::move(cells);
    }
    void merge(Group& ga, Group& gb, double dth, P2 shift) {
        transform_group(gb, dth, shift);
        gb.agents.each([&](int64_t aid) { M(aid).group = ga.id; ga.agents.add(aid); });
        for (const int64_t& tk : gb.trees.key_list()) {
            TreeP t = gb.trees.at(tk);
            auto nt = ga.near_trees(t->p, 12);
            if (nt.empty()) {
                t->id = ga.next_tree; ga.next_tree++; ga.add_tree(t);
                t->assigned.each([&](int64_t aid) { M(aid).has_post = true; M(aid).post = t->id; });
            } else {
                TreeP same = nt[0];
                same->first = pmin(same->first, t->first); same->last = pmax(same->last, t->last);
                same->fruit_seen = pmax(same->fruit_seen, t->fruit_seen); same->fresh = same->fresh || t->fresh;
                t->assigned.each([&](int64_t aid) { M(aid).has_post = true; M(aid).post = same->id; same->assigned.add(aid); });
            }
        }
        for (const int64_t& fk : gb.fruits.key_list()) {
            FruitP f = gb.fruits.at(fk);
            auto nf = ga.near_fruits(f->p, 4);
            if (nf.empty()) {
                f->id = ga.next_fruit; ga.next_fruit++; ga.add_fruit(f);
                if (f->has_claim) { M(f->claimed).has_fruit = true; M(f->claimed).fruit = f->id; }
            } else {
                FruitP same = nf[0];
                same->born_lo = pmax(same->born_lo, f->born_lo); same->born_hi = pmin(same->born_hi, f->born_hi);
                if (!same->has_claim && f->has_claim) { same->has_claim = true; same->claimed = f->claimed; }
                if (f->has_claim) {
                    Mind& cm = M(f->claimed);
                    if (same->has_claim && same->claimed == f->claimed) { cm.has_fruit = true; cm.fruit = same->id; }
                    else cm.has_fruit = false;
                }
            }
        }
        gb.cells.each([&](const CellK& c, CellV& v) {
            CellV* e = ga.cells.get(c);
            if (e) { e->last = pmax(e->last, v.last); if (e->biome < 0) e->biome = v.biome; }
            else ga.cells.set(c, v);
        });
        ga.anchored = ga.anchored || gb.anchored;
        groups.erase(gb.id);
    }
    void merge_groups() {
        if (P.merge_anchored > 0.) {   // nightsim: anchored groups share the absolute frame, so merge them at once
            GroupP first;
            std::vector<GroupP> rest;
            groups.each([&](const int64_t&, GroupP& g) { if (g->anchored) { if (!first) first = g; else rest.push_back(g); } });
            for (auto& g : rest) merge(*first, *g, 0., P2{0., 0.});
        }
        while (true) {
            bool changed = false;
            for (const AState& s : states) {
                MindP mp = minds.at(s.aid);
                Mind& m = *mp;
                for (const Obs& o : *s.obs) {
                    if (o.type != 1 || !minds.has(o.id)) continue;
                    Mind& mb = M(o.id);
                    if (mb.group == m.group) continue;
                    GroupP ga = groups.at(m.group), gb = groups.at(mb.group);
                    Pose pb = pose_from_observer(*m.pose, o);
                    P2 pB = pb->p; double thB = pb->theta;
                    if (gb->anchored && !ga->anchored) {
                        double dth = wrap(thB - mb.pose->theta);
                        P2 shift = sub(pB, rot(mb.pose->p, dth));
                        double inv_dth = -dth;
                        P2 inv_shift = mul(rot(shift, inv_dth), -1.);
                        merge(*gb, *ga, inv_dth, inv_shift);
                    } else if (ga->anchored && gb->anchored) {
                        merge(*ga, *gb, 0., P2{0., 0.});
                    } else {
                        double dth = wrap(thB - mb.pose->theta);
                        P2 shift = sub(pB, rot(mb.pose->p, dth));
                        merge(*ga, *gb, dth, shift);
                    }
                    changed = true;
                    break;
                }
                if (changed) break;
            }
            if (!changed) return;
        }
    }

    // ------------------------------------------------------------ registration
    static Pose pose_from_observer(const PoseObj& op, const Obs& o) {
        if (o.distance < 1e-6) return mkpose(op.p, wrap(-o.rel_dir));
        return mkpose(polar(op, o), wrap(op.theta + o.angle + OPI - o.rel_dir));
    }
    static const Obs* find_agent_obs(const std::vector<Obs>& obs, int64_t id) {
        for (const Obs& o : obs) if (o.type == 1 && o.id == id) return &o;
        return nullptr;
    }
    void register_new(const std::vector<int64_t>& new_ids) {
        std::vector<int64_t> spawners;
        for (int64_t a : last_spawners) if (minds.has(a)) spawners.push_back(a);
        for (size_t k = 0; k < new_ids.size(); k++) {
            int64_t cid = new_ids[k];
            bool has_parent = k < spawners.size();
            int64_t parent = has_parent ? spawners[k] : 0;
            Pose pose;
            if (has_parent) {
                Mind& pm = M(parent);
                const Obs* o = find_agent_obs(*st(parent).obs, cid);
                if (o) pose = pose_from_observer(*pm.pose, *o);
                else {
                    o = find_agent_obs(*st(cid).obs, parent);
                    if (o) {
                        if (o->distance < 1e-6) pose = mkpose(pm.pose->p, wrap(-o->angle));
                        else {
                            double th = wrap(pm.pose->theta - o->angle - OPI + o->rel_dir);
                            pose = mkpose(sub(pm.pose->p, mul(unit(th + o->angle), o->distance)), th);
                        }
                    }
                }
            }
            if (!pose) {
                for (const AState& s : states) {
                    if (s.aid == cid || !minds.has(s.aid)) continue;
                    const Obs* o = find_agent_obs(*s.obs, cid);
                    if (o) { Mind& pm = M(s.aid); parent = s.aid; has_parent = true; pose = pose_from_observer(*pm.pose, *o); break; }
                }
            }
            GroupP g;
            if (!pose) { g = new_group(); pose = mkpose(P2{0., 0.}, 0.); }
            else g = groups.at(M(parent).group);
            auto m = std::make_shared<Mind>();
            m->aid = cid; m->group = g->id; m->pose = pose; m->born = time;
            m->has_eprev = true; m->energy_prev = st(cid).energy;
            minds.set(cid, m); g->agents.add(cid);
        }
    }

    // ------------------------------------------------------------ odometry
    void odometry(Mind& m) {
        if (!m.has_last) return;
        const LastAction& a = m.last_action;
        double d = pmax(0., pmin(a.dist, a.sprint));
        if (a.energy < a.max_e / 5 && d > a.speed) d = a.speed;
        d *= MOVE_PENALTY[a.biome];
        PoseObj& pose = *m.pose;
        double ang = pose.theta + a.direction;
        if (d > 0) {
            std::vector<std::pair<P2, P2>> near;
            for (const EdgeMem& e : m.edges)
                if (time - e.t < 4. && point_segment(pose.p, e.a, e.b) < d + 8) near.push_back({e.a, e.b});
            auto blocked = [&](P2 q) {
                for (auto& ab : near)
                    if (point_segment(q, ab.first, ab.second) < 5.5 || segments_cross(pose.p, q, ab.first, ab.second)) return true;
                return false;
            };
            P2 q = add(pose.p, mul(unit(ang), d));
            if (!near.empty() && blocked(q)) {
                double step = OPI / 18; bool moved = false;
                for (int i = 0; i < 36; i++) {
                    double aa = ang + step * (double)((i + 1) / 2) * ((i % 2) ? -1.0 : 1.0);
                    P2 q2 = add(pose.p, mul(unit(aa), d));
                    if (!blocked(q2)) { q = q2; moved = true; break; }
                }
                if (!moved) q = pose.p;
            }
            pose.p = q;
        }
        pose.theta = wrap(pose.theta + a.turn);
        if (G(m.group).anchored) pose.p = P2{pmin(pmax(pose.p.x, 5.), W - 5.), pmin(pmax(pose.p.y, 5.), H - 5.)};
    }

    // ------------------------------------------------------------ perception
    static bool in_view(const PoseObj& pose, double hear, double cone, double vr, P2 p, double margin) {
        double d, ang; local_of(pose, p, d, ang);
        if (d <= hear - margin - 1) return true;
        return d <= vr - margin - 5 && std::fabs(ang) <= cone / 2 - 0.06;
    }
    void anchor(Mind& m, const Obs& o) {
        double x1 = o.c[0], y1 = o.c[1], x2 = o.c[2], y2 = o.c[3];
        double L = hypot2(x2 - x1, y2 - y1);
        double phi = std::atan2(y2 - y1, x2 - x1);
        double theta; P2 cands[4];
        if (L > 1500) {
            theta = wrap(-phi);
            cands[0] = {0., 30.}; cands[1] = {0., H - 30.}; cands[2] = {0., 0.}; cands[3] = {0., H};
        } else {
            theta = wrap(OPI / 2 - phi);
            cands[0] = {30., 0.}; cands[1] = {W - 30., 0.}; cands[2] = {0., 0.}; cands[3] = {W, 0.};
        }
        bool found = false; P2 best{};
        for (P2 sxy : cands) {
            P2 pp = sub(sxy, rot(P2{x1, y1}, theta));
            if (4 <= pp.x && pp.x <= W - 4 && 4 <= pp.y && pp.y <= H - 4) { best = pp; found = true; break; }
        }
        if (!found) return;
        Group& g = G(m.group);
        if (!g.anchored) {
            double dth = wrap(theta - m.pose->theta);
            P2 shift = sub(best, rot(m.pose->p, dth));
            transform_group(g, dth, shift); g.anchored = true;
        } else {
            P2 err = sub(best, m.pose->p);
            double ne = norm(err);
            if (0.5 < ne && ne < 40) m.pose->p = best;
        }
    }
    void marks_of(const PoseObj& pose, const std::vector<Obs>& obs, double wf, double wt, std::vector<Mark>& out) {
        out.clear();
        for (const Obs& o : obs) {
            if (o.type == 0) out.push_back({polar(pose, o), wf});
            else if (o.type == 3) out.push_back({polar(pose, o), wt});
            else if (o.type == 4) {
                out.push_back({transform(pose, P2{o.c[0], o.c[1]}), wt});
                out.push_back({transform(pose, P2{o.c[2], o.c[3]}), wt});
            }
        }
    }
    void observe(Mind& m, const AState& s) {
        Group& g = G(m.group);
        Pose posep = m.pose;  // may go stale if anchoring transforms the group (as in Python)
        PoseObj& pose = *posep;
        double hear = s.hear, cone = s.cone, vr = s.vr;
        const std::vector<Obs>& obs = *s.obs;
        bool moved = m.has_last && m.last_action.dist > 0;
        double wf = P.vo_win_fruit, wt = P.vo_win_far;
        std::vector<Mark> marks;
        marks_of(pose, obs, wf, wt, marks);
        if (moved && !marks.empty() && !m.prev_marks.empty()) {
            std::vector<P2> pairs;
            for (const Mark& mk : marks) {
                bool hb = false; double bd = 0; P2 br{};
                for (const Mark& r : m.prev_marks) {
                    double dd = dist(mk.q, r.q);
                    if (dd < mk.win && (!hb || dd < bd)) { hb = true; bd = dd; br = r.q; }
                }
                if (hb) pairs.push_back(P2{br.x - mk.q.x, br.y - mk.q.y});
            }
            if (pairs.size() >= 2) {
                std::vector<double> xs, ys;
                for (auto& pr : pairs) { xs.push_back(pr.x); ys.push_back(pr.y); }
                std::sort(xs.begin(), xs.end()); std::sort(ys.begin(), ys.end());
                double ex = xs[xs.size() / 2], ey = ys[ys.size() / 2];
                int64_t agree = 0;
                for (auto& pr : pairs) if (std::fabs(pr.x - ex) < 0.8 && std::fabs(pr.y - ey) < 0.8) agree++;
                double hh = hypot2(ex, ey);
                int64_t need = std::max<int64_t>(2, ((int64_t)pairs.size() + 1) / 2);
                if (0.05 < hh && hh < P.vo_cap && agree >= need) {
                    pose.p = add(pose.p, P2{ex, ey});
                    for (auto& mk : marks) mk.q = add(mk.q, P2{ex, ey});
                }
            }
        }
        m.prev_marks = marks;
        for (const Obs& o : obs) {
            if (o.type != 4) continue;
            if (hypot2(o.c[2] - o.c[0], o.c[3] - o.c[1]) > 1000) anchor(m, o);
        }
        for (const Obs& o : obs) {
            if (o.type != 4) continue;
            P2 a = transform(pose, P2{o.c[0], o.c[1]}), b = transform(pose, P2{o.c[2], o.c[3]});
            bool found = false;
            for (auto& e : m.edges)
                if (dist_lt(a, e.a, 6) && dist_lt(b, e.b, 6)) { e = EdgeMem{a, b, time}; found = true; break; }
            if (!found) m.edges.push_back(EdgeMem{a, b, time});
        }
        {
            std::vector<EdgeMem> keep;
            for (auto& e : m.edges) if (time - e.t < 40.) keep.push_back(e);
            if (keep.size() > 150) keep.erase(keep.begin(), keep.end() - 150);
            m.edges.swap(keep);
        }
        for (const Obs& o : obs) {
            if (o.type != 3) continue;
            P2 p = polar(pose, o);
            TreeP t;
            double td = 0;
            for (auto& c : g.near_trees(p, 12)) {
                if (c->dead) continue;
                double dd = dist(c->p, p);
                if (!t || dd < td) { t = c; td = dd; }
            }
            if (!t) {
                bool fresh = m.prev_pose && in_view(*m.prev_pose, m.prev_hear, m.prev_cone, m.prev_vis, p, 3.);
                t = std::make_shared<TreeM>();
                t->id = g.next_tree; t->p = p; t->first = time; t->last = time; t->fresh = fresh;
                g.add_tree(t); g.next_tree++;
            } else {
                P2 err = sub(t->p, p);
                double ne = norm(err);
                if (moved && 0.3 < ne && ne < 8 && time - t->last < 2.) { pose.p = add(pose.p, err); moved = false; }
                else if (norm(err) >= 1.0 && time - t->last >= 2.) g.move_tree(t, p);
            }
            t->last = time; g.seen_trees.insert(t->id);
        }
        for (const Obs& o : obs) {
            if (o.type != 0) continue;
            P2 p = polar(pose, o);
            FruitP f; double fd = 0;
            for (auto& c : g.near_fruits(p, 5)) {
                double dd = dist(c->p, p);
                if (!f || dd < fd) { f = c; fd = dd; }
            }
            if (!f) {
                double lo = -OINF;
                if (m.prev_pose && in_view(*m.prev_pose, m.prev_hear, m.prev_cone, m.prev_vis, p, 3.)) lo = time - 0.1;
                else {
                    g.agents.each([&](int64_t a) {
                        auto& hh = M(a).hear_hist;
                        for (size_t i = hh.size(); i-- > 0;)
                            if (dist_lt(hh[i].p, p, hh[i].r - 2.)) { lo = pmax(lo, hh[i].t); break; }
                    });
                }
                lo = pmax(lo, 0.);
                f = std::make_shared<FruitM>();
                f->id = g.next_fruit; f->p = p; f->born_lo = lo; f->born_hi = time; f->last = time;
                g.add_fruit(f); g.next_fruit++;
                for (auto& t : g.near_trees(p, 70)) if (!t->dead) t->fruit_seen = time;
            }
            f->last = time; g.seen_fruits.insert(f->id);
        }
        CellK c0 = cell_of(pose.p);
        CellV* v = g.cells.get(c0);
        if (!v) g.cells.set(c0, CellV{time, s.biome});
        else { v->last = time; v->biome = s.biome; }
        const double rs[3] = {60., 120., 175.};
        for (double r : rs) {
            if (r > vr) break;
            const double as[3] = {-cone / 3, 0., cone / 3};
            for (double a : as) {
                CellK c = cell_of(add(pose.p, mul(unit(pose.theta + a), r)));
                CellV* cv = g.cells.get(c);
                if (!cv) g.cells.set(c, CellV{time, -1});
                else cv->last = time;
            }
        }
        if (m.has_eprev && m.has_last && !m.old) {
            const LastAction& la = m.last_action;
            double d = pmax(0., pmin(la.dist, la.sprint));
            if (la.energy < la.max_e / 5 && d > la.speed) d = la.speed;
            double cost = d <= la.speed ? d * 0.05 : la.speed * 0.05 + (d - la.speed) * 0.5;
            cost += pmin(OPI, std::fabs(la.turn)) / TAU + 0.1 + (m.spawned_ok ? 100. : 0.);
            double dev = (m.energy_prev - s.energy) - cost;
            if ((0.45 < dev && dev < 2.5 && s.energy < s.max_energy - 0.5) || s.age > 120.5) {
                m.old = true; m.old_since = time;
            }
        }
        m.has_eprev = true; m.energy_prev = s.energy;
        m.hear_hist.push_back(Hear{time, pose.p, hear});
        if (m.hear_hist.size() > 400) m.hear_hist.erase(m.hear_hist.begin(), m.hear_hist.begin() + 100);
        if (!m.prev_marks.empty()) marks_of(pose, obs, wf, wt, m.prev_marks);
        m.prev_pose = mkpose(pose.p, pose.theta); m.prev_hear = hear; m.prev_cone = cone; m.prev_vis = vr;
    }

    void maintain(Group& g) {
        double now = time;
        std::unordered_set<int64_t> vis_t, vis_f;
        g.agents.each([&](int64_t a) {
            Mind& m = M(a); const AState& s = st(a);
            double h = s.hear, c = s.cone, v = s.vr;
            std::vector<std::pair<P2, P2>> edges;
            for (auto& e : m.edges) if (now - e.t < 40.) edges.push_back({e.a, e.b});
            P2 mp = m.pose->p;
            auto occluded = [&](P2 p) {
                if (dist_le(p, mp, h - 3.)) return false;
                for (auto& e : edges) if (segments_cross(mp, p, e.first, e.second)) return true;
                return false;
            };
            for (auto& t : g.near_trees(mp, v))
                if (!vis_t.count(t->id) && in_view(*m.pose, h, c, v, t->p, 20.) && !occluded(t->p)) vis_t.insert(t->id);
            for (auto& f : g.near_fruits(mp, v))
                if (!vis_f.count(f->id) && in_view(*m.pose, h, c, v, f->p, 8.) && !occluded(f->p)) vis_f.insert(f->id);
        });
        for (int64_t tid : g.trees.key_list()) {
            TreeP t = g.trees.at(tid);
            if (t->dead) {
                if (now - t->last > 55.) g.del_tree(tid);
                continue;
            }
            if (now > t->first + 62.5 || (!g.seen_trees.count(tid) && vis_t.count(tid))) { t->dead = true; continue; }
            IntSet na;
            t->assigned.each([&](int64_t a) {
                if (minds.has(a) && M(a).has_post && M(a).post == tid) na.add(a);
            });
            t->assigned = na;
        }
        for (int64_t fid : g.fruits.key_list()) {
            FruitP f = g.fruits.at(fid);
            bool gone = now > f->born_hi + 50.05 || (!g.seen_fruits.count(fid) && vis_f.count(fid));
            if (gone) {
                if (f->has_claim && minds.has(f->claimed) && M(f->claimed).has_fruit && M(f->claimed).fruit == fid)
                    M(f->claimed).has_fruit = false;
                g.del_fruit(fid);
                continue;
            }
            if (f->has_claim && (!minds.has(f->claimed) || !M(f->claimed).has_fruit || M(f->claimed).fruit != fid))
                f->has_claim = false;
        }
        g.trees.each([&](const int64_t&, TreeP& t) {
            auto near = g.near_fruits(t->p, 70.);
            t->fruit_here = (int64_t)near.size();
            int64_t fr = 0;
            for (auto& f : near) if (!f->has_claim) fr++;
            t->fruit_free = fr;
        });
        g.seen_trees.clear(); g.seen_fruits.clear();
    }

    // ------------------------------------------------------------ economy
    double n_est() const { return P.n0 * std::pow(0.5, time / P.tree_half); }
    int64_t cap() {
        double r = (double)py_round(P.cap_mult * n_est());
        double c = pmax(P.cap_min, pmin(P.cap_max, r));
        int64_t ci = (int64_t)c;
        if (P.cap_tree_slack >= 0 && !groups.empty()) {
            int64_t known = 0; bool first = true;
            groups.each([&](const int64_t&, GroupP& g) {
                int64_t cnt = 0;
                g->trees.each([&](const int64_t&, TreeP& t) { if (!t->dead) cnt++; });
                if (first || cnt > known) { known = cnt; first = false; }
            });
            double lim = pmax(P.cap_hard_min, (double)known * P.tree_slots + P.cap_tree_slack);
            ci = (int64_t)pmin((double)ci, lim);
        }
        return ci;
    }
    double fitness(const AState& s) const {
        return (P.fit_vision * std::pow(s.vr / 200., 2.0) * pmin(1.5, s.cone / 1.0472)
                + P.fit_hear * std::pow(s.hear / 50., 2.0)
                + P.fit_energy * pmin(2., s.max_energy / 500.) + P.fit_speed * pmin(1.5, pmin(s.speed, s.sprint) / 10.));
    }
    bool ready(const FruitM& f, double energy = OINF, bool old = false) const {
        if (time < f.born_hi + P.fruit_min_wait) return false;
        if (f.born_lo == -OINF) return true;
        double t_eat = pmin(f.born_hi + P.ripen_wait, f.born_lo + P.rot_margin);
        double left = t_eat - time;
        if (left <= 0.) return true;
        if (old) return false;
        return energy < left + P.hungry_margin;
    }
    double reserve() const {
        double t = time;
        if (t <= P.reserve_t0) return P.breed_reserve;
        if (t >= P.reserve_t1) return P.breed_reserve_late;
        return P.breed_reserve + (P.breed_reserve_late - P.breed_reserve) * (t - P.reserve_t0) / (P.reserve_t1 - P.reserve_t0);
    }
    static bool site_ok(const TreeM& t) { return (!t.dead) || t.fruit_here > 0; }
    static int64_t others_n(const IntSet& s, int64_t aid) { return (int64_t)s.size() - (s.has(aid) ? 1 : 0); }

    double tree_value(Group& g, TreeM& t, Mind& m, const AState& s) {
        int64_t n = others_n(t.assigned, m.aid);
        double reach = P.tree_reach * (g.agents.size() <= 1 ? P.lone_reach_mult : 1.);
        if (dist_gt(t.p, m.pose->p, reach)) return -OINF;
        double d = dist(t.p, m.pose->p);
        double walk = pmax(1., pmin(s.speed, s.sprint) * MOVE_PENALTY[s.biome]);
        double travel_t = d / walk / 10.;
        double travel_e = d * 0.05 + travel_t;
        double wait = 0., future = 0.;
        if (!t.dead) {
            if ((double)n >= P.tree_slots) return -OINF;
            double remaining = t.fresh ? (t.first + 58. - time) : (t.first + 55. - time);
            remaining -= travel_t;
            if (remaining < 6.) return -OINF;
            wait = t.fresh ? pmax(0., t.first + 20. - time - travel_t) : 0.;
            CellV* cell = g.cells.get(cell_of(t.p));
            int biome = cell ? cell->biome : -1;
            double rate = fruit_rate_or(biome, 0.08) * 60.;
            double known = time - t.first;
            if (!t.fresh && known > 15. && time - t.fruit_seen > known) rate *= 0.5;
            future = pmax(0., remaining - wait) * rate;
        }
        double here = 55. * (double)t.fruit_free;
        if (P.cluster_radius > 0.) {
            // neighbours of t are fixed during one assign_posts pass: cache them per tree
            const std::vector<TreeP>* nl;
            auto ci = cluster_cache.find(t.id);
            if (ci != cluster_cache.end()) nl = &ci->second;
            else nl = &(cluster_cache[t.id] = g.near_trees(t.p, P.cluster_radius));
            for (auto& u : *nl) {
                if (u->id == t.id || u->dead || (double)others_n(u->assigned, m.aid) >= P.tree_slots) continue;
                double ur = u->fresh ? (u->first + 58. - time) : (u->first + 55. - time);
                if (ur > 6.) {
                    CellV* ucell = g.cells.get(cell_of(u->p));
                    int ub = ucell ? ucell->biome : -1;
                    future += 0.7 * pmax(0., ur) * fruit_rate_or(ub, 0.08) * 60.;
                }
                here += 55. * (double)u->fruit_free;
            }
        }
        if (future + here <= 0.) return -OINF;
        if (s.energy - travel_e - wait - 12. < 0.) return -OINF;
        double value = (future + here) / (double)(n + 1) - travel_e - 0.5 * wait - P.dist_pen * d;
        if (P.nursery_bonus > 0. && !m.heir_done && s.age >= P.heir_age - 8.)
            value += P.nursery_bonus * (double)std::min<int64_t>(4, t.fruit_free);
        if (P.spread_weight > 0.) {
            bool any = false; double gap = 0;
            g.agents.each([&](int64_t a) {
                if (a == m.aid) return;
                double dd = dist(t.p, M(a).pose->p);
                if (!any || dd < gap) { gap = dd; any = true; }
            });
            if (any) value += P.spread_weight * 60. * pmin(1., gap / pmax(1., s.vr));
        }
        return value;
    }

    std::unordered_map<int64_t, std::vector<TreeP>> cluster_cache;
    void assign_posts(Group& g) {
        cluster_cache.clear();
        std::vector<TreeP> sites;
        g.trees.each([&](const int64_t&, TreeP& t) { sites.push_back(t); });
        std::vector<int64_t> agents = g.agents.list();
        std::stable_sort(agents.begin(), agents.end(), [&](int64_t a, int64_t b) { return st(a).energy < st(b).energy; });
        for (int64_t a : agents) {
            Mind& m = M(a);
            if (m.old) continue;
            bool keep = m.has_post && g.trees.has(m.post) && site_ok(*g.trees.at(m.post));
            if (keep && (time < m.repost_at || time - m.post_since < P.min_stay)) continue;
            double cur_v = -OINF;
            if (keep) {
                cur_v = tree_value(g, *g.trees.at(m.post), m, st(a));
                m.repost_at = time + P.repost_every;
            }
            bool hb = false; double bv = 0; TreeP bt;
            for (auto& t : sites) {
                if (keep && t->id == m.post) continue;
                if (!site_ok(*t)) continue;
                double v = tree_value(g, *t, m, st(a));
                if (v > P.site_min && (!hb || v > bv)) { hb = true; bv = v; bt = t; }
            }
            if (keep && (!hb || bv < cur_v + P.switch_gain)) continue;
            if (m.has_post && g.trees.has(m.post)) g.trees.at(m.post)->assigned.discard(a);
            m.has_post = false;
            if (hb) {
                m.has_post = true; m.post = bt->id; bt->assigned.add(a);
                m.has_explore = false; m.has_tkey = false; m.post_since = time;
            }
        }
        cluster_cache.clear();
    }

    struct FPair { int64_t bucket; double nf, d; int64_t a, fid; };
    void assign_fruits(Group& g) {
        g.agents.each([&](int64_t a) {
            Mind& m = M(a);
            if (m.has_fruit && (!g.fruits.has(m.fruit) || !g.fruits.at(m.fruit)->has_claim || g.fruits.at(m.fruit)->claimed != a))
                m.has_fruit = false;
        });
        std::vector<FPair> pairs;
        g.agents.each([&](int64_t a) {
            Mind& m = M(a);
            if (m.has_fruit) return;
            const AState& s = st(a);
            if (s.age > P.no_eat_age) return;
            bool full = s.energy > s.max_energy - 30.;
            double reach = m.old ? P.old_reach : P.fruit_reach * (g.agents.size() <= 1 ? P.lone_reach_mult : 1.);
            for (auto& f : g.near_fruits(m.pose->p, reach)) {
                if (f->has_claim) continue;
                double d = dist(f->p, m.pose->p);
                if (!ready(*f, s.energy, m.old)) continue;
                bool owe_heir = (!m.heir_done) && s.age >= P.heir_age - 5. && s.energy < P.heir_reserve + 20.;
                int64_t bucket;
                if (m.old) bucket = P.old_eat_last ? 10 : 5;
                else if (culled.count(a)) bucket = 10;
                else if (owe_heir) bucket = 0;
                else if (full) bucket = 9;
                else if (P.feed_breed) bucket = s.energy < reserve() + 20. ? 1 : 2 + int_floordiv(s.energy, 120);
                else bucket = int_floordiv(s.energy, 60);
                pairs.push_back(FPair{bucket, -fitness(s), d, a, f->id});
            }
        });
        std::sort(pairs.begin(), pairs.end(), [](const FPair& x, const FPair& y) {
            if (x.bucket != y.bucket) return x.bucket < y.bucket;
            if (x.nf != y.nf) return x.nf < y.nf;
            if (x.d != y.d) return x.d < y.d;
            if (x.a != y.a) return x.a < y.a;
            return x.fid < y.fid;
        });
        std::unordered_set<int64_t> taken;
        for (auto& pr : pairs) {
            FruitM& f = *g.fruits.at(pr.fid);
            if (taken.count(pr.a) || f.has_claim) continue;
            f.has_claim = true; f.claimed = pr.a;
            M(pr.a).has_fruit = true; M(pr.a).fruit = pr.fid; taken.insert(pr.a);
        }
    }

    // ------------------------------------------------------------ navigation
    void go_to(Mind& m, const AState& s, P2 target, double stop, double& o_step, double& o_dir, double& o_turn) {
        const PoseObj& pose = *m.pose;
        double d, ang; local_of(pose, target, d, ang);
        if (d <= stop) { o_step = 0.; o_dir = 0.; o_turn = 0.; return; }
        double walk = pmin(s.speed, s.sprint);
        double step = pmin(walk, pmax(0., d - stop * 0.5));
        double heading = pose.theta + ang;
        double look = pmin(45., d);
        std::vector<std::pair<P2, P2>> recent;
        for (auto& e : m.edges)
            if (time - e.t < 25. && point_segment(pose.p, e.a, e.b) < look + 10.) recent.push_back({e.a, e.b});
        Group& g = G(m.group);
        std::vector<P2> unripe;
        for (auto& f : g.near_fruits(pose.p, look + 20.))
            if (!ready(*f) && !(m.has_fruit && f->id == m.fruit)) unripe.push_back(f->p);
        auto clear = [&](double h, double lk) {
            P2 q = add(pose.p, mul(unit(h), lk));
            for (auto& ab : recent)
                if (segments_cross(pose.p, q, ab.first, ab.second) || point_segment(q, ab.first, ab.second) < 7.) return false;
            for (auto& fp : unripe)
                if (point_segment(fp, pose.p, q) < 15.) return false;
            return true;
        };
        if ((!recent.empty() || !unripe.empty()) && !clear(heading, look)) {
            bool done = false;
            for (int k = 1; k < 12 && !done; k++) {
                for (int sgn : {1, -1}) {
                    double h = heading + (double)(sgn * k) * OPI / 12;
                    if (clear(h, look)) { heading = h; done = true; break; }
                }
            }
        }
        double direction = wrap(heading - pose.theta);
        double tt = P.travel_turn;
        o_step = step; o_dir = direction; o_turn = pmax(-tt, pmin(tt, direction));
    }
    bool progress(Mind& m, P2 target, double d) {
        int64_t kx = py_round(target.x / 10), ky = py_round(target.y / 10);
        if (!m.has_tkey || m.tkey_x != kx || m.tkey_y != ky) {
            m.has_tkey = true; m.tkey_x = kx; m.tkey_y = ky; m.best_d = d; m.no_progress = 0;
            return false;
        }
        if (d < m.best_d - 1.5) { m.best_d = d; m.no_progress = 0; }
        else m.no_progress++;
        if (m.no_progress >= 15) {
            m.blocked.push_back(Blocked{target, time + 40.});
            int64_t c = rng.randbelow(2) == 0 ? -1 : 1;  // choice([-1, 1])
            double u = rng.uniform(-.4, .4);
            m.has_detour = true;
            m.detour_dir = m.pose->theta + (double)c * OPI / 2 + u;
            m.detour_until = time + 2.;
            m.has_tkey = false; m.no_progress = 0;
            return true;
        }
        return false;
    }
    bool explore_target(Mind& m, Group& g, const AState& s, P2& out_p, double& out_until) {
        const PoseObj& pose = *m.pose; double R = P.explore_radius;
        std::vector<P2> others;
        g.agents.each([&](int64_t a) {
            if (a == m.aid) return;
            Mind& om = M(a);
            if (om.has_post && g.trees.has(om.post)) others.push_back(g.trees.at(om.post)->p);
            else if (om.has_explore) others.push_back(om.explore_p);
            else others.push_back(om.pose->p);
        });
        CellK c0 = cell_of(pose.p); int64_t n = int_floordiv(R, CELL) + 1;
        bool hb = false; double bs = 0; P2 bc{};
        for (int64_t dx = -n; dx <= n; dx++)
            for (int64_t dy = -n; dy <= n; dy++) {
                CellK c{c0.x + dx, c0.y + dy};
                P2 center{((double)c.x + .5) * CELL, ((double)c.y + .5) * CELL};
                if (g.anchored && !(40 < center.x && center.x < W - 40 && 40 < center.y && center.y < H - 40)) continue;
                double d = dist(center, pose.p);
                if (d > R || d < 60) continue;
                CellV* v = g.cells.get(c);
                double stale = !v ? 1.2 : pmin(1., (time - v->last) / 200.);
                int biome = v ? v->biome : -1;
                double w = biome >= 0 ? tree_rate_or(biome, 0.6) : 0.6;
                double score = stale * w - 0.6 * d / R;
                for (auto& q : others) if (dist_lt(center, q, 130)) { score -= 0.5; break; }
                for (auto& b : m.blocked) if (dist_lt(center, b.p, 40) && b.until > time) { score -= 1.; break; }
                for (auto& e : m.edges)
                    if (time - e.t < 30. && segments_cross(pose.p, center, e.a, e.b)) { score -= 0.4; break; }
                score += rng.uniform(0, .08);
                if (!hb || score > bs) { hb = true; bs = score; bc = center; }
            }
        if (!hb) return false;
        double jx = rng.uniform(-20, 20);
        double jy = rng.uniform(-20, 20);
        P2 target = add(bc, P2{jx, jy});
        double walk = pmax(1., pmin(s.speed, s.sprint) * MOVE_PENALTY[s.biome]);
        out_p = target; out_until = time + dist(target, pose.p) / walk / 10. + 10.;
        return true;
    }
    bool watch_post(Mind& m, Group& g, const AState& s, P2& out) {
        const PoseObj& pose = *m.pose; double vr = pmin(s.vr, 400.); double R = P.watch_reach;
        std::vector<P2> others;
        g.agents.each([&](int64_t a) {
            if (a == m.aid) return;
            Mind& om = M(a);
            if (om.has_post && g.trees.has(om.post)) others.push_back(g.trees.at(om.post)->p);
            else if (om.has_watch) others.push_back(om.watch_p);
            else others.push_back(om.pose->p);
        });
        CellK c0 = cell_of(pose.p); int64_t n = int_floordiv(R, CELL) + 1, k = int_floordiv(vr, CELL) + 1;
        bool hb = false; double bs = 0; P2 bc{};
        double vr08 = vr * 0.8;
        for (int64_t dx = -n; dx <= n; dx++)
            for (int64_t dy = -n; dy <= n; dy++) {
                CellK c{c0.x + dx, c0.y + dy};
                P2 center{((double)c.x + .5) * CELL, ((double)c.y + .5) * CELL};
                if (g.anchored && !(60 < center.x && center.x < W - 60 && 60 < center.y && center.y < H - 60)) continue;
                double d = dist(center, pose.p);
                if (d > R) continue;
                double cover = 0.;
                for (int64_t ex = -k; ex <= k; ex++)
                    for (int64_t ey = -k; ey <= k; ey++) {
                        CellK cc{c.x + ex, c.y + ey};
                        P2 q{((double)cc.x + .5) * CELL, ((double)cc.y + .5) * CELL};
                        if (g.anchored && !(30 < q.x && q.x < W - 30 && 30 < q.y && q.y < H - 30)) continue;
                        if (dist_gt(q, center, vr)) continue;
                        CellV* v = g.cells.get(cc);
                        double w = (v && v->biome >= 0) ? tree_rate_or(v->biome, 0.5) : 0.5;
                        for (auto& o : others) if (dist_lt(q, o, vr08)) { w *= 0.3; break; }
                        cover += w;
                    }
                double score = cover - d * 0.02 + rng.uniform(0, .2);
                if (!hb || score > bs) { hb = true; bs = score; bc = center; }
            }
        if (!hb) return false;
        out = bc;
        return true;
    }

    // ------------------------------------------------------------ per-agent behaviour
    struct Plan { double dist, direction, turn; };
    Plan act(Mind& m, const AState& s) {
        Group& g = G(m.group);
        const PoseObj& pose = *m.pose;
        double walk = pmin(s.speed, s.sprint);
        {
            std::vector<Blocked> keep;
            for (auto& b : m.blocked) if (b.until > time) keep.push_back(b);
            m.blocked.swap(keep);
        }
        if (m.has_detour && time <= m.detour_until) {
            double direction = wrap(m.detour_dir - pose.theta);
            return {walk, direction, pmax(-.3, pmin(.3, direction))};
        }
        if (m.has_fruit && g.fruits.has(m.fruit)) {
            FruitP f = g.fruits.at(m.fruit);
            double d, ang; local_of(pose, f->p, d, ang);
            if (progress(m, f->p, d)) { f->has_claim = false; m.has_fruit = false; }
            else {
                double dd, dir, turn; go_to(m, s, f->p, 0., dd, dir, turn);
                return {pmin(walk, d + 1.), dir, turn};
            }
        }
        if (time >= P.late_still_t) return {0., 0., P.sweep_rate};
        if (m.has_post && g.trees.has(m.post) && site_ok(*g.trees.at(m.post))) {
            TreeP t = g.trees.at(m.post);
            m.last_site = time;
            double d, ang; local_of(*m.pose, t->p, d, ang);
            if (d > P.post_radius) {
                if (progress(m, t->p, d)) { t->assigned.discard(m.aid); m.has_post = false; }
                else {
                    double dd, dir, turn; go_to(m, s, t->p, P.post_radius - 8., dd, dir, turn);
                    return {dd, dir, turn};
                }
            }
            m.has_tkey = false;
            const PoseObj& ps = *m.pose;
            std::vector<FruitP> close;
            for (auto& f : g.near_fruits(ps.p, 16.)) if (!ready(*f)) close.push_back(f);
            if (!close.empty()) {
                FruitP f; double fd = 0;
                for (auto& c : close) { double dd = dist(c->p, ps.p); if (!f || dd < fd) { f = c; fd = dd; } }
                double away = std::atan2(ps.p.y - f->p.y, ps.p.x - f->p.x);
                double direction = wrap(away - ps.theta);
                return {pmin(walk, 18. - dist(f->p, ps.p) + 1.), direction, 0.};
            }
            double turn = P.idle_sweep ? P.sweep_rate * m.sweep_sign : 0.;
            return {0., 0., turn};
        }
        if (m.old) return {0., 0., P.sweep_rate};
        if (s.energy < P.explore_energy || time - m.last_site < P.watch_patience) {
            m.has_explore = false;
            if (!m.has_watch || time - m.watch_t > P.watch_refresh) {
                P2 tgt;
                if (watch_post(m, g, s, tgt)) { m.has_watch = true; m.watch_p = tgt; m.watch_t = time; }
                else m.has_watch = false;
                m.has_tkey = false;
            }
            if (m.has_watch && s.energy >= P.explore_min) {
                double d, ang; local_of(*m.pose, m.watch_p, d, ang);
                if (d > 25.) {
                    if (progress(m, m.watch_p, d)) m.has_watch = false;
                    else {
                        double dd, dir, turn; go_to(m, s, m.watch_p, 20., dd, dir, turn);
                        return {dd, dir, turn};
                    }
                }
            }
            return {0., 0., P.sweep_rate * 3.};
        }
        if (m.sweep_left > 0) { m.sweep_left--; return {0., 0., 0.32}; }
        if (!m.has_explore || time > m.explore_until || dist(m.explore_p, m.pose->p) < 25.) {
            bool arrived = m.has_explore && dist(m.explore_p, m.pose->p) < 25.;
            P2 tp; double tu;
            if (explore_target(m, g, s, tp, tu)) { m.has_explore = true; m.explore_p = tp; m.explore_until = tu; }
            else m.has_explore = false;
            m.has_tkey = false;
            if (arrived) { m.sweep_left = 20; return {0., 0., 0.32}; }
        }
        if (!m.has_explore) return {0., 0., 0.32};
        P2 target = m.explore_p;
        double d, ang; local_of(*m.pose, target, d, ang);
        if (progress(m, target, d)) { m.has_explore = false; return {0., 0., 0.}; }
        double dd, dir, turn; go_to(m, s, target, 20., dd, dir, turn);
        return {dd, dir, turn};
    }

    // ------------------------------------------------------------ predators (nightsim)
    // A predator is a threat when it is within pred_r, or within pred_face_r and facing us (rel_dir small:
    // rel_dir = bearing(predator->agent) - predator heading). Response: move directly away from the
    // inverse-distance-weighted threats; face the nearest one (a faced predator beyond 90 uses the slow 45-degree
    // pivot approach); sprint only inside pred_sprint_r. Plans stay the odometry source, so poses remain exact.
    int64_t n_evading = 0;
    bool evade(const AState& s, Plan& pl) {
        const Obs* nr = nullptr; double vx = 0., vy = 0.;
        for (const Obs& o : *s.obs) {
            if (o.type != 2) continue;
            bool facing = o.has_rel_dir && std::fabs(o.rel_dir) < 0.5;
            if (!(o.distance < P.pred_r || (facing && o.distance < P.pred_face_r))) continue;
            double w = 1.0 / pmax(o.distance, 15.);
            vx -= std::cos(o.angle) * w; vy -= std::sin(o.angle) * w;
            if (!nr || o.distance < nr->distance) nr = &o;
        }
        if (!nr) return false;
        double away = std::atan2(vy, vx);
        double walk = pmin(s.speed, s.sprint);
        double step = nr->distance < P.pred_sprint_r ? s.sprint : walk;
        double turn = P.pred_face > 0. ? nr->angle : 0.;
        pl = Plan{step, away, turn};
        n_evading++;
        return true;
    }

    // ------------------------------------------------------------ main
    double cost_now(double dist_, double turn, const AState& s) const {
        double d = pmax(0., pmin(dist_, s.sprint));
        if (s.energy < s.max_energy / 5 && d > s.speed) d = s.speed;
        double c = d <= s.speed ? d * 0.05 : s.speed * 0.05 + (d - s.speed) * 0.5;
        return c + pmin(OPI, std::fabs(turn)) / TAU;
    }

    bool late_on = false;
    void apply_late() {
        auto ov = [](double& dst, double v) { if (!std::isnan(v)) dst = v; };
        ov(P.fruit_reach, P.l_fruit_reach); ov(P.tree_reach, P.l_tree_reach); ov(P.watch_reach, P.l_watch_reach); ov(P.explore_energy, P.l_explore_energy); ov(P.cap_min, P.l_cap_min); ov(P.cap_mult, P.l_cap_mult); ov(P.cap_tree_slack, P.l_cap_tree_slack); ov(P.cap_hard_min, P.l_cap_hard_min); ov(P.sweep_rate, P.l_sweep_rate); ov(P.watch_patience, P.l_watch_patience); ov(P.explore_radius, P.l_explore_radius); ov(P.old_reach, P.l_old_reach); ov(P.dist_pen, P.l_dist_pen);
    }
    std::vector<Act> call(std::vector<AState>&& sts, double sim_time) {
        time = sim_time;
        if (!late_on && time >= P.late_t) { late_on = true; apply_late(); }
        states = std::move(sts);
        sidx.clear();
        for (size_t i = 0; i < states.size(); i++) sidx[states[i].aid] = i;
        for (int64_t aid : minds.key_list()) {
            if (in_states(aid)) continue;
            MindP m = minds.at(aid); minds.erase(aid);
            GroupP g = groups.at(m->group);
            g->agents.discard(aid);
            g->trees.each([&](const int64_t&, TreeP& t) { t->assigned.discard(aid); });
            g->fruits.each([&](const int64_t&, FruitP& f) { if (f->has_claim && f->claimed == aid) f->has_claim = false; });
            if (g->agents.empty()) groups.erase(g->id);
        }
        for (const AState& s : states) if (minds.has(s.aid)) odometry(M(s.aid));
        std::vector<int64_t> nw;
        for (const AState& s : states) if (!minds.has(s.aid)) nw.push_back(s.aid);
        std::sort(nw.begin(), nw.end());
        if (!nw.empty()) register_new(nw);
        merge_groups();
        for (const AState& s : states) observe(M(s.aid), s);
        {
            std::vector<GroupP> gl;
            groups.each([&](const int64_t&, GroupP& g) { gl.push_back(g); });
            for (auto& g : gl) maintain(*g);
        }
        culled.clear();
        if (P.cull) {
            std::vector<int64_t> young_ids;
            minds.each([&](const int64_t& a, MindP& m) { if (!m->old) young_ids.push_back(a); });
            int64_t surplus = (int64_t)young_ids.size() - cap();
            if (surplus > 0) {
                std::stable_sort(young_ids.begin(), young_ids.end(), [&](int64_t a, int64_t b) {
                    double fa = fitness(st(a)), fb = fitness(st(b));
                    if (fa != fb) return fa < fb;
                    return st(a).energy < st(b).energy;
                });
                for (int64_t i = 0; i < surplus; i++) culled.insert(young_ids[i]);
            }
        }
        {
            std::vector<GroupP> gl;
            groups.each([&](const int64_t&, GroupP& g) { gl.push_back(g); });
            for (auto& g : gl) { assign_posts(*g); assign_fruits(*g); }
        }
        std::vector<int64_t> young;
        minds.each([&](const int64_t& a, MindP& m) { if (!m->old) young.push_back(a); });
        int64_t capv = cap(); int64_t pop = (int64_t)states.size();
        std::unordered_set<int64_t> spawn_set;
        std::unordered_map<int64_t, Plan> plans;
        for (const AState& s : states) plans[s.aid] = act(M(s.aid), s);
        if (P.pred_mode > 0.) for (const AState& s : states) evade(s, plans[s.aid]);
        int64_t young_now = (int64_t)young.size();
        std::unordered_map<int64_t, double> fit;
        for (const AState& s : states) fit[s.aid] = fitness(s);
        std::vector<int64_t> elders;
        minds.each([&](const int64_t& a, MindP& m) { if (m->old || st(a).age >= P.heir_age) elders.push_back(a); });
        std::stable_sort(elders.begin(), elders.end(), [&](int64_t a, int64_t b) { return -st(a).energy < -st(b).energy; });
        std::vector<double> yfit;
        for (int64_t a : young) yfit.push_back(fit[a]);
        std::sort(yfit.begin(), yfit.end());
        if (yfit.empty()) yfit.push_back(0.);
        double median_fit = yfit[yfit.size() / 2];
        for (int64_t aid : elders) {
            Mind& m = M(aid); const AState& s = st(aid);
            if (m.heir_done || s.biome == RIVER || (culled.count(aid) && !m.old)) continue;
            if (P.heir_select && (double)young.size() >= P.select_min_young && fit[aid] < median_fit - P.heir_slack) continue;
            const Plan& pl = plans[aid];
            Group& g = G(m.group);
            bool at_food = m.has_post && g.trees.has(m.post) && (g.trees.at(m.post)->fruit_here > 0 || !g.trees.at(m.post)->dead);
            double left = s.energy - cost_now(pl.dist, pl.turn, s);
            bool ok;
            if (m.old) ok = left > 101.;
            else ok = left > P.heir_reserve || (P.heir_at_food && at_food && left > 130.);
            if (ok && P.heir_needs_site && !at_food && young_now >= capv && pop > 2) ok = false;
            if (ok) { spawn_set.insert(aid); m.heir_done = true; young_now++; }
        }
        for (int64_t aid : elders) {
            Mind& m = M(aid); const AState& s = st(aid);
            if (!m.old || spawn_set.count(aid)) continue;
            Group& g = G(m.group);
            TreeP site;
            if (m.has_post && g.trees.has(m.post)) site = g.trees.at(m.post);
            bool late = time >= P.dump_after_t;
            if (!late && (!site || (double)site->fruit_here < P.dump_food_site)) continue;
            const Plan& pl = plans[aid];
            if (s.energy - cost_now(pl.dist, pl.turn, s) > 101. && (double)young_now < (double)capv * P.dump_mult) {
                spawn_set.insert(aid); young_now++;
            }
        }
        if (pop <= 2) {
            for (const AState& s : states) {
                const Plan& pl = plans[s.aid];
                if (!spawn_set.count(s.aid) && s.energy - cost_now(pl.dist, pl.turn, s) > P.emergency_reserve) {
                    spawn_set.insert(s.aid); young_now++;
                }
            }
        }
        int64_t slots = std::min<int64_t>(capv - young_now, (int64_t)P.births_per_tick);
        if (slots > 0) {
            struct Cand { double fit, left; int64_t aid; };
            std::vector<Cand> cands;
            for (const AState& s : states) {
                Mind& m = M(s.aid);
                if (spawn_set.count(s.aid) || s.biome == RIVER) continue;
                const Plan& pl = plans[s.aid];
                double left = s.energy - cost_now(pl.dist, pl.turn, s);
                if (m.old) {
                    if (!P.extra_old || left <= 101.) continue;
                } else {
                    double thr = (double)young.size() < P.cap_min ? pmin(reserve(), P.low_pop_reserve) : reserve();
                    if (left <= thr) continue;
                }
                Group& g = G(m.group);
                bool food = (m.has_post && g.trees.has(m.post) && !g.trees.at(m.post)->dead) || !g.near_fruits(m.pose->p, 90.).empty();
                if (!food) continue;
                cands.push_back(Cand{fit[s.aid], left, s.aid});
            }
            std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
                if (x.fit != y.fit) return x.fit > y.fit;
                if (x.left != y.left) return x.left > y.left;
                return x.aid > y.aid;
            });
            for (int64_t i = 0; i < slots && i < (int64_t)cands.size(); i++) spawn_set.insert(cands[i].aid);
        }
        std::vector<Act> actions;
        last_spawners.clear();
        std::vector<int64_t> order;
        for (const AState& s : states) order.push_back(s.aid);
        std::sort(order.begin(), order.end());
        for (int64_t aid : order) {
            const AState& s = st(aid); Mind& m = M(aid);
            const Plan& pl = plans[aid];
            bool spawn = spawn_set.count(aid) > 0;
            bool ok = spawn && s.energy - cost_now(pl.dist, pl.turn, s) > 100.;
            m.spawned_ok = ok;
            if (ok) last_spawners.push_back(aid);
            m.has_last = true;
            m.last_action = LastAction{pl.dist, pl.direction, pl.turn, s.biome, s.energy, s.speed, s.sprint, s.max_energy};
            actions.push_back(Act{aid, pl.dist, pl.direction, pl.turn, spawn});
        }
        return actions;
    }
};

}  // namespace orchard
