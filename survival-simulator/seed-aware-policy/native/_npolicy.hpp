// Native port of the orchard policy (fastsim/policy/orchard_ref.py + best-config.json).
// Included by _nengine.cpp after the Engine class. The orchard foundation
// reproduces Python arithmetic/container semantics; the entrapment additions
// are a native adaptation, not a decision-identical port of models/core.py.
// Foundation details: CPython's math.hypot/math.dist algorithm, round(), float % and //, dict
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
    double fruit_seen = -OINF; bool dead = false; IntSet assigned; int64_t fruit_here = 0, fruit_free = 0; int64_t misses = 0;
};
struct FruitM { int64_t id; P2 p; double born_lo, born_hi, last; bool has_claim = false; int64_t claimed = 0; int64_t misses = 0; };
using TreeP = std::shared_ptr<TreeM>;
using FruitP = std::shared_ptr<FruitM>;

struct EdgeMem { P2 a, b; double t; };
struct Mark { P2 q; double win; };
struct Hear { double t; P2 p; double r; };
struct Blocked { P2 p; double until; };
struct LastAction { double dist, direction, turn; int biome; double energy, speed, sprint, max_e; };
struct CellV { double last; int biome; };  // biome -1 = None

struct Mind {
    int corner_phase=0; double corner_start=0.,corner_retry=0.,corner_heading=0.,corner_want=0.,corner_error=0.; P2 corner_goal{},corner_pred{};
    int64_t aid; int64_t group; Pose pose; double born;
    bool guide_sample_valid=false; int guide_sample_tick=0; P2 guide_sample{};
    std::vector<std::pair<P2,double>> terrain_samples;
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
    int hide_idx = -1; double hide_t = -1e9;   // nightsim: crevice pass-through escape
    bool refuge_in = false; double refuge_pred_t = -1e9; int refuge_bad = 0;   // nightsim: refuge (hold inside a narrow gap)
    double evade_t = -1e9;   // nightsim diagnostics: last tick this agent evaded
    double dodge_head = 0.; int64_t dodge_left = 0;   // nightsim: committed dodge heading (pred_dodge_hold)
};
using MindP = std::shared_ptr<Mind>;

struct Group {
    int corner_index=-1;
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
    struct Wall { bool horiz; double c, lo, hi; double solid; double t; int64_t n = 1;
                  std::vector<double> cs, los, his; int64_t obs1 = -1, obs2 = -1; int64_t n_obs = 0; int64_t n_conf = 0; };   // axis-aligned face; solid = +1: solid on +axis side; n_conf = observations claiming the opposite solid side
    std::vector<Wall> walls;
#include "wall_index_fields.hpp"          // nightsim: permanent wall faces (anchored frame only)
    struct Site { P2 goal, mouth, out, rear; double overlap, gap, score; bool rear_ok; };
    std::vector<Site> sites; double sites_t = -1e9;
    bool has_trap = false; Site trap{}; int64_t bait = -1, rep = -1; double trap_since = 0.;
    std::vector<int64_t> retired;   // former baits: stay frozen in the crevice until they die
    std::vector<int64_t> leaving;   // nightsim bait_rotate: former baits walking out through the rear to forage again
    int64_t guide = -1; int guide_state = 0; double guide_since = 0., guide_seen = -1e9; P2 guide_pred{}; int64_t guide_done = 0;
    double guide_dprev = -1., guide_closing_t = -1e9; P2 guide_pred_prev{}; bool guide_has_prev = false;
    // funnel counters (diagnostics): episodes started / reached LEAD with a real chase / reached the lane point (state 3) / ended by death / lost / handoff position reached
    int64_t baits_rotated = 0;
    int64_t ep_start = 0, ep_chase = 0, ep_state3 = 0, ep_died = 0, ep_lost = 0, ep_hand = 0; bool ep_chased = false, ep_s3 = false, ep_h = false;
    // last known guide status (for death attribution): distance to lane point, predators within 120, walk speed, stuck ticks, ticks alive
    double gl_dT = 0., gl_speed = 0.; int64_t gl_npred = 0, gl_stuck = 0, gl_ticks = 0; P2 gl_pos{};
    int64_t d_far = 0, d_multi = 0, d_slow = 0, d_stuck = 0, d_early = 0, d_state1 = 0;
    bool guide_sprinting = false; int64_t held_max = 0; double guide_end_t = -1e9; int64_t ep_deliv = 0; double wait_since = -1.;
    int64_t keeper = -1; bool keeper_spawn = false; double keeper_spawn_t = -1e9; int64_t baits_born = 0;
    int64_t relay = -1; int64_t relays_done = 0;
    P2 rep_pos{}; int64_t rep_stuck = 0; double rep_since = 0.;
    bool rep_entered_rear = false;
    double rep_progress = OINF, rep_progress_t = -1e9;
    double bait_gap_since = -1.;
    int64_t bait_arrivals = 0, bait_overlaps = 0, replacement_failures = 0;
    double bait_gap_seconds = 0.;
    struct PredSeen { P2 p; double heading; };
    std::vector<PredSeen> pseen;   // nightsim: predators seen by any member this tick (group frame)
    struct PredMark { P2 p; double t; };
    std::vector<PredMark> pmem;   // nightsim: recent predator sightings (pred_avoid_*)
    std::unordered_set<int64_t> seen_trees, seen_fruits;
    struct NavGrid { uint64_t version = 0; double step = 0., radius = 0.; int nx = 0, ny = 0; std::vector<char> blocked; };
    uint64_t wall_version = 1;
    std::vector<NavGrid> nav_grids;
    uint64_t route_cache_version = 0;
    std::unordered_map<std::string, std::vector<P2>> route_cache;
    std::unordered_set<std::string> failed_route_cache;

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
    double corner_mode=0.,corner_tolerance=10.,corner_budget=3.,corner_gaze=.25,corner_start_angle=180.;
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
    double entrapment_lookahead = 0., explore_until_trap = 0.;
    double merge_anchored = 0., no_spawn = 0., fit_speed_cap = 1.5;
    double hide_mode = 0., hide_r = 150., hide_trigger = 80., trap_post_w = 0., trap_post_r = 400., refuge_mode = 0., refuge_r = 60., refuge_trigger = 80., refuge_leave = 8., refuge_slow_only = 0., refuge_post_w = 0., refuge_post_r = 250., refuge_clear = 0., refuge_sprint = 0., site_safe = 0., refuge_verify = 0., wall_conflict = 1., guide_clear = 0., pred_avoid_w = 0., pred_avoid_r = 250., pred_avoid_t = 90., child_prio = 0., sprint_floor = 0., sprint_floor_breed = 1., sprint_floor_unripe = 1., guide_route = 0., guide_mapclear = 0., guide_ctrl = 0., guide_gap = 40., guide_ctrl_acq = 110., guide_lag = 0., guide_chase_cos = 0.8, guide_pv = 0., guide_pv_near = 110., guide_pv_far = 150., guide_pv_dT = 150., guide_plan = 0., guide_safe = 30., guide_keep = 70., guide_sprint_pen = 4., guide_chased = 0., guide_chase_r = 100., guide_release = 200., trap_rear_only = 1., bait_rotate = 0., bait_rot_e = 0.;
    double decoy_old = 0., decoy_e = 0., decoy_r = 150., evade_closest = 0., spawn_pred_r = 0.;
    double keeper_mode = 0., keeper_r = 120., keeper_reserve = 60., rep_timeout = 45., keeper_post_w = 0., keeper_post_r = 250., site_dist_w = 0.02;
    double trap_bait_fixed = -1., guide_near = 45., guide_far = 70., guide_acq_sprint = 0., guide_block_ang = 2.5, guide_slow = 1., guide_fastclose = 8., guide_side_pen = 300., bait_on_sight = 0., guide_sprint_until = 45., guide_max_dist = 0., guide_lane_w = 0., guide_pred_lane_max = 0., guide_wait_max = 6., guide_relay = 0., guide_relay_min = 200., guide_relay_ahead = 180., guide_relay_r = 150., guide_wallclear = 0., pred_wallclear = 0., guide_lead_sprint = 0., guide_acq = 55., guide_min_e = 120., guide_lost = 10., guide_hand = 40.;
    double oracle_r = 600., age_infer = 0., age_fruit = 0., dead_misses = 1., fruit_misses = 1., occ_walls = 0., vis_margin_tree = 20., vis_margin_fruit = 8.;
    double oracle_trees = 0., trap_mode = 0., test_freeze = 0., wall_min_n = 6., trap_depth = 5., wall_tol = 8., wall_min_obs = 2.,
           trap_start = 60., bait_margin = 15., bait_min_life = 25., bait_young_pen = 50., trap_keepout = 80.,
           bait_overlap_seconds = 20., bait_food_lead_seconds = 6., bait_progress_timeout = 8.;   // DIAGNOSTIC ONLY (engine truth): anchored groups know every live tree and its age   // no_spawn: tests only
    double pred_mode = 0., pred_r = 200., pred_sprint_r = 90., pred_face = 1., pred_face_r = 260., pred_share = 0.,
           pred_dodge_r = 0., pred_dodge_ang = 1.5708, pred_dodge_hold = 0., pred_dodge_hold_face = 1.;
    double late_t = OINF, l_fruit_reach = NAN, l_tree_reach = NAN, l_watch_reach = NAN, l_explore_energy = NAN, l_cap_min = NAN, l_cap_mult = NAN, l_cap_tree_slack = NAN, l_cap_hard_min = NAN, l_sweep_rate = NAN, l_watch_patience = NAN, l_explore_radius = NAN, l_old_reach = NAN, l_dist_pen = NAN, l_births_per_tick = NAN, l_emergency_reserve = NAN, l_low_pop_reserve = NAN;
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

    #include "resource_integration.hpp"
    #include "model_world.hpp"
    // ------------------------------------------------------------ groups
    GroupP new_group() {
        auto g = std::make_shared<Group>(); g->id = next_group;
        groups.set(g->id, g); next_group++;
        return g;
    }
    bool dbg_log = false;
    void transform_group(Group& g, double dth, P2 shift) {
        if (dbg_log) fprintf(stderr, "[t=%.1f] transform_group g%lld n=%zu dth=%.3f shift=(%.1f,%.1f)\n", time, (long long)g.id, g.agents.size(), dth, shift.x, shift.y);
        auto T = [&](P2 q) { return add(rot(q, dth), shift); };
        g.agents.each([&](int64_t aid) {
            Mind& m = M(aid);
            m.pose = mkpose(T(m.pose->p), wrap(m.pose->theta + dth));
            if (m.prev_pose) m.prev_pose = mkpose(T(m.prev_pose->p), wrap(m.prev_pose->theta + dth));
            for (auto& e : m.edges) { e.a = T(e.a); e.b = T(e.b); }
            for (auto& b : m.blocked) b.p = T(b.p);
            for (auto& h : m.hear_hist) h.p = T(h.p);
            for (auto& t : m.terrain_samples) t.first = T(t.first);
            if (m.guide_sample_valid) m.guide_sample = T(m.guide_sample);
            if (m.has_explore) m.explore_p = T(m.explore_p);
            if (m.has_watch) m.watch_p = T(m.watch_p);
            if (m.has_detour) m.detour_dir = wrap(m.detour_dir + dth);
            m.has_tkey = false;
        });
        g.trees.each([&](const int64_t&, TreeP& t) { t->p = T(t->p); });
        g.fruits.each([&](const int64_t&, FruitP& f) { f->p = T(f->p); });
        for (auto& w : g.walls) {
            P2 a = w.horiz ? P2{w.lo, w.c} : P2{w.c, w.lo};
            P2 b = w.horiz ? P2{w.hi, w.c} : P2{w.c, w.hi};
            P2 normal = w.horiz ? P2{0., w.solid} : P2{w.solid, 0.};
            a = T(a); b = T(b); normal = rot(normal, dth);
            w.horiz = std::fabs(a.y - b.y) < std::fabs(a.x - b.x);
            w.c = w.horiz ? .5 * (a.y + b.y) : .5 * (a.x + b.x);
            w.lo = w.horiz ? pmin(a.x, b.x) : pmin(a.y, b.y);
            w.hi = w.horiz ? pmax(a.x, b.x) : pmax(a.y, b.y);
            w.solid = (w.horiz ? normal.y : normal.x) < 0. ? -1. : 1.;
            w.cs = {w.c}; w.los = {w.lo}; w.his = {w.hi};
        }
        if (!g.walls.empty()) g.wall_version++;
        g.nav_grids.clear(); g.route_cache.clear(); g.failed_route_cache.clear(); g.route_cache_version = 0;
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
        auto occupied = [&](const Group& g) {
            return g.has_trap && (g.bait >= 0 || g.rep >= 0 || g.guide >= 0 || !g.retired.empty() || !g.leaving.empty());
        };
        bool ga_active = occupied(ga), gb_active = occupied(gb);
        bool same_trap = ga.has_trap && gb.has_trap && dist_lt(ga.trap.goal, gb.trap.goal, 6.);
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
        // Both anchored maps are in the same public frame here. Preserve all
        // confirmed wall evidence and coalesce repeated observations of a face.
        for (auto& src : gb.walls) {
            Group::Wall* dst = nullptr;
            for (auto& w : ga.walls)
                if (w.horiz == src.horiz && w.solid == src.solid && std::fabs(w.c - src.c) <= P.wall_tol
                        && std::fabs(w.lo - src.lo) <= 2. * P.wall_tol && std::fabs(w.hi - src.hi) <= 2. * P.wall_tol) { dst = &w; break; }
            if (!dst) { ga.walls.push_back(src); continue; }
            dst->n += src.n; dst->n_conf += src.n_conf; dst->n_obs += src.n_obs; dst->t = pmax(dst->t, src.t);
            dst->cs.insert(dst->cs.end(), src.cs.begin(), src.cs.end());
            dst->los.insert(dst->los.end(), src.los.begin(), src.los.end());
            dst->his.insert(dst->his.end(), src.his.begin(), src.his.end());
            auto trim_median = [](std::vector<double>& v) {
                std::sort(v.begin(), v.end());
                double m = v[v.size() / 2];
                if (v.size() > 31) { size_t lo = (v.size() - 31) / 2; v = std::vector<double>(v.begin() + lo, v.begin() + lo + 31); }
                return m;
            };
            dst->c = trim_median(dst->cs); dst->lo = trim_median(dst->los); dst->hi = trim_median(dst->his);
        }
        if (!gb.walls.empty()) ga.wall_version++;
        ga.nav_grids.clear(); ga.route_cache.clear(); ga.failed_route_cache.clear(); ga.route_cache_version = 0;

        auto append_unique = [](std::vector<int64_t>& out, int64_t aid) {
            if (aid >= 0 && std::find(out.begin(), out.end(), aid) == out.end()) out.push_back(aid);
        };
        if (gb_active && !ga_active) {
            ga.sites = gb.sites; ga.sites_t = gb.sites_t; ga.has_trap = gb.has_trap; ga.trap = gb.trap;
            ga.trap_since = gb.trap_since; ga.bait = gb.bait; ga.rep = gb.rep; ga.retired = gb.retired; ga.leaving = gb.leaving;
            ga.guide = gb.guide; ga.guide_state = gb.guide_state; ga.guide_since = gb.guide_since; ga.guide_seen = gb.guide_seen;
            ga.guide_pred = gb.guide_pred; ga.relay = gb.relay; ga.keeper = gb.keeper;
            ga.rep_pos = gb.rep_pos; ga.rep_stuck = gb.rep_stuck; ga.rep_since = gb.rep_since;
            ga.rep_entered_rear = gb.rep_entered_rear; ga.rep_progress = gb.rep_progress; ga.rep_progress_t = gb.rep_progress_t;
            ga.bait_gap_since = gb.bait_gap_since;
        } else if (gb_active && ga_active) {
            if (same_trap) {
                if (ga.bait < 0) ga.bait = gb.bait; else if (gb.bait != ga.bait) append_unique(ga.retired, gb.bait);
                if (ga.rep < 0) { ga.rep = gb.rep; ga.rep_entered_rear = gb.rep_entered_rear; ga.rep_progress = gb.rep_progress; ga.rep_progress_t = gb.rep_progress_t; }
                for (int64_t a : gb.retired) append_unique(ga.retired, a);
                for (int64_t a : gb.leaving) append_unique(ga.leaving, a);
            } else {
                // merge() cannot retain two independent trap state machines. Keep
                // the secondary bait holding its already occupied site; release
                // its travelling replacement and guide rather than redirect them.
                append_unique(ga.retired, gb.bait);
                for (int64_t a : gb.retired) append_unique(ga.retired, a);
            }
        }
        ga.bait_arrivals += gb.bait_arrivals; ga.bait_overlaps += gb.bait_overlaps;
        ga.replacement_failures += gb.replacement_failures; ga.bait_gap_seconds += gb.bait_gap_seconds;
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
                    if (dbg_log) fprintf(stderr, "[t=%.1f] MERGE observer %lld (g%lld anch %d pose %.0f,%.0f th %.2f) sees %lld (g%lld anch %d, its pose %.0f,%.0f th %.2f) at d=%.0f ang=%.2f rel=%.2f -> implied (%.0f,%.0f th %.2f)\n",
                                        time, (long long)s.aid, (long long)ga->id, ga->anchored, m.pose->p.x, m.pose->p.y, m.pose->theta, (long long)o.id, (long long)gb->id, gb->anchored, mb.pose->p.x, mb.pose->p.y, mb.pose->theta, o.distance, o.angle, o.rel_dir, pB.x, pB.y, thB);
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
            if (!pose) { g = new_group(); pose = mkpose(P2{0., 0.}, 0.); if (dbg_log) fprintf(stderr, "[t=%.1f] newborn %lld: NO observer -> new group %lld (spawners %zu, new %zu)\n", time, (long long)cid, (long long)g->id, spawners.size(), new_ids.size()); }
            else { g = groups.at(M(parent).group); if (dbg_log) fprintf(stderr, "[t=%.1f] newborn %lld: parent %lld (%s) pose (%.0f,%.0f)\n", time, (long long)cid, (long long)parent, k < spawners.size() ? "spawner" : "observer", pose->p.x, pose->p.y); }
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
        if (dbg_log && !G(m.group).anchored) fprintf(stderr, "[t=%.1f] anchor: agent %lld group %lld (n=%zu) edge L=%.0f\n", time, (long long)m.aid, (long long)m.group, G(m.group).agents.size(), hypot2(x2 - x1, y2 - y1));
        double L = hypot2(x2 - x1, y2 - y1);
        double phi = std::atan2(y2 - y1, x2 - x1);
        double theta; P2 cands[2];
        // the edge in the agent frame rotated by theta is axis-aligned; its offset tells which wall it is:
        // a horizontal edge below the agent (+y after rotation) is the bottom wall's inner face (y = H-30), above it the top face (y = 30)
        if (L > 1500) {
            theta = wrap(-phi);
            P2 r1 = rot(P2{x1, y1}, theta);
            if (r1.y > 0) { cands[0] = {0., H - 30.}; cands[1] = {0., H}; } else { cands[0] = {0., 30.}; cands[1] = {0., 0.}; }
        } else {
            theta = wrap(OPI / 2 - phi);
            P2 r1 = rot(P2{x1, y1}, theta);
            if (r1.x > 0) { cands[0] = {W - 30., 0.}; cands[1] = {W, 0.}; } else { cands[0] = {30., 0.}; cands[1] = {0., 0.}; }
        }
        bool found = false; P2 best{};
        for (P2 sxy : cands) {
            P2 pp = sub(sxy, rot(P2{x1, y1}, theta));
            if (35. <= pp.x && pp.x <= W - 35. && 35. <= pp.y && pp.y <= H - 35.) { best = pp; found = true; break; }
        }
        if (!found) return;
        Group& g = G(m.group);
        if (!g.anchored) {
            double dth = wrap(theta - m.pose->theta);
            P2 shift = sub(best, rot(m.pose->p, dth));
            transform_group(g, dth, shift); g.anchored = true;
            if (dbg_log) fprintf(stderr, "[t=%.1f] anchored agent %lld at (%.0f,%.0f) theta %.2f from edge (%.0f,%.0f)-(%.0f,%.0f) L=%.0f\n", time, (long long)m.aid, m.pose->p.x, m.pose->p.y, m.pose->theta, x1, y1, x2, y2, L);
        } else {
            P2 err = sub(best, m.pose->p);
            double ne = norm(err);
            if (ne > 0.5) {   // boundary evidence is exact: snap the agent (any size of error), keep its heading
                if (dbg_log && ne >= 40.) fprintf(stderr, "[t=%.1f] SNAP agent %lld by %.0f to (%.0f,%.0f)\n", time, (long long)m.aid, ne, best.x, best.y);
                m.pose->p = best;
            }
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
            if ((P.trap_mode > 0. || P.occ_walls > 0. || P.hide_mode > 0. || P.refuge_mode > 0.) && g.anchored && !(model_full()&&resource_mode>=66&&resource_mode<=68)) add_wall(g, a, b, pose.p, m.aid);
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
                if (!fresh && P.age_infer > 0.) {   // nightsim: the tree's cell was in view dt ago without it => age <= dt
                    CellV* cv = g.cells.get(cell_of(p));
                    if (cv && cv->last < time && time - cv->last < P.age_infer) { t->first = 0.5 * (cv->last + time); t->fresh = true; }
                }
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
                if (P.age_fruit > 0.) {   // nightsim: a fruit born by born_hi proves its tree was >= 20 s old then
                    TreeP tn; double tdn = 0;
                    for (auto& t : g.near_trees(p, 70)) { if (t->dead) continue; double dd = dist(t->p, p); if (!tn || dd < tdn) { tn = t; tdn = dd; } }
                    if (tn && !tn->fresh && f->born_hi - 20. < tn->first) tn->first = f->born_hi - 20.;
                }
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
                if (P.occ_walls > 0.)   // nightsim: also the family's permanent confirmed wall map
                    for (auto& w : g.walls) {
                        if (!confirmed(w)) continue;
                        P2 a = w.horiz ? P2{w.lo, w.c} : P2{w.c, w.lo}, b = w.horiz ? P2{w.hi, w.c} : P2{w.c, w.hi};
                        if (segments_cross(mp, p, a, b)) return true;
                    }
                return false;
            };
            for (auto& t : g.near_trees(mp, v))
                if (!vis_t.count(t->id) && in_view(*m.pose, h, c, v, t->p, P.vis_margin_tree) && !occluded(t->p)) vis_t.insert(t->id);
            for (auto& f : g.near_fruits(mp, v))
                if (!vis_f.count(f->id) && in_view(*m.pose, h, c, v, f->p, P.vis_margin_fruit) && !occluded(f->p)) vis_f.insert(f->id);
        });
        for (int64_t tid : g.trees.key_list()) {
            TreeP t = g.trees.at(tid);
            if (t->dead) {
                if (now - t->last > 55.) g.del_tree(tid);
                continue;
            }
            if (g.seen_trees.count(tid)) t->misses = 0;
            else if (vis_t.count(tid)) t->misses++;
            if ((!model_tree_alive(*t) && now > t->first + 62.5) || t->misses >= (int64_t)P.dead_misses) { t->dead = true; continue; }
            IntSet na;
            t->assigned.each([&](int64_t a) {
                if (minds.has(a) && M(a).has_post && M(a).post == tid) na.add(a);
            });
            t->assigned = na;
        }
        for (int64_t fid : g.fruits.key_list()) {
            FruitP f = g.fruits.at(fid);
            if (g.seen_fruits.count(fid)) f->misses = 0;
            else if (vis_f.count(fid)) f->misses++;
            bool gone = now > f->born_hi + 50.05 || f->misses >= (int64_t)P.fruit_misses;
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
        if((resource_mode==13||resource_mode==14)&&resource_synchronized){
            int64_t food_cap=std::max<int64_t>(2,(int64_t)std::floor(.65*resource_trees.size()+.10*resource_fruits.size()));
            ci=std::min(ci,food_cap);
        }
        return ci;
    }
    double fitness(const AState& s) const {
        return (P.fit_vision * std::pow(s.vr / 200., 2.0) * pmin(1.5, s.cone / 1.0472)
                + P.fit_hear * std::pow(s.hear / 50., 2.0)
                + P.fit_energy * pmin(2., s.max_energy / 500.) + P.fit_speed * pmin(P.fit_speed_cap, pmin(s.speed, s.sprint) / 10.));
    }
    // nightsim (Oscar 09:45): keep agents above the engine's sprint cap (20% of max energy) plus a margin
    bool below_floor(const AState& s) const { return P.sprint_floor > 0. && s.energy < 0.2 * s.max_energy + P.sprint_floor; }
    bool ready(const FruitM& f, double energy = OINF, bool old = false) const {
        if (auto decision=forecast_ready(f,energy,old); decision.first) return decision.second;
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
        double reach = global_scarcity_search()?2000.:P.tree_reach * (g.agents.size() <= 1 ? P.lone_reach_mult : 1.);
        if (dist_gt(t.p, m.pose->p, reach)) return -OINF;
        double d = dist(t.p, m.pose->p);
        double walk = pmax(1., pmin(s.speed, s.sprint) * MOVE_PENALTY[s.biome]);
        double travel_t = d / walk / 10.;
        double travel_e = d * 0.05 + travel_t;
        double wait = 0., future = 0.;
        if (!t.dead) {
            if ((double)n >= P.tree_slots) return -OINF;
            double remaining = model_remaining(t, t.fresh ? (t.first + 58. - time) : (t.first + 55. - time));
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
                double ur = model_remaining(*u,u->fresh ? (u->first + 58. - time) : (u->first + 55. - time));
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
        value += forecast_post_adjustment(t,m,s,travel_t);
        if (P.nursery_bonus > 0. && !m.heir_done && s.age >= heir_age_for(s.aid) - 8.)
            value += P.nursery_bonus * (double)std::min<int64_t>(4, t.fruit_free);
        if (P.pred_avoid_w > 0. && !g.pmem.empty()) {   // nightsim: avoid posts where predators were seen recently
            double pen = 0.;
            for (auto& q : g.pmem) if (dist_lt(q.p, t.p, P.pred_avoid_r)) pen += 1. - (time - q.t) / P.pred_avoid_t;
            value -= P.pred_avoid_w * 60. * pmin(pen, 3.);
        }
        if (P.trap_post_w > 0. && g.has_trap) {   // nightsim: prefer posts around the trap so hunting predators pass its mouth
            double dm = dist(t.p, g.trap.mouth);
            if (dm > P.trap_keepout) value += P.trap_post_w * 60. * pmax(0., 1. - dm / P.trap_post_r);
        }
        if (P.refuge_post_w > 0. && !g.sites.empty()) {   // nightsim: prefer posts near a known narrow gap (refuge within reach when chased)
            double dm = OINF; for (auto& st_ : g.sites) dm = pmin(dm, dist(t.p, st_.mouth));
            if (dm > 20.) value += P.refuge_post_w * 60. * pmax(0., 1. - dm / P.refuge_post_r);
        }
        if (P.keeper_post_w > 0. && g.has_trap && g.keeper == m.aid) {   // the keeper prefers posts near the rear entrance
            double dr = dist(t.p, g.trap.rear);
            value += P.keeper_post_w * 60. * pmax(0., 1. - dr / P.keeper_post_r);
        }
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
            if (m.old || is_trap_role(a)) continue;
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
    #include "fruit_assignment.hpp"
    void assign_fruits(Group& g) {
        g.agents.each([&](int64_t a) {
            Mind& m = M(a);
            if (m.has_fruit && (!g.fruits.has(m.fruit) || !g.fruits.at(m.fruit)->has_claim || g.fruits.at(m.fruit)->claimed != a))
                m.has_fruit = false;
        });
        std::vector<FPair> pairs;
        g.agents.each([&](int64_t a) {
            Mind& m = M(a);
            if (m.has_fruit || is_trap_role(a)) return;
            const AState& s = st(a);
            if (s.age > P.no_eat_age) return;
            bool full = s.energy > s.max_energy - 30.;
            // Delayed-model policy experiments: avoid reserving a full fruit
            // when little of its energy fits. Baseline remains unchanged.
            if(resource_synchronized&&((resource_mode==18||resource_mode==20||resource_mode==181||resource_mode==183))&&s.energy>s.max_energy-60.)return;
            double reach = m.old ? P.old_reach : global_scarcity_search()?2000.:P.fruit_reach * (g.agents.size() <= 1 ? P.lone_reach_mult : 1.);
            for (auto& f : g.near_fruits(m.pose->p, reach)) {
                if (f->has_claim) continue;
                double d = dist(f->p, m.pose->p);
                if(global_scarcity_search()&&d>P.fruit_reach){
                    double penalty=MOVE_PENALTY[s.biome];double travel=d/pmax(1.,pmin(s.speed,s.sprint)*penalty*10.);
                    if(s.energy<d*.05/pmax(.1,penalty)+2.*travel+20.)continue;
                    auto it=matched_fruit.find(f.get());if(it!=matched_fruit.end()&&travel+.5>=(100.-it->second.age)/2.)continue;
                }
                bool bf = !m.old && below_floor(s);
                if (!harvest_arrival_ready(*f,m,s,ready(*f, (bf && P.sprint_floor_unripe > 0.) ? -OINF : s.energy, m.old))) continue;
                bool owe_heir = (!m.heir_done) && s.age >= heir_age_for(s.aid) - 5. && s.energy < P.heir_reserve + 20.;
                int64_t bucket;
                if (m.old) bucket = (P.old_eat_last || (exact_lifecycle()&&(resource_mode==41||resource_mode==42))) ? 10 : 5;
                else if (culled.count(a)) bucket = 10;
                else if (P.child_prio > 0. && s.age < 60. && s.energy < 0.2 * s.max_energy + P.child_prio) bucket = -1;
                else if (bf) bucket = -1;   // below the sprint floor: eat first   // nightsim: walk-capped newborns eat first (the engine forbids sprinting below 20% of max energy)
                else if (owe_heir) bucket = 0;
                else if (full) bucket = 9;
                else if (P.feed_breed) bucket = s.energy < reserve() + 20. ? 1 : 2 + int_floordiv(s.energy, 120);
                else bucket = int_floordiv(s.energy, 60);
                double priority=-fitness(s);
                if(resource_synchronized&&(resource_mode==19||resource_mode==20||resource_mode==181||resource_mode==182)){
                    double speed=pmax(1.,pmin(s.speed,s.sprint)*MOVE_PENALTY[s.biome]*10.);
                    double travel=d/speed;
                    double gain=pmin(60.,pmax(0.,s.max_energy-s.energy));
                    // Preserve Lucas's urgent/child/heir buckets; within one
                    // bucket allocate by usable energy per travel second.
                    priority=-gain/(1.+travel);
                }
                pairs.push_back(FPair{bucket, priority, d, a, f->id});
            }
        });
        if(economic_fruit_mode()){economic_assign(g,pairs);return;}
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
        if(routing_mode()&&model_full()&&G(m.group).anchored){P2 next; if(route_next(G(m.group),pose.p,target,5.01,next))target=next;}
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
                double dd, dir, turn; double wait_radius=harvest_wait_radius(*f,s);
                go_to(m, s, f->p, wait_radius, dd, dir, turn);
                return {(wait_radius>0.||(routing_mode()&&model_full()))?dd:pmin(walk, d + 1.), dir, turn};
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

    // ------------------------------------------------------------ oracle (diagnostic upper bound only)
    std::vector<P2> oracle_p; std::vector<double> oracle_age;
    void apply_oracle() {
        // 1: every live tree; 2: only trees within oracle_r of a group member; 3: ages of already-known trees only
        int mode = (int)P.oracle_trees;
        groups.each([&](const int64_t&, GroupP& g) {
            if (!g->anchored) return;
            std::vector<P2> mem; std::vector<double> memr;
            g->agents.each([&](int64_t a) { mem.push_back(M(a).pose->p); memr.push_back(in_states(a) ? st(a).vr : 0.); });
            for (size_t k = 0; k < oracle_p.size(); k++) {
                P2 p = oracle_p[k]; TreeP t; double td = 0;
                if (mode == 2 || mode == 4) {   // 4: within each member's own vision range (ignores cone and walls)
                    bool nearm = false;
                    for (size_t q = 0; q < mem.size(); q++) if (dist_lt(mem[q], p, mode == 2 ? P.oracle_r : memr[q])) { nearm = true; break; }
                    if (!nearm) continue;
                }
                for (auto& c : g->near_trees(p, 12)) {
                    if (c->dead) continue;
                    double dd = dist(c->p, p);
                    if (!t || dd < td) { t = c; td = dd; }
                }
                if (mode == 3) {
                    if (t) { t->first = time - oracle_age[k]; t->fresh = true; }
                    continue;
                }
                if (mode == 5) {   // protect already-known live trees from false 'dead' marks; nothing new, no ages
                    if (t) g->seen_trees.insert(t->id);
                    continue;
                }
                if (!t) {
                    t = std::make_shared<TreeM>();
                    t->id = g->next_tree; t->p = p; t->first = time - oracle_age[k]; t->last = time; t->fresh = true;
                    g->add_tree(t); g->next_tree++;
                }
                t->last = time; g->seen_trees.insert(t->id);
            }
        });
    }

    // ------------------------------------------------------------ trap sites (nightsim)
    static double seg_dist(P2 p, const Group::Wall& w) {
        if (w.horiz) { double x = pmax(w.lo, pmin(w.hi, p.x)); return hypot2(p.x - x, p.y - w.c); }
        double y = pmax(w.lo, pmin(w.hi, p.y)); return hypot2(p.x - w.c, p.y - y);
    }
    static double median_of(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
    void add_wall(Group& g, P2 a, P2 b, P2 obs, int64_t who) {
        if(routing_mode()&&model_full())return;
        bool horiz = std::fabs(a.y - b.y) < std::fabs(a.x - b.x);
        double c = horiz ? 0.5 * (a.y + b.y) : 0.5 * (a.x + b.x);
        double lo = horiz ? pmin(a.x, b.x) : pmin(a.y, b.y), hi = horiz ? pmax(a.x, b.x) : pmax(a.y, b.y);
        if (hi - lo < 8.) return;
        double oc = horiz ? obs.y : obs.x;
        double solid = oc < c ? 1. : -1.;
        for (auto& w : g.walls) {
            if (w.horiz != horiz || std::fabs(w.c - c) > P.wall_tol) continue;
            if (std::fabs(w.lo - lo) > 2. * P.wall_tol || std::fabs(w.hi - hi) > 2. * P.wall_tol) continue;
            if (w.solid != solid) {   // same face seen with the opposite solid side: a mis-posed observer, not a second wall
                if (P.wall_conflict > 0.) {
                    bool was = w.n >= P.wall_min_n && w.n_obs >= P.wall_min_obs && w.n_conf * 4 <= w.n;
                    double old_solid = w.solid; w.n_conf++; if (w.n_conf > w.n) { w.solid = solid; std::swap(w.n, w.n_conf); }
                    bool now = w.n >= P.wall_min_n && w.n_obs >= P.wall_min_obs && w.n_conf * 4 <= w.n;
                    if (was != now || old_solid != w.solid) g.wall_version++;
                    return;
                }
                continue;
            }
            bool was = w.n >= P.wall_min_n && w.n_obs >= P.wall_min_obs && (P.wall_conflict <= 0. || w.n_conf * 4 <= w.n);
            double old_c = w.c, old_lo = w.lo, old_hi = w.hi;
            size_t k = (size_t)(w.n % 31);
            if (w.cs.size() < 31) { w.cs.push_back(c); w.los.push_back(lo); w.his.push_back(hi); }
            else { w.cs[k] = c; w.los[k] = lo; w.his[k] = hi; }
            w.c = median_of(w.cs); w.lo = median_of(w.los); w.hi = median_of(w.his);
            // Failed searches require exact geometry, unlike revalidated successful routes.
            if (old_c != w.c || old_lo != w.lo || old_hi != w.hi) g.failed_route_cache.clear();
            if (who != w.obs1 && who != w.obs2) { if (w.obs1 < 0) w.obs1 = who; else if (w.obs2 < 0) w.obs2 = who; w.n_obs++; }
            w.n++; w.t = time;
            bool now = w.n >= P.wall_min_n && w.n_obs >= P.wall_min_obs && (P.wall_conflict <= 0. || w.n_conf * 4 <= w.n);
            if (was != now || (now && (std::fabs(old_c - w.c) > .5 || std::fabs(old_lo - w.lo) > .5 || std::fabs(old_hi - w.hi) > .5))) g.wall_version++;
            return;
        }
        if (g.walls.size() < 4000) {
            Group::Wall w{horiz, c, lo, hi, solid, time, 1};
            w.cs = {c}; w.los = {lo}; w.his = {hi}; w.obs1 = who; w.n_obs = 1;
            g.walls.push_back(w);
            g.wall_version++;
        }
    }
    bool confirmed(const Group::Wall& w) const { return w.n >= P.wall_min_n && w.n_obs >= P.wall_min_obs && (P.wall_conflict <= 0. || w.n_conf * 4 <= w.n); }
#include "wall_index_methods.hpp"
    bool clear_of(const Group& g, P2 p, double r, int skip1 = -1, int skip2 = -1) const {
        if (p.x < r || p.y < r || p.x > W - r || p.y > H - r) return false;
        for (int i : wi_query(g,p.x-r,p.y-r,p.x+r,p.y+r)) {
            if ((int)i == skip1 || (int)i == skip2 || !confirmed(g.walls[i])) continue;
            const auto& w = g.walls[i];
            double xmin = w.horiz ? w.lo : w.c, xmax = w.horiz ? w.hi : w.c;
            double ymin = w.horiz ? w.c : w.lo, ymax = w.horiz ? w.c : w.hi;
            if (p.x < xmin - r || p.x > xmax + r || p.y < ymin - r || p.y > ymax + r) continue;
            if (seg_dist(p, w) < r) return false;
        }
        return true;
    }
    bool clear_segment(const Group& g, P2 a, P2 b, double r, int skip1 = -1, int skip2 = -1) const {
        if (!clear_of(g, a, r, skip1, skip2) || !clear_of(g, b, r, skip1, skip2)) return false;
        double ab_min_x = pmin(a.x, b.x), ab_max_x = pmax(a.x, b.x);
        double ab_min_y = pmin(a.y, b.y), ab_max_y = pmax(a.y, b.y);
        for (int i : wi_query(g,ab_min_x-r,ab_min_y-r,ab_max_x+r,ab_max_y+r)) {
            if ((int)i == skip1 || (int)i == skip2 || !confirmed(g.walls[i])) continue;
            const auto& w = g.walls[i];
            P2 c = w.horiz ? P2{w.lo, w.c} : P2{w.c, w.lo};
            P2 d = w.horiz ? P2{w.hi, w.c} : P2{w.c, w.hi};
            double cd_min_x = pmin(c.x, d.x), cd_max_x = pmax(c.x, d.x);
            double cd_min_y = pmin(c.y, d.y), cd_max_y = pmax(c.y, d.y);
            if (ab_max_x < cd_min_x - r || ab_min_x > cd_max_x + r
                    || ab_max_y < cd_min_y - r || ab_min_y > cd_max_y + r) continue;
            if (segments_cross(a, b, c, d) || point_segment(c, a, b) < r || point_segment(d, a, b) < r
                    || point_segment(a, c, d) < r || point_segment(b, c, d) < r) return false;
        }
        return true;
    }
    void find_sites(Group& g) {
        g.sites.clear();
        const auto& Wl = g.walls;
        P2 cen{0, 0}; int64_t n = 0;
        g.agents.each([&](int64_t a) { cen = add(cen, M(a).pose->p); n++; });
        if (n) cen = mul(cen, 1.0 / (double)n);
        for (size_t i = 0; i < Wl.size(); i++)
            for (size_t j = 0; j < Wl.size(); j++) {
                const auto& A = Wl[i]; const auto& B = Wl[j];
                if (i == j || A.horiz != B.horiz || !confirmed(A) || !confirmed(B)) continue;
                // A is the face with free space on its +axis side (solid on -), B the face with solid on its + side
                if (!(A.solid < 0 && B.solid > 0)) continue;
                double gap = B.c - A.c;
                if (gap < 10.1 || gap > 19.9) continue;
                double lo = pmax(A.lo, B.lo), hi = pmin(A.hi, B.hi);
                if (hi - lo < 20.) continue;
                double depth = P.trap_depth;
                double xc = A.c + gap / 2;
                for (int end = 0; end < 2; end++) {
                    double m_along = end == 0 ? lo : hi, inward = end == 0 ? 1. : -1.;
                    auto pt = [&](double along, double across) { return A.horiz ? P2{along, across} : P2{across, along}; };
                    P2 mouth = pt(m_along, xc), goal = pt(m_along + depth * inward, xc);
                    P2 rear = pt((end == 0 ? hi : lo) + 20. * inward, xc);
                    if (!clear_of(g, goal, 5.01, (int)i, (int)j)) continue;
                    // predator approach lane outside the mouth: all points at 15..125 out must be clear for r=11
                    bool lane = false; double lane_off = 0.;
                    for (double off : {0., 8., -8., 16., -16., 24., -24.}) {
                        bool ok = true;
                        for (double d : {15., 30., 60., 90., 125.})
                            if (!clear_of(g, pt(m_along - d * inward, xc + off), 11.)) { ok = false; break; }
                        if (ok) { lane = true; lane_off = off; break; }
                    }
                    if (!lane) continue;
                    // The full radius-5.01 rear route must be known clear, not merely its endpoints.
                    bool rear_ok = clear_segment(g, rear, goal, 5.01);
                    if (!rear_ok) continue;
                    Group::Site st{goal, mouth, pt(m_along - 60. * inward, xc + lane_off), rear, hi - lo, gap, 0., rear_ok};
                    st.score = (hi - lo) + (rear_ok ? 30. : 0.) - (n ? P.site_dist_w * dist(goal, cen) : 0.);
                    g.sites.push_back(st);
                }
            }
        std::sort(g.sites.begin(), g.sites.end(), [](const Group::Site& a, const Group::Site& b) { return a.score > b.score; });
        g.sites_t = time;
    }

    // ------------------------------------------------------------ bait (nightsim, trap_mode >= 2)
    bool is_trap_role(int64_t aid) {
        if (P.trap_mode < 2. || !minds.has(aid)) return false;
        Group& g = G(M(aid).group);
        return g.has_trap && (g.bait == aid || g.rep == aid || g.guide == aid || g.relay == aid || std::find(g.retired.begin(), g.retired.end(), aid) != g.retired.end() || std::find(g.leaving.begin(), g.leaving.end(), aid) != g.leaving.end());
    }
    double life_left(const AState& s, const Mind&, double energy_cost = 0.) const {
        // Same conservative closed form as models/core.py: reserve two energy,
        // then solve the quadratic senescence loss after age 60.
        double energy = pmax(0., s.energy - energy_cost - 2.);
        double young = pmin(energy, pmax(0., 60. - s.age));
        energy -= young;
        double age = s.age + young, b = 1. + .1 * age;
        double old = energy > 0. ? 2. * energy / (b + std::sqrt(b * b + .2 * energy)) : 0.;
        return young + old;
    }
    struct BaitEstimate { double distance, seconds, energy_cost; };
    BaitEstimate bait_estimate(Group& g, const Mind& m, const AState& s, const Group::Site& site, bool entered) {
        P2 unused{}; double route = 0.;
        if (entered) {
            if (!route_next(g, m.pose->p, site.goal, 5.01, unused, &route)) return BaitEstimate{OINF, OINF, OINF};
        } else {
            double approach = 0.;
            if (!route_next(g, m.pose->p, site.rear, 5.01, unused, &approach)) return BaitEstimate{OINF, OINF, OINF};
            route = approach + dist(site.rear, site.goal);
        }
        // River speed and walking cost make the time/energy estimate conservative
        // without consulting hidden terrain.
        double walk = pmax(.1, pmin(s.speed, s.sprint));
        return BaitEstimate{route, route / (walk * .3 * 10.), route / .3 * .05};
    }
    void pick_trap(Group& g) {
        if (g.sites.empty()) return;
        if (g.has_trap) {   // keep the current site while it is still reported (within 6 units)
            for (auto& st : g.sites) if (dist_lt(st.goal, g.trap.goal, 6.)) { g.trap = st; return; }
            if (g.bait >= 0 && minds.has(g.bait) && dist_lt(M(g.bait).pose->p, g.trap.goal, 6.)) return;   // bait already holding
        }
        g.trap = g.sites[0]; g.has_trap = true; g.trap_since = time; g.bait = -1; g.rep = -1;
        g.rep_entered_rear = false; g.rep_progress = OINF; g.rep_progress_t = time;
    }
    void bait_plan(Group& g, Mind& m, const AState& s, const Group::Site& st, double deadline, Plan& pl) {
        // Always enter through the verified rear route; never wait at its entrance.
        P2 in = sub(st.goal, st.mouth); double nl = norm(in); in = mul(in, 1.0 / pmax(nl, 1e-6));
        const PoseObj& ps = *m.pose;
        double dg = dist(ps.p, st.goal);
        if (dg <= 3.) {
            double d, ang; local_of(ps, sub(st.mouth, mul(in, 50.)), d, ang);   // stand still, face out of the front mouth
            pl = Plan{0., 0., std::fabs(ang) > 0.2 ? ang : 0.};
            return;
        }
        double walk = pmin(s.speed, s.sprint);
        // A collision detour can push the replacement back through the exposed front.
        if (m.aid == g.rep && g.rep_entered_rear && (sub(ps.p, st.mouth).x * in.x + sub(ps.p, st.mouth).y * in.y) < 0.)
            g.rep_entered_rear = false;
        if (m.aid == g.rep && !g.rep_entered_rear && dist_lt(ps.p, st.rear, 3.)) g.rep_entered_rear = true;
        bool entered_rear = m.aid == g.bait || g.rep_entered_rear;
        P2 target = entered_rear ? st.goal : st.rear;

        // A non-full replacement may take one nearby, observed ripe fruit only
        // when the detour still meets both the handoff deadline and energy reserve.
        if (!entered_rear && s.energy < s.max_energy && deadline < OINF) {
            BaitEstimate base = bait_estimate(g, m, s, st, false);
            bool have = false; double best_extra = OINF; P2 best{};
            for (const Obs& o : *s.obs) {
                if (o.type != 0 || o.distance > 60.) continue;
                P2 fruit = polar(ps, o); FruitP known;
                for (auto& f : g.near_fruits(fruit, 8.)) if (ready(*f) && (!known || dist(f->p, fruit) < dist(known->p, fruit))) known = f;
                if (!known || !clear_segment(g, ps.p, fruit, 5.01) || !clear_segment(g, fruit, target, 5.01)) continue;
                double extra = dist(ps.p, fruit) + dist(fruit, target) - dist(ps.p, target);
                double travel = base.seconds + extra / (pmax(.1, pmin(s.speed, s.sprint)) * .3 * 10.) + .2;
                double cost = base.energy_cost + extra / .3 * .05;
                if (extra <= 30. && travel + 5. < deadline && life_left(s, m, cost) >= travel + 15. && extra < best_extra) {
                    have = true; best_extra = extra; best = fruit;
                }
            }
            if (have) target = best;
        }
        P2 waypoint{};
        if (!route_next(g, ps.p, target, 5.01, waypoint)) { pl = Plan{0., 0., .2}; return; }
        double d, ang; local_of(ps, waypoint, d, ang);
        bool axis_leg = entered_rear && dist_lt(target, st.goal, 1.);
        if (axis_leg) { pl = Plan{pmin(walk, d), ang, 0.}; return; }   // inside the channel: straight along the axis
        // A* already proves radius-5.01 clearance. The orchard navigator's
        // separate 7-unit heuristic would steer away from valid narrow routes.
        double distance, direction; local_of(ps, waypoint, distance, direction);
        pl = Plan{pmin(walk, distance), direction, pmax(-P.travel_turn,pmin(P.travel_turn,direction))};
    }
    void run_trap(std::unordered_map<int64_t, Plan>& plans) {
        if (time < P.trap_start) return;
        groups.each([&](const int64_t&, GroupP& gp) {
            Group& g = *gp;
            if (!g.anchored) return;
            pick_trap(g);
            if (!g.has_trap) return;
            for (auto& q : g.pseen) if (!dist_lt(q.p, g.trap.mouth, 40.)) { g.guide_seen = time; break; }
            if (P.trap_bait_fixed >= 0. && minds.has((int64_t)P.trap_bait_fixed)) g.bait = (int64_t)P.trap_bait_fixed;
            if (g.bait >= 0 && (!minds.has(g.bait) || M(g.bait).group != g.id)) g.bait = -1;
            if (g.rep >= 0 && (!minds.has(g.rep) || M(g.rep).group != g.id)) {
                g.rep = -1; g.rep_entered_rear = false; g.replacement_failures++;
            }
            // replacement reached the goal -> it becomes the bait (the old bait stays there until it dies)
            {
                std::vector<int64_t> keep;
                for (int64_t a : g.retired) if (minds.has(a) && M(a).group == g.id) keep.push_back(a);
                g.retired.swap(keep);
            }
            {
                std::vector<int64_t> keep;
                for (int64_t a : g.leaving) if (minds.has(a) && M(a).group == g.id && !dist_lt(M(a).pose->p, g.trap.rear, 6.)) keep.push_back(a);
                g.leaving.swap(keep);
            }
            if (g.rep >= 0 && dist_lt(M(g.rep).pose->p, g.trap.goal, 4.)) {
                bool overlap = g.bait >= 0 && dist_lt(M(g.bait).pose->p, g.trap.goal, 4.);
                if (overlap) {
                    g.bait_overlaps++;
                    if (P.bait_rotate > 0. && g.trap.rear_ok) { g.leaving.push_back(g.bait); g.baits_rotated++; }
                    else g.retired.push_back(g.bait);
                }
                g.bait = g.rep; g.rep = -1;
                g.rep_entered_rear = false; g.rep_progress = OINF; g.bait_arrivals++;
            }
            bool bait_ready = g.bait >= 0 && dist_lt(M(g.bait).pose->p, g.trap.goal, 4.);
            if (!bait_ready && g.bait_gap_since < 0.) g.bait_gap_since = time;
            if (bait_ready && g.bait_gap_since >= 0.) { g.bait_gap_seconds += time - g.bait_gap_since; g.bait_gap_since = -1.; }
            if (g.rep >= 0) {   // release a replacement after eight seconds without route progress
                Mind& rm = M(g.rep);
                BaitEstimate estimate = bait_estimate(g, rm, st(g.rep), g.trap, g.rep_entered_rear);
                if (estimate.distance < g.rep_progress - 2.) { g.rep_progress = estimate.distance; g.rep_progress_t = time; }
                g.rep_pos = rm.pose->p;
                if (time - g.rep_progress_t > P.bait_progress_timeout) {
                    g.rep = -1; g.rep_stuck = 0; g.rep_entered_rear = false; g.rep_progress = OINF;
                    g.replacement_failures++;
                }
            }
            double need = OINF;
            if (g.bait >= 0) need = life_left(st(g.bait), M(g.bait));
            if (g.bait >= 0 && P.bait_rot_e > 0. && g.trap.rear_ok && st(g.bait).energy < P.bait_rot_e) need = pmin(need, 0.);   // nightsim: ask for a replacement early so the bait can leave and eat
            bool want_bait = P.bait_on_sight <= 0. || time - g.guide_seen < P.bait_on_sight || g.bait >= 0;   // trap on demand: only after a recent sighting
            if (P.keeper_mode > 0.) {
                // keeper: the member nearest the rear entrance holds a post there and spawns the next bait as a child
                if (g.keeper >= 0 && (!minds.has(g.keeper) || M(g.keeper).group != g.id || M(g.keeper).old || is_trap_role(g.keeper))) g.keeper = -1;
                if (g.keeper < 0) {
                    int64_t bk = -1; double bd = OINF;
                    g.agents.each([&](int64_t a) {
                        if (is_trap_role(a) || M(a).old || st(a).age > P.heir_age - 10.) return;
                        double d = dist(M(a).pose->p, g.trap.rear) - 0.2 * st(a).energy;   // near the rear, energetic, young
                        if (d < bd) { bd = d; bk = a; }
                    });
                    g.keeper = bk;
                }
                // adopt the keeper's newborn as the replacement bait
                if (g.keeper_spawn && g.rep < 0) {
                    int64_t nb = -1; double bd = 60.;
                    g.agents.each([&](int64_t a) {
                        if (is_trap_role(a) || a == g.keeper || st(a).age > 2.) return;
                        double d = dist(M(a).pose->p, M(g.keeper).pose->p);
                        if (d < bd) { bd = d; nb = a; }
                    });
                    if (nb >= 0) { g.rep = nb; g.rep_since = time; g.rep_stuck = 0; g.rep_entered_rear = false; g.rep_progress = OINF; g.rep_progress_t = time; g.keeper_spawn = false; g.baits_born++; Mind& m = M(nb); m.has_post = false; m.has_fruit = false; }
                    else if (time - g.keeper_spawn_t > 1.5) g.keeper_spawn = false;
                }
                if (g.keeper >= 0 && g.rep < 0 && want_bait && (g.bait < 0 || need < P.bait_margin + 60.)) {
                    const AState& ks = st(g.keeper);
                    if (dist_lt(M(g.keeper).pose->p, g.trap.rear, P.keeper_r) && ks.energy > 100. + P.keeper_reserve) { g.keeper_spawn = true; g.keeper_spawn_t = time; }
                }
            }
            if (P.keeper_mode <= 0. && P.trap_bait_fixed < 0. && g.rep < 0 && want_bait) {
                int64_t best = -1; double bs = -OINF;
                double best_travel = OINF;
                g.agents.each([&](int64_t a) {
                    if (is_trap_role(a)) return;
                    const AState& s = st(a); Mind& m = M(a);
                    // Best-case lower bounds: straight rear route, fastest public
                    // terrain and minimum walking cost. Failure here proves the
                    // radius-aware conservative route cannot be viable either.
                    double lower_distance = dist(m.pose->p, g.trap.rear) + dist(g.trap.rear, g.trap.goal);
                    double walk = pmax(.1, pmin(s.speed, s.sprint));
                    double lower_travel = lower_distance / (walk * 10.);
                    double upper_life = life_left(s, m, lower_distance * .05);
                    if (upper_life < pmax(P.bait_min_life, lower_travel + 15.)) return;
                    BaitEstimate estimate = bait_estimate(g, m, s, g.trap, false);
                    double travel = estimate.seconds;
                    double life = life_left(s, m, estimate.energy_cost);
                    if (life < pmax(P.bait_min_life, travel + 15.)) return;
                    bool meets_deadline = g.bait < 0 || travel + 5. < need;
                    double score = (meets_deadline ? 1000000. : 0.) + (m.old ? 1000. : 0.)
                                 + (m.old ? 0. : s.age) - travel - (m.old ? 0. : P.bait_young_pen);
                    if (score > bs) { bs = score; best = a; best_travel = travel; }
                });
                double food_lead = best >= 0 && st(best).energy < st(best).max_energy - 30. ? P.bait_food_lead_seconds : 0.;
                bool due = g.bait < 0 || (best >= 0 && need <= best_travel + P.bait_overlap_seconds + food_lead);
                if (best >= 0 && due) {
                    g.rep = best; g.rep_since = time; g.rep_stuck = 0; g.rep_entered_rear = false;
                    g.rep_progress = OINF; g.rep_progress_t = time; Mind& m = M(best);
                    if (m.has_post && g.trees.has(m.post)) g.trees.at(m.post)->assigned.discard(best);
                    m.has_post = false;
                    if (m.has_fruit && g.fruits.has(m.fruit)) g.fruits.at(m.fruit)->has_claim = false;
                    m.has_fruit = false;
                }
            }
            if (P.keeper_mode > 0. && g.keeper >= 0) {
                Mind& km = M(g.keeper); const AState& ks = st(g.keeper);
                double dk = dist(km.pose->p, g.trap.rear);
                bool needed = g.rep < 0 && (g.bait < 0 || need < P.bait_margin + 60.) && ks.energy > 100. + P.keeper_reserve;
                if (dk > P.keeper_r && needed) { double dd, dir, turn; go_to(km, ks, g.trap.rear, P.keeper_r * 0.6, dd, dir, turn); plans[g.keeper] = Plan{dd, dir, turn}; }
            }
            if (g.bait >= 0) bait_plan(g, M(g.bait), st(g.bait), g.trap, OINF, plans[g.bait]);
            if (g.rep >= 0) bait_plan(g, M(g.rep), st(g.rep), g.trap, need, plans[g.rep]);
            for (int64_t a : g.retired) plans[a] = Plan{0., 0., 0.};
            for (int64_t a : g.leaving) { double dd, dir, turn; go_to(M(a), st(a), g.trap.rear, 0., dd, dir, turn); plans[a] = Plan{dd, dir, turn}; }
            if (P.trap_mode >= 3.) run_guide(g, plans);
            // everyone else keeps clear of the mouth so the held predators' closest agent stays the bait
            g.agents.each([&](int64_t a) {
                if (a == g.bait || a == g.rep || a == g.guide || a == g.keeper || std::find(g.retired.begin(), g.retired.end(), a) != g.retired.end()) return;
                Mind& m = M(a); double d = dist(m.pose->p, g.trap.mouth);
                if (d < P.trap_keepout) {
                    double dd, ang; local_of(*m.pose, g.trap.mouth, dd, ang);
                    const AState& s = st(a);
                    plans[a] = Plan{pmin(s.speed, s.sprint), wrap(ang + OPI), 0.};
                }
            });
        });
    }


    // Public sensing only: no simulator state or predator identity is available here.
    bool predator_detects(Group& g, P2 predator, double heading, P2 agent) const {
        double d=dist(predator,agent);
        if (d<=60.) return true;
        if (d>250. || std::fabs(wrap(std::atan2(agent.y-predator.y,agent.x-predator.x)-heading))>OPI/6.) return false;
        for (const auto& w:g.walls) if (confirmed(w)) {
            P2 a=w.horiz?P2{w.lo,w.c}:P2{w.c,w.lo};
            P2 b=w.horiz?P2{w.hi,w.c}:P2{w.c,w.hi};
            if (segments_cross(predator,agent,a,b)) return false;
        }
        return true;
    }
    void run_native_guide(Group& g, std::unordered_map<int64_t, Plan>& plans) {
        if (g.guide>=0 && (!minds.has(g.guide) || M(g.guide).group!=g.id)) {
            g.ep_died++; g.guide=-1; g.guide_state=0;
        }
        if (g.bait<0 || !minds.has(g.bait) || dist(M(g.bait).pose->p,g.trap.goal)>6.) {
            g.guide=-1; g.guide_state=0; return;
        }
        auto held=[&](P2 p) { return dist(p,g.trap.goal)<=60.; };
        if (g.guide<0) {
            int64_t best=-1; double score=-OINF; P2 target{};
            g.agents.each([&](int64_t aid) {
                if (is_trap_role(aid) || frozen.count(aid)) return;
                Mind& m=M(aid); const AState& s=st(aid);
                if (s.energy < pmax(P.guide_min_e,s.max_energy*.2+10.)) return;
                for (const auto& o:*s.obs) if (o.type==2) {
                    P2 p=polar(*m.pose,o);
                    double heading=m.pose->theta+o.angle+OPI-(o.has_rel_dir?o.rel_dir:0.);
                    if (held(p) || !predator_detects(g,p,heading,m.pose->p)) continue;
                    // Reuse an agent already pursued. Prefer retirement only when
                    // enough life remains to be useful; proximity still matters.
                    double travel=dist(m.pose->p,g.trap.out)/(pmax(.1,pmin(s.speed,s.sprint))*.5*10.);
                    if (life_left(s,m)<travel+3.) continue;
                    double value=(m.old?30.:0.)-o.distance+.02*s.energy;
                    if (value>score) { score=value; best=aid; target=p; }
                }
            });
            if (best<0) return;
            g.guide=best; g.guide_state=2; g.guide_since=time; g.guide_seen=time;
            g.guide_pred=target; g.guide_has_prev=false; g.ep_start++; g.ep_h=false; g.ep_chased=true;
            Mind& m=M(best);
            if (m.has_post && g.trees.has(m.post)) g.trees.at(m.post)->assigned.discard(best);
            if (m.has_fruit && g.fruits.has(m.fruit)) g.fruits.at(m.fruit)->has_claim=false;
            m.has_post=false; m.has_fruit=false; m.guide_sample_valid=false;
        }
        Mind& m=M(g.guide); const AState& s=st(g.guide); const auto& ps=*m.pose;
        const Obs* target=nullptr; double closest=OINF;
        for (const auto& o:*s.obs) if (o.type==2) {
            P2 p=polar(ps,o);
            if (held(p) && dist(p,g.guide_pred)>40.) continue;
            double d=dist(p,g.guide_pred);
            if (d<closest) { closest=d; target=&o; }
        }
        if (!target) {
            double d,ang; local_of(ps,g.guide_pred,d,ang);
            if (time-g.guide_seen>2.) { g.guide=-1; g.guide_state=0; g.ep_lost++; return; }
            // Reacquire from the last public sighting; the shared safety pass
            // still runs if the role is released. Never use an unseen true pose.
            double dd,dir,turn; go_to(m,s,g.guide_pred,100.,dd,dir,turn);
            plans[g.guide]=Plan{dd,dir,ang}; return;
        }
        g.guide_pred=polar(ps,*target); g.guide_seen=time;
        double ph=ps.theta+target->angle+OPI-(target->has_rel_dir?target->rel_dir:0.);
        bool following=predator_detects(g,g.guide_pred,ph,ps.p);
        P2 outward=sub(g.trap.mouth,g.trap.goal); outward=mul(outward,1./pmax(norm(outward),1e-6));
        P2 delivery=add(g.trap.goal,mul(outward,P.guide_hand-1.));
        double dB=dist(ps.p,g.trap.goal);
        bool in_front=sub(ps.p,g.trap.mouth).x*outward.x+sub(ps.p,g.trap.mouth).y*outward.y>=0.;
        if (dB<=P.guide_hand && in_front && following) {
            g.guide_state=3;
            if (!g.ep_h) { g.ep_h=true; g.ep_hand++; }
            plans[g.guide]=Plan{0.,0.,target->angle}; return;
        }
        g.guide_state=dB<100.?3:2; g.ep_h=false;
        P2 destination=following?delivery:g.guide_pred;
        P2 waypoint=destination;
        route_next(g,ps.p,destination,5.01,waypoint);
        double dd,dir,turn; go_to(m,s,waypoint,following?1.:50.,dd,dir,turn);
        entrapment_guide::Input input;
        input.agent={s.energy,s.max_energy,s.age,s.speed,s.sprint,MOVE_PENALTY[s.biome]};
        input.nominal={dd,dir,target->angle}; input.tick=(int)std::llround(time*10.);
        input.reacquiring=!following; input.preferred_min=P.guide_near; input.preferred_max=P.guide_far;
        auto local=[&](P2 p) { P2 q=rot(sub(p,ps.p),-ps.theta); return entrapment_guide::P2{q.x,q.y}; };
        input.bait=local(g.trap.goal); input.route_waypoints.push_back(local(waypoint));
        for (const auto& o:*s.obs) if (o.type==2) {
            if (&o==target) input.target_index=input.predators.size();
            input.predators.push_back({o.distance,o.angle,o.rel_dir,o.has_rel_dir});
        }
        auto wall=[&](P2 a,P2 b) { if (point_segment(ps.p,a,b)<350.) input.walls.push_back({local(a),local(b)}); };
        for (const auto& w:g.walls) if (confirmed(w)) wall(w.horiz?P2{w.lo,w.c}:P2{w.c,w.lo},w.horiz?P2{w.hi,w.c}:P2{w.c,w.hi});
        for (const auto& t:m.terrain_samples) input.terrain_samples.push_back({local(t.first),t.second,0.});
        if (m.guide_sample_valid) input.previous={true,m.guide_sample_tick,local(m.guide_sample)};
        auto result=entrapment_guide::search(input);
        m.guide_sample_valid=true; m.guide_sample_tick=input.tick; m.guide_sample=g.guide_pred;
        plans[g.guide]=result.found?Plan{result.plan.move_distance,result.plan.move_direction,result.plan.turn_angle}:Plan{dd,dir,target->angle};
    }

    // ------------------------------------------------------------ guide (nightsim, trap_mode >= 3): one predator to the trap
    // states: 1 ACQUIRE (get within guide_acq so it locks on), 2 LEAD (face it, back toward the lane point, keep
    // guide_near..guide_far), 3 DELIVER (back through the mouth past the bait, out the rear or stop deeper), 4 DONE.
    void run_guide(Group& g, std::unordered_map<int64_t, Plan>& plans) {
        if (P.entrapment_lookahead > 0.) { run_native_guide(g, plans); return; }
        if (g.guide < 0 && g.relay >= 0) g.relay = -1;
        if (g.guide >= 0 && (!minds.has(g.guide) || M(g.guide).group != g.id)) {
            g.ep_died++; gstat[6]++;
            if (g.gl_dT > 300.) g.d_far++;
            if (g.gl_npred >= 2) g.d_multi++;
            if (g.gl_speed < 12.) g.d_slow++;
            if (g.gl_stuck >= 5) g.d_stuck++;
            if (g.gl_ticks < 30) g.d_early++;
            if (g.guide_state == 1) g.d_state1++;
            g.guide = -1; g.guide_state = 0; g.guide_end_t = time;
        }
        // nearest shared predator sighting that is not already held at the mouth
        bool have = false; P2 pp{}; double best = OINF;
        for (auto& q : g.pseen) {
            if (dist_lt(q.p, g.trap.mouth, 40.)) continue;
            if (g.guide < 0 && P.guide_pred_lane_max > 0. && dist_gt(q.p, g.trap.out, P.guide_pred_lane_max)) continue;   // only predators already near the trap
            double d = g.guide >= 0 ? dist(q.p, M(g.guide).pose->p) : dist(q.p, g.trap.mouth);
            if (d < best) { best = d; pp = q.p; have = true; }
        }
        if (have) { g.guide_pred = pp; g.guide_seen = time; }
        {   // deliveries: count sightings held at the mouth; a new one during/just after a guide episode is a delivery
            int64_t hn = 0; for (auto& q : g.pseen) if (dist_lt(q.p, g.trap.mouth, 40.)) hn++;
            if (hn > g.held_max && (g.guide >= 0 || time - g.guide_end_t < 6.)) { g.guide_done++; if (g.guide >= 0) { g.guide = -1; g.guide_state = 0; g.guide_end_t = time; g.ep_deliv++; gstat[7]++; } }
            if (hn > g.held_max) g.held_max = hn;
            if (hn < g.held_max) g.held_max = hn;   // follow drops so the next arrival counts again
        }
        if (g.bait < 0) return;
        if (g.guide < 0 && P.guide_chased > 0.) {   // nightsim (Oscar 11:25): nobody walks toward a predator; the agent a predator is chasing becomes the guide
            int64_t bg = -1; double bd = OINF; P2 bq{};
            for (auto& q : g.pseen) {
                if (dist_lt(q.p, g.trap.mouth, 40.)) continue;
                if (P.guide_pred_lane_max > 0. && dist_gt(q.p, g.trap.out, P.guide_pred_lane_max)) continue;   // only predators already near the trap
                int64_t na = -1; double nd = OINF;   // the predator's target = its closest agent (bait/retired included: they cannot guide)
                g.agents.each([&](int64_t a) { double d = dist(M(a).pose->p, q.p); if (d < nd) { nd = d; na = a; } });
                if (na < 0 || nd > P.guide_chase_r || is_trap_role(na) || frozen.count(na)) continue;
                if (nd < bd) { bd = nd; bg = na; bq = q.p; }
            }
            if (bg < 0) return;
            g.guide = bg; g.guide_state = 2; g.guide_since = time; g.ep_start++; gstat[11]++; g.guide_sprinting = false; g.ep_chased = g.ep_s3 = g.ep_h = false;
            g.gl_ticks = 0; g.gl_stuck = 0; g.gl_pos = M(bg).pose->p; g.guide_pred = bq; g.guide_seen = time; g.guide_closing_t = time;
            Mind& m = M(bg);
            if (m.has_post && g.trees.has(m.post)) g.trees.at(m.post)->assigned.discard(bg);
            m.has_post = false;
            if (m.has_fruit && g.fruits.has(m.fruit)) g.fruits.at(m.fruit)->has_claim = false;
            m.has_fruit = false;
        }
        if (g.guide < 0) {
            if (!have) return;
            int64_t bg = -1; double bs = -OINF;
            g.agents.each([&](int64_t a) {
                if (is_trap_role(a) || frozen.count(a)) return;
                const AState& s = st(a); Mind& m = M(a);
                if (s.energy < P.guide_min_e || m.old) return;   // old agents drain 10+/s: they cannot guide
                if (P.guide_max_dist > 0. && dist(m.pose->p, pp) > P.guide_max_dist) return;   // do not send guides across the map
                if (P.guide_route > 0.) { P2 nx; if (!route_next(g, m.pose->p, g.trap.out, 6., nx)) return; }
                else if (P.guide_clear > 0. && !path_clear(g, m.pose->p, g.trap.out, 6.)) return;   // nightsim: the straight walk to the lane point must not cross known walls
                double sc = s.energy * 0.1 - dist(m.pose->p, pp) * 0.05 - dist(m.pose->p, g.trap.out) * P.guide_lane_w + pmin(s.speed, s.sprint) * 10.;
                {   // prefer candidates that are NOT on the far side of the predator from the lane (predator between them and the trap)
                    P2 a = sub(g.trap.out, m.pose->p), b = sub(pp, m.pose->p);
                    double na = norm(a), nb = norm(b);
                    if (na > 1. && nb > 1. && (a.x * b.x + a.y * b.y) / (na * nb) > 0.45) sc -= P.guide_side_pen;
                }
                if (sc > bs) { bs = sc; bg = a; }
            });
            if (bg < 0) return;
            g.guide = bg; g.guide_state = 1; g.guide_since = time; g.ep_start++; gstat[11]++; g.guide_sprinting = false; g.ep_chased = g.ep_s3 = g.ep_h = false; g.gl_ticks = 0; g.gl_stuck = 0; g.gl_pos = M(bg).pose->p;
            Mind& m = M(bg);
            if (m.has_post && g.trees.has(m.post)) g.trees.at(m.post)->assigned.discard(bg);
            m.has_post = false;
            if (m.has_fruit && g.fruits.has(m.fruit)) g.fruits.at(m.fruit)->has_claim = false;
            m.has_fruit = false;
        }
        Mind& m = M(g.guide); const AState& s = st(g.guide); const PoseObj& ps = *m.pose;
        double walk = pmin(s.speed, s.sprint);
        double dP, angP; local_of(ps, g.guide_pred, dP, angP);
        {   // status for death attribution
            g.gl_dT = dist(ps.p, g.trap.out); g.gl_speed = walk; g.gl_ticks++;
            int64_t np_ = 0; for (auto& q : g.pseen) if (dist_lt(q.p, ps.p, 120.)) np_++;
            g.gl_npred = np_;
            if (dist_lt(ps.p, g.gl_pos, 3.)) g.gl_stuck++; else g.gl_stuck = 0;
            g.gl_pos = ps.p;
        }
        bool fresh = time - g.guide_seen < 0.15;
        double closing_rate = 0.;   // units per tick the gap shrank since the last sighting
        if (fresh) {
            if (g.guide_has_prev) {   // predator approaching us: its own displacement points at us
                P2 mv = sub(g.guide_pred, g.guide_pred_prev); P2 to = sub(ps.p, g.guide_pred_prev);
                double nm = norm(mv), nt = norm(to);
                if (nm > 0.5 && nt > 1. && (mv.x * to.x + mv.y * to.y) / (nm * nt) > P.guide_chase_cos && dP < g.guide_dprev + 0.5) g.guide_closing_t = time;
                if (g.guide_dprev >= 0.) closing_rate = g.guide_dprev - dP;
            }
            g.guide_pred_prev = g.guide_pred; g.guide_has_prev = true; g.guide_dprev = dP;
        }
        bool chasing = time - g.guide_closing_t < 1.5;
        if (g.guide_state >= 1 && g.guide_state <= 3) gstat[7 + g.guide_state]++;
        if (dbg_log) fprintf(stderr, "[t=%.1f] run_guide g%lld guide %lld state %d relay_p %.0f relay %lld\n", time, (long long)g.id, (long long)g.guide, g.guide_state, P.guide_relay, (long long)g.relay);
        if (P.guide_relay > 0. && g.guide_state == 2) {
            // relay guiding: a fresh member waits on the lane ahead of the guide; when the predator comes within guide_acq of it,
            // it becomes the guide (the predator switches to its closest agent) and the old guide is released
            if (g.relay >= 0 && (!minds.has(g.relay) || M(g.relay).group != g.id || g.relay == g.guide)) g.relay = -1;
            P2 to_lane = sub(g.trap.out, ps.p); double dl = norm(to_lane);
            if (g.relay < 0 && dl > P.guide_relay_min) {
                P2 u = mul(to_lane, 1.0 / pmax(dl, 1e-6));
                P2 spot = add(ps.p, mul(u, pmin(P.guide_relay_ahead, dl - 20.)));
                int64_t br = -1; double bs = OINF;
                g.agents.each([&](int64_t a) {
                    if (is_trap_role(a) || frozen.count(a) || M(a).old || st(a).energy < P.guide_min_e) return;
                    double d = dist(M(a).pose->p, spot);
                    if (d < P.guide_relay_r && d < bs) { bs = d; br = a; }
                });
                if (dbg_log) fprintf(stderr, "[t=%.1f] relay search: guide %lld dl=%.0f spot=(%.0f,%.0f) members=%zu -> %lld (best d %.0f)\n", time, (long long)g.guide, dl, spot.x, spot.y, g.agents.size(), (long long)br, bs);
                if (br >= 0) { g.relay = br; Mind& rm = M(br); if (rm.has_post && g.trees.has(rm.post)) g.trees.at(rm.post)->assigned.discard(br); rm.has_post = false; rm.has_fruit = false; }
            }
            if (g.relay >= 0) {
                Mind& rm = M(g.relay); const AState& rs = st(g.relay);
                P2 u = mul(to_lane, 1.0 / pmax(dl, 1e-6));
                P2 spot = add(ps.p, mul(u, pmin(P.guide_relay_ahead, dl - 20.)));
                double dpr = dist(rm.pose->p, g.guide_pred);
                if (dbg_log) fprintf(stderr, "[t=%.1f] relay %lld dpr=%.0f fresh=%d chasing=%d\n", time, (long long)g.relay, dpr, fresh, chasing);
                if (fresh && dpr < P.guide_acq && chasing) {   // hand over
                    int64_t old_g = g.guide; g.guide = g.relay; g.relay = -1; g.relays_done++;
                    g.guide_dprev = -1.; g.guide_has_prev = false; g.guide_sprinting = false; g.wait_since = -1.;
                    (void)old_g;   // the old guide returns to normal duty (it is now behind the predator)
                    return;
                }
                double dd, dir, turn; go_to(rm, rs, spot, 6., dd, dir, turn);
                double dP2, angP2; local_of(*rm.pose, g.guide_pred, dP2, angP2);
                plans[g.relay] = Plan{dd, dir, dP2 < 150. ? angP2 : turn};
            }
        }
        if (g.guide_state == 1 && P.guide_chased > 0.) {   // chased mode: no acquiring; the chase is over -> release the guide to normal duty
            gstat[4]++; g.guide = -1; g.guide_state = 0; g.guide_dprev = -1.; g.guide_has_prev = false; g.ep_lost++; return;
        }
        if (g.guide_state == 1) {
            if (time - g.guide_seen > P.guide_lost) { gstat[4]++; g.guide = -1; g.guide_state = 0; g.guide_dprev = -1.; g.guide_has_prev = false; g.ep_lost++; return; }
            double acq_r = P.guide_ctrl > 0. ? P.guide_ctrl_acq : P.guide_acq;   // nightsim guide_ctrl: never walk into a charging predator
            if (fresh && (dP <= acq_r || chasing)) { g.guide_state = 2; }
            else {   // get into its hearing range fast: sprint when it is not coming to us
                double dd, dir, turn; go_to(m, s, g.guide_pred, acq_r - 10., dd, dir, turn);
                if (P.guide_acq_sprint > 0. && dP > 80. && !chasing) dd = pmin(s.sprint, pmax(0., dP - 45.));
                plans[g.guide] = Plan{dd, dir, turn}; return;
            }
        }
        if (g.guide_state == 2) {
            if (time - g.guide_seen > P.guide_lost) { gstat[0]++; g.guide_state = 1; return; }
            P2 lead_target = g.trap.out;
            if (P.guide_route > 0.) { P2 nx; if (!route_next(g, ps.p, g.trap.out, 6., nx)) { gstat[5]++; g.guide = -1; g.guide_state = 0; g.guide_dprev = -1.; g.guide_has_prev = false; g.ep_lost++; return; } lead_target = nx; }
            else if (P.guide_clear > 0. && !path_clear(g, ps.p, g.trap.out, 6.)) { gstat[5]++; g.guide = -1; g.guide_state = 0; g.guide_dprev = -1.; g.guide_has_prev = false; g.ep_lost++; return; }
            if (P.guide_chased > 0.) { if (dP > P.guide_release) { gstat[1]++; g.guide_state = 1; return; } }   // chased mode: keep leading while it is in sight and within guide_release
            else if (dP > P.guide_acq && !chasing && time - g.guide_closing_t > 2.) { gstat[1]++; g.guide_state = 1; return; }
            double dT, angT; local_of(ps, g.trap.out, dT, angT);
            if (P.guide_route > 0. && !(lead_target.x == g.trap.out.x && lead_target.y == g.trap.out.y)) { double dV, angV; local_of(ps, lead_target, dV, angV); angT = angV; }   // steer toward the via point; dT stays the true remaining distance
            if (chasing && !g.ep_chased) { g.ep_chased = true; g.ep_chase++; }
            if (dT < 12. && chasing && dP < P.guide_far + 30.) { g.guide_state = 3; if (!g.ep_s3) { g.ep_s3 = true; g.ep_state3++; } }
            else {
                // stay in front of the predator: never pass it, keep it in its senses (< ~200), never let it reach 15
                double step = (P.guide_lead_sprint > 0. && dP < P.guide_far) ? s.sprint : walk, dir = angT;   // nightsim: sprint-lead inside the band
                double off = wrap(angT - angP);   // lane direction relative to the predator direction
                double sgn = off > 0 ? 1. : -1.;
                bool blocked_ = std::fabs(off) < 1.1;                                               // predator between us and the lane
                // nightsim guide_pv: in the open (far from the lane point, clear line of sight to the predator in the map) keep it
                // beyond 1.5 x hearing so it only pivots (~10.6/tick) and a walking guide holds the gap; near walls use the tight band
                double b_near = P.guide_near, b_until = P.guide_sprint_until, b_far = P.guide_far;
                if (P.guide_pv > 0. && dT > P.guide_pv_dT && path_clear(g, ps.p, g.guide_pred, 1.)) { b_near = P.guide_pv_near; b_until = P.guide_pv_near + 2.; b_far = P.guide_pv_far; }
                if (P.guide_ctrl > 0.) {   // nightsim: sprint exactly when a walk step would leave less than guide_gap after the predator's worst-case 15 step
                    g.guide_sprinting = dP - P.guide_lag + walk * MOVE_PENALTY[s.biome] - 15. < P.guide_gap;   // guide_lag: the sighting is one predator step old
                } else if (dP < b_near) g.guide_sprinting = true;                                    // too close: start sprinting
                else if (dP > b_until) g.guide_sprinting = false;                     // hysteresis: keep sprinting until this far
                if (g.guide_sprinting) { step = s.sprint; dir = wrap(angP + (blocked_ ? sgn * P.guide_block_ang : OPI)); }
                else if (blocked_) { step = walk; dir = wrap(angP + sgn * 1.9); }                              // predator in the way: circle it, drifting away
                else if (dP > b_far && P.guide_slow > 0. && !(P.guide_fastclose > 0. && closing_rate > P.guide_fastclose && dP < b_far + 40.)) {
                    step = walk * pmax(0., (b_far + 20. - dP) / 20.);   // slow down beyond the band, unless it is charging in fast
                    if (step < 1.) { if (g.wait_since < 0.) g.wait_since = time; if (time - g.wait_since > P.guide_wait_max && P.guide_chased <= 0.) { gstat[2]++; g.wait_since = -1.; g.guide_state = 1; return; } }
                    else g.wait_since = -1.;
                } else g.wait_since = -1.;
                int n_near_ = 0, n_ahead_ = 0;
                {
                    P2 ld = sub(g.trap.out, ps.p); double nl_ = norm(ld);
                    for (auto& q : g.pseen) if (dist_lt(q.p, ps.p, 150.) && !dist_lt(q.p, g.trap.mouth, 40.)) {
                        n_near_++;
                        P2 v = sub(q.p, ps.p); double nv = norm(v);
                        if (nl_ > 1. && nv > 1. && (ld.x * v.x + ld.y * v.y) / (nl_ * nv) > 0.) n_ahead_++;   // predator between us and the lane point
                    }
                }
                bool plan_on = P.guide_plan > 0. && (P.guide_plan < 1.5 || (n_near_ >= 2 && (P.guide_plan < 2.5 || n_ahead_ >= 1)));   // mode 3: 2+ near and one ahead
                if (plan_on) {   // nightsim: local planner against ALL sensed predators (+ known walls); mode 2 = only with 2+ predators within 150
                    std::vector<P2> preds;
                    for (auto& q : g.pseen) if (dist_lt(q.p, ps.p, 250.) && !dist_lt(q.p, g.trap.mouth, 40.)) preds.push_back(q.p);
                    if (preds.empty()) preds.push_back(g.guide_pred);
                    double pen = MOVE_PENALTY[s.biome];
                    P2 lane = g.trap.out; double base = std::atan2(lane.y - ps.p.y, lane.x - ps.p.x);
                    double best_sc = -OINF, best_h = ps.theta + dir, best_st = step;
                    for (int k = 0; k < 24; k++) {
                        double h = base + (double)k * 2. * OPI / 24.;
                        for (int spr = 0; spr < 2; spr++) {
                            double st_ = spr ? s.sprint : walk;
                            P2 q = add(ps.p, mul(unit(h), st_ * pen));
                            if (!path_clear(g, ps.p, add(ps.p, mul(unit(h), st_ * pen + 6.)), 5.)) continue;
                            double mg = OINF;
                            for (const P2& pj : preds) mg = pmin(mg, dist(q, pj) - 15.);   // sighting is one step old: its next step closes ~15 more
                            double prog = dist(ps.p, lane) - dist(q, lane);
                            double sc;
                            if (mg >= P.guide_safe) sc = 1000. + 3. * prog - (spr ? P.guide_sprint_pen : 0.) - (mg > P.guide_keep ? 2. * (mg - P.guide_keep) : 0.);
                            else sc = 10. * mg + 0.5 * prog - (spr ? 0.5 : 0.);
                            if (sc > best_sc) { best_sc = sc; best_h = h; best_st = st_; }
                        }
                    }
                    dir = wrap(best_h - ps.theta); step = best_st; g.guide_sprinting = best_st > walk + 0.1;
                }
                if (dbg_log) fprintf(stderr, "[t=%.1f] LEAD guide %lld pose (%.0f,%.0f) pred_belief (%.0f,%.0f) dP=%.1f fresh=%d chasing=%d sprinting=%d blocked=%d step=%.1f dT=%.0f\n", time, (long long)g.guide, ps.p.x, ps.p.y, g.guide_pred.x, g.guide_pred.y, dP, fresh ? 1 : 0, chasing ? 1 : 0, g.guide_sprinting ? 1 : 0, blocked_ ? 1 : 0, step, dT);
                if (P.guide_wallclear > 0.) dir = steer_clear(m, dir, 30.);
                if (P.guide_mapclear > 0.) {   // nightsim: keep the next guide_mapclear units of the (backwards) walk clear of known walls in the group map
                    auto clear_h = [&](double h) { return path_clear(g, ps.p, add(ps.p, mul(unit(h), P.guide_mapclear)), 6.); };
                    double h0 = ps.theta + dir;
                    if (!clear_h(h0)) {
                        for (int k = 1; k <= 11; k++) {
                            bool done = false;
                            for (int sg : {1, -1}) { double h = h0 + (double)(sg * k) * OPI / 12.; if (clear_h(h)) { dir = wrap(h - ps.theta); done = true; break; } }
                            if (done) break;
                        }
                    }
                }
                plans[g.guide] = Plan{pmin(step, pmax(dT, 1.)), dir, angP};
                return;
            }
        }
        if (g.guide_state == 3) {
            // Lucas's handoff: back toward the mouth and stop guide_hand from the bait, facing the predator; being
            // eaten here is allowed (the predator then hears the bait and holds at the mouth)
            if (time - g.guide_seen > P.guide_lost || dP > P.guide_far + 120. || (P.guide_chased <= 0. && !chasing && time - g.guide_closing_t > 3.)) { gstat[3]++; g.guide_state = 1; return; }
            double dB = dist(ps.p, g.trap.goal);
            if (dB <= P.guide_hand) { if (!g.ep_h) { g.ep_h = true; g.ep_hand++; } plans[g.guide] = Plan{0., 0., fresh ? angP : 0.}; return; }
            double dM, angM; local_of(ps, g.trap.mouth, dM, angM);
            plans[g.guide] = Plan{pmin(walk, dB - P.guide_hand), angM, fresh ? angP : 0.};
            return;
        }
    }


    // nightsim: adjust a desired heading (agent-relative) so the next `look` units do not cross a recently seen wall
    double steer_clear(Mind& m, double dir_rel, double look) {
        const PoseObj& pose = *m.pose;
        std::vector<std::pair<P2, P2>> recent;
        for (auto& e : m.edges)
            if (time - e.t < 25. && point_segment(pose.p, e.a, e.b) < look + 10.) recent.push_back({e.a, e.b});
        if (recent.empty()) return dir_rel;
        auto clear = [&](double h) {
            P2 q = add(pose.p, mul(unit(h), look));
            for (auto& ab : recent)
                if (segments_cross(pose.p, q, ab.first, ab.second) || point_segment(q, ab.first, ab.second) < 7.) return false;
            return true;
        };
        double heading = pose.theta + dir_rel;
        if (clear(heading)) return dir_rel;
        for (int k = 1; k < 12; k++)
            for (int sgn : {1, -1}) {
                double h = heading + (double)(sgn * k) * OPI / 12;
                if (clear(h)) return wrap(h - pose.theta);
            }
        return dir_rel;
    }

    // ------------------------------------------------------------ predators (nightsim)
    // A predator is a threat when it is within pred_r, or within pred_face_r and facing us (rel_dir small:
    // rel_dir = bearing(predator->agent) - predator heading). Response: move directly away from the
    // inverse-distance-weighted threats; face the nearest one (a faced predator beyond 90 uses the slow 45-degree
    // pivot approach); sprint only inside pred_sprint_r. Plans stay the odometry source, so poses remain exact.
    int64_t n_evading = 0;
    void share_predators() {
        groups.each([&](const int64_t&, GroupP& g) { g->pseen.clear(); });
        if (P.pred_avoid_w > 0.) groups.each([&](const int64_t&, GroupP& g) {   // roll the sighting memory
            auto& pm = g->pmem; size_t k = 0;
            for (size_t i = 0; i < pm.size(); i++) if (time - pm[i].t < P.pred_avoid_t) pm[k++] = pm[i];
            pm.resize(k);
        });
        for (const AState& s : states) {
            Mind& m = M(s.aid); Group& g = G(m.group);
            for (const Obs& o : *s.obs) {
                if (o.type != 2) continue;
                P2 p = polar(*m.pose, o);
                double hd = m.pose->theta + o.angle + OPI - o.rel_dir;
                bool dup = false;
                for (auto& q : g.pseen) if (dist_lt(q.p, p, 20.)) { dup = true; break; }
                if (!dup) g.pseen.push_back(Group::PredSeen{p, hd});
                if (!dup && P.pred_avoid_w > 0.) {
                    bool near = false; for (auto& q : g.pmem) if (time - q.t < 5. && dist_lt(q.p, p, 30.)) { near = true; break; }
                    if (!near && g.pmem.size() < 400) g.pmem.push_back(Group::PredMark{p, time});
                }
            }
        }
    }
    // Threat: within pred_r, or within pred_face_r while facing us. Response: move away from the inverse-distance-
    // weighted threats (sprint inside pred_sprint_r), face the nearest (pred_face), and inside pred_dodge_r step
    // sideways (perpendicular, away from the predator's heading) to exploit its 0.3 rad/tick turn cap.
    #include "native_corner.hpp"
    bool evade(const AState& s, Plan& pl) {
        if(P.corner_mode>0. && !is_trap_role(s.aid) && corner_steer(s,pl))return true;
        Mind& m = M(s.aid);
        struct Th { double d, ang, rel; };
        std::vector<Th> th;
        if(model_full()&&(resource_mode==26||resource_mode==27||resource_mode==28||(resource_mode>=39&&resource_mode<=42))){
            const auto& ps=*m.pose;auto& g=G(m.group);
            for(auto& q:model_predators){
                double d,ang;local_of(ps,q.p,d,ang);
                if(d>pmax(P.pred_r,P.pred_face_r)+20.)continue;
                if(resource_mode==28&&q.resting)continue;
                if((resource_mode==26||resource_mode==28)&&!path_clear(g,q.p,ps.p,10.01))continue;
                double rel=wrap(std::atan2(ps.p.y-q.p.y,ps.p.x-q.p.x)-q.heading);
                th.push_back(Th{d,ang,rel});
            }
        }else if (P.pred_share > 0.) {
            const PoseObj& ps = *m.pose;
            for (auto& q : G(m.group).pseen) {
                double d, ang; local_of(ps, q.p, d, ang);
                double rel = wrap(std::atan2(ps.p.y - q.p.y, ps.p.x - q.p.x) - q.heading);
                th.push_back(Th{d, ang, rel});
            }
        } else {
            for (const Obs& o : *s.obs) if (o.type == 2) th.push_back(Th{o.distance, o.angle, o.has_rel_dir ? o.rel_dir : OPI});
        }
        const Th* nr = nullptr; double vx = 0., vy = 0.;
        for (const Th& t : th) {
            bool facing = std::fabs(t.rel) < OPI/6.;
            bool danger=t.d<P.pred_r || (facing && t.d<P.pred_face_r);
            if (P.entrapment_lookahead>0.) {
                const auto& ps=*m.pose; Group& g=G(m.group);
                P2 pred=add(ps.p,mul(unit(ps.theta+t.ang),t.d));
                double ph=ps.theta+t.ang+OPI-t.rel;
                P2 next=add(ps.p,mul(unit(ps.theta+pl.direction),pl.dist*MOVE_PENALTY[s.biome]));
                P2 pred_next=add(pred,mul(unit(ph),15.));
                danger=predator_detects(g,pred,ph,ps.p) || predator_detects(g,pred_next,ph,next);
            }
            if (!danger) continue;
            double w = 1.0 / pmax(t.d, 15.);
            vx -= std::cos(t.ang) * w; vy -= std::sin(t.ang) * w;
            if (!nr || t.d < nr->d) nr = &t;
        }
        if (!nr) return false;
        if (P.evade_closest > 0. && nr->d > P.evade_closest) {   // predators chase their closest agent: if another visible agent is nearer to it, keep foraging
            P2 pp{std::cos(nr->ang) * nr->d, std::sin(nr->ang) * nr->d};
            for (const Obs& o : *s.obs) {
                if (o.type != 1) continue;
                P2 q{std::cos(o.angle) * o.distance, std::sin(o.angle) * o.distance};
                if (dist(q, pp) < nr->d - 8.) return false;
            }
        }
        if (P.refuge_mode > 0. && refuge(s, pl, nr->d, nr->ang)) return true;
        if (P.hide_mode > 0. && hide_through(s, pl, nr->d)) return true;
        {   // decoy (nightsim): an agent that is dying anyway walks TOWARD the nearest predator so it becomes the
            // predator's closest target instead of a young forager; its low energy costs little score
            Mind& m = M(s.aid);
            bool dying = (P.decoy_old > 0. && m.old) || (P.decoy_e > 0. && s.energy < P.decoy_e && s.age > 40.);
            if (dying && nr->d < P.decoy_r) {
                double walk = pmin(s.speed, s.sprint);
                pl = Plan{pmin(walk, nr->d), nr->ang, nr->ang};
                n_evading++;
                return true;
            }
        }
        double away = std::atan2(vy, vx);
        Mind& mm = M(s.aid);
        if (mm.dodge_left > 0) {   // committed: keep the chosen absolute heading (walk after the first sprint ticks)
            mm.dodge_left--;
            double dir = wrap(mm.dodge_head - mm.pose->theta);
            double walk = pmin(s.speed, s.sprint);
            double step = (nr->d < P.pred_sprint_r) ? s.sprint : walk;
            pl = Plan{step, dir, P.pred_face > 0. && P.pred_dodge_hold_face > 0. ? nr->ang : 0.};
            n_evading++;
            return true;
        }
        if (nr->d < P.pred_dodge_r) {
            // predator heading in agent frame points along (bearing to predator + pi - rel); step perpendicular to it,
            // on the side that increases the angle the predator must turn
            double side = nr->rel >= 0. ? 1. : -1.;
            away = wrap(nr->ang + OPI + side * P.pred_dodge_ang);
            if (P.pred_dodge_hold > 0.) { mm.dodge_head = wrap(mm.pose->theta + away); mm.dodge_left = (int64_t)P.pred_dodge_hold; }
        }
        double walk = pmin(s.speed, s.sprint);
        double step = nr->d < P.pred_sprint_r ? s.sprint : walk;
        double turn = P.pred_face > 0. ? nr->ang : 0.;
        if (P.pred_wallclear > 0.) away = steer_clear(mm, away, 25.);
        pl = Plan{step, away, turn};
        n_evading++;
        return true;
    }


    bool path_clear(Group& g, P2 a, P2 b, double r) { return clear_segment(g, a, b, r); }
    Group::NavGrid& nav_grid(Group& g, double step, double r) {
        for (auto& grid : g.nav_grids)
            if (grid.version == g.wall_version && grid.step == step && std::fabs(grid.radius - r) < .01) return grid;
        if (g.nav_grids.size() >= 6) g.nav_grids.erase(g.nav_grids.begin());
        Group::NavGrid grid; grid.version = g.wall_version; grid.step = step; grid.radius = r;
        grid.nx = (int)std::floor(W / step) + 1; grid.ny = (int)std::floor(H / step) + 1;
        grid.blocked.assign((size_t)grid.nx * (size_t)grid.ny, 1);
        for (int x = 0; x < grid.nx; x++) for (int y = 0; y < grid.ny; y++) {
            P2 p{(double)x * step, (double)y * step};
            grid.blocked[(size_t)x * (size_t)grid.ny + (size_t)y] = clear_of(g, p, r) ? 0 : 1;
        }
        g.nav_grids.push_back(std::move(grid)); return g.nav_grids.back();
    }
    static std::string route_key(P2 a, P2 b, double r, double step) {
        auto qs = [&](double v) { return (int64_t)std::floor(v / step); };
        auto qg = [](double v) { return (int64_t)std::llround(v / 2.); };
        return std::to_string(qs(a.x)) + ":" + std::to_string(qs(a.y)) + ":" + std::to_string(qg(b.x)) + ":"
             + std::to_string(qg(b.y)) + ":" + std::to_string((int64_t)std::llround(r * 10.)) + ":" + std::to_string((int)step);
    }
    static std::string exact_route_key(P2 a, P2 b, double r, double step) {
        auto bits = [](double v) { uint64_t u; std::memcpy(&u, &v, sizeof(u)); return std::to_string(u); };
        return bits(a.x) + ":" + bits(a.y) + ":" + bits(b.x) + ":" + bits(b.y) + ":" + bits(r) + ":" + bits(step);
    }
    bool astar_route(Group& g, P2 a, P2 b, double r, double step, std::vector<P2>& route) {
        if (path_clear(g, a, b, r)) { route = {a, b}; return true; }
        if (!clear_of(g, a, r) || !clear_of(g, b, r)) return false;
        if (g.route_cache_version != g.wall_version) {
            g.route_cache.clear(); g.failed_route_cache.clear(); g.route_cache_version = g.wall_version;
        }
        std::string failed_key = exact_route_key(a, b, r, step) + ":" + std::to_string(g.wall_version);
        if (g.failed_route_cache.count(failed_key)) return false;
        std::string key = route_key(a, b, r, step);
        auto cached = g.route_cache.find(key);
        if (cached != g.route_cache.end()) {
            route = cached->second; if (!route.empty()) { route[0] = a; route.back() = b; }
            bool valid = route.size() >= 2;
            for (size_t i = 0; valid && i + 1 < route.size(); i++) valid = path_clear(g, route[i], route[i + 1], r);
            if (valid) return true;
            g.route_cache.erase(cached);
        }
        auto& grid = nav_grid(g, step, r); int N = grid.nx * grid.ny;
        auto id = [&](int x, int y) { return x * grid.ny + y; };
        auto point = [&](int n) { return P2{(double)(n / grid.ny) * step, (double)(n % grid.ny) * step}; };
        struct Q { double f, cost; int n; };
        struct Greater { bool operator()(const Q& x, const Q& y) const { return x.f > y.f || (x.f == y.f && x.n > y.n); } };
        std::vector<Q> open;
        auto push = [&](Q q) { open.push_back(q); std::push_heap(open.begin(), open.end(), Greater{}); };
        auto pop = [&]() { std::pop_heap(open.begin(), open.end(), Greater{}); Q q = open.back(); open.pop_back(); return q; };
        std::vector<double> cost((size_t)N, OINF); std::vector<int> parent((size_t)N, -1);
        int sx = (int)std::llround(a.x / step), sy = (int)std::llround(a.y / step);
        for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) {
            int x = sx + dx, y = sy + dy; if (x < 0 || y < 0 || x >= grid.nx || y >= grid.ny) continue;
            int n = id(x, y); P2 p = point(n); if (grid.blocked[(size_t)n] || !path_clear(g, a, p, r)) continue;
            double c = dist(a, p); if (c < cost[(size_t)n]) { cost[(size_t)n] = c; parent[(size_t)n] = -2; push(Q{c + dist(p, b), c, n}); }
        }
        int goal = -1;
        static const int D[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
        while (!open.empty()) {
            Q q = pop(); if (q.cost != cost[(size_t)q.n]) continue;
            P2 p = point(q.n); if (path_clear(g, p, b, r)) { goal = q.n; break; }
            int x = q.n / grid.ny, y = q.n % grid.ny;
            for (auto& d : D) {
                int xx = x + d[0], yy = y + d[1]; if (xx < 0 || yy < 0 || xx >= grid.nx || yy >= grid.ny) continue;
                int nn = id(xx, yy); if (grid.blocked[(size_t)nn]) continue;
                P2 np = point(nn); if (!path_clear(g, p, np, r)) continue;
                double nc = q.cost + dist(p, np);
                if (nc < cost[(size_t)nn]) { cost[(size_t)nn] = nc; parent[(size_t)nn] = q.n; push(Q{nc + dist(np, b), nc, nn}); }
            }
        }
        if (goal < 0) {
            if (g.failed_route_cache.size() >= 256) g.failed_route_cache.clear();
            g.failed_route_cache.insert(std::move(failed_key));
            return false;
        }
        std::vector<P2> raw{b}; for (int n = goal; n >= 0; n = parent[(size_t)n]) raw.push_back(point(n)); raw.push_back(a);
        std::reverse(raw.begin(), raw.end());
        route.clear(); route.push_back(a); size_t i = 0;
        while (i + 1 < raw.size()) {
            size_t j = raw.size() - 1; while (j > i + 1 && !path_clear(g, raw[i], raw[j], r)) j--;
            route.push_back(raw[j]); i = j;
        }
        if (g.route_cache.size() >= 256) g.route_cache.clear();
        g.route_cache[key] = route; return true;
    }
    // Confirmed-wall A*: 20-unit grid first, 10-unit retry; greedy visibility smoothing.
    bool route_next(Group& g, P2 a, P2 b, double r, P2& next, double* route_length = nullptr) {
        std::vector<P2> route;
        if (!astar_route(g, a, b, r, 20., route) && !astar_route(g, a, b, r, 10., route)) return false;
        next = route.size() > 1 ? route[1] : b;
        if (route_length) { *route_length = 0.; for (size_t i = 1; i < route.size(); i++) *route_length += dist(route[i - 1], route[i]); }
        return true;
    }
    // refuge (nightsim, refuge_mode): a chased agent that is already close to a narrow gap steps 9 deep into it and
    // holds there; the predator (radius 10) cannot enter a 10.1-19.9 gap and stays pressed at the mouth while it hears
    // the agent. No guide, no bait child: the chase itself delivers the predator. Returns true when it planned.
    int64_t refuge_events = 0, refuge_holds = 0, refuge_died_route = 0, refuge_died_hold = 0, refuge_died_exit = 0, refuge_exits = 0, refuge_aborts = 0;
    int64_t gstat[12] = {};   // guide diagnostics: 0 lead unseen->acq, 1 lead far/not chasing->acq, 2 wait timeout->acq, 3 hand state back->acq, 4 acq lost (end), 5 route/clear fail (end), 6 died (end), 7 delivered (end), 8-10 ticks in state 1/2/3, 11 episodes started
    bool refuge(const AState& s, Plan& pl, double dP, double angP) {
        Mind& m = M(s.aid); Group& g = G(m.group);
        if (!g.anchored || g.sites.empty()) { m.hide_idx = -1; m.refuge_in = false; return false; }
        const PoseObj& ps = *m.pose;
        if (P.refuge_slow_only > 0. && !m.refuge_in && m.hide_idx < 0) {   // only agents that cannot outrun a predator (15/tick)
            bool low = s.energy < 0.2 * s.max_energy;
            if (!low && s.sprint > 15.5) return false;
        }
        if (m.hide_idx < 0 || m.hide_idx >= (int)g.sites.size()) {
            if (dP > P.refuge_trigger) return false;
            P2 pp = add(ps.p, mul(unit(ps.theta + angP), dP));
            int best = -1; double bd = P.refuge_r;
            for (size_t i = 0; i < g.sites.size(); i++) {
                const auto& st_ = g.sites[i];
                double d = dist(st_.mouth, ps.p);
                if (d >= bd) continue;
                if (dist(st_.mouth, pp) < d + 5.) continue;   // the predator is nearer to the mouth than we are: do not run at it
                if (P.refuge_clear > 0.) {   // the straight run to the pre-point and the mouth must not cross known walls
                    P2 in_ = sub(st_.goal, st_.mouth); in_ = mul(in_, 1.0 / pmax(norm(in_), 1e-6));
                    P2 pre_ = sub(st_.mouth, mul(in_, 25.));
                    if (!path_clear(g, ps.p, pre_, 5.5) || !path_clear(g, pre_, st_.mouth, 4.5)) continue;
                }
                bd = d; best = (int)i;
            }
            if (best < 0) return false;
            m.hide_idx = best; m.hide_t = time; m.refuge_in = false; refuge_events++;
        }
        const Group::Site& st_ = g.sites[m.hide_idx];
        m.refuge_pred_t = time;
        if (dist_lt(ps.p, st_.goal, 3.)) {   // hold, facing the mouth
            if (P.refuge_verify > 0.) {   // the two faces must be observed right now at gap/2 +- 3 on both sides; otherwise the map is wrong here
                P2 axis = sub(st_.goal, st_.mouth); axis = mul(axis, 1.0 / pmax(norm(axis), 1e-6));
                int left = 0, right = 0;
                for (auto& e : m.edges) {
                    if (time - e.t > 0.05) continue;
                    P2 ev = sub(e.b, e.a); double n = norm(ev); if (n < 3.) continue; ev = mul(ev, 1.0 / n);
                    if (std::fabs(ev.x * axis.x + ev.y * axis.y) < 0.9) continue;
                    double d = point_segment(ps.p, e.a, e.b);
                    if (d < st_.gap / 2 - 3. || d > st_.gap / 2 + 3.) continue;
                    P2 mid = mul(add(e.a, e.b), 0.5);
                    double side = axis.x * (mid.y - ps.p.y) - axis.y * (mid.x - ps.p.x);
                    if (side > 0) left++; else right++;
                }
                if (left > 0 && right > 0) m.refuge_bad = 0;
                else if (++m.refuge_bad >= 2) { m.hide_idx = -1; m.refuge_in = false; m.refuge_bad = 0; refuge_aborts++; return false; }
            }
            m.refuge_in = true; refuge_holds++; double dm_, am_; local_of(ps, st_.mouth, dm_, am_); pl = Plan{0., 0., am_}; n_evading++; return true;
        }
        P2 in = sub(st_.goal, st_.mouth); in = mul(in, 1.0 / pmax(norm(in), 1e-6));
        P2 pre = sub(st_.mouth, mul(in, 25.));
        P2 rel = sub(ps.p, pre); double along = rel.x * in.x + rel.y * in.y, across = std::fabs(rel.x * in.y - rel.y * in.x);
        P2 target = (along > -3. && across < 4.) ? st_.goal : pre;
        double d, ang; local_of(ps, target, d, ang);
        double walk = pmin(s.speed, s.sprint);
        double step = (P.refuge_sprint > 0. || dP < P.pred_sprint_r) ? s.sprint : walk;
        pl = Plan{pmin(step, d), ang, ang};
        n_evading++;
        return true;
    }
    // refuge exit (called for every agent each tick): keep holding for refuge_leave s after the last sensed predator,
    // then walk out to the lane point and return to normal duty
    bool refuge_exit(const AState& s, Plan& pl) {
        Mind& m = M(s.aid);
        if (m.hide_idx < 0) return false;
        Group& g = G(m.group);
        if (m.hide_idx >= (int)g.sites.size()) { m.hide_idx = -1; m.refuge_in = false; return false; }
        if (time - m.refuge_pred_t < P.refuge_leave) { if (m.refuge_in) { pl = Plan{0., 0., 0.}; return true; } return false; }
        const Group::Site& st_ = g.sites[m.hide_idx];
        P2 in = sub(st_.goal, st_.mouth); in = mul(in, 1.0 / pmax(norm(in), 1e-6));
        P2 pre = sub(st_.mouth, mul(in, 25.));
        if (!m.refuge_in || dist_lt(m.pose->p, pre, 4.)) { if (m.refuge_in) refuge_exits++; m.hide_idx = -1; m.refuge_in = false; return false; }
        double d, ang; local_of(*m.pose, pre, d, ang);
        pl = Plan{pmin(pmin(s.speed, s.sprint), d), ang, ang};
        return true;
    }

    // crevice pass-through escape (nightsim, hide_mode): run into the nearest known crevice with an open rear and out the
    // other side; the predator (radius 10) cannot follow through a 10-20 wide gap and must go around the obstacle
    bool hide_through(const AState& s, Plan& pl, double dP) {
        Mind& m = M(s.aid); Group& g = G(m.group);
        if (!g.anchored || g.sites.empty()) return false;
        const PoseObj& ps = *m.pose;
        if (m.hide_idx < 0 || m.hide_idx >= (int)g.sites.size() || time - m.hide_t > 12.) {
            if (dP > P.hide_trigger) return false;
            int best = -1; double bd = P.hide_r;
            for (size_t i = 0; i < g.sites.size(); i++) {
                const auto& st_ = g.sites[i];
                if (!st_.rear_ok) continue;
                double d = dist(st_.mouth, ps.p);
                if (d < bd) { bd = d; best = (int)i; }
            }
            if (best < 0) return false;
            m.hide_idx = best; m.hide_t = time;
        }
        const Group::Site& st_ = g.sites[m.hide_idx];
        P2 in = sub(st_.goal, st_.mouth); in = mul(in, 1.0 / pmax(norm(in), 1e-6));
        P2 pre = sub(st_.mouth, mul(in, 25.));
        P2 rel = sub(ps.p, pre); double along = rel.x * in.x + rel.y * in.y, across = std::fabs(rel.x * in.y - rel.y * in.x);
        P2 target = (along > -3. && across < 4.) ? st_.rear : pre;
        double d, ang; local_of(ps, target, d, ang);
        if (dist_lt(ps.p, st_.rear, 6.)) { m.hide_idx = -1; return false; }   // through: back to normal (evade again if needed)
        double walk = pmin(s.speed, s.sprint);
        double step = dP < P.pred_sprint_r ? s.sprint : walk;
        pl = Plan{pmin(step, d), ang, ang};
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

    std::unordered_set<int64_t> frozen;   // tests only
    bool late_on = false;
    void apply_late() {
        auto ov = [](double& dst, double v) { if (!std::isnan(v)) dst = v; };
        ov(P.fruit_reach, P.l_fruit_reach); ov(P.tree_reach, P.l_tree_reach); ov(P.watch_reach, P.l_watch_reach); ov(P.explore_energy, P.l_explore_energy); ov(P.cap_min, P.l_cap_min); ov(P.cap_mult, P.l_cap_mult); ov(P.cap_tree_slack, P.l_cap_tree_slack); ov(P.cap_hard_min, P.l_cap_hard_min); ov(P.sweep_rate, P.l_sweep_rate); ov(P.watch_patience, P.l_watch_patience); ov(P.explore_radius, P.l_explore_radius); ov(P.old_reach, P.l_old_reach); ov(P.dist_pen, P.l_dist_pen); ov(P.births_per_tick, P.l_births_per_tick); ov(P.emergency_reserve, P.l_emergency_reserve); ov(P.low_pop_reserve, P.l_low_pop_reserve);
    }
    std::vector<Act> call(std::vector<AState>&& sts, double sim_time) {
        time = sim_time;groups.each([&](const int64_t&, GroupP& g){g->wi_ready=false;});
        if (!late_on && time >= P.late_t) { late_on = true; apply_late(); }
        states = std::move(sts);
        sidx.clear();
        for (size_t i = 0; i < states.size(); i++) sidx[states[i].aid] = i;
        for (int64_t aid : minds.key_list()) {
            if (in_states(aid)) continue;
            MindP m = minds.at(aid); minds.erase(aid);
            if (m->hide_idx >= 0) { if (!m->refuge_in) refuge_died_route++; else if (time - m->refuge_pred_t < P.refuge_leave) refuge_died_hold++; else refuge_died_exit++; }
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
        model_localize();
        merge_groups();
        for (const AState& s : states) {
            observe(M(s.aid), s);
            auto& m=M(s.aid);
            if (m.terrain_samples.empty() || dist(m.terrain_samples.back().first,m.pose->p)>4.) {
                if (m.terrain_samples.size()>=64) m.terrain_samples.erase(m.terrain_samples.begin());
                m.terrain_samples.push_back({m.pose->p,MOVE_PENALTY[s.biome]});
            }
        }
        model_localize();model_map();model_lifecycle();
        groups.each([&](const int64_t&, GroupP& g){wi_prepare(*g);});
        if (P.oracle_trees > 0.) apply_oracle();
        ingest_resource_forecast();
        if (P.pred_mode > 0.) share_predators();
        model_predator_map();
        if (P.trap_mode > 0. || P.hide_mode > 0. || P.refuge_mode > 0.) groups.each([&](const int64_t&, GroupP& g) { if (g->anchored && time - g->sites_t >= 2.) find_sites(*g); });
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
        if (P.explore_until_trap > 0.) for (const AState& s : states) {
            Mind& m=M(s.aid); Group& g=G(m.group);
            if (g.has_trap || !g.sites.empty() || m.old || s.energy < P.explore_energy || m.has_fruit) continue;
            if (m.has_post && g.trees.has(m.post)) g.trees.at(m.post)->assigned.discard(s.aid);
            m.has_post=false;
            if (!m.has_explore || time>m.explore_until || dist(m.explore_p,m.pose->p)<25.) {
                P2 target{}; double until=0.;
                m.has_explore=explore_target(m,g,s,target,until);
                if (m.has_explore) { m.explore_p=target; m.explore_until=until; }
            }
            if (m.has_explore) {
                P2 target=m.explore_p, next{};
                if (g.anchored && route_next(g,m.pose->p,target,5.01,next)) target=next;
                double dd,dir,turn; go_to(m,s,target,3.,dd,dir,turn);
                plans[s.aid]=Plan{dd,dir,turn};
            }
        }
        if (P.pred_mode > 0.) for (const AState& s : states) if (!is_trap_role(s.aid) && evade(s, plans[s.aid])) M(s.aid).evade_t = time;
        if (P.refuge_mode > 0.) for (const AState& s : states) if (!is_trap_role(s.aid)) refuge_exit(s, plans[s.aid]);
        if (P.trap_mode >= 2.) run_trap(plans);
        if (P.entrapment_lookahead > 0.) for (const AState& s:states) {
            Group& g=G(M(s.aid).group);
            bool incoming=g.rep==s.aid && !g.rep_entered_rear;
            if ((!is_trap_role(s.aid) || incoming) && evade(s,plans[s.aid])) M(s.aid).evade_t=time;
        }
        if (P.test_freeze > 0.) for (const AState& s : states) plans[s.aid] = Plan{0., 0., 0.};
        for (int64_t a : frozen) if (plans.count(a)) plans[a] = Plan{0., 0., 0.};
        int64_t young_now = (int64_t)young.size();
        std::unordered_map<int64_t, double> fit;
        for (const AState& s : states) fit[s.aid] = (model_full()&&resource_mode==65) ? 2.*pmin(2.,pmin(s.speed,s.sprint)/10.)+.6*pmin(2.,s.max_energy/500.) : fitness(s);
        std::vector<int64_t> elders;
        minds.each([&](const int64_t& a, MindP& m) { if (m->old || st(a).age >= heir_age_for(a)) elders.push_back(a); });
        std::stable_sort(elders.begin(), elders.end(), [&](int64_t a, int64_t b) { return -st(a).energy < -st(b).energy; });
        std::vector<double> yfit;
        for (int64_t a : young) yfit.push_back(fit[a]);
        std::sort(yfit.begin(), yfit.end());
        if (yfit.empty()) yfit.push_back(0.);
        double median_fit = yfit[yfit.size() / 2];
        for (int64_t aid : elders) {
            Mind& m = M(aid); const AState& s = st(aid);
            if (is_trap_role(aid) || m.heir_done || s.biome == RIVER || (culled.count(aid) && !m.old)) continue;
            if (P.heir_select && (double)young.size() >= P.select_min_young && fit[aid] < median_fit - P.heir_slack) continue;
            const Plan& pl = plans[aid];
            Group& g = G(m.group);
            bool at_food = m.has_post && g.trees.has(m.post) && (g.trees.at(m.post)->fruit_here > 0 || !g.trees.at(m.post)->dead);
            double left = s.energy - cost_now(pl.dist, pl.turn, s);
            bool ok;
            if (m.old) ok = left > 101.;
            else ok = left > P.heir_reserve || (P.heir_at_food && at_food && left > 130.);
            if (ok && !m.old && P.sprint_floor > 0. && P.sprint_floor_breed > 0. && left - 100. < 0.2 * s.max_energy + P.sprint_floor) ok = false;
            if (ok && P.heir_needs_site && !at_food && young_now >= capv && pop > 2) ok = false;
            if (ok) { spawn_set.insert(aid); m.heir_done = true; young_now++; }
        }
        for (int64_t aid : elders) {
            Mind& m = M(aid); const AState& s = st(aid);
            if (is_trap_role(aid) || !m.old || spawn_set.count(aid)) continue;
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
                if (!is_trap_role(s.aid) && !spawn_set.count(s.aid) && s.energy - cost_now(pl.dist, pl.turn, s) > P.emergency_reserve) {
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
                if (is_trap_role(s.aid) || spawn_set.count(s.aid) || s.biome == RIVER) continue;
                const Plan& pl = plans[s.aid];
                double left = s.energy - cost_now(pl.dist, pl.turn, s);
                if (m.old) {
                    if (!P.extra_old || left <= 101.) continue;
                } else {
                    double thr = (double)young.size() < P.cap_min ? pmin(reserve(), P.low_pop_reserve) : reserve();
                    if (left <= thr) continue;
                }
                if (P.sprint_floor > 0. && P.sprint_floor_breed > 0. && left - 100. < 0.2 * s.max_energy + P.sprint_floor) continue;   // nightsim: stay above the sprint floor after paying for the child
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
            bool spawn = spawn_set.count(aid) > 0 && P.no_spawn <= 0. && !is_trap_role(aid);
            if (P.keeper_mode > 0.) { Group& gk = G(m.group); if (gk.keeper == aid && gk.keeper_spawn && s.energy > 101.) spawn = true; }
            if (spawn && P.spawn_pred_r > 0.) for (const Obs& o : *s.obs) if (o.type == 2 && o.distance < P.spawn_pred_r) { spawn = false; break; }
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
