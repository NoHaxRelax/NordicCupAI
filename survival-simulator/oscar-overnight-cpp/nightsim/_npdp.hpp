// nightsim pdp layer (x3 perception hunt; included inside the Policy class body; off unless pdp_r > 0).
// Exact-chase-law escape: the predator's move is capped at +-0.3 rad off its heading per tick and it only perceives
// hearing <= 60 all around plus a +-30 deg cone beyond. A pessimistic value-iteration table (survival/exploits/perception/
// pdp_tables.py) gives, for an agent position in the predator body frame, the ticks to FORCE an escape into its
// unobserved rear (d > 60 outside the cone) with agent step A (walk) against predator step S. In grass a walker (A=10)
// beats the sprinting predator (S=15) from >= ~40 in front, from anywhere at the side/rear. Tables are raw uint8
// 525x525 files pdp_A{A}_S{S}.bin in $NIGHT_PDP_DIR (default /workspace/night/pdp); 255 = not forceable.
// pdp_r: engage when the nearest predator (after its pending step) is within this; pdp_iso: skip when a second predator
// is within this; pdp_face: 1 face the predator after moving (keeps it in view; chase mode below 90 anyway).
struct PdpTab { int n = 0; std::vector<uint8_t> v; };
std::unordered_map<int64_t, std::shared_ptr<PdpTab>> pdp_cache;
int64_t pdp_cnt[4] = {0, 0, 0, 0};   // 0 engaged, 1 doomed (fallback), 2 no table, 3 multi-predator skip
std::shared_ptr<PdpTab> pdp_table(int A, double S) {
    int64_t key = (int64_t)A * 100000 + (int64_t)std::lround(S * 10.);
    auto it = pdp_cache.find(key);
    if (it != pdp_cache.end()) return it->second;
    auto t = std::make_shared<PdpTab>();
    const char* dir = getenv("NIGHT_PDP_DIR");
    char path[512]; char sb[32];
    snprintf(sb, sizeof sb, "%g", S);
    char sfx[16] = ""; if (P.pdp_set > 0.) snprintf(sfx, sizeof sfx, "%d", (int)P.pdp_set);
    snprintf(path, sizeof path, "%s%s/pdp_A%d_S%s.bin", dir ? dir : "/workspace/night/pdp", sfx, A, sb);
    FILE* f = fopen(path, "rb");
    if (f) {
        t->n = 525; t->v.resize((size_t)t->n * t->n);
        if (fread(t->v.data(), 1, t->v.size(), f) != t->v.size()) { t->n = 0; t->v.clear(); }
        fclose(f);
    }
    pdp_cache[key] = t;
    return t;
}
static int pdp_look(const PdpTab& t, double rx, double ry) {
    const double R = 262.;
    int ix = (int)std::floor(rx + R), iy = (int)std::floor(ry + R);
    if (ix < 0 || iy < 0 || ix >= t.n - 1 || iy >= t.n - 1) return 255;
    auto g = [&](int a, int b) { return (int)t.v[(size_t)a * t.n + b]; };
    return std::max(std::max(g(ix, iy), g(ix + 1, iy)), std::max(g(ix, iy + 1), g(ix + 1, iy + 1)));
}
// exact predator.step against one agent at (ax, ay) with heading ah (all in one frame); returns observed flag
static bool pdp_pstep(double& px, double& py, double& ph, double ax, double ay, double ah) {
    double dx = ax - px, dy = ay - py, d = std::sqrt(dx * dx + dy * dy);
    double a = wrap(std::atan2(dy, dx) - ph);
    if (!(d <= 60. || (d <= 250. && std::fabs(a) <= OPI / 6.))) return false;
    double rel = wrap(std::atan2(py - ay, px - ax) - ah);
    if (std::fabs(rel) > OPI / 2. || d < 90.) {
        double ts = std::max(-0.3, std::min(0.3, a * 0.5)), st = std::min(15., d);
        if (std::fabs(a) > 0.05) { px += st * std::cos(ph + ts); py += st * std::sin(ph + ts); ph += ts; }
        else { px += st * std::cos(ph + a); py += st * std::sin(ph + a); }
        return true;
    }
    double sg = rel > 0. ? -1. : (rel < 0. ? 1. : 0.), md = a + sg * OPI / 4.;
    double xa = d * std::cos(a) - 15. * std::cos(md), ya = d * std::sin(a) - 15. * std::sin(md);
    px += 15. * std::cos(ph + md); py += 15. * std::sin(ph + md); ph += std::atan2(ya, xa);
    return true;
}
template <class TH>
bool pdp_evade(const AState& s, const std::vector<TH>& th, Plan& pl) {
    if (th.empty()) return false;
    Mind& m = M(s.aid);
    // predator poses in the agent frame (agent at origin, heading 0), after their pending step
    struct PP { double x, y, h, d; };
    std::vector<PP> ps;
    for (const TH& t : th) {
        PP q{t.d * std::cos(t.ang), t.d * std::sin(t.ang), wrap(t.ang + OPI - t.rel), 0.};
        pdp_pstep(q.x, q.y, q.h, 0., 0., 0.);
        q.d = std::hypot(q.x, q.y); ps.push_back(q);
    }
    std::sort(ps.begin(), ps.end(), [](const PP& a, const PP& b) { return a.d < b.d; });
    if (ps[0].d >= P.pdp_r) return false;
    if (ps.size() > 1 && ps[1].d < P.pdp_iso) { pdp_cnt[3]++; return false; }
    if (P.pdp_excl > 0.) {   // only when we are its target: it must perceive us and no other visible agent is nearer to it
        const PP& q = ps[0];
        double a0 = wrap(std::atan2(-q.y, -q.x) - q.h);
        if (!(q.d <= 60. || (q.d <= 250. && std::fabs(a0) <= OPI / 6.))) return false;
        for (const Obs& o : *s.obs) {
            if (o.type != 1) continue;
            double ox = o.distance * std::cos(o.angle), oy = o.distance * std::sin(o.angle);
            if (std::hypot(ox - q.x, oy - q.y) < q.d + P.pdp_excl) { pdp_cnt[3]++; return false; }
        }
    }
    double pen = MOVE_PENALTY[s.biome];
    double S = pen >= 0.99 ? 15. : (pen >= 0.79 ? 12. : (pen >= 0.49 ? 7.5 : 4.5));
    const PP& q0 = ps[0];
    const PoseObj& pso = *m.pose; Group& gw = G(m.group);
    bool usew = gw.anchored && !gw.walls.empty();
    double c0 = std::cos(pso.theta), s0 = std::sin(pso.theta);
    double best_v = OINF, best_tb = OINF; double bd = 0., bdir = 0., bturn = 0.; bool found = false;
    auto search = [&](int A) {
        auto tab = pdp_table(A, S);
        if (!tab || tab->n == 0) { pdp_cnt[2]++; return; }
        for (int k = -1; k < 72; k++) {
            double step = k < 0 ? 0. : (k < 48 ? (double)A : A / 2.);
            double dir = k < 0 ? 0. : (k < 48 ? TAU * k / 48. : TAU * (k - 48) / 24.);
            double ax = step * std::cos(dir), ay = step * std::sin(dir);
            if (step > 0. && usew) {
                P2 w{pso.p.x + ax * c0 - ay * s0, pso.p.y + ax * s0 + ay * c0};
                if (!clear_of(gw, w, 5.5)) continue;
            }
            double ah = P.pdp_face > 0. ? std::atan2(q0.y - ay, q0.x - ax) : 0.;
            double px = q0.x, py = q0.y, ph = q0.h;
            bool seen = pdp_pstep(px, py, ph, ax, ay, ah);
            double dn = std::hypot(ax - px, ay - py);
            double v;
            if (!seen) v = 0.;
            else if (dn < 15.) v = 1000.;
            else {
                double c = std::cos(-ph), sn = std::sin(-ph);
                double rx = (ax - px) * c - (ay - py) * sn, ry = (ax - px) * sn + (ay - py) * c;
                v = pdp_look(*tab, rx, ry);
            }
            for (size_t j = 1; j < ps.size(); j++) {   // other predators: never step into their capture lobe
                double qx = ps[j].x, qy = ps[j].y, qh = ps[j].h;
                pdp_pstep(qx, qy, qh, ax, ay, ah);
                if (std::hypot(ax - qx, ay - qy) < 15.) v = 1000.;
            }
            double tb = -dn + P.pdp_cost * step;
            if (v < best_v || (v == best_v && tb < best_tb)) { best_v = v; best_tb = tb; bd = step; bdir = dir; bturn = ah; found = true; }
        }
    };
    double walk = pmin(s.speed, s.sprint);
    int A = std::min(20, (int)std::floor(walk * pen + 1e-9));
    if (A >= 2) search(A);
    if ((!found || best_v >= 255.) && P.pdp_sprint > 0. && s.energy > s.max_energy / 5. + P.pdp_sprint) {   // sprint table
        int As = std::min(20, (int)std::floor(s.sprint * pen + 1e-9));
        if (As > A) { best_v = OINF; best_tb = OINF; found = false; search(As); }
    }
    if (!found || best_v >= 255.) { pdp_cnt[1]++; return false; }
    pdp_cnt[0]++;
    pl = Plan{pen > 0. ? bd / pen : bd, wrap(bdir), wrap(bturn)};
    return true;
}
