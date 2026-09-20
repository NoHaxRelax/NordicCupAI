// nightsim stuck-spot guiding (sg_*; included inside the Policy class body; everything off unless sg_mode > 0).
// A stuck spot ("sink") is a limit cycle of the unmodified predator AI with no agent in sight: pressed within 12 of a
// wall its edge-avoid turn is pi/2 per tick, and next to a second wall the 10-degree collision deflection closes the
// orbit (period 2 or 4); from that state it never leaves (verified: 100% of exact-state drops hold 300 s). Only some
// (position, heading) states near the vertex fall into the cycle, so each sink carries a delivery spec: kill point G,
// approach heading th, lane start L = G - 70 u(th) (straight, predator-free). Engine-truth ceiling: specs come from
// nightsim/stuckspot.py (--mode spec) through dbg_sg_sinks. Early in the game (sg_t0..sg_t1) a member near an awake
// predator that is within sg_pred_r of a free sink becomes the guide:
//   1 ACQUIRE  walk toward the predator until it is within sg_acq (so it locks on) or closing in
//   2 LEAD     keep it chasing inside sg_near..sg_far (hearing 60 goes through walls) and walk to L
//   3 LANE     walk from L to G along the lane, facing the predator (it follows along the lane heading ~ th)
//   4 HOLD     stand at G facing it and get eaten there (sg_escape > 0: walk on sg_escape units instead)
struct SgSpec { P2 G, L; double th; };
struct SgSink { P2 p, G, L; double th = 0.; bool spec = false; std::vector<SgSpec> specs; double occ_t = -1e9; int64_t sent = 0, held_seen = 0; };
struct SgState {
    int64_t guide = -1; int sink = -1, spec = 0; int state = 0; P2 pred{}; double seen = -1e9, since = 0., closing_t = -1e9, dprev = -1.;
    bool sprinting = false; double wait_since = -1.; P2 last_pos{};
};
struct SgRect { double x, y, w, h; };
std::vector<SgSink> sg_sinks;
std::vector<SgRect> sg_rects;   // engine-truth ceiling only: obstacle rectangles for the straight-path check (sg_clear)
std::unordered_map<int64_t, SgState> sg_state;
// counters: 0 episodes, 1 reached L, 2 reached G (hold), 3 lost (predator unseen), 4 timeout, 5 waited out, 6 escaped, 7 released (other)
int64_t sg_cnt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
// diagnostics (group-ticks in the window): 0 unanchored with sightings, 1 anchored with sightings, 2 a sighting near a usable sink, 3 a guide found
int64_t sg_diag[4] = {0, 0, 0, 0};

bool sg_role(int64_t aid) const {
    if (P.sg_mode <= 0.) return false;
    for (auto& kv : sg_state) if (kv.second.guide == aid) return true;
    return false;
}
bool sg_seg_clear(P2 a, P2 b, double r) const {
    if (sg_rects.empty()) return true;
    double L = dist(a, b); int n = std::max(1, (int)(L / 4.));
    for (int i = 0; i <= n; i++) {
        double x = a.x + (b.x - a.x) * i / n, y = a.y + (b.y - a.y) * i / n;
        for (auto& o : sg_rects) if (o.x - r < x && x < o.x + o.w + r && o.y - r < y && y < o.y + o.h + r) return false;
    }
    return true;
}
void sg_end(SgState& S, int why) { if (why >= 0) sg_cnt[why]++; S.guide = -1; S.sink = -1; S.state = 0; S.dprev = -1.; S.sprinting = false; S.wait_since = -1.; }

void run_sg(std::unordered_map<int64_t, Plan>& plans) {
    if (P.sg_mode <= 0. || sg_sinks.empty()) return;
    if (P.sg_keep_r > 0.) {   // keep-out: members stay out of sg_keep_r around sinks believed to hold a predator (or all sinks, sg_keep_all)
        for (const AState& s : states) {
            Mind& m = M(s.aid); if (!G(m.group).anchored || sg_role(s.aid)) continue;
            for (auto& sk : sg_sinks) {
                if (P.sg_keep_all <= 0. && time - sk.occ_t > P.sg_keep_t) continue;
                double d, ang; local_of(*m.pose, sk.p, d, ang);
                if (d < P.sg_keep_r) { plans[s.aid] = Plan{pmin(s.speed, s.sprint), wrap(ang + OPI), 0.}; break; }
            }
        }
    }
    groups.each([&](const int64_t& gid, GroupP& gp) {
        Group& g = *gp;
        bool inwin = time >= P.sg_t0 && time <= P.sg_t1 && !g.pseen.empty();
        if (!g.anchored) { if (inwin) sg_diag[0]++; return; }
        if (inwin) sg_diag[1]++;
        SgState& S = sg_state[gid];
        auto at_sink = [&](P2 p) { for (auto& sk : sg_sinks) if (dist_lt(p, sk.p, P.sg_occ_r)) return true; return false; };
        for (auto& q : g.pseen) for (auto& sk : sg_sinks) if (dist_lt(q.p, sk.p, P.sg_occ_r)) { sk.occ_t = time; sk.held_seen++; }
        if (S.guide >= 0 && (!minds.has(S.guide) || M(S.guide).group != gid)) sg_end(S, -1);
        if (S.guide < 0) {
            if (time < P.sg_t0 || time > P.sg_t1) return;
            int64_t bg = -1; int bs_i = -1; double bsc = -OINF; P2 bq{};
            for (auto& q : g.pseen) {
                if (at_sink(q.p)) continue;
                int si = -1; double sd = OINF;
                for (size_t i = 0; i < sg_sinks.size(); i++) {
                    const SgSink& sk = sg_sinks[i];
                    if ((double)sk.sent >= P.sg_cap || (P.sg_need_spec > 0. && !sk.spec)) continue;
                    if (P.sg_skip_occ > 0. && time - sk.occ_t < P.sg_skip_occ) continue;
                    double d = dist(q.p, sk.L);
                    if (d < P.sg_pred_r && d < sd) { sd = d; si = (int)i; }
                }
                if (si < 0) continue;
                sg_diag[2]++;
                const SgSink& sk = sg_sinks[si];
                g.agents.each([&](int64_t a) {
                    if (frozen.count(a) || (P.trap_mode >= 2. && is_trap_role(a))) return;
                    const AState& s = st(a); Mind& m = M(a);
                    if (s.energy < P.sg_min_e || m.old) return;
                    double dq = dist(m.pose->p, q.p);
                    if (dq > P.sg_guide_r) return;
                    if (P.sg_clear > 0. && !sg_seg_clear(m.pose->p, sk.L, 6.)) return;
                    double sc = -dq - P.sg_dist_w * dist(m.pose->p, sk.L) + pmin(s.speed, s.sprint) * 5.;
                    P2 a1 = sub(sk.L, m.pose->p), b1 = sub(q.p, m.pose->p);
                    double na = norm(a1), nb = norm(b1);
                    if (na > 1. && nb > 1. && (a1.x * b1.x + a1.y * b1.y) / (na * nb) > 0.45) sc -= P.sg_side_pen;
                    if (sc > bsc) { bsc = sc; bg = a; bs_i = si; bq = q.p; }
                });
            }
            if (bg < 0) return;
            sg_diag[3]++;
            {   // lane choice: the spec whose lane the guide can enter from behind (guide on the L side of G, heading along u)
                const SgSink& sk = sg_sinks[bs_i]; P2 gp = M(bg).pose->p; double bc = -OINF; S.spec = 0;
                for (size_t k = 0; k < sk.specs.size(); k++) {
                    const SgSpec& c = sk.specs[k]; P2 u{std::cos(c.th), std::sin(c.th)}; P2 v = sub(c.G, gp); double nv = norm(v);
                    double cs = nv > 1. ? (u.x * v.x + u.y * v.y) / nv : 0.;
                    if (P.sg_clear > 0. && !sg_seg_clear(gp, c.L, 6.)) cs -= 10.;
                    if (cs > bc) { bc = cs; S.spec = (int)k; }
                }
            }
            S.guide = bg; S.sink = bs_i; S.state = 1; S.pred = bq; S.seen = time; S.since = time; S.closing_t = -1e9; S.dprev = -1.;
            sg_sinks[bs_i].sent++; sg_cnt[0]++;
            Mind& m = M(bg);
            if (m.has_post && g.trees.has(m.post)) g.trees.at(m.post)->assigned.discard(bg);
            m.has_post = false;
            if (m.has_fruit && g.fruits.has(m.fruit)) g.fruits.at(m.fruit)->has_claim = false;
            m.has_fruit = false;
        }
        Mind& m = M(S.guide); const AState& s = st(S.guide); const PoseObj& ps = *m.pose;
        S.last_pos = ps.p;
        const SgSink& sk0 = sg_sinks[S.sink];
        SgSpec skc = sk0.specs.empty() ? SgSpec{sk0.G, sk0.L, sk0.th} : sk0.specs[(size_t)S.spec];
        struct { P2 G, L; double th; } sk{skc.G, skc.L, skc.th};
        {   // track the guided predator: the sighting nearest the guide (a held one only once we are at the lane)
            bool have = false; P2 pp{}; double best = OINF;
            for (auto& q : g.pseen) { if (S.state < 3 && at_sink(q.p)) continue; double d = dist(q.p, ps.p); if (d < best) { best = d; pp = q.p; have = true; } }
            if (have) { S.pred = pp; S.seen = time; }
        }
        double walk = pmin(s.speed, s.sprint);
        double dP, angP; local_of(ps, S.pred, dP, angP);
        bool fresh = time - S.seen < 0.15;
        if (fresh) { if (S.dprev >= 0. && dP < S.dprev - 0.5) S.closing_t = time; S.dprev = dP; }
        bool chasing = time - S.closing_t < 1.5;
        if (time - S.seen > P.sg_lost) { sg_end(S, 3); return; }
        if (time - S.since > P.sg_timeout) { sg_end(S, 4); return; }
        double turn_to_p = angP;   // always keep facing the (believed) predator: our 90-degree cone keeps it in sight
        if (S.state == 1) {   // ACQUIRE
            if (dP <= P.sg_acq || chasing) S.state = 2;
            else {
                double dd, dir, turn; go_to(m, s, S.pred, P.sg_acq - 10., dd, dir, turn);
                plans[S.guide] = Plan{dd, dir, turn_to_p};
                return;
            }
        }
        if (S.state == 2) {   // LEAD to L
            if (dP > P.sg_release && !chasing) { S.state = 1; return; }
            double dT, angT; local_of(ps, sk.L, dT, angT);
            if (dT < 12. && dP < P.sg_far + 30.) { S.state = 3; sg_cnt[1]++; }
            else {
                double off = wrap(angT - angP), sgn = off > 0 ? 1. : -1.;
                bool blocked_ = std::fabs(off) < 1.1;
                double step = walk, dir = angT;
                if (dP < P.sg_near) S.sprinting = true; else if (dP > P.sg_near + 3.) S.sprinting = false;
                if (S.sprinting) { step = s.sprint; dir = wrap(angP + (blocked_ ? sgn * P.guide_block_ang : OPI)); }
                else if (blocked_) { dir = wrap(angP + sgn * 1.9); }
                else if (dP > P.sg_far) {
                    step = walk * pmax(0., (P.sg_far + 20. - dP) / 20.);
                    if (step < 1.) { if (S.wait_since < 0.) S.wait_since = time; if (time - S.wait_since > P.sg_wait) { S.wait_since = -1.; S.state = 1; return; } }
                    else S.wait_since = -1.;
                } else S.wait_since = -1.;
                plans[S.guide] = Plan{pmin(step, pmax(dT, 1.)), dir, turn_to_p};
                return;
            }
        }
        if (S.state == 3) {   // LANE: L -> G
            double dG, angG; local_of(ps, sk.G, dG, angG);
            if (dG < P.sg_arrive) { S.state = 4; sg_cnt[2]++; }
            else {
                double step = walk;
                if (dP < P.sg_near - 15.) step = s.sprint;          // it is right behind us: keep ahead
                else if (dP > P.sg_far + 10.) step = walk * 0.5;     // let it close in along the lane
                plans[S.guide] = Plan{pmin(step, dG), angG, turn_to_p};
                return;
            }
        }
        if (S.state == 4) {   // HOLD at G (or walk on)
            if (P.sg_escape > 0.) {
                P2 u{std::cos(sk.th), std::sin(sk.th)}; P2 E = add(sk.G, mul(u, P.sg_escape));
                double dE, angE; local_of(ps, E, dE, angE);
                if (dE < 4.) { sg_end(S, 6); return; }
                plans[S.guide] = Plan{pmin(s.sprint, dE), angE, 0.};
                return;
            }
            double dG, angG; local_of(ps, sk.G, dG, angG);
            plans[S.guide] = Plan{pmin(dG, walk), angG, turn_to_p};
            return;
        }
    });
}
