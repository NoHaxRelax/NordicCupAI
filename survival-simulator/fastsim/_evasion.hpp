// Native port of models/orchard_evasion_policy.py: the thin own-sighting predator
// evasion layer on top of the vendored orchard foraging/population core.
//
// Decision-identical to the Python subclass by construction: it overrides the same
// single method (act / _act), takes the same two branches in the same order, and
// returns instead of delegating, so none of the base act()'s side effects (blocked
// pruning, progress() odometry updates, last_site) run on a fleeing or facing tick.
//
// Structure follows survival-simulator/oscar-overnight-cpp nightsim/_npolicy.hpp's
// evade(), but only loosely: that port weights several threats by inverse distance,
// picks its dodge side from the predator's reported rel_dir, keeps a committed dodge
// heading, applies itself as a post-hoc patch over an already-computed plan, and has
// no fruit release. Ours uses the single nearest own-sighting, picks the dodge side
// from the bearing sign, replaces the plan outright, and releases the claimed fruit.
// Only the parameter names are shared. Where the two disagree, the Python file wins.

namespace orchard {

struct PredParams {
    int64_t mode = 1;
    double r = 70., face_r = 80., sprint_r = 40., dodge_r = 80., dodge_ang = 1.4, turn_max = 1.0;
    // React only to a predator hunting THIS agent; see the Python constructor.
    // Off by default - a mechanism to be measured, not an assumed improvement.
    int64_t evade_closest = 0;
    // Face the predator while retreating instead of turning our back to it.
    // predator.py:37 charges when `abs(agent_looking_dir) > pi/2 || distance < 90`,
    // and agent_looking_dir is the same number as our own observed `angle`. Turning
    // away drives it to pi and satisfies the gate; facing holds it near 0 and leaves
    // the predator in its pivot branch, which closes 10.6/tick instead of 15.
    int64_t face_threat = 0;
};

// ---------------------------------------------------------------- deflection
// Port of the wall_* layer in models/orchard_evasion_policy.py. Constants carry
// the same meaning and the same reasons; see that file's module docstring.
//
// Floor of the deflection band: the engine's charge gate is distance < 90 tested
// after the agent moves, a pivot step closes at most 10.6, so 105 survives one
// fully blocked retreat tick without handing over a charge. Not tunable - it
// guards an engine threshold, not a preference.
constexpr double WALL_MIN_R = 105.;
// Sign changes per engagement before the sign freezes. Re-optimising every tick
// oscillates: a greedy controller walks the predator to a 28-unit gap and back out.
constexpr int64_t WALL_MAX_FLIPS = 2;
// Remembered edges older than this are ignored (what the base _goto trusts).
constexpr double WALL_EDGE_AGE = 25.;
// Engagement state is dropped after this long with no steering tick.
constexpr double WALL_RESET = 1.;

struct WallParams {
    int64_t mode = 0;                 // 0 off / 1 steer at a wall / 2 hold pivot only
    double engage_r = 190., target_gap = 35., epsilon = 0.25;
    int64_t max_ticks = 120;
};

// Python keeps this on the mind object as `m.wsteer`; here it is keyed by agent id.
// Equivalent because an entry older than WALL_RESET is discarded before use, so a
// reused id can never inherit a live engagement.
struct WSteer {
    double sign = 0.;
    int64_t flips = 0, ticks = 0;
    bool locked = false, done = false;
    double at = 0.;
};

class EvasionPolicy : public Policy {
public:
    PredParams PRED;
    WallParams WALL;
    // Live engagement per agent; see WSteer. Entries are dropped on reset, so this
    // never grows past the agents currently steering.
    std::unordered_map<int64_t, WSteer> wsteer;
    std::unordered_map<int64_t, double> harvest_probe_after;
    int64_t harvest_probe_agent = -1;
    double harvest_probe_time = -1.;
    // Reporting only; never read by a decision (mirrors self.metrics in Python).
    int64_t flee_ticks = 0, face_ticks = 0, sprint_ticks = 0;
    int64_t steer_ticks = 0, steer_locked = 0, steer_flips = 0, steer_aborts = 0,
            steer_no_target = 0, steer_not_target = 0;

    EvasionPolicy(const std::vector<uint32_t>& seed_key, const Params& p, const PredParams& pp,
                  const WallParams& wp)
        : Policy(seed_key, p), PRED(pp), WALL(wp) {}

    // Nearest predator in this agent's own observations, or nullptr.
    // Strict <, so the first of equally distant sightings wins, exactly as the
    // Python loop's `o['distance'] < nearest['distance']` does.
    const Obs* threat(const AState& s) const {
        const Obs* nearest = nullptr;
        for (const Obs& o : *s.obs) {
            if (o.type != 2) continue;  // 2 == 'Predator'
            if (!nearest || o.distance < nearest->distance) nearest = &o;
        }
        return nearest;
    }

    // Free a claimed fruit before fleeing/facing, or it sits reserved and unreachable
    // by hungrier group-mates for as long as the threat lasts.
    void release_fruit(Mind& m) {
        if (!m.has_fruit) return;
        Group& g = G(m.group);
        if (g.fruits.has(m.fruit)) g.fruits.at(m.fruit)->has_claim = false;
        m.has_fruit = false;
    }

    // ------------------------------------------------------------ deflection
    // Group-frame point the maneuver must never drag a predator towards. Posts are
    // the orchard itself and win; without any, the other members' dead-reckoned
    // positions stand in. A lone agent with no post has nothing to protect (false).
    // Accumulation order follows the group's set iteration, which PySetEmu
    // reproduces exactly, so the float sums match Python's term for term.
    bool colony_point(const Mind& m, P2& out) {
        Group& g = G(m.group);
        std::vector<P2> posts, others;
        g.agents.each([&](int64_t aid) {
            if (!minds.has(aid)) return;
            const Mind& om = *minds.at(aid);
            if (om.has_post && g.trees.has(om.post)) posts.push_back(g.trees.at(om.post)->p);
            if (aid != m.aid) others.push_back(om.pose->p);
        });
        const std::vector<P2>& pts = posts.empty() ? others : posts;
        if (pts.empty()) return false;
        double sx = 0., sy = 0.;
        for (const P2& p : pts) sx += p.x;
        for (const P2& p : pts) sy += p.y;
        out = P2{sx / (double)pts.size(), sy / (double)pts.size()};
        return true;
    }

    // Nearest remembered wall point to the predator that leads away from home.
    // Mirrors the predator's own edge-avoidance geometry - globally nearest point
    // across candidate segments, not a corner search - because the engine picks
    // exactly that point and corners are not special to it.
    bool wall_target(Mind& m, P2 pred_p, double retreat_dir, P2& out_pt, double& out_gap) {
        const PoseObj& pose = *m.pose;
        P2 colony{}, cd{};
        bool have_colony = colony_point(m, colony);
        if (have_colony) {
            cd = P2{colony.x - pose.p.x, colony.y - pose.p.y};
            double cn = hypot2(cd.x, cd.y);
            if (cn > 1e-6) {
                cd = P2{cd.x / cn, cd.y / cn};
                // Retreating towards home would tow the predator into the orchard
                // whichever wall is chosen, so refuse the whole engagement.
                if (std::cos(retreat_dir) * cd.x + std::sin(retreat_dir) * cd.y > 0.) return false;
            } else {
                have_colony = false;
            }
        }
        const double reach = 2. * WALL.engage_r;
        bool found = false;
        double best_gap = 0.;
        P2 best_pt{};
        for (const EdgeMem& e : m.edges) {
            if (time - e.t >= WALL_EDGE_AGE) continue;
            double vx = e.b.x - e.a.x, vy = e.b.y - e.a.y;
            double L2 = vx * vx + vy * vy;
            if (L2 < 1e-9) continue;
            double t = ((pred_p.x - e.a.x) * vx + (pred_p.y - e.a.y) * vy) / L2;
            t = pmax(0., pmin(1., t));
            P2 pt{e.a.x + t * vx, e.a.y + t * vy};
            double gap = dist(pt, pred_p);
            if (gap > reach) continue;
            if (have_colony) {
                double wx = pt.x - pose.p.x, wy = pt.y - pose.p.y;
                double wn = hypot2(wx, wy);
                if (wn < 1e-6 || (wx / wn) * cd.x + (wy / wn) * cd.y > 0.) continue;
            }
            if (!found || gap < best_gap) { found = true; best_gap = gap; best_pt = pt; }
        }
        if (!found) return false;
        out_pt = best_pt;
        out_gap = best_gap;
        return true;
    }

    // Is this agent the one the predator is actually chasing? The engine picks
    // min(agents, key=distance), so only its nearest visible agent has any control
    // over it; steering on somebody else's predator spends movement and turn energy
    // for nothing. Own observations only, so a nearer agent this one cannot see
    // still slips through - conservative, but it never wrongly suppresses the
    // true target.
    bool is_target(const AState& s, const Obs& t) const {
        const double d = t.distance, a = t.angle;
        const double px = d * std::cos(a), py = d * std::sin(a);
        for (const Obs& o : *s.obs) {
            if (o.type != 1) continue;  // 1 == 'Agent'
            double ox = o.distance * std::cos(o.angle), oy = o.distance * std::sin(o.angle);
            if (hypot2(px - ox, py - oy) < d) return false;
        }
        return true;
    }

    // Hold the predator in pivot mode and aim its 45-degree step at a wall.
    // Returns false to fall through to the unchanged flee/face/forage branches.
    bool steer(Mind& m, const AState& s, const Obs& t, Plan& out) {
        const double distance = t.distance, angle = t.angle;
        if (distance <= pmax(WALL_MIN_R, PRED.r) || distance > WALL.engage_r) return false;
        if (!is_target(s, t)) { steer_not_target++; return false; }
        auto it = wsteer.find(m.aid);
        WSteer* st = (it == wsteer.end()) ? nullptr : &it->second;
        if (st && time - st->at > WALL_RESET) {  // contact lost long enough: new engagement
            wsteer.erase(it);
            st = nullptr;
        }
        if (st && st->done) return false;  // tick cap already spent on this engagement
        const PoseObj& pose = *m.pose;
        // Retreat straight away from the predator: the heading that holds the band
        // open longest (pivot closes 10.6/tick, walking away gives back 10).
        const double retreat = wrap(angle + OPI);
        const double walk = pmin(s.speed, s.sprint);
        const bool held = (WALL.mode != 1) || (st && (st->locked || st->flips >= WALL_MAX_FLIPS));
        P2 target{};
        double gap = 0.;
        if (!held) {
            // Activation gate (mode 1, only while still re-optimising): a wall has to
            // be in sight and it has to lead away from home. An engagement already
            // under way may keep aiming at a remembered wall, because holding the
            // predator in pivot means facing it, which routinely swings the agent's
            // 60-degree cone off the wall it started on.
            if (!st) {
                bool any_edge = false;
                for (const Obs& o : *s.obs) if (o.type == 4) { any_edge = true; break; }  // 4 == 'Edge'
                if (!any_edge) return false;
            }
            if (!wall_target(m, polar(pose, t), pose.theta + retreat, target, gap)) {
                steer_no_target++;
                return false;
            }
        }
        if (!st) {
            WSteer fresh;
            fresh.at = time;
            wsteer[m.aid] = fresh;
            st = &wsteer[m.aid];
        }
        // --- sign: which 45-degree pivot offset the predator takes this tick ---
        double sign;
        if (held) {
            sign = st->sign != 0. ? st->sign : 1.;  // frozen: ablation, lock-in, or flip-flop
        } else {
            P2 pred_p = polar(pose, t);
            double bearing = std::atan2(pose.p.y - pred_p.y, pose.p.x - pred_p.x);
            bool have = false;
            double best_d = 0., best_c = 0.;
            const double cands[2] = {1., -1.};
            for (double cand : cands) {
                double a = bearing + cand * OPI / 4;
                P2 step{pred_p.x + 15. * std::cos(a), pred_p.y + 15. * std::sin(a)};
                double d = dist(step, target);
                if (!have || d < best_d) { have = true; best_d = d; best_c = cand; }
            }
            sign = best_c;
            if (st->sign != 0. && sign != st->sign) {
                st->flips++;
                steer_flips++;
                if (st->flips >= WALL_MAX_FLIPS) sign = st->sign;  // oscillating: freeze
            }
            if (gap <= WALL.target_gap) { st->locked = true; steer_locked++; }
        }
        st->sign = sign;
        st->at = time;
        st->ticks++;
        if (st->ticks >= WALL.max_ticks) { st->done = true; steer_aborts++; }
        // --- facing: put the predator's own view of us at -sign*epsilon so that
        // pivot_sign == -sign(agent_looking_dir) == sign. The engine moves with the
        // old facing and turns afterwards, so retreat and aim do not fight: predict
        // where the predator sits after our own step, then turn onto it.
        const double step = walk * MOVE_PENALTY[s.biome];
        const double px = distance * std::cos(angle) - step * std::cos(retreat);
        const double py = distance * std::sin(angle) - step * std::sin(retreat);
        double turn = wrap(std::atan2(py, px) + sign * WALL.epsilon);
        turn = pmax(-PRED.turn_max, pmin(PRED.turn_max, turn));
        steer_ticks++;
        release_fruit(m);
        out = Plan{walk, retreat, turn};
        return true;
    }

    Plan act(Mind& m, const AState& s) override {
        if (P.harvest_ready && harvest_probe_time != time) {
            harvest_probe_time = time;
            harvest_probe_agent = -1;
            // A stationary observation removes ambiguous ego translation when
            // landmarks are sparse. Probe one agent outside contact range after
            // both lagged predator moves; other agents retain normal behavior.
            if (states.size() >= 8) {
                int64_t first = states.front().aid;
                for (const AState& q : states) first = std::min(first, q.aid);
                double best_energy = OINF;
                for (const AState& q : states) {
                    if (q.aid == first || q.energy <= 10. || q.energy >= best_energy ||
                        harvest_probe_after[q.aid] > time) continue;
                    const Obs* t = threat(q);
                    if (!t || t->distance < 50. || t->distance > 65. ||
                        std::abs(t->rel_dir) >= 0.6) continue;
                    best_energy = q.energy;
                    harvest_probe_agent = q.aid;
                }
            }
        }
        if (P.harvest_ready && s.aid == harvest_probe_agent) {
            const Obs* t = threat(s);
            harvest_probe_after[s.aid] = time + 1.;
            release_fruit(m);
            return Plan{0., 0., pmax(-PRED.turn_max, pmin(PRED.turn_max, t->angle))};
        }
        if (PRED.mode) {
            const Obs* t = threat(s);
            // Somebody else's predator: fall through to foraging untouched. Applied
            // once here rather than inside each branch so flee, deflection and face
            // all agree on whose predator this is.
            if (t && PRED.evade_closest && !is_target(s, *t)) t = nullptr;
            if (t) {
                const double distance = t->distance, angle = t->angle;
                const double tm = PRED.turn_max;
                if (distance <= PRED.r) {
                    double away = wrap(angle + OPI);
                    // A predator outruns an agent in a straight line, so break across
                    // its approach instead of directly away once it is this close.
                    if (distance <= PRED.dodge_r)
                        away = wrap(away + (angle >= 0. ? PRED.dodge_ang : -PRED.dodge_ang));
                    bool sprinting = distance <= PRED.sprint_r;
                    double reach = sprinting ? s.sprint : pmin(s.speed, s.sprint);
                    flee_ticks++;
                    sprint_ticks += sprinting ? 1 : 0;
                    release_fruit(m);
                    // Retreat along `away` either way; only the facing differs.
                    const double aim = PRED.face_threat ? angle : away;
                    return Plan{reach, away, pmax(-tm, pmin(tm, aim))};
                }
                // Deflection sits below flee and above face: its band floor is
                // WALL_MIN_R > the engine's 90-unit charge gate and >= pred_r, so a
                // predator already close enough to charge is still handled by flee.
                if (WALL.mode) {
                    Plan p{};
                    if (steer(m, s, *t, p)) return p;
                }
                if (distance <= PRED.face_r) {
                    face_ticks++;
                    release_fruit(m);
                    return Plan{0., 0., pmax(-tm, pmin(tm, angle))};
                }
            }
        }
        return Policy::act(m, s);
    }
};

}  // namespace orchard
