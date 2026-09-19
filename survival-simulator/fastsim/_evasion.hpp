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
    double gaze = 0., cone_gate = 0., cone_margin = 0.1;
    double wall_escape = 0., wall_look = 30., wall_reward = 80.;
};

class EvasionPolicy : public Policy {
public:
    PredParams PRED;
    // Reporting only; never read by a decision (mirrors self.metrics in Python).
    int64_t flee_ticks = 0, face_ticks = 0, sprint_ticks = 0;

    EvasionPolicy(const std::vector<uint32_t>& seed_key, const Params& p, const PredParams& pp)
        : Policy(seed_key, p), PRED(pp) {}

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

    Plan act(Mind& m, const AState& s) override {
        if (PRED.mode) {
            const Obs* t = threat(s);
            if (t) {
                const double distance = t->distance, angle = t->angle;
                const double tm = PRED.turn_max;
                // Public relative heading only. Hearing remains omnidirectional.
                if (PRED.cone_gate && distance > 60. && t->has_rel_dir &&
                    std::abs(wrap(angle + OPI - t->rel_dir)) > OPI / 6 + PRED.cone_margin)
                    return Policy::act(m, s);
                if (distance <= PRED.r) {
                    double away = wrap(angle + OPI);
                    // A predator outruns an agent in a straight line, so break across
                    // its approach instead of directly away once it is this close.
                    if (distance <= PRED.dodge_r)
                        away = wrap(away + (angle >= 0. ? PRED.dodge_ang : -PRED.dodge_ang));
                    bool sprinting = distance <= PRED.sprint_r;
                    double reach = sprinting ? s.sprint : pmin(s.speed, s.sprint);
                    // Search short reachable headings around OBSERVED edges. Reward
                    // breaking line of sight only beyond the predator's hearing
                    // radius. This is a geometric heuristic, not privileged rollout.
                    if (PRED.wall_escape && !m.edges.empty()) {
                        P2 pos = m.pose->p, pred = polar(*m.pose, *t);
                        double best = -OINF, selected = away;
                        for (int k = 0; k < 24; ++k) {
                            double h = m.pose->theta + away + k * OPI / 12.;
                            P2 q = add(pos, mul(unit(h), PRED.wall_look));
                            bool blocked = false, hidden = false;
                            for (const auto& e : m.edges) {
                                if (time - e.t > 25.) continue;
                                if (segments_cross(pos, q, e.a, e.b) || point_segment(q, e.a, e.b) < 7.) blocked = true;
                                if (segments_cross(pred, q, e.a, e.b)) hidden = true;
                            }
                            if (blocked) continue;
                            double d = dist(q, pred);
                            double value = d - 0.3 * std::abs(wrap(h - m.pose->theta - away));
                            if (distance > 60. && d > 65. && hidden) value += PRED.wall_reward;
                            if (value > best) { best = value; selected = wrap(h - m.pose->theta); }
                        }
                        away = selected;
                    }
                    flee_ticks++;
                    sprint_ticks += sprinting ? 1 : 0;
                    release_fruit(m);
                    double gaze = PRED.gaze ? angle : away;
                    return Plan{reach, away, pmax(-tm, pmin(tm, gaze))};
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
