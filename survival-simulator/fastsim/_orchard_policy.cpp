// The policy translation unit. Its ENTIRE include list is the four lines below:
// the observation-only boundary, the calling convention, and the two policy headers.
// No Python.h, no numpy, no _engine.cpp - so engine internals, true coordinates, the
// world seed, fruit/tree ages, predator state and the engine RNG are not merely
// unused here, they are undeclared. fastsim/check_boundary.py checks this
// mechanically (include list, plus the compiled object's undefined symbols).
#include "policy_abi.hpp"
#include "policy_iface.hpp"

using namespace polabi;  // Obs, AState, Act and the CPython-semantics helpers

#include "_orchard.hpp"
#include "_evasion.hpp"

namespace polabi {
namespace {

double cfg_get(const Cfg& c, const char* k, double dflt, bool* found = nullptr) {
    for (size_t i = 0; i < c.n; i++)
        if (std::strcmp(c.items[i].key, k) == 0) { if (found) *found = true; return c.items[i].value; }
    if (found) *found = false;
    return dflt;
}

void parse_params(const Cfg& c, orchard::Params& P) {
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
              {"nursery_bonus", &P.nursery_bonus}};
    for (F& f : fs) { bool got = false; double v = cfg_get(c, f.k, 0., &got); if (got) *f.v = v; }
    struct B { const char* k; bool* v; };
    B bs[] = {{"idle_sweep", &P.idle_sweep}, {"extra_old", &P.extra_old}, {"cull", &P.cull}, {"heir_select", &P.heir_select},
              {"heir_at_food", &P.heir_at_food}, {"old_eat_last", &P.old_eat_last}, {"heir_needs_site", &P.heir_needs_site}};
    for (B& b : bs) { bool got = false; double v = cfg_get(c, b.k, 0., &got); if (got) *b.v = (v != 0.); }
    P.feed_breed = c.feed_mode && std::strcmp(c.feed_mode, "breed") == 0;
}

class PolicyImpl : public IPolicy {
public:
    orchard::EvasionPolicy pol;
    PolicyImpl(const std::vector<uint32_t>& key, const orchard::Params& P, const orchard::PredParams& PR)
        : pol(key, P, PR) {}

    const std::vector<Act>& call(const AState* states, size_t n, double sim_time) override {
        return pol.call(states, n, sim_time);
    }
    void metrics(int64_t& flee, int64_t& face, int64_t& sprint) const override {
        flee = pol.flee_ticks; face = pol.face_ticks; sprint = pol.sprint_ticks;
    }
    void set_profile(bool on) override { pol.profile_phases = on; }
    void phases(double* out) const override {
        for (int i = 0; i < PH_N; i++) out[i] = pol.ph[i];
        // PH_REST timed the whole call; make it the unattributed remainder.
        for (int i = 0; i < PH_REST; i++) out[PH_REST] -= out[i];
    }
    size_t dump_minds(MindRow* out, size_t cap) const override {
        size_t i = 0;
        const_cast<orchard::EvasionPolicy&>(pol).minds.each([&](const int64_t& aid, orchard::MindP& m) {
            if (i >= cap) { i++; return; }
            MindRow& r = out[i++];
            r.aid = aid; r.group = m->group;
            r.px = m->pose->p.x; r.py = m->pose->p.y; r.theta = m->pose->theta;
            r.post = m->post; r.has_post = m->has_post;
            r.fruit = m->fruit; r.has_fruit = m->has_fruit;
            r.old = m->old;
            r.ex = m->explore_p.x; r.ey = m->explore_p.y; r.has_explore = m->has_explore;
            r.wx = m->watch_p.x; r.wy = m->watch_p.y; r.has_watch = m->has_watch;
            r.n_edges = (int64_t)m->edges.size();
            r.best_d = m->best_d; r.energy_prev = m->energy_prev;
            r.n_prev_marks = (int64_t)m->prev_marks.size();
            r.n_hear_hist = (int64_t)m->hear_hist.size();
            r.prev_theta = m->prev_pose ? m->prev_pose->theta : 0.0;
        });
        return i;
    }
    size_t dump_groups(GroupRow* out, size_t cap) const override {
        size_t i = 0;
        const_cast<orchard::EvasionPolicy&>(pol).groups.each([&](const int64_t& gid, orchard::GroupP& g) {
            if (i >= cap) { i++; return; }
            GroupRow& r = out[i++];
            r.gid = gid; r.anchored = g->anchored;
            int64_t nt = 0, nf = 0;
            g->trees.each([&](const int64_t&, orchard::TreeP&) { nt++; });
            g->fruits.each([&](const int64_t&, orchard::FruitP&) { nf++; });
            r.n_trees = nt; r.n_fruits = nf;
            r.n_cells = (int64_t)g->cells.size();
            r.next_tree = g->next_tree; r.next_fruit = g->next_fruit;
        });
        return i;
    }
};

}  // namespace

IPolicy* make_policy(const uint32_t* seed_key, size_t nkey, const Cfg& cfg, const char** err) {
    // Sharing predator sightings across a group is deliberately not implemented:
    // upstream measured it worse than own-sightings-only, so pred_share accepts 0 and
    // rejects anything else rather than silently ignoring the setting. Same rule and
    // same message as models/orchard_evasion_policy.py, and it lives on the policy
    // side because it is a statement about the policy, not about the engine.
    if (cfg_get(cfg, "pred_share", 0.) != 0.) {
        if (err) *err = "pred_share>0 (group-shared predator sightings) is not implemented";
        return nullptr;
    }
    orchard::Params P;
    parse_params(cfg, P);
    orchard::PredParams PR;
    PR.mode = (int64_t)cfg_get(cfg, "pred_mode", (double)PR.mode);  // int(pred_mode)
    PR.r = cfg_get(cfg, "pred_r", PR.r);
    PR.face_r = cfg_get(cfg, "pred_face_r", PR.face_r);
    PR.sprint_r = cfg_get(cfg, "pred_sprint_r", PR.sprint_r);
    PR.dodge_r = cfg_get(cfg, "pred_dodge_r", PR.dodge_r);
    PR.dodge_ang = cfg_get(cfg, "pred_dodge_ang", PR.dodge_ang);
    PR.turn_max = cfg_get(cfg, "pred_turn_max", PR.turn_max);
    PR.gaze = cfg_get(cfg, "pred_gaze", PR.gaze);
    PR.cone_gate = cfg_get(cfg, "pred_cone_gate", PR.cone_gate);
    PR.cone_margin = cfg_get(cfg, "pred_cone_margin", PR.cone_margin);
    PR.wall_escape = cfg_get(cfg, "pred_wall_escape", PR.wall_escape);
    PR.wall_look = cfg_get(cfg, "pred_wall_look", PR.wall_look);
    PR.wall_reward = cfg_get(cfg, "pred_wall_reward", PR.wall_reward);
    std::vector<uint32_t> key(seed_key, seed_key + nkey);
    if (key.empty()) key.push_back(0);
    return new PolicyImpl(key, P, PR);
}

void destroy_policy(IPolicy* p) { delete p; }

}  // namespace polabi
