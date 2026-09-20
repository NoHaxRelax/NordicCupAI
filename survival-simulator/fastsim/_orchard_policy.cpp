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
#include "../models/late_activation.hpp"

namespace polabi {
namespace {

double cfg_get(const Cfg& c, const char* k, double dflt, bool* found = nullptr) {
    for (size_t i = 0; i < c.n; i++)
        if (std::strcmp(c.items[i].key, k) == 0) { if (found) *found = true; return c.items[i].value; }
    if (found) *found = false;
    return dflt;
}

void parse_params(const Cfg& c, orchard::Params& P, const char* prefix = "") {
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
              {"nursery_bonus", &P.nursery_bonus},{"share_obs",&P.share_obs},{"econ_start",&P.econ_start},{"econ_radius",&P.econ_radius},{"econ_horizon",&P.econ_horizon},{"cap_budget",&P.cap_budget},{"crowd_weight",&P.crowd_weight},{"fruit_auction",&P.fruit_auction},{"auction_cost",&P.auction_cost},{"fruit_net",&P.fruit_net},{"food_risk",&P.food_risk},{"post_opt",&P.post_opt},{"rock_penalty",&P.rock_penalty},{"relocate_after",&P.relocate_after},{"relocate_energy",&P.relocate_energy},{"renewal_weight",&P.renewal_weight},{"budget_reserve",&P.budget_reserve},{"aging_food",&P.aging_food}};
    for (F& f : fs) { bool got = false; double v = cfg_get(c, (std::string(prefix) + f.k).c_str(), 0., &got); if (got) *f.v = v; }
    struct B { const char* k; bool* v; };
    B bs[] = {{"idle_sweep", &P.idle_sweep}, {"extra_old", &P.extra_old}, {"cull", &P.cull}, {"heir_select", &P.heir_select},
              {"heir_at_food", &P.heir_at_food}, {"old_eat_last", &P.old_eat_last}, {"heir_needs_site", &P.heir_needs_site}};
    for (B& b : bs) { bool got = false; double v = cfg_get(c, (std::string(prefix) + b.k).c_str(), 0., &got); if (got) *b.v = (v != 0.); }
    P.feed_breed = c.feed_mode && std::strcmp(c.feed_mode, "breed") == 0;
}

class PolicyImpl : public IPolicy {
public:
    orchard::EvasionPolicy pol;
    orchard::Params base_params, late_params;
    LateActivation gate;
    PolicyImpl(const std::vector<uint32_t>& key, const orchard::Params& P, const orchard::PredParams& PR, const Cfg& cfg)
        : pol(key, P, PR), base_params(P), late_params(P) {
        parse_params(cfg, late_params, "late_");
        gate.enabled = cfg_get(cfg, "gate_enabled", 0.) != 0.;
        gate.after_ticks = cfg_get(cfg, "gate_ticks", 12000.);
        gate.below_population = cfg_get(cfg, "gate_population", 10.);
        gate.logic = (int)cfg_get(cfg, "gate_logic", 2.);
        gate.persistence = (int)cfg_get(cfg, "gate_persistence", 0.);
    }
    void copy_parameters(const IPolicy& other) override {
        const auto& source = static_cast<const PolicyImpl&>(other);
        base_params = source.base_params;
        late_params = source.late_params;
        gate.enabled = source.gate.enabled; gate.after_ticks = source.gate.after_ticks;
        gate.below_population = source.gate.below_population; gate.logic = source.gate.logic;
        gate.persistence = source.gate.persistence;
        pol.P = base_params;
        pol.PRED = source.pol.PRED;
        // Keep RNG, odometry, groups, fruit claims and every agent's memory.
        pol.cluster_cache.clear();
    }

    const std::vector<Act>& call(const AState* states, size_t n, double sim_time) override {
        pol.P=gate.step(n) ? late_params : base_params;
        const auto& x=pol.PRED;
        if(sim_time>=x.phase_start && (x.phase_population<=0 || n<=x.phase_population)) {
            if(x.late_cap>=0) pol.P.cap_mult=x.late_cap;
            if(x.late_retire>=0) pol.P.no_eat_age=x.late_retire;
            if(x.late_reach>=0) pol.P.fruit_reach=x.late_reach;
            if(x.late_reserve>=0) pol.P.breed_reserve_late=x.late_reserve;
        }
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
    // pred_share was an unsupported Python compatibility knob. The native
    // shared-map extension is explicitly selected with share_obs instead.
    if (cfg_get(cfg, "pred_share", 0.) != 0.) {
        if (err) *err = "Use share_obs for native shared map and predator observations";
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
    PR.stuck_mode = cfg_get(cfg, "stuck_mode", PR.stuck_mode);
    PR.stuck_radius = cfg_get(cfg, "stuck_radius", PR.stuck_radius);
    PR.stuck_gap = cfg_get(cfg, "stuck_gap", PR.stuck_gap);
    PR.stuck_reward = cfg_get(cfg, "stuck_reward", PR.stuck_reward);
    PR.stuck_release = cfg_get(cfg, "stuck_release", PR.stuck_release);
    PR.stuck_patience = cfg_get(cfg, "stuck_patience", PR.stuck_patience);
    PR.stuck_energy = cfg_get(cfg, "stuck_energy", PR.stuck_energy);
    PR.wall_escape = cfg_get(cfg, "pred_wall_escape", PR.wall_escape);
    PR.wall_look = cfg_get(cfg, "pred_wall_look", PR.wall_look);
    PR.wall_reward = cfg_get(cfg, "pred_wall_reward", PR.wall_reward);
        PR.pulse_degrees=cfg_get(cfg,"pulse_degrees",PR.pulse_degrees);
    PR.pulse_ticks=cfg_get(cfg,"pulse_ticks",PR.pulse_ticks);
    PR.pulse_idle=cfg_get(cfg,"pulse_idle",PR.pulse_idle);
    PR.feature_start=cfg_get(cfg,"feature_start",PR.feature_start);
    PR.feature_population=cfg_get(cfg,"feature_population",PR.feature_population);
    PR.look_steps=cfg_get(cfg,"look_steps",PR.look_steps);
    PR.look_radius=cfg_get(cfg,"look_radius",PR.look_radius);
    PR.risk_margin=cfg_get(cfg,"risk_margin",PR.risk_margin);
    PR.behind_weight=cfg_get(cfg,"behind_weight",PR.behind_weight);
    PR.energy_weight=cfg_get(cfg,"energy_weight",PR.energy_weight);
    PR.phase_start=cfg_get(cfg,"phase_start",PR.phase_start);
    PR.phase_population=cfg_get(cfg,"phase_population",PR.phase_population);
    PR.late_cap=cfg_get(cfg,"late_cap",PR.late_cap);
    PR.late_retire=cfg_get(cfg,"late_retire",PR.late_retire);
    PR.late_reach=cfg_get(cfg,"late_reach",PR.late_reach);
    PR.late_reserve=cfg_get(cfg,"late_reserve",PR.late_reserve);
    std::vector<uint32_t> key(seed_key, seed_key + nkey);
    if (key.empty()) key.push_back(0);
    return new PolicyImpl(key, P, PR, cfg);
}

void destroy_policy(IPolicy* p) { delete p; }

}  // namespace polabi
