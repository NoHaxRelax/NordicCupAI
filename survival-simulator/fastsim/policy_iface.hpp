// The engine <-> policy calling convention. Both sides include this; it adds nothing
// to what policy_abi.hpp already exposes, so including it does not widen the policy's
// view. Everything crossing the boundary is POD or a reference to POD: no Python
// objects, no dicts, no serialization, no per-tick heap allocation.
#pragma once

#include "policy_abi.hpp"

namespace polabi {

// models/orchard_evasion_policy.py's evasion keywords.
struct PredCfg {
    int64_t mode = 1;
    double r = 70., face_r = 80., sprint_r = 40., dodge_r = 80., dodge_ang = 1.4, turn_max = 1.0;
};

// Debug rows, matching what fastsim/debug_policy.py compares against the Python minds.
struct MindRow {
    int64_t aid, group;
    double px, py, theta;
    int64_t post; bool has_post;
    int64_t fruit; bool has_fruit;
    bool old;
    double ex, ey; bool has_explore;
    double wx, wy; bool has_watch;
    int64_t n_edges;
    double best_d, energy_prev;
    int64_t n_prev_marks, n_hear_hist;
    double prev_theta;
};
struct GroupRow {
    int64_t gid; bool anchored;
    int64_t n_trees, n_fruits, n_cells, next_tree, next_fruit;
};

class IPolicy {
public:
    virtual ~IPolicy() {}
    virtual void copy_parameters(const IPolicy& other) = 0;
    // Decide for `n` agents. The returned reference points at an internal buffer that
    // stays valid until the next call; nothing is copied out and nothing is allocated
    // once the buffers have grown to their working size.
    virtual const std::vector<Act>& call(const AState* states, size_t n, double sim_time) = 0;
    virtual void metrics(int64_t& flee, int64_t& face, int64_t& sprint) const = 0;
    // Optional per-phase accounting; `out` receives PH_N seconds. Reporting only.
    virtual void set_profile(bool on) = 0;
    virtual void phases(double* out) const = 0;
    virtual size_t dump_minds(MindRow* out, size_t cap) const = 0;
    virtual size_t dump_groups(GroupRow* out, size_t cap) const = 0;
};

// Returns nullptr and points *err at a static message on a rejected configuration
// (currently: pred_share != 0, which is not implemented and must not silently no-op).
IPolicy* make_policy(const uint32_t* seed_key, size_t nkey, const Cfg& cfg, const char** err);
void destroy_policy(IPolicy* p);

}  // namespace polabi
