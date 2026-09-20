// Native C++ full-game benchmark for the duplicate-action interaction.
//
// The policy and engine are both Lucas's C++ implementation.  Python is used
// only during startup by native_common.hpp to bind NumPy math kernels; no
// Python policy or per-tick action construction runs after initialization.
#define SURVIVAL_ENGINE_SOURCE "../lucas_integration/native/_nengine.cpp"
#include "../../research/seed_inference/native_common.hpp"

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double TURN_COST = 0.5;
constexpr double KILL_RADIUS = 15.0;

struct View {
    int64_t aid;
    double energy;
    double age;
    double pred_distance;
};

struct Attempt {
    double t;
    int64_t farm;
    int64_t doomed;
    double pred_distance;
    double farm_energy;
    double doomed_energy;
    size_t actions;
    std::string outcome = "pending";
    double death_energy = 0.;
    double death_t = 0.;
};

double nearest_predator(const orchard::AState& state) {
    double result = OINF;
    for (const Obs& obs : *state.obs) {
        if (obs.type == 2) result = std::min(result, obs.distance);
    }
    return result;
}

J attempt_json(const Attempt& attempt) {
    return {
        {"t", attempt.t},
        {"farm", attempt.farm},
        {"doomed", attempt.doomed},
        {"pred_distance", attempt.pred_distance},
        {"farm_energy", attempt.farm_energy},
        {"doomed_energy", attempt.doomed_energy},
        {"actions", attempt.actions},
        {"outcome", attempt.outcome},
        {"death_energy", attempt.death_energy},
        {"death_t", attempt.death_t},
    };
}

void usage() {
    std::cerr
        << "action_burst_bench CONFIG SEED BUDGET MAX_HARVESTS HORIZON OUT "
        << "[--cooldown seconds] [--trigger distance] [--closing distance] "
        << "[--min-free agents] [--no-spawn] [--cap-min n] [--cap-max n] "
        << "[--cap-mult x]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        usage();
        return 2;
    }
    const fs::path config_path = argv[1];
    const uint32_t seed = static_cast<uint32_t>(std::stoull(argv[2]));
    const int budget = std::stoi(argv[3]);
    const int max_harvests = std::stoi(argv[4]);
    const double horizon = std::stod(argv[5]);
    const fs::path output_path = argv[6];
    double cooldown = 50.;
    double trigger = 30.;
    double closing = 6.;
    int min_free = 6;

    J config = read_json(config_path);
    // Lucas distributes named one-config files (for example
    // {"with_predators_best": {...}}); accept those without accidentally
    // treating the outer label as a policy parameter.
    if (config.size() == 1 && config.begin().value().is_object()) {
        config = config.begin().value();
    }
    auto params = initialize_math(config);
    for (int i = 7; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&]() -> const char* {
            if (++i >= argc) throw std::runtime_error("Missing value for " + flag);
            return argv[i];
        };
        if (flag == "--cooldown") cooldown = std::stod(value());
        else if (flag == "--trigger") trigger = std::stod(value());
        else if (flag == "--closing") closing = std::stod(value());
        else if (flag == "--min-free") min_free = std::stoi(value());
        else if (flag == "--cap-min") params.cap_min = std::stod(value());
        else if (flag == "--cap-max") params.cap_max = std::stod(value());
        else if (flag == "--cap-mult") params.cap_mult = std::stod(value());
        else if (flag == "--no-spawn") {
            params.no_spawn = 1.;
        } else {
            throw std::runtime_error("Unknown flag " + flag);
        }
    }

    auto engine = make_engine(seed);
    orchard::Policy policy({0}, params);
    const auto started = Clock::now();
    std::unordered_map<int64_t, double> previous_pred_distance;
    std::unordered_map<int64_t, size_t> pending_attempts;
    std::vector<Attempt> attempts;
    int transfers = 0;
    int max_actions = 0;
    int total_actions = 0;
    int ticks = 0;
    double last_harvest_time = -OINF;

    while (!engine->agents.empty() && engine->time + 1e-6 < horizon) {
        auto states = policy_states(engine.get());
        std::vector<View> views;
        views.reserve(states.size());
        std::unordered_map<int64_t, View> by_id;
        std::unordered_map<int64_t, double> current_pred_distance;
        for (const auto& state : states) {
            const double distance = nearest_predator(state);
            View view{state.aid, state.energy, state.age, distance};
            views.push_back(view);
            by_id.emplace(view.aid, view);
            current_pred_distance.emplace(view.aid, distance);
        }

        auto actions = policy.call(std::move(states), engine->time);
        const bool enough_agents = static_cast<int>(views.size()) >= min_free + 2;
        if (budget > 0 && static_cast<int>(attempts.size()) < max_harvests && enough_agents &&
            engine->time - last_harvest_time >= cooldown) {
            struct Candidate {
                double distance;
                double closing_gain;
                double doomed_energy;
                int64_t farm;
                int64_t doomed;
            };
            std::vector<Candidate> candidates;
            for (size_t index = 0; index < views.size(); ++index) {
                const View& farm = views[index];
                if (index == 0 || farm.pred_distance > trigger) continue;
                const auto previous = previous_pred_distance.find(farm.aid);
                const bool first_contact = previous == previous_pred_distance.end();
                const double closing_gain = first_contact ? 0. : previous->second - farm.pred_distance;
                if (first_contact ? farm.pred_distance > KILL_RADIUS : closing_gain < closing) continue;
                const View& doomed = views[index - 1];
                if (doomed.pred_distance <= KILL_RADIUS) continue;
                candidates.push_back({farm.pred_distance, closing_gain, doomed.energy, farm.aid, doomed.aid});
            }
            if (!candidates.empty()) {
                const auto chosen = *std::min_element(candidates.begin(), candidates.end(),
                    [](const Candidate& a, const Candidate& b) {
                        if (a.distance != b.distance) return a.distance < b.distance;
                        if (a.closing_gain != b.closing_gain) return a.closing_gain > b.closing_gain;
                        if (a.doomed_energy != b.doomed_energy) return a.doomed_energy < b.doomed_energy;
                        return a.farm < b.farm;
                    });
                actions.erase(std::remove_if(actions.begin(), actions.end(), [&](const orchard::Act& action) {
                    return action.aid == chosen.farm || action.aid == chosen.doomed;
                }), actions.end());
                const int n_doomed = std::max(0, static_cast<int>(std::ceil((chosen.doomed_energy + 1.) / TURN_COST)));
                for (int i = 0; i < n_doomed; ++i) actions.push_back({chosen.doomed, 0., 0., PI, false});
                for (int i = 0; i < budget; ++i) actions.push_back({chosen.farm, 0., 0., PI, false});
                attempts.push_back({engine->time, chosen.farm, chosen.doomed, chosen.distance,
                                    by_id.at(chosen.farm).energy, chosen.doomed_energy, actions.size()});
                pending_attempts[chosen.farm] = attempts.size() - 1;
                last_harvest_time = engine->time;
            }
        }
        previous_pred_distance = std::move(current_pred_distance);

        const size_t event_start = engine->events.size();
        max_actions = std::max(max_actions, static_cast<int>(actions.size()));
        total_actions += static_cast<int>(actions.size());
        for (const auto& action : actions) {
            engine->agent_step({action.aid, action.dist, true, action.direction, action.turn, action.spawn});
        }
        engine->non_agent_step();
        ++ticks;
        for (size_t event_index = event_start; event_index < engine->events.size(); ++event_index) {
            const auto& event = engine->events[event_index];
            const auto pending = pending_attempts.find(event.id);
            if (pending == pending_attempts.end() || (event.kind != 0 && event.kind != 1)) continue;
            Attempt& attempt = attempts[pending->second];
            attempt.outcome = event.kind == 1 ? "predator" : "starvation";
            attempt.death_energy = event.energy;
            attempt.death_t = event.t;
            if (event.kind == 1 && event.energy < -1000.) ++transfers;
            pending_attempts.erase(pending);
        }
    }

    J output = {
        {"seed", seed},
        {"score", engine->score},
        {"survival", engine->time},
        {"population", engine->agents.size()},
        {"predators", engine->predators.size()},
        {"budget", budget},
        {"harvest_attempts", attempts.size()},
        {"confirmed_predator_transfers", transfers},
        {"pending_attempts", pending_attempts.size()},
        {"max_actions", max_actions},
        {"mean_actions", ticks ? static_cast<double>(total_actions) / ticks : 0.},
        {"ticks", ticks},
        {"wall_seconds", seconds(started)},
        {"no_spawn", params.no_spawn > 0.},
        {"cap_min", params.cap_min},
        {"cap_max", params.cap_max},
        {"cap_mult", params.cap_mult},
        {"min_free", min_free},
        {"attempts", J::array()},
    };
    for (const auto& attempt : attempts) output["attempts"].push_back(attempt_json(attempt));
    save_json(output_path, output);
    std::cout << output.dump() << '\n';
}
