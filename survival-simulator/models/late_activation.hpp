#pragma once
// Public inputs only: number of policy calls and number of agents in the request.
struct LateActivation {
    bool enabled = false, active = false, used = false, retired = false;
    long long calls = 0;
    double after_ticks = 12000., below_population = 10.;
    int logic = 2, persistence = 0; // time, population, AND, OR; reversible, latch, once
    bool step(size_t population) {
        const bool old = active;
        const bool t = calls >= after_ticks;
        const bool p = population < below_population;
        const bool condition = logic == 0 ? t : logic == 1 ? p : logic == 2 ? t && p : t || p;
        ++calls;
        if (!enabled) return active = false;
        if (persistence == 1) active = active || condition;
        else if (persistence == 2) {
            if (used && old && !condition) retired = true;
            active = !retired && condition;
        } else active = condition;
        used = used || active;
        return active;
    }
};
