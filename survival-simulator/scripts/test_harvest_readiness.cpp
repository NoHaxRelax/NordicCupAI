// Standalone observation-only regressions; no simulator or Python API linked.
#include "../fastsim/policy_abi.hpp"
using namespace polabi;
#include "../fastsim/_orchard.hpp"
#include "../fastsim/_evasion.hpp"
#include <cassert>
#include <iostream>

struct World {
    std::vector<std::vector<Obs>> observations;
    std::vector<AState> states;
    World(int n, bool predator=false) : observations(n), states(n) {
        for (int i=0; i<n; ++i) {
            if (predator && i==1) {
                Obs o{}; o.type=2; o.distance=60.; o.has_rel_dir=true;
                observations[i].push_back(o);
            }
            states[i]=AState{i+1, &observations[i], 150., GRASSLAND, 0.,
                             10., 20., 60., 1.047197551, 200., 500.};
        }
    }
};

orchard::EvasionPolicy policy(bool enabled=true) {
    orchard::Params p; p.harvest_ready=enabled;
    return orchard::EvasionPolicy({0},p,orchard::PredParams{},orchard::WallParams{});
}

int main() {
    {
        World w(8,true); auto p=policy();
        auto a=p.call(w.states.data(),w.states.size(),.1);
        assert(a[1].dist==0.);
        a=p.call(w.states.data(),w.states.size(),.2);
        assert(a[1].dist>0.);
    }
    {
        World w(7,true); auto p=policy();
        auto a=p.call(w.states.data(),w.states.size(),.1);
        assert(a[1].dist>0.);
    }
    {
        World w(5); auto p=policy();
        auto a=p.call(w.states.data(),w.states.size(),.1);
        int births=0; for (const auto& x:a) births+=x.spawn;
        assert(births==3);
    }
    {
        World w(5); auto p=policy();
        for (auto& s:w.states) s.energy=125.;
        auto a=p.call(w.states.data(),w.states.size(),.1);
        for (const auto& x:a) assert(!x.spawn);
    }
    {
        World w(5); auto p=policy(false);
        auto a=p.call(w.states.data(),w.states.size(),.1);
        for (const auto& x:a) assert(!x.spawn);
    }
    {
        World w(8,true); auto p=policy(false);
        auto a=p.call(w.states.data(),w.states.size(),.1);
        assert(a[1].dist>0.);
    }
    {
        auto p=policy(); p.time=1.;
        orchard::FruitM fruit{}; fruit.born_lo=0.; fruit.born_hi=1.;
        assert(p.ready(fruit,50.));
        auto old=policy(false); old.time=1.;
        assert(!old.ready(fruit,50.));
    }
    {
        World w(8);
        Obs fruit{}; fruit.type=0; fruit.distance=50.;
        fruit.has_id=true; fruit.id=1;
        w.observations[0].push_back(fruit);
        auto p=policy(); auto a=p.call(w.states.data(),w.states.size(),.1);
        assert(a[0].dist>0.);
    }
    std::cout << "8 native harvest-readiness regressions passed\n";
}
