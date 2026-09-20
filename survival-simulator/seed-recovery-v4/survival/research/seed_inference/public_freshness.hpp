#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <map>
#include <set>
#include <cstdint>

namespace public_freshness {
using J=nlohmann::json;
inline bool observations_equal(J a,J b) {
    if(a==b)return true;
    auto less=[](const J& x,const J& y){return x.dump()<y.dump();};
    std::sort(a.begin(),a.end(),less);std::sort(b.begin(),b.end(),less);return a==b;
}
struct Frame {
    J pose_input;
    std::set<int64_t> excluded_agent_ids;
    bool has_prior_tick=false;
};
// Public data only. Never pass the filtered frame to model-parity checking;
// filtering exists solely for spatial inference from fresh observations.
class Guard {
    J previous;
    Frame last;
public:
    Frame update(const J& current) {
        double t=current.at("sim_time");
        if(!previous.is_null() && t==previous.at("sim_time").get<double>() && current==previous)return last;
        Frame out{current,{},false};
        if(!previous.is_null() && t>previous.at("sim_time").get<double>()) {
            out.has_prior_tick=true;
            std::map<int64_t,const J*> old;
            for(const auto& a:previous.at("agent_status"))old[a.at("agent_id")]=&a;
            J keep=J::array();
            for(const auto& a:current.at("agent_status")) {
                int64_t id=a.at("agent_id");auto p=old.find(id);
                bool stale=p!=old.end() && a.at("age")==p->second->at("age")
                    && observations_equal(a.at("observations"),p->second->at("observations"));
                if(stale)out.excluded_agent_ids.insert(id);else keep.push_back(a);
            }
            out.pose_input["agent_status"]=std::move(keep);
            if(out.pose_input.contains("n_agents"))out.pose_input["n_agents"]=out.pose_input["agent_status"].size();
        }
        // A lower time starts a new game. Newborns and the first input have no
        // prior-tick freshness proof; this helper makes no claim beyond that.
        previous=current;last=out;return out;
    }
};
}
