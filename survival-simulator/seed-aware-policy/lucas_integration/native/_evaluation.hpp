// Spectator-only measurements. Never passed to orchard::Policy.
struct NativeEvaluation {
    struct Before { std::string role; bool sprint = false, sacrifice = false; };
    struct Trap { orchard::P2 goal; bool started = false; double gap = 0.; };
    std::unordered_map<int64_t, Before> before;
    std::unordered_map<std::string, int64_t> deaths_role, deaths_cause_role;
    std::unordered_set<int64_t> last_guides, last_holds;
    std::vector<Trap> traps;
    std::unordered_map<int32_t, double> near_since;
    int64_t premature = 0, sprint_failures = 0, sacrifices = 0;
    int64_t attempts = 0, deliveries = 0, fruits = 0, ripe = 0, samples = 0;
    int64_t maximum_held30 = 0, final_held30 = 0, maximum_near = 0, peak = 0;
    double energy_fraction = 0., gap_total = 0., gap_max = 0.;

    void start(Engine& e, orchard::Policy& p, const std::vector<orchard::Act>& acts) {
        before.clear();
        std::unordered_map<int64_t, double> moves;
        for (const auto& a : acts) moves[a.aid] = a.dist;
        for (const auto& a : e.agents) {
            Before b; b.role = "gatherer"; b.sprint = a.energy >= a.max_energy/5.;
            if (p.minds.has(a.id)) {
                auto& m = *p.minds.at(a.id);
                if (m.old) b.role = "retired";
                if (e.time-m.evade_t<.15) b.role = "avoiding_predator";
            }
            before[a.id] = b;
            energy_fraction += a.energy/a.max_energy; samples++;
        }
        std::unordered_set<int64_t> guides, holds;
        p.groups.each([&](const int64_t&, orchard::GroupP& gp) {
            auto& g=*gp;
            if (!g.has_trap) return;
            auto mark=[&](int64_t aid,const char* role) { if (before.count(aid)) before[aid].role=role; };
            mark(g.bait,"bait"); mark(g.rep,"replacement"); mark(g.guide,"guide");
            mark(g.relay,"relay");
            for (auto aid:g.retired) mark(aid,"retired_bait");
            if (before.count(g.guide)) {
                guides.insert(g.guide);
                bool holding=g.guide_state==3 && g.ep_h && moves[g.guide]<=1e-9
                    && orchard::dist(p.M(g.guide).pose->p,g.trap.goal)<=p.P.guide_hand+1e-6;
                if (holding) { holds.insert(g.guide); before[g.guide].sacrifice=true; }
            }
            bool known=false;
            for (auto& t:traps) if (orchard::dist(t.goal,g.trap.goal)<10.) { known=true; break; }
            if (!known) traps.push_back(Trap{g.trap.goal});
        });
        for (auto aid:guides) attempts += !last_guides.count(aid);
        for (auto aid:holds) deliveries += !last_holds.count(aid);
        last_guides=std::move(guides); last_holds=std::move(holds);
        peak=std::max<int64_t>(peak,e.agents.size());
    }

    void finish(Engine& e, size_t first_event) {
        for (size_t i=first_event;i<e.events.size();i++) {
            const auto& v=e.events[i];
            if (v.kind==2) { fruits++; ripe+=v.age>=20.; continue; }
            auto found=before.find(v.id);
            Before b=found==before.end()?Before{"newborn",false,false}:found->second;
            deaths_role[b.role]++;
            deaths_cause_role[std::string(v.kind==1?"predator:":"starvation:")+b.role]++;
            if (v.kind==1) {
                if (b.sacrifice) sacrifices++;
                else { premature++; sprint_failures+=b.sprint; }
            }
        }
        std::vector<orchard::P2> occupied;
        for (auto& t:traps) {
            bool present=false;
            for (const auto& a:e.agents) {
                auto found=before.find(a.id);
                if (found==before.end()) continue;
                const auto& role=found->second.role;
                if (role!="bait" && role!="replacement" && role!="retired_bait") continue;
                if (std::hypot(a.x-t.goal.x,a.y-t.goal.y)<=4.) { present=true; break; }
            }
            if (present) { t.started=true; t.gap=0.; occupied.push_back(t.goal); }
            else if (t.started) { t.gap+=e.dt; gap_total+=e.dt; gap_max=std::max(gap_max,t.gap); }
        }
        int64_t near=0,held=0;
        for (const auto& pred:e.predators) {
            bool close=false;
            for (const auto& q:occupied) if (std::hypot(pred.x-q.x,pred.y-q.y)<=40.) { close=true; break; }
            if (close) {
                near++;
                if (!near_since.count(pred.key)) near_since[pred.key]=e.time;
                held+=e.time-near_since[pred.key]>=30.;
            } else near_since.erase(pred.key);
        }
        maximum_near=std::max(maximum_near,near);
        maximum_held30=std::max(maximum_held30,held); final_held30=held;
        peak=std::max<int64_t>(peak,e.agents.size());
    }

    PyObject* as_dict() const {
        PyObject* d=PyDict_New();
        auto set=[&](const char* k,PyObject* v) { PyDict_SetItemString(d,k,v); Py_DECREF(v); };
        auto integer=[&](const char* k,int64_t v) { set(k,PyLong_FromLongLong(v)); };
        auto real=[&](const char* k,double v) { set(k,PyFloat_FromDouble(v)); };
        auto counts=[&](const char* k,const auto& values) {
            PyObject* c=PyDict_New();
            for (const auto& item:values) { PyObject* v=PyLong_FromLongLong(item.second); PyDict_SetItemString(c,item.first.c_str(),v); Py_DECREF(v); }
            set(k,c);
        };
        counts("deaths_by_role",deaths_role); counts("deaths_by_cause_and_role",deaths_cause_role);
        integer("total_premature_captures",premature);
        integer("premature_captures_with_sprint_available",sprint_failures);
        integer("intentional_delivery_sacrifices",sacrifices);
        integer("guide_attempts",attempts); integer("guide_deliveries",deliveries);
        integer("fruit_eaten",fruits); integer("ripe_fruit_eaten",ripe);
        if (fruits) real("ripe_fraction",double(ripe)/fruits);
        if (samples) real("mean_agent_energy_fraction",energy_fraction/samples);
        real("bait_gap_seconds_total",gap_total); real("bait_gap_seconds_max",gap_max);
        bool started=false; for (const auto& t:traps) started|=t.started;
        set("bait_occupancy_started",PyBool_FromLong(started));
        integer("maximum_predators_within_40_of_bait",maximum_near);
        integer("maximum_predators_near_bait_30s",maximum_held30);
        integer("final_predators_near_bait_30s",final_held30);
        integer("peak_agents",peak);
        Py_INCREF(Py_None); set("confirmed_predator_retention",Py_None);
        set("metric_definitions",PyUnicode_FromString(
            "Spectator only. Capture means an agent eaten; premature excludes intentional stationary delivery holds. "
            "Sprint availability uses energy before the action, across all roles. Guide deliveries mean reaching a stationary handoff, not capture success. "
            "Bait gaps sum trap-seconds after first actual role-holder occupancy within 4 of an observed trap goal; nearby goals within 10 are the same site. "
            "A held-30s count is continuous actual proximity within 40 of an occupied bait site, not proof of permanent entrapment. "
            "Ripe fruit is age>=20. No evaluator values enter decisions."));
        return d;
    }
};
