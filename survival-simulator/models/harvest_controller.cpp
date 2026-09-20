// Native harvest decisions. Inputs are public dictionaries and the existing
// observation tracker; this translation unit cannot access simulator objects.
// Built as exploit_lab.reliable_harvest, replacing the historical Python module.
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <vector>
namespace py=pybind11;
using namespace pybind11::literals;
using Z=std::complex<double>;
constexpr double PI=3.14159265358979323846;

double num(py::handle d,const char* k) { return py::cast<double>(d[py::str(k)]); }
long long aid(py::handle d) { return py::cast<long long>(d[py::str("agent_id")]); }
py::dict action(long long id,double dist=0,double direction=0,double turn=0,bool spawn=false) {
    return py::dict("agent_id"_a=id,"move_distance"_a=dist,"move_direction"_a=direction,
                    "turn_angle"_a=turn,"spawn_agent"_a=spawn);
}
long long minimum_drain(double energy,double remaining,double meals=0) {
    double debt=std::max(200.000001,100.+30.*(std::max(0.,remaining)+.1))+meals;
    return 2*std::max(0LL,(long long)std::ceil(energy+debt));
}
py::list drain(long long id,long long count) {
    if(count<0 || count%2) throw py::value_error("drain count must be nonnegative and even");
    py::list out(count); auto a=action(id,0,0,PI), b=action(id,0,0,-PI);
    for(long long i=0;i<count;i+=2) { out[i]=a; out[i+1]=b; }
    return out;
}
double energy_after(py::handle s,py::iterable actions) {
    double energy=num(s,"energy"),speed=num(s,"speed"),sprint=num(s,"sprint_speed");
    for(auto a:actions) {
        double d=std::max(0.,std::min(num(a,"move_distance"),sprint));
        if(energy<num(s,"max_energy")/5) d=std::min(d,speed);
        energy-=std::min(d,speed)*.05+std::max(0.,d-speed)*.5;
        energy-=std::min(PI,std::abs(num(a,"turn_angle")))/(2*PI);
        if(py::cast<bool>(a[py::str("spawn_agent")]) && energy>100) energy-=100;
    }
    return energy;
}
bool will_visit(py::iterable living,py::dict energies,long long target,double dt=.1) {
    bool skip=false;
    for(auto v:living) {
        long long id=py::cast<long long>(v);
        if(id==target) return !skip;
        if(skip) skip=false;
        else if(py::cast<double>(energies[v])<=dt) skip=true;
    }
    return false;
}
double factor(py::handle s) {
    auto b=py::cast<std::string>(s[py::str("biome")]);
    return b=="river"?.3:b=="swamp"?.5:b=="desert"?.8:1.;
}
struct Attempt {
    long long farm,sacrifice,target_key;
    py::list witnesses;
    double expected_gain,min_gain;
    int ticks=1;
};

struct HarvestPolicy {
    double horizon,engage_range;
    long long extra,cap;
    int min_free,max_ticks;
    py::object memory;
    std::shared_ptr<Attempt> pending;
    double previous_score=std::numeric_limits<double>::quiet_NaN(),next_commit_at=0;
    py::dict metrics;
    py::object last_event=py::none();

    HarvestPolicy(double h=3000,long long ex=0,int free=6,int ticks=2,
                  py::object capacity=py::none(),double range=45)
      :horizon(h),engage_range(range),extra(ex),cap(capacity.is_none()?250000+ex:py::cast<long long>(capacity)),
       min_free(free),max_ticks(ticks) {
        if(ex<0 || ex%2) throw py::value_error("extra_drain_actions must be a nonnegative even integer");
        if(!std::isfinite(h)||h<=0||ticks<1||free<1||cap<1) throw py::value_error("invalid harvest limits");
        memory=py::module_::import("exploit_lab.predator_memory").attr("PredatorMemory")();
        for(auto k:{"attempts","confirmed","failed","retries","sacrifices","cap_rejections",
                    "floor_rejections","skip_rejections"}) metrics[py::str(k)]=0;
    }
    void inc(const char* key) { metrics[py::str(key)]=py::cast<long long>(metrics[py::str(key)])+1; }
    py::list observe(py::list states,double now,double score) {
        last_event=py::none();
        py::tuple stale=pending?py::make_tuple(pending->farm):py::tuple();
        memory.attr("update")(states,now,stale);
        bool present=false;
        if(pending) for(auto s:states) if(aid(s)==pending->farm) present=true;
        if(pending && !present) {
            double gain=score-previous_score;
            bool confirmed=gain>=pending->min_gain;
            if(confirmed) memory.attr("certify")(pending->witnesses,horizon,now);
            // Any disappearance ages out pre-meal sightings, even if other meals
            // obscure the public score confirmation. Never immediately retarget.
            next_commit_at=now+.5;
            inc(confirmed?"confirmed":"failed");
            last_event=py::dict("kind"_a=confirmed?"confirmed":"failed","farm"_a=pending->farm,
                                "gain"_a=gain,"time"_a=now);
            pending.reset();
        }
        previous_score=score;
        return memory.attr("filtered")(states,now).cast<py::list>();
    }
    py::list finish(py::list states,py::list out) { memory.attr("record")(states,out); return out; }
    py::list actions(py::list states,double now,py::list native) {
        std::map<long long,py::dict> by;
        std::map<long long,py::list> grouped;
        py::list living;
        py::dict energies;
        for(auto s:states) by.emplace(aid(s),py::reinterpret_borrow<py::dict>(s));
        for(auto a:native) {
            auto id=aid(a);
            if(!grouped.count(id)) grouped.emplace(id,py::list());
            grouped.at(id).append(a);
        }
        for(auto& [id,s]:by) {
            living.append(id);
            if(!grouped.count(id)) grouped.emplace(id,py::list());
            energies[py::int_(id)]=energy_after(s,grouped.at(id));
        }
        struct Candidate { long long id; py::object track; };
        std::vector<Candidate> candidates;
        py::dict current=memory.attr("current");
        auto tracks=[&](long long id) { return current.contains(py::int_(id))?
            py::cast<py::list>(current[py::int_(id)]):py::list(); };
        if(pending) {
            if(pending->ticks>=max_ticks) { inc("failed");pending.reset();return finish(states,native); }
            for(auto t:tracks(pending->farm))
                if(py::cast<long long>(t.attr("key"))==pending->target_key &&
                   !py::cast<bool>(memory.attr("sleeping")(t,now)))
                    candidates.push_back({pending->farm,py::reinterpret_borrow<py::object>(t)});
        } else {
            if(now<next_commit_at) return finish(states,native);
            for(auto& [id,s]:by) {
                if(num(s,"energy")<=0) continue;
                for(auto t:tracks(id)) {
                    Z point=py::cast<Z>(t.attr("point")),motion=py::cast<Z>(t.attr("motion"));
                    py::dict obs=t.attr("obs");
                    double magnitude=std::abs(motion);
                    // Full sprint stride in any shipped terrain. Walking strides
                    // (11, 8.8, 5.5, 3.3) are excluded, including imminent rests.
                    bool sprint=false;
                    for(double stride:{15.,12.,7.5,4.5}) if(std::abs(magnitude-stride)<.02) sprint=true;
                    if(py::cast<bool>(t.attr("moving")) && sprint &&
                       (std::conj(point)*motion).real()<0 && std::abs(num(obs,"rel_dir"))<.6 &&
                       num(obs,"distance")<=engage_range && py::cast<double>(t.attr("certified_until"))<=now)
                        candidates.push_back({id,py::reinterpret_borrow<py::object>(t)});
                }
            }
            std::stable_sort(candidates.begin(),candidates.end(),[&](auto& a,auto& b) {
                double ea=num(by.at(a.id),"energy"),eb=num(by.at(b.id),"energy");
                if(ea!=eb) return ea<eb;
                double da=num(a.track.attr("obs"),"distance"),db=num(b.track.attr("obs"),"distance");
                return da!=db?da<db:a.id<b.id;
            });
        }
        for(auto& candidate:candidates) {
            auto fid=candidate.id;auto it=by.find(fid);
            if(it==by.begin() || it==by.end()) continue;
            auto did=std::prev(it)->first;
            auto farm=it->second;auto target=candidate.track;
            int survivors=0;
            for(auto& [id,s]:by) if(id!=fid && id!=did && py::cast<double>(energies[py::int_(id)])>.1) ++survivors;
            if(survivors<min_free) {inc("floor_rejections");continue;}
            if(!will_visit(living,energies,did)) {inc("skip_rejections");continue;}
            Z velocity=std::polar(std::abs(py::cast<Z>(target.attr("motion"))),py::cast<double>(target.attr("heading")));
            Z point=pending?Z():py::cast<Z>(target.attr("point"))+2.*velocity;
            double direction=std::arg(point),travel=std::abs(point)/factor(farm);
            py::list moves;
            while(travel>1e-8 && py::len(moves)<20) {
                double step=std::min({num(farm,"speed"),num(farm,"sprint_speed"),travel});
                if(step<=0) break;
                moves.append(action(fid,step,direction));travel-=step;
            }
            if(travel>1e-8) continue;
            py::dict frames=memory.attr("relative_frames")(states,fid);
            std::set<long long> exposed;
            double meals=0,global_meals=0;
            Z target_point=py::cast<Z>(target.attr("point"));
            for(auto& [id,s]:by) {
                if(id==fid || id==did) continue;
                global_meals+=num(s,"max_energy");
                double movement=0;
                for(auto a:grouped.at(id)) movement+=std::max(0.,std::min(num(a,"move_distance"),num(s,"sprint_speed")));
                bool near=!frames.contains(py::int_(id));
                if(!near) {
                    py::tuple frame=frames[py::int_(id)];
                    near=std::abs(py::cast<Z>(frame[0])-target_point)<45.+movement;
                }
                if(near) {exposed.insert(id);meals+=num(s,"max_energy");}
            }
            for(auto a:native) if(py::cast<bool>(a[py::str("spawn_agent")])) {
                if(aid(a)!=fid && aid(a)!=did) global_meals+=75;
                if(exposed.count(aid(a))) meals+=75;
            }
            double after_move=energy_after(farm,moves);
            long long count=pending?0:minimum_drain(after_move,horizon-now,meals)+extra;
            long long sacrifice_count=2*std::max(0LL,(long long)std::ceil(num(by.at(did),"energy")));
            py::list kept;
            for(auto a:native) if(aid(a)!=fid && aid(a)!=did) kept.append(a);
            if((long long)py::len(kept)+sacrifice_count+(long long)py::len(moves)+count>cap) {inc("cap_rejections");continue;}
            py::list out;
            for(auto a:kept) out.append(a);
            for(auto a:drain(did,sacrifice_count)) out.append(a);
            for(auto a:moves) out.append(a);
            for(auto a:drain(fid,count)) out.append(a);
            if(pending) {++pending->ticks;pending->sacrifice=did;inc("retries");}
            else {
                double expected=-(after_move-count*.5)/100.;
                pending=std::make_shared<Attempt>(Attempt{fid,did,py::cast<long long>(target.attr("key")),
                    memory.attr("witnesses")(states,fid,target).cast<py::list>(),expected,
                    std::max(1.,expected-global_meals/100.-5.)});
                inc("attempts");
            }
            inc("sacrifices");
            py::dict obs=target.attr("obs");
            last_event=py::dict("kind"_a=count?"attempt":"retry","farm"_a=fid,"sacrifice"_a=did,
                "time"_a=now,"drain"_a=count,"target_distance"_a=num(obs,"distance"),
                "target_motion"_a=std::abs(py::cast<Z>(target.attr("motion"))),"target_angle"_a=num(obs,"angle"));
            return finish(states,out);
        }
        if(pending) {inc("failed");pending.reset();}
        return finish(states,native);
    }
};

PYBIND11_MODULE(reliable_harvest,m) {
    m.attr("IMPLEMENTATION")="cpp";
    m.def("action",&action,py::arg("aid"),py::arg("dist")=0.,py::arg("direction")=0.,py::arg("turn")=0.,py::arg("spawn")=false);
    m.def("drain_actions",&drain);
    m.def("minimum_drain_actions",&minimum_drain,py::arg("energy"),py::arg("remaining"),py::arg("positive_meals")=0.);
    m.def("energy_after",&energy_after);
    m.def("will_visit",&will_visit,py::arg("living"),py::arg("energies"),py::arg("target"),py::arg("dt")=.1);
    py::class_<Attempt,std::shared_ptr<Attempt>>(m,"Attempt")
      .def_readwrite("farm",&Attempt::farm).def_readwrite("sacrifice",&Attempt::sacrifice)
      .def_readwrite("target_key",&Attempt::target_key).def_readwrite("witnesses",&Attempt::witnesses)
      .def_readwrite("expected_gain",&Attempt::expected_gain).def_readwrite("ticks",&Attempt::ticks);
    py::class_<HarvestPolicy>(m,"HarvestPolicy")
      .def(py::init<double,long long,int,int,py::object,double>(),py::arg("horizon")=3000.,
           py::arg("extra_drain_actions")=0,py::arg("min_free_agents")=6,py::arg("max_harvest_ticks")=2,
           py::arg("max_actions_per_tick")=py::none(),py::arg("engage_range")=45.)
      .def("observe",&HarvestPolicy::observe).def("actions",&HarvestPolicy::actions)
      .def_readwrite("memory",&HarvestPolicy::memory).def_readwrite("pending",&HarvestPolicy::pending)
      .def_readwrite("metrics",&HarvestPolicy::metrics).def_readwrite("last_event",&HarvestPolicy::last_event)
      .def_readwrite("previous_score",&HarvestPolicy::previous_score).def_readwrite("extra",&HarvestPolicy::extra)
      .def_readwrite("cap",&HarvestPolicy::cap).def_readwrite("min_free",&HarvestPolicy::min_free)
      .def_readwrite("max_ticks",&HarvestPolicy::max_ticks).def_readwrite("horizon",&HarvestPolicy::horizon);
}
