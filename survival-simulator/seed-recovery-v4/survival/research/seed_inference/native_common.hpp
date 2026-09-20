#pragma once
// C++ simulation/controller runtime. CPython is initialized once to obtain
// NumPy's compiled math kernels; no Python code runs in the tick/search loop.
#ifndef SURVIVAL_ENGINE_SOURCE
#define SURVIVAL_ENGINE_SOURCE "../../fastsim/_engine.cpp"
#endif
#include SURVIVAL_ENGINE_SOURCE
#include <nlohmann/json.hpp>
#include <httplib.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <queue>
#include <climits>
#include <ctime>
#include <sys/wait.h>
#include <unistd.h>
using J=nlohmann::json;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
int64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
std::string utc_now(){auto t=std::time(nullptr);std::tm tm;gmtime_r(&t,&tm);char s[32];std::strftime(s,sizeof(s),"%Y-%m-%dT%H:%M:%SZ",&tm);return s;}
std::string read_text(const fs::path& p){std::ifstream f(p);return {std::istreambuf_iterator<char>(f),{}};}
J read_json(const fs::path& p){return J::parse(read_text(p));}
void save_json(const fs::path& p,const J& value){fs::create_directories(p.parent_path());auto tmp=p.string()+".tmp";{std::ofstream f(tmp);f<<value.dump();}fs::rename(tmp,p);}
void touch(const fs::path& p){std::ofstream f(p);}
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
const char* TAGS[]={"Fruit","Agent","Predator","Tree","Edge"};
int biome_index(const std::string& name){for(int i=0;i<5;i++)if(name==BIOMES[i].name)return i;return -1;}
int obs_index(const std::string& name){for(int i=0;i<5;i++)if(name==TAGS[i])return i;throw std::runtime_error("Unknown observation type");}
J observations(const std::vector<Obs>& obs){
    J out=J::array();for(const auto& o:obs){J r={{"type",TAGS[o.type]}};
        if(o.type==4)r["coords"]={{o.c[0],o.c[1]},{o.c[2],o.c[3]}};
        else{r["distance"]=o.distance;r["angle"]=o.angle;if(o.has_rel_dir)r["rel_dir"]=o.rel_dir;if(o.has_id)r["id"]=o.id;}
        out.push_back(r);}return out;
}
J public_state(Engine& e){
    J agents=J::array();for(const auto& a:e.agents)agents.push_back({{"agent_id",a.id},{"energy",a.energy},{"age",a.age},
        {"biome",BIOMES[e.biome_at(a.x,a.y)].name},{"speed",a.speed},{"sprint_speed",a.sprint_speed},{"hearing_radius",a.hearing_radius},
        {"vision_angle",a.cone_angle},{"vision_range",a.vision_radius},{"max_energy",a.max_energy},
        {"observations",observations(e.agent_observations[a.id])}});
    return J{{"game_status",e.agents.empty()?"game_over":"ok"},{"score",e.score},{"sim_time",e.time},{"n_agents",e.agents.size()},{"agent_status",agents}};
}
J canonical(J value){
    if(value.is_object()){
        for(auto& v:value.items())v.value()=canonical(v.value());
        if(value.contains("observations")){auto& a=value["observations"];std::sort(a.begin(),a.end(),[](const J& x,const J& y){return x.dump()<y.dump();});}
    }else if(value.is_array())for(auto& v:value)v=canonical(v);
    return value;
}
bool same(const J& a,const J& b,std::string* error=nullptr,const std::string& path=""){
    auto bad=[&](){if(error&&error->empty())*error=path+": "+a.dump().substr(0,200)+" != "+b.dump().substr(0,200);return false;};
    if(a.is_number()&&b.is_number())return std::abs(a.get<double>()-b.get<double>())<=1e-7||bad();
    if(a.type()!=b.type())return bad();
    if(a.is_object()){
        if(a.size()!=b.size())return bad();for(auto& item:a.items())if(!b.contains(item.key())||!same(item.value(),b[item.key()],error,path+"."+item.key()))return false;return true;
    }
    if(a.is_array()){
        if(a.size()!=b.size())return bad();
        if(path.size()>=13&&path.substr(path.size()-13)==".observations"){
            std::vector<bool> used(b.size(),false);for(size_t i=0;i<a.size();i++){bool found=false;
                for(size_t j=0;j<b.size();j++)if(!used[j]&&same(a[i],b[j])){used[j]=true;found=true;break;}
                if(!found){if(error)*error=path+": unmatched "+a[i].dump();return false;}}return true;
        }
        for(size_t i=0;i<a.size();i++)if(!same(a[i],b[i],error,path+"["+std::to_string(i)+"]"))return false;return true;
    }
    return a==b||bad();
}
// Corner-ray hit edges can differ across NumPy/CPU floating-point kernels.
// Keep full comparison receipts, but synchronize the dynamic model using every
// agent attribute and all non-edge observations. No dynamic field is removed.
J dynamic_state(J state){for(auto& a:state["agent_status"]){J obs=J::array();for(auto& o:a["observations"])if(o["type"]!="Edge")obs.push_back(o);a["observations"]=obs;}return state;}
void advance(Engine& e,const J& actions){
    for(const auto& a:actions)e.agent_step({a.at("agent_id").get<int64_t>(),a.at("move_distance"),true,a.at("move_direction"),a.at("turn_angle"),a.at("spawn_agent")});
    e.non_agent_step();
}
std::unique_ptr<Engine> make_engine(uint32_t seed){return std::make_unique<Engine>(1600,1200,400,5,0,32,50,std::vector<uint32_t>{seed},.1,true);}
struct Inputs{
    std::vector<std::vector<Obs>> obs;
    std::vector<orchard::AState> states;
    Inputs(const J& body){
        auto& agents=body.at("agent_status");obs.resize(agents.size());states.reserve(agents.size());
        for(size_t i=0;i<agents.size();i++){auto& a=agents[i];for(auto& r:a.at("observations")){
            Obs o{};o.type=obs_index(r.at("type"));if(o.type==4){o.c[0]=r["coords"][0][0];o.c[1]=r["coords"][0][1];o.c[2]=r["coords"][1][0];o.c[3]=r["coords"][1][1];}
            else{o.distance=r.at("distance");o.angle=r.at("angle");o.has_rel_dir=r.contains("rel_dir");o.rel_dir=r.value("rel_dir",0.);o.has_id=r.contains("id");o.id=r.value("id",int64_t(-1));}obs[i].push_back(o);}
            states.push_back({a.at("agent_id"),&obs[i],a.at("energy"),biome_index(a.at("biome")),a.at("age"),a.at("speed"),a.at("sprint_speed"),a.at("hearing_radius"),a.at("vision_angle"),a.at("vision_range"),a.at("max_energy")});}
    }
};
J policy_actions(orchard::Policy& policy,const J& body){if(body.at("agent_status").empty())return J::array();Inputs in(body);J out=J::array();for(auto a:policy.call(std::move(in.states),body.at("sim_time")))out.push_back({{"agent_id",a.aid},{"move_distance",a.dist},{"move_direction",a.direction},{"turn_angle",a.turn},{"spawn_agent",a.spawn}});return out;}
orchard::Params initialize_math(const J& config){
    Py_Initialize();if(_import_array()<0||_import_umath()<0)throw std::runtime_error("Cannot load NumPy native C API");
    PyObject* np=PyImport_ImportModule("numpy");if(!np)throw std::runtime_error("Cannot import NumPy");
    const char* names[]={"sin","cos","arctan2","hypot"};NpLoop* loops[]={&np_sin_l,&np_cos_l,&np_atan2_l,&np_hypot_l};
    for(int i=0;i<4;i++){auto u=PyObject_GetAttrString(np,names[i]);if(!u||!find_loop(u,i<2?1:2,*loops[i]))throw std::runtime_error("Cannot bind native NumPy loop");}
    auto module=PyImport_ImportModule("json");auto text=PyUnicode_FromString(config.dump().c_str());auto params=PyObject_CallMethod(module,"loads","O",text);
    orchard::Params p;if(!parse_params(params,p))throw std::runtime_error("Invalid policy config");
    // Imported ufuncs remain alive. Numerical loops can run without the GIL,
    // just as the existing C++ engine's run_policy method does.
    PyEval_SaveThread();return p;
}
J creature_record(const Creature& a,int64_t id){return J{{"id",id},{"x",a.x},{"y",a.y},{"size",a.size},{"energy",a.energy},{"max_energy",a.max_energy},{"speed",a.speed},{"sprint_speed",a.sprint_speed},{"direction",a.direction},{"hearing_radius",a.hearing_radius},{"vision_range",a.vision_radius},{"vision_angle",a.cone_angle}};}
struct NativeReplay{
    J data;size_t tick=0;std::set<int64_t> previous_agents,previous_predators;
    NativeReplay(Engine& e,uint32_t seed,const std::string& title){
        J obstacles=J::array();for(auto o:e.obstacles)obstacles.push_back({{"x",o.x},{"y",o.y},{"width",o.w},{"height",o.h}});
        data={{"format","survival-replay"},{"version",1},{"world",{{"width",e.W},{"height",e.H},{"obstacles",obstacles}}},
          {"meta",{{"title",title},{"policy","native-seed-inference"},{"seed",seed},{"dt",e.dt},{"record_interval",10.},
          {"scenario","Native inferred world, not hosted ground truth"},{"renderer","state-only C++"},{"created_at",utc_now()},
          {"notes","Native C++ recorder follows ReplayRecorder schema. State-only exception for requested all-C++ live runtime. Full-precision public requests and prior forecasts are separate receipts."}}},
          {"frames",J::array()},{"events",J::array()}};
    }
    void capture(Engine& e,const J& actions=J::array(),bool force=false){
        std::set<int64_t> aa,pp;for(auto& a:e.agents)aa.insert(a.id);for(auto& p:e.predators)pp.insert(p.key);
        bool important=aa!=previous_agents||pp!=previous_predators;previous_agents=aa;previous_predators=pp;
        bool keep=force||important||tick%100==0;tick++;if(!keep)return;
        auto& frames=data["frames"];if(!frames.empty()&&frames.back()["t"].get<double>()==e.time)return;
        J agents=J::array(),predators=J::array(),fruits=J::array(),trees=J::array();
        for(auto& a:e.agents){J r=creature_record(a,a.id);r.update({{"age",a.age},{"max_age",a.max_age},{"biome",BIOMES[e.biome_at(a.x,a.y)].name},{"observations",observations(e.agent_observations[a.id])}});agents.push_back(r);}
        for(auto& p:e.predators){J r=creature_record(p,p.key);r["resting"]=p.resting;predators.push_back(r);}
        for(auto& f:e.fruits)fruits.push_back({{"id",f.fruit_id},{"x",f.x},{"y",f.y},{"radius",f.radius},{"age",f.age},{"energy",f.energy}});
        for(auto& t:e.trees)trees.push_back({{"id",t.key},{"x",t.x},{"y",t.y},{"radius",t.radius},{"age",t.age}});
        frames.push_back({{"t",e.time},{"score",e.score},{"agents",agents},{"predators",predators},{"fruits",fruits},{"trees",trees}});
    }
    void save(Engine& e,const fs::path& p,const std::string& reason){capture(e,J::array(),true);data["summary"]={{"duration",e.time},{"score",e.score},{"frames",data["frames"].size()},{"reason",reason}};if(fs::exists(p))throw std::runtime_error("Replay already exists");save_json(p,data);}
};
using Points=std::map<std::pair<int,int>,int>;
struct Pose{double x,y,h;};
std::map<int64_t,Pose> frame_poses(const J& body){
    std::map<int64_t,Pose> poses;std::map<int64_t,const J*> agents;std::vector<int64_t> todo;
    for(auto& a:body.at("agent_status")){int64_t id=a.at("agent_id");agents[id]=&a;
        for(auto& o:a.at("observations")){if(o["type"]!="Edge")continue;double ax=o["coords"][0][0],ay=o["coords"][0][1],bx=o["coords"][1][0],by=o["coords"][1][1];
            double length=std::hypot(bx-ax,by-ay);bool vertical=std::abs(length-1200)<1e-6,horizontal=std::abs(length-1600)<1e-6;if(!vertical&&!horizontal)continue;
            double h=(vertical?PI/2:0.)-std::atan2(by-ay,bx-ax),rx=std::cos(h)*ax-std::sin(h)*ay,ry=std::sin(h)*ax+std::cos(h)*ay;
            std::vector<Pose> candidates;
            for(double wall:vertical?std::vector<double>{30.,1570.}:std::vector<double>{30.,1170.}){
                double x=vertical?wall-rx:-rx,y=vertical?-ry:wall-ry;
                if(30+1e-6<x&&x<1570-1e-6&&30+1e-6<y&&y<1170-1e-6)candidates.push_back({x,y,h});
            }
            if(candidates.size()==1){poses[id]=candidates[0];todo.push_back(id);break;}}
    }
    for(size_t i=0;i<todo.size();i++){auto p=poses[todo[i]];for(auto& o:agents[todo[i]]->at("observations")){
        if(o["type"]!="Agent"||!o.contains("id"))continue;int64_t id=o["id"];if(!agents.count(id)||poses.count(id))continue;
        double angle=p.h+o["angle"].get<double>(),d=o["distance"];poses[id]={p.x+d*std::cos(angle),p.y+d*std::sin(angle),angle+PI-o["rel_dir"].get<double>()};todo.push_back(id);}}
    return poses;
}
void add_samples(Points& points,const J& body){auto poses=frame_poses(body);for(auto& a:body.at("agent_status")){
    int64_t id=a["agent_id"];int label=biome_index(a["biome"]);if(label<0||label>=4||!poses.count(id))continue;auto p=poses[id];
    if(!(30<p.x&&p.x<1570&&30<p.y&&p.y<1170)||std::min(std::abs(p.x-std::round(p.x)),std::abs(p.y-std::round(p.y)))<1e-7)continue;
    auto key=std::make_pair(int(p.x),int(p.y));if(points.count(key)&&points[key]!=label)throw std::runtime_error("Contradictory public terrain samples");points[key]=label;}}
J samples_json(const Points& points){std::map<int,std::vector<J>> groups;for(auto [p,label]:points)groups[label].push_back({p.first,p.second,label});J out=J::array();bool added=true;while(added){added=false;for(auto& [label,v]:groups)if(!v.empty()){out.push_back(v.back());v.pop_back();added=true;}}return out;}
bool terrain_matches(uint32_t seed,const Points& points){PyRandom rng;rng.init_by_array({seed});int x[10],y[10],labels[10];for(int i=0;i<10;i++){x[i]=rng.randbelow(1600);y[i]=rng.randbelow(1200);}for(auto& label:labels)label=rng.randbelow(4);for(auto [p,label]:points){int best=0,d=INT_MAX;for(int i=0;i<10;i++){int dx=x[i]-p.first,dy=y[i]-p.second,dd=dx*dx+dy*dy;if(dd<d){d=dd;best=i;}}if(labels[best]!=label)return false;}return true;}
