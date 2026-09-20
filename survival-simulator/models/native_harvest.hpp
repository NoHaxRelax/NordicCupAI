#pragma once
#include "harvest_memory.hpp"
#include <optional>

namespace native_harvest {
struct Config { double horizon=3000, engage=45; int min_free=6,max_ticks=2; int64_t cap=250000,extra=0; };
inline int64_t minimum_drain(double energy,double remaining,double meals=0) {
    return 2*std::max<int64_t>(0,(int64_t)std::ceil(energy+std::max(200.000001,100+30*(std::max(0.,remaining)+.1))+meals));
}
inline double after(const AState& s,const std::vector<Act>& actions) {
    double e=s.energy;
    for(auto& a:actions)if(a.aid==s.aid){double d=std::max(0.,std::min(a.dist,s.sprint));if(e<s.max_energy/5)d=std::min(d,s.speed);
        e-=std::min(d,s.speed)*.05+std::max(0.,d-s.speed)*.5+std::min(pi,std::abs(a.turn))/(2*pi);
        if(a.spawn&&e>100)e-=100;
    }return e;
}
inline bool will_visit(const std::map<int64_t,double>& energies,int64_t target) {
    bool skip=false;for(auto [id,e]:energies){if(id==target)return !skip;if(skip)skip=false;else if(e<=.1)skip=true;}return false;
}
inline void drain(std::vector<Act>& out,int64_t id,int64_t n) {
    for(int64_t i=0;i<n;i+=2){out.push_back({id,0,0,pi,false});out.push_back({id,0,0,-pi,false});}
}
struct Attempt {int64_t farm,sacrifice,key;std::set<int64_t> witnesses;double min_gain;int ticks=1;};
struct Controller {
    Config cfg;Memory memory;std::optional<Attempt> pending;
    double previous_score=0,next_commit=0;
    int64_t attempts=0,confirmed=0,failed=0,retries=0,sacrifices=0,cap_rejections=0,floor_rejections=0,skip_rejections=0,total_dupes=0;
    std::vector<AState> filtered;std::vector<std::vector<Obs>> filtered_obs;
    explicit Controller(Config c={}):cfg(c){}
    const std::vector<AState>& observe(const std::vector<AState>& states,double now,double score) {
        memory.update(states,now,pending?pending->farm:-1);
        bool present=false;if(pending)for(auto& s:states)if(s.aid==pending->farm)present=true;
        if(pending&&!present){if(score-previous_score>=pending->min_gain){++confirmed;memory.certify(pending->witnesses,cfg.horizon,now);}else ++failed;next_commit=now+.5;pending.reset();}
        previous_score=score;
        filtered=states;filtered_obs.resize(states.size());
        for(size_t i=0;i<states.size();++i){std::set<int> ignored;
            for(auto& t:memory.tracks[states[i].aid])if(Memory::sleeping(t,now))ignored.insert(t.obs_index);
            memory.ignored+=ignored.size();filtered_obs[i].clear();
            for(size_t j=0;j<states[i].obs->size();++j)if(!ignored.count((int)j))filtered_obs[i].push_back((*states[i].obs)[j]);
            filtered[i].obs=&filtered_obs[i];
        }return filtered;
    }
    std::vector<Act> actions(const std::vector<AState>& states,double now,const std::vector<Act>& native) {
        auto finish=[&](std::vector<Act> out){memory.record(states,out);return out;};
        std::map<int64_t,const AState*> by;std::map<int64_t,double> energies;
        for(auto& s:states){by[s.aid]=&s;energies[s.aid]=after(s,native);}
        struct Candidate {int64_t id;Track* t;};std::vector<Candidate> candidates;
        if(pending){if(pending->ticks>=cfg.max_ticks){++failed;pending.reset();return finish(native);}
            for(auto& t:memory.tracks[pending->farm])if(t.key==pending->key&&!Memory::sleeping(t,now))candidates.push_back({pending->farm,&t});
        }else{if(now<next_commit)return finish(native);
            for(auto [id,s]:by)if(s->energy>0)for(auto& t:memory.tracks[id]){
                bool sprint=false;for(double stride:{15.,12.,7.5,4.5})if(std::abs(std::abs(t.motion)-stride)<.02)sprint=true;
                if(t.moving&&sprint&&(std::conj(t.point)*t.motion).real()<0&&std::abs(t.obs.rel_dir)<.6&&t.obs.distance<=cfg.engage&&t.until<=now)candidates.push_back({id,&t});
            }
            std::stable_sort(candidates.begin(),candidates.end(),[&](auto a,auto b){double ea=by[a.id]->energy,eb=by[b.id]->energy;if(ea!=eb)return ea<eb;if(a.t->obs.distance!=b.t->obs.distance)return a.t->obs.distance<b.t->obs.distance;return a.id<b.id;});
        }
        for(auto candidate:candidates){auto it=by.find(candidate.id);if(it==by.begin()||it==by.end())continue;
            auto fid=candidate.id,did=std::prev(it)->first;auto& farm=*it->second;auto& target=*candidate.t;
            int survivors=0;for(auto [id,e]:energies)if(id!=fid&&id!=did&&e>.1)++survivors;
            if(survivors<cfg.min_free){++floor_rejections;continue;}
            if(!will_visit(energies,did)){++skip_rejections;continue;}
            Z point=pending?Z{}:target.point+2.*rot(target.heading)*std::abs(target.motion);
            double direction=std::arg(point),travel=std::abs(point)/biome_factor(farm.biome);
            std::vector<Act> moves;
            while(travel>1e-8&&moves.size()<20){double step=std::min({farm.speed,farm.sprint,travel});if(step<=0)break;moves.push_back({fid,step,direction,0,false});travel-=step;}
            if(travel>1e-8)continue;
            auto frames=memory.frames(states,fid);std::set<int64_t> exposed;double meals=0,global_meals=0;
            for(auto [id,s]:by){if(id==fid||id==did)continue;global_meals+=s->max_energy;double movement=0;
                for(auto& a:native)if(a.aid==id)movement+=std::max(0.,std::min(a.dist,s->sprint));
                if(!frames.count(id)||std::abs(frames[id].first-target.point)<45.+movement){exposed.insert(id);meals+=s->max_energy;}
            }
            for(auto& a:native)if(a.spawn){if(a.aid!=fid&&a.aid!=did)global_meals+=75;if(exposed.count(a.aid))meals+=75;}
            double e=after(farm,moves);int64_t count=pending?0:minimum_drain(e,cfg.horizon-now,meals)+cfg.extra;
            int64_t sacrifice=2*std::max<int64_t>(0,(int64_t)std::ceil(by[did]->energy));
            std::vector<Act> out;for(auto& a:native)if(a.aid!=fid&&a.aid!=did)out.push_back(a);
            if((int64_t)out.size()+sacrifice+(int64_t)moves.size()+count>cfg.cap){++cap_rejections;continue;}
            drain(out,did,sacrifice);out.insert(out.end(),moves.begin(),moves.end());drain(out,fid,count);
            if(pending){++pending->ticks;pending->sacrifice=did;++retries;}
            else{pending=Attempt{fid,did,target.key,memory.witnesses(states,fid,target),std::max(1.,-(e-count*.5)/100.-global_meals/100.-5.)};++attempts;}
            ++sacrifices;total_dupes+=sacrifice+count;return finish(std::move(out));
        }
        if(pending){++failed;pending.reset();}return finish(native);
    }
};
}
