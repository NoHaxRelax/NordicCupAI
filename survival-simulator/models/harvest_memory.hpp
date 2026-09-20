// Public-observation-only native predator memory. No simulator/Python includes.
#pragma once
#include "../fastsim/policy_abi.hpp"
#include <array>
#include <complex>
#include <map>
#include <set>

namespace native_harvest {
using namespace polabi;
using Z = std::complex<double>;
constexpr double pi = 3.14159265358979323846;
inline Z rot(double a) { return {pm::cos(a), pm::sin(a)}; }
inline Z polar(const Obs& o) { return rot(o.angle) * o.distance; }
inline double biome_factor(int b) { return b==RIVER?.3:b==SWAMP?.5:b==DESERT?.8:1.; }
struct Track {
    int64_t key=0;
    Z point{}, motion{};
    double heading=0, at=0, until=0, confirm_after=0;
    bool moving=false;
    int stationary=0, obs_index=-1;
    Obs obs{};
};
struct Ego { Z displacement{}; double turn=0; bool reliable=false; };
struct Memory {
    using Marks=std::vector<std::pair<int,Z>>;
    std::map<int64_t,std::vector<Track>> tracks;
    std::map<int64_t,Marks> previous;
    std::map<int64_t,Ego> ego;
    std::map<int64_t,double> ages;
    std::map<std::array<long long,4>,double> sites;
    int64_t next_key=0, ignored=0;
    static bool sleeping(const Track& t,double now) {
        return t.until>now && now>=t.confirm_after && t.stationary>=2 && !t.moving;
    }
    static Marks marks(const AState& s) {
        Marks out;
        for(const auto& o:*s.obs) if(o.type==0||o.type==3) out.emplace_back(o.type,polar(o));
        return out;
    }
    static std::vector<std::array<long long,4>> signatures(const AState& s,Z point) {
        std::vector<Z> m;
        for(const auto& o:*s.obs) {
            if(o.type==3) m.push_back(polar(o));
            if(o.type==4) {m.emplace_back(o.c[0],o.c[1]);m.emplace_back(o.c[2],o.c[3]);}
        }
        std::sort(m.begin(),m.end(),[](Z a,Z b){return a.real()!=b.real()?a.real()<b.real():a.imag()<b.imag();});
        m.erase(std::unique(m.begin(),m.end()),m.end());
        std::stable_sort(m.begin(),m.end(),[&](Z a,Z b){return std::abs(a-point)<std::abs(b-point);});
        if(m.size()>5) m.resize(5);
        std::vector<std::array<long long,4>> out;
        for(size_t i=0;i<m.size();++i) for(size_t j=i+1;j<m.size();++j) {
            if(std::abs(m[i]-m[j])<10) continue;
            Z x=m[i]-point,y=m[j]-point;
            if(std::abs(x)>std::abs(y)) std::swap(x,y);
            out.push_back({(long long)std::nearbyint(std::abs(x)*100),
                           (long long)std::nearbyint(std::abs(y)*100),
                           (long long)std::nearbyint(std::abs(x-y)*100),
                           (std::conj(x)*y).imag()>=0?1LL:-1LL});
        }
        return out;
    }
    void update(const std::vector<AState>& states,double now,int64_t preserve=-1) {
        std::set<int64_t> living;
        for(auto& s:states) living.insert(s.aid);
        for(auto it=tracks.begin();it!=tracks.end();) {
            if(!living.count(it->first)) {previous.erase(it->first);ego.erase(it->first);ages.erase(it->first);it=tracks.erase(it);}
            else ++it;
        }
        for(auto& s:states) {
            auto id=s.aid;
            bool stale=ages.count(id)&&ages[id]==s.age; ages[id]=s.age;
            if(stale&&id==preserve) continue;
            bool predators=false; for(auto& o:*s.obs) if(o.type==2) predators=true;
            if(stale||!predators) {tracks.erase(id);previous.erase(id);continue;}
            Ego e=ego[id]; Z r=rot(-e.turn);
            auto m=marks(s); std::vector<Z> offsets;
            for(auto& [kind,old]:previous[id]) {
                Z expected=(old-e.displacement)*r,match{};int count=0;
                for(auto& [k,p]:m) if(k==kind&&std::abs(p-expected)<12) {match=p;++count;}
                if(count==1) offsets.push_back(old-match/r);
            }
            std::vector<Z> best;
            for(Z q:offsets) {std::vector<Z> cluster;for(Z p:offsets) if(std::abs(p-q)<.05) cluster.push_back(p);if(cluster.size()>best.size()) best=cluster;}
            if(best.size()>=2) {e.displacement={};for(Z p:best)e.displacement+=p;e.displacement/=best.size();e.reliable=true;}
            previous[id]=m;
            auto old=tracks[id];
            for(auto& t:old) {t.point=(t.point-e.displacement)*r;t.heading-=e.turn;t.motion*=r;t.obs_index=-1;}
            std::vector<int> sightings;
            for(size_t i=0;i<s.obs->size();++i) if((*s.obs)[i].type==2) sightings.push_back((int)i);
            auto nearest=[&](int i){double d=std::numeric_limits<double>::infinity();for(auto& t:old)d=std::min(d,std::abs(t.point-polar((*s.obs)[i])));return d;};
            std::stable_sort(sightings.begin(),sightings.end(),[&](int a,int b){return nearest(a)<nearest(b);});
            std::set<int64_t> used;std::vector<Track> current;
            for(int index:sightings) {
                auto o=(*s.obs)[index]; Z p=polar(o);double heading=o.angle+pi-o.rel_dir;
                std::vector<std::pair<double,size_t>> candidates;
                for(size_t j=0;j<old.size();++j) if(!used.count(old[j].key)&&now-old[j].at<.21&&std::abs(old[j].point-p)<=16)
                    candidates.emplace_back(std::abs(old[j].point-p),j);
                std::sort(candidates.begin(),candidates.end(),[&](auto a,auto b){return a.first!=b.first?a.first<b.first:old[a.second].key<old[b.second].key;});
                Track t;
                if(!candidates.empty()&&(candidates.size()==1||candidates[1].first-candidates[0].first>3)) {
                    t=old[candidates[0].second]; Z delta=p-t.point;
                    t.moving=e.reliable&&std::abs(delta)>1&&std::abs(delta)<=15.1;
                    t.motion=t.moving?delta:Z{};
                    bool stationary=e.reliable&&std::abs(delta)<.1&&std::abs(std::remainder(heading-t.heading,2*pi))<.02;
                    t.stationary=stationary?t.stationary+1:0;
                    if(e.reliable&&!stationary&&now>t.confirm_after+1e-8)t.until=0;
                    used.insert(t.key);
                } else t.key=next_key++;
                t.point=p;t.heading=heading;t.at=now;t.obs=o;t.obs_index=index;current.push_back(t);
            }
            tracks[id]=std::move(current);
        }
        for(auto& s:states) for(auto& o:*s.obs) if(o.type==1&&living.count(o.id)) {
            Z origin=polar(o),r=rot(-(o.angle+pi-o.rel_dir));
            for(auto& t:tracks[s.aid]) if(sleeping(t,now)) {
                Track* match=nullptr;int count=0;
                for(auto& u:tracks[o.id]) if(std::abs((t.point-origin)*r-u.point)<.1) {match=&u;++count;}
                if(count==1&&!match->moving) {match->until=t.until;match->stationary=std::max(2,match->stationary);}
            }
        }
        for(auto& s:states) for(auto& t:tracks[s.aid]) if(t.stationary>=2&&!t.moving) {
            auto keys=signatures(s,t.point);
            if(sleeping(t,now)) for(auto k:keys)sites[k]=t.until;
            else {double until=0;for(auto k:keys){auto it=sites.find(k);if(it!=sites.end())until=std::max(until,it->second);}if(until>now){t.until=until;t.confirm_after=now;}}
        }
    }
    using Frames=std::map<int64_t,std::pair<Z,Z>>;
    Frames frames(const std::vector<AState>& states,int64_t farm) const {
        std::map<int64_t,const AState*> by;for(auto& s:states)by[s.aid]=&s;
        Frames out{{farm,{{},{1,0}}}};std::vector<int64_t> queue{farm};
        while(!queue.empty()) {auto id=queue.back();queue.pop_back();auto [origin,r]=out[id];
            for(auto& o:*by.at(id)->obs) if(o.type==1&&by.count(o.id)&&!out.count(o.id)) {
                out[o.id]={origin+polar(o)*r,r*rot(o.angle+pi-o.rel_dir)};queue.push_back(o.id);
            }
        }return out;
    }
    std::set<int64_t> witnesses(const std::vector<AState>& states,int64_t farm,const Track& target) {
        std::set<int64_t> out;
        for(auto& [id,f]:frames(states,farm)) {Z p=(target.point-f.first)/f.second;int n=0;int64_t key=0;
            for(auto& t:tracks[id])if(std::abs(t.point-p)<.1){++n;key=t.key;}if(n==1)out.insert(key);
        }return out;
    }
    void certify(const std::set<int64_t>& keys,double until,double now) {
        for(auto& [id,ts]:tracks)for(auto& t:ts)if(keys.count(t.key)){t.until=until;t.confirm_after=now+.1;t.stationary=0;}
    }
    void record(const std::vector<AState>& states,const std::vector<Act>& acts) {
        for(auto& s:states) if(!tracks[s.aid].empty()) {
            Ego e; e.reliable=true;double energy=s.energy;
            for(auto& a:acts) if(a.aid==s.aid) {
                double d=std::max(0.,std::min(a.dist,s.sprint));if(energy<s.max_energy/5)d=std::min(d,s.speed);
                energy-=std::min(d,s.speed)*.05+std::max(0.,d-s.speed)*.5;
                if(d){e.reliable=false;e.displacement+=rot(e.turn+a.direction)*d*biome_factor(s.biome);}
                e.turn+=a.turn;energy-=std::min(pi,std::abs(a.turn))/(2*pi);
                if(a.spawn&&energy>100)energy-=100;
            }ego[s.aid]=e;
        }
    }
};
}
