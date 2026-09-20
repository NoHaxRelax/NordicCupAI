#pragma once

// Observation-only three-tick predator-guide steering.  This header is
// intentionally independent of _npolicy.hpp: callers translate their P2/Plan
// and DTO observations into Input.  Delivery holds and target association stay
// with the caller.  Coordinates and headings are guide-local at the current
// tick; +x is forward and angles use the policy's ordinary math convention.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace entrapment_guide {

constexpr double PI = 3.14159265358979323846;
constexpr double TAU = 2.0 * PI;
constexpr int HORIZON = 3;
constexpr int BEAM = 4;
constexpr double CAPTURE_RADIUS = 15.0;

struct P2 { double x = 0., y = 0.; };
struct Segment { P2 a, b; };
struct Plan { double move_distance = 0., move_direction = 0., turn_angle = 0.; };
struct PredatorObservation {
    double distance = 0., angle = 0., relative_heading = 0.;
    bool has_relative_heading = false;
};
struct TerrainSample { P2 point; double progress = 1., uncertainty = 0.; };
struct Agent {
    double energy = 0., max_energy = 500., age = 0.;
    double speed = 10., sprint_speed = 20., terrain_progress = 1.;
};
struct PreviousObservation {
    bool valid = false;
    int tick = 0;
    P2 target_in_current_frame; // caller transforms its fixed-frame sample
};
struct Input {
    Plan nominal;
    Agent agent;
    P2 bait;
    std::vector<Segment> walls;
    std::vector<P2> route_waypoints;
    std::vector<PredatorObservation> predators;
    std::size_t target_index = 0;
    std::vector<TerrainSample> terrain_samples;
    PreviousObservation previous;
    int tick = 0;
    bool observation_is_lagged = true;
    bool reacquiring = false;
    double preferred_min = 100., preferred_max = 120.;
    double guide_clearance = 5.01, predator_clearance = 10.;
};
struct Forecast {
    bool predicted_safe = false;
    bool velocity_estimated = false;
    bool lag_guard_triggered = false;
    int capture_scenarios = 0, sampled_scenarios = 0, expanded = 0;
    double minimum_separation = std::numeric_limits<double>::infinity();
    std::array<P2,HORIZON+1> guide_path{};
    std::array<P2,HORIZON+1> target_path{};
    int path_size = 0;
};
struct Result { bool found = false; Plan plan; Forecast forecast; };

inline double wrap(double a) { return std::atan2(std::sin(a),std::cos(a)); }
inline P2 add(P2 a,P2 b) { return {a.x+b.x,a.y+b.y}; }
inline P2 sub(P2 a,P2 b) { return {a.x-b.x,a.y-b.y}; }
inline P2 mul(P2 a,double s) { return {a.x*s,a.y*s}; }
inline double dot(P2 a,P2 b) { return a.x*b.x+a.y*b.y; }
inline double dist(P2 a,P2 b) { return std::hypot(a.x-b.x,a.y-b.y); }
inline P2 polar(double r,double a) { return {r*std::cos(a),r*std::sin(a)}; }

inline double point_segment(P2 p,const Segment& s) {
    P2 d=sub(s.b,s.a); double n=dot(d,d);
    double t=n>1e-12?std::clamp(dot(sub(p,s.a),d)/n,0.,1.):0.;
    return dist(p,add(s.a,mul(d,t)));
}
inline double cross(P2 a,P2 b,P2 c) {
    return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
}
inline bool on_segment(P2 a,P2 b,P2 p) {
    constexpr double eps=1e-9;
    return std::fabs(cross(a,b,p))<=eps &&
           p.x>=std::min(a.x,b.x)-eps && p.x<=std::max(a.x,b.x)+eps &&
           p.y>=std::min(a.y,b.y)-eps && p.y<=std::max(a.y,b.y)+eps;
}
inline bool intersects(P2 a,P2 b,const Segment& s) {
    double x1=cross(a,b,s.a),x2=cross(a,b,s.b),x3=cross(s.a,s.b,a),x4=cross(s.a,s.b,b);
    bool proper=((x1>1e-9&&x2<-1e-9)||(x1<-1e-9&&x2>1e-9)) &&
                ((x3>1e-9&&x4<-1e-9)||(x3<-1e-9&&x4>1e-9));
    return proper || on_segment(a,b,s.a) || on_segment(a,b,s.b) ||
           on_segment(s.a,s.b,a) || on_segment(s.a,s.b,b);
}
inline double segment_distance(P2 a,P2 b,const Segment& s) {
    if(intersects(a,b,s)) return 0.;
    Segment q{a,b};
    return std::min({point_segment(a,s),point_segment(b,s),
                     point_segment(s.a,q),point_segment(s.b,q)});
}
inline bool clear_segment(const Input& in,P2 a,P2 b,double clearance) {
    for(const auto& w:in.walls) {
        if(intersects(a,b,w)) return false;
        double required=std::min(clearance,point_segment(a,w));
        if(segment_distance(a,b,w)+1e-6<required) return false;
    }
    return true;
}
inline bool free_point(const Input& in,P2 p,double clearance) {
    for(const auto& w:in.walls) if(point_segment(p,w)<clearance) return false;
    return true;
}
inline bool visible(const Input& in,P2 p,double heading,P2 target) {
    double d=dist(p,target); if(d<=60.) return true;
    if(d>250.||std::fabs(wrap(std::atan2(target.y-p.y,target.x-p.x)-heading))>PI/6.) return false;
    for(const auto& w:in.walls) if(intersects(p,target,w)) return false;
    return true;
}

struct PredState { P2 p; double heading=0.; };
inline PredState predator_step(const Input& in,PredState s,P2 guide,double guide_heading,
                               double cap=15.,double bias=0.,double terrain=1.) {
    if(cap==0.) return s;
    P2 target=guide; double target_heading=guide_heading;
    if(dist(s.p,in.bait)<dist(s.p,guide)&&visible(in,s.p,s.heading,in.bait)) {
        target=in.bait; target_heading=0.;
    }
    double d=dist(s.p,target), angle=wrap(std::atan2(target.y-s.p.y,target.x-s.p.x)-s.heading);
    double requested,direction,turn;
    if(visible(in,s.p,s.heading,target)) {
        double looking=wrap(std::atan2(s.p.y-target.y,s.p.x-target.x)-target_heading);
        if(std::fabs(looking)>PI/2.||d<90.) {
            turn=std::fabs(angle)>.05?std::clamp(angle*.5,-.3,.3):0.;
            direction=std::fabs(angle)>.05?turn:angle; requested=std::min(15.,d);
        } else {
            double sign=std::fabs(looking)<1e-7?bias:(looking>0.?-1.:1.);
            direction=angle+sign*PI/4.;
            turn=std::atan2(d*std::sin(angle)-15.*std::sin(direction),
                            d*std::cos(angle)-15.*std::cos(direction)); requested=15.;
        }
    } else { requested=11.;direction=0.;turn=0.; }
    double length=std::min(requested,cap)*terrain, absolute=s.heading+direction;
    P2 result=s.p;
    for(int i=0;i<36;i++) {
        double adjusted=absolute+PI/18.*((i+1)/2)*(i%2?-1.:1.);
        P2 q=add(s.p,polar(length,adjusted));
        if(free_point(in,q,in.predator_clearance)){result=q;break;}
    }
    return {result,wrap(s.heading+turn)};
}

inline double terrain_at(const Input& in,P2 q) {
    double best=dist(q,{0.,0.}), factor=in.agent.terrain_progress, uncertainty=0.;
    for(const auto& s:in.terrain_samples) if(dist(q,s.point)<best) {
        best=dist(q,s.point);factor=s.progress;uncertainty=s.uncertainty;
    }
    return best+uncertainty<=12.?factor:in.agent.terrain_progress;
}

struct Node {
    P2 point{}; double heading=0.,energy=0.;
    std::vector<PredState> predators;
    Plan first{}; bool has_first=false;
    std::array<P2,HORIZON+1> path{};
    std::vector<std::array<P2,HORIZON+1>> predator_paths;
    std::array<std::pair<double,double>,HORIZON> commands{};
    int depth=0,survival=HORIZON+1; double separation=std::numeric_limits<double>::infinity(),spent=0.;
};
struct Rank {
    int a=0; double b=0.; int c=0; double d=0.,e=0.;
    bool operator<(const Rank& o)const{return std::tie(a,b,c,d,e)<std::tie(o.a,o.b,o.c,o.d,o.e);}
};

inline Result search_impl(const Input& in,bool project_lag) {
    Result out; if(in.predators.empty()||in.target_index>=in.predators.size()) return out;
    std::vector<P2> positions; std::vector<PredState> initial;
    for(const auto& o:in.predators) {
        P2 p=polar(o.distance,o.angle); positions.push_back(p);
        initial.push_back({p,wrap(std::atan2(-p.y,-p.x)-(o.has_relative_heading?o.relative_heading:0.))});
    }
    const std::vector<PredState> observed=initial;
    bool velocity=false;
    if(!project_lag&&in.previous.valid&&in.tick-in.previous.tick==1) {
        P2 v=sub(positions[in.target_index],in.previous.target_in_current_frame);
        P2 q=add(positions[in.target_index],v);
        if(std::hypot(v.x,v.y)<=16.&&free_point(in,q,in.predator_clearance)) {
            initial[in.target_index].p=q;velocity=true;
        }
    }
    auto lag_start=[&](double cap,double terrain,double bias) {
        auto p=observed;
        if(in.observation_is_lagged)
            for(auto& s:p)s=predator_step(in,s,{0.,0.},0.,cap,bias,terrain);
        return p;
    };
    if(project_lag) initial=lag_start(15.,1.,0.);
    std::vector<P2> waypoints=in.route_waypoints;
    if(waypoints.empty()) waypoints.push_back(polar(HORIZON*in.nominal.move_distance*
                                                    in.agent.terrain_progress,in.nominal.move_direction));
    double pref_min=in.reacquiring?0.:in.preferred_min,pref_max=in.reacquiring?55.:in.preferred_max;
    Node root;root.energy=in.agent.energy;root.predators=initial;root.path[0]={0.,0.};
    root.predator_paths.resize(initial.size());
    for(std::size_t i=0;i<initial.size();i++)root.predator_paths[i][0]=initial[i].p;
    std::vector<Node> nodes{root};int expanded=0;
    for(int depth=0;depth<HORIZON;depth++) {
        std::vector<std::pair<Rank,Node>> choices;
        for(const Node& n:nodes) {
            P2 g=n.point,p=n.predators[in.target_index].p;double mod=terrain_at(in,g);
            P2 goal=waypoints.back();for(P2 w:waypoints)if(dist(w,g)>3.){goal=w;break;}
            double bearing=std::atan2(goal.y-g.y,goal.x-g.x),away=std::atan2(g.y-p.y,g.x-p.x);
            std::vector<double> angles{bearing,bearing-PI/4.,bearing+PI/4.,away,away-PI/2.,away+PI/2.};
            for(int i=0;i<12;i++)angles.push_back(i*TAU/12.);
            std::vector<double> unique;for(double a:angles){a=wrap(a);bool seen=false;for(double b:unique)if(std::fabs(wrap(a-b))<1e-7)seen=true;if(!seen)unique.push_back(a);}
            double cap=n.energy>=in.agent.max_energy/5.?in.agent.sprint_speed:std::min(in.agent.speed,in.agent.sprint_speed);
            std::vector<double> lengths{std::min(in.agent.speed,cap)};if(cap!=lengths[0])lengths.push_back(cap);
            double reserve=in.agent.max_energy/5.+3.*(.05*in.agent.speed+.5*std::max(0.,cap-in.agent.speed)+1.);
            if(n.energy<reserve)lengths.push_back((std::min(in.agent.speed,cap)+cap)/2.);
            std::sort(lengths.begin(),lengths.end());
            lengths.erase(std::unique(lengths.begin(),lengths.end()),lengths.end());
            std::vector<std::pair<double,double>> commands{{0.,bearing}};
            for(double len:lengths)for(double a:unique)commands.push_back({std::fabs(wrap(a-bearing))<1e-7?std::min(len,dist(goal,g)/mod):len,a});
            for(auto [len,angle]:commands) {
                P2 q=add(g,polar(len*mod,angle));if(len&& !clear_segment(in,g,q,in.guide_clearance))continue;
                double heading=std::atan2(p.y-q.y,p.x-q.x),turning=std::fabs(wrap(heading-n.heading));
                double cost=.05*std::min(len,in.agent.speed)+.5*std::max(0.,len-in.agent.speed)+turning/TAU+.15;
                double age=in.agent.age+.1*depth;if(age>=60.)cost+=.01*age;if(cost>=n.energy)continue;
                Node c=n;c.point=q;c.heading=heading;c.energy-=cost;c.depth=depth+1;c.commands[depth]={len,angle};c.path[depth+1]=q;c.spent+=cost;
                double sep=std::numeric_limits<double>::infinity();
                for(std::size_t i=0;i<c.predators.size();i++) {c.predators[i]=predator_step(in,c.predators[i],q,heading);c.predator_paths[i][depth+1]=c.predators[i].p;sep=std::min(sep,dist(q,c.predators[i].p));}
                c.separation=std::min(c.separation,sep);if(sep<CAPTURE_RADIUS)c.survival=std::min(c.survival,depth);
                bool contact=visible(in,c.predators[in.target_index].p,c.predators[in.target_index].heading,q);
                double d=dist(q,c.predators[in.target_index].p),spacing=std::max({0.,pref_min-d,d-pref_max});
                double route=dist(q,goal)+(dist(q,in.bait)<=80.?0.:1.5)*spacing;
                if(!c.has_first){c.first={len,angle,heading};c.has_first=true;}
                Rank rank{-c.survival,c.survival<=depth?-c.separation:0.,contact?0:1,route,c.spent};
                choices.push_back({rank,std::move(c)});expanded++;
            }
        }
        if(choices.empty()) return out;
        std::sort(choices.begin(),choices.end(),[](auto&a,auto&b){return a.first<b.first;});
        nodes.clear();std::set<std::pair<int,int>> seen;
        for(auto& row:choices){auto key=std::make_pair((int)std::lround(row.second.first.move_distance*100.),(int)std::lround(row.second.first.move_direction*1000.));if(seen.insert(key).second)nodes.push_back(std::move(row.second));if((int)nodes.size()>=BEAM)break;}
    }
    struct Checked{std::tuple<int,int,int,int> rank;Node* node;int captures,scenarios;double minimum;};
    std::vector<Checked> checked;int order=0;
    const std::array<double,2> caps{15.,11.};
    for(auto& n:nodes){int captures=0,scenarios=0,worst=HORIZON+1;double minimum=std::numeric_limits<double>::infinity();
        std::vector<std::pair<double,double>> motions;for(double c:caps)for(double t:{1.,.8,.5,.3})motions.push_back({c,t});if(project_lag)motions.push_back({0.,1.});
        for(auto [cap,terrain]:motions)for(double bias:{-1.,0.,1.})for(int slow_at:{HORIZON,1,2}){
            auto preds=project_lag?lag_start(cap,terrain,bias):initial;P2 q{};bool caught=false;
            for(int d=0;d<HORIZON;d++){auto [len,angle]=n.commands[d];double factor=d>=slow_at?.3:terrain_at(in,q);P2 nq=add(q,polar(len*factor,angle));if(clear_segment(in,q,nq,in.guide_clearance))q=nq;double h=std::atan2(preds[in.target_index].p.y-q.y,preds[in.target_index].p.x-q.x);for(auto& p:preds)p=predator_step(in,p,q,h,cap,bias,terrain);double sep=std::numeric_limits<double>::infinity();for(auto&p:preds)sep=std::min(sep,dist(q,p.p));minimum=std::min(minimum,sep);if(sep<CAPTURE_RADIUS){caught=true;worst=std::min(worst,d);break;}}
            captures+=caught;scenarios++;}
        checked.push_back({{captures>0?1:0,-worst,captures,order++},&n,captures,scenarios,minimum});}
    auto best=std::min_element(checked.begin(),checked.end(),[](auto&a,auto&b){return a.rank<b.rank;});
    Node& win=*best->node;
    if(!project_lag){P2 q=polar(win.first.move_distance*in.agent.terrain_progress,win.first.move_direction);bool danger=false;
        std::vector<std::pair<double,double>> motions;for(double c:{15.,11.})for(double t:{1.,.8,.5,.3})motions.push_back({c,t});motions.push_back({0.,1.});
        for(auto [cap,t]:motions)for(double bias:{-1.,0.,1.}){auto ps=lag_start(cap,t,bias);for(auto p:ps)if(dist(predator_step(in,p,q,win.first.turn_angle,cap,bias,t).p,q)<CAPTURE_RADIUS)danger=true;}
        if(danger)return search_impl(in,true);}
    out.found=true;out.plan=win.first;out.forecast.predicted_safe=best->captures==0;out.forecast.velocity_estimated=velocity;
    out.forecast.lag_guard_triggered=project_lag;out.forecast.capture_scenarios=best->captures;out.forecast.sampled_scenarios=best->scenarios;out.forecast.minimum_separation=best->minimum;out.forecast.expanded=expanded;out.forecast.path_size=HORIZON+1;out.forecast.guide_path=win.path;out.forecast.target_path=win.predator_paths[in.target_index];return out;
}

inline Result search(const Input& input) { return search_impl(input,false); }

} // namespace entrapment_guide
