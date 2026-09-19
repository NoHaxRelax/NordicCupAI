// Native port of models/orchard_evasion_policy.py: the thin own-sighting predator
// evasion layer on top of the vendored orchard foraging/population core.
//
// Decision-identical to the Python subclass by construction: it overrides the same
// single method (act / _act), takes the same two branches in the same order, and
// returns instead of delegating, so none of the base act()'s side effects (blocked
// pruning, progress() odometry updates, last_site) run on a fleeing or facing tick.
//
// Structure follows survival-simulator/oscar-overnight-cpp nightsim/_npolicy.hpp's
// evade(), but only loosely: that port weights several threats by inverse distance,
// picks its dodge side from the predator's reported rel_dir, keeps a committed dodge
// heading, applies itself as a post-hoc patch over an already-computed plan, and has
// no fruit release. Ours uses the single nearest own-sighting, picks the dodge side
// from the bearing sign, replaces the plan outright, and releases the claimed fruit.
// Only the parameter names are shared. Where the two disagree, the Python file wins.

namespace orchard {

struct PredParams {
    int64_t mode = 1;
    double r = 70., face_r = 80., sprint_r = 40., dodge_r = 80., dodge_ang = 1.4, turn_max = 1.0;
    double gaze = 0., cone_gate = 0., cone_margin = 0.1;
    double pulse_degrees=0., pulse_ticks=8., pulse_idle=0.;
    double feature_start=0., feature_population=0.;
    double look_steps=0., look_radius=130., risk_margin=3., behind_weight=0., energy_weight=.3;
    double phase_start=900., phase_population=0., late_cap=-1., late_retire=-1., late_reach=-1., late_reserve=-1.;
    double wall_escape = 0., wall_look = 30., wall_reward = 80.;
    double trap_bias=0., trap_protect=0., trap_reach=180., trap_active_r=130.;
};

class EvasionPolicy : public Policy {
public:
    PredParams PRED;
    struct Patch { P2 p; double first, last; };
    std::unordered_map<int64_t,std::vector<Patch>> patches;

    // Spatial occupancy, not predator identity: public sightings have no stable ID.
    // Confirmation requires repeated sightings for 60 s with no gap over 2 s.
    Plan edge_migration(Mind& m,const AState& s,Plan base) {
        if(PRED.trap_bias==0 && PRED.trap_protect==0) return base;
        auto& ps=patches[m.group];
        ps.erase(std::remove_if(ps.begin(),ps.end(),[&](const Patch& p){return time-p.last>120.;}),ps.end());
        for(const Obs& o:*s.obs) if(o.type==2) {
            P2 p=polar(*m.pose,o);double wall=OINF;
            for(const auto& e:m.edges)wall=pmin(wall,point_segment(p,e.a,e.b));
            if(wall>35.)continue;
            Patch* found=nullptr;
            for(auto& q:ps)if(dist(q.p,p)<20.){found=&q;break;}
            if(found){if(time-found->last>2.)found->first=time;found->last=time;}
            else if(ps.size()<128)ps.push_back({p,time,time});
        }
        const Obs* t=threat(s);P2 pos=m.pose->p,pred{},target{};
        bool steering=false;double ph=0.;
        if(t && t->has_rel_dir && t->distance<=PRED.trap_active_r && PRED.trap_bias>0) {
            pred=polar(*m.pose,*t);
            ph=wrap(m.pose->theta+t->angle+OPI-t->rel_dir);
            double best=PRED.trap_reach;
            // Two nearby surfaces characterize the inspected stalls. Offset towards
            // the observed free side; never invent unseen rock/boundary geometry.
            for(const auto& e:m.edges) {
                P2 v=sub(e.b,e.a);double l2=v.x*v.x+v.y*v.y;if(l2<1.)continue;
                P2 d=sub(pred,e.a);double u=pmax(0.,pmin(1.,(d.x*v.x+d.y*v.y)/l2));
                P2 foot=add(e.a,mul(v,u));double gap=dist(pred,foot);
                if(gap<12. || gap>best)continue;
                P2 q=add(foot,mul(sub(pred,foot),13./gap));
                bool blocked=false,second=false;
                for(const auto& other:m.edges) {
                    double separation=point_segment(q,other.a,other.b);
                    if(segments_cross(pred,q,other.a,other.b)||separation<10.){blocked=true;break;}
                    // Duplicate/collinear observations of one wall don't count.
                    P2 w=sub(other.b,other.a);
                    double cross=v.x*w.y-v.y*w.x;
                    double offset=std::abs(v.x*(other.a.y-e.a.y)-v.y*(other.a.x-e.a.x))/std::sqrt(l2);
                    if(separation<35. && (offset>3. || cross*cross>.01*l2*(w.x*w.x+w.y*w.y)))second=true;
                }
                if(!blocked && second){best=gap;target=q;steering=true;}
            }
        }
        auto danger=[&](P2 q) {
            double risk=0.;
            if(!PRED.trap_protect)return risk;
            for(const auto& patch:ps) if(patch.last-patch.first>=60.) {
                double d=dist(q,patch.p);risk+=pmax(0.,85.-d);
                // Fresh headings permit cone avoidance; stale patches use only a
                // small exclusion disc instead of pretending to know the heading.
                for(const Obs& o:*s.obs)if(o.type==2 && o.has_rel_dir && dist(polar(*m.pose,o),patch.p)<20.) {
                    P2 p=polar(*m.pose,o);double h=m.pose->theta+o.angle+OPI-o.rel_dir;
                    if(dist(q,p)<260. && std::abs(wrap(pm::atan2(q.y-p.y,q.x-p.x)-h))<OPI/6.+.1)
                        risk+=80.;
                }
            }
            return risk;
        };
        double terrain=s.biome==RIVER?.3:s.biome==SWAMP?.5:s.biome==DESERT?.8:1.;
        double length=pmin(base.dist,s.energy<s.max_energy/5.?s.speed:s.sprint)*terrain;
        if(length<=0.)return base;
        P2 original=add(pos,mul(unit(m.pose->theta+base.direction),length));
        if(!steering && danger(original)==0.)return base;
        auto score=[&](double a) {
            P2 q=add(pos,mul(unit(m.pose->theta+a),length));
            for(const auto& e:m.edges)if(segments_cross(pos,q,e.a,e.b)||point_segment(q,e.a,e.b)<5.)return -1e12;
            double value=length*pm::cos(wrap(a-base.direction))*.25-danger(q)*PRED.trap_protect;
            for(const Obs& o:*s.obs)if(o.type==2) {
                P2 p=polar(*m.pose,o);double d=dist(q,p);
                // One unseen DTO move plus the next chase move: conservative
                // safety preference, not a collision guarantee or full rollout.
                value-=pmax(0.,45.-d)*100.;
                value+=pmin(d,80.)*.25;
            }
            if(steering) {
                double bearing=pm::atan2(q.y-pred.y,q.x-pred.x);
                double h=ph+pmax(-.3,pmin(.3,wrap(bearing-ph)*.5));
                double want=pm::atan2(target.y-pred.y,target.x-pred.x);
                value+=PRED.trap_bias*pm::cos(wrap(h-want));
            }
            return value;
        };
        double best=score(base.direction);Plan selected=base;
        for(int k=0;k<24;k++) {
            double a=-OPI+k*TAU/24.,v=score(a);
            if(v>best){best=v;selected.direction=a;selected.turn=pmax(-PRED.turn_max,pmin(PRED.turn_max,a));}
        }
        if(selected.direction!=base.direction)release_fruit(m);
        return selected;
    }
    // Reporting only; never read by a decision (mirrors self.metrics in Python).
    int64_t flee_ticks = 0, face_ticks = 0, sprint_ticks = 0;

    EvasionPolicy(const std::vector<uint32_t>& seed_key, const Params& p, const PredParams& pp)
        : Policy(seed_key, p), PRED(pp) {}

    // Nearest predator in this agent's own observations, or nullptr.
    // Strict <, so the first of equally distant sightings wins, exactly as the
    // Python loop's `o['distance'] < nearest['distance']` does.
    const Obs* threat(const AState& s) const {
        const Obs* nearest = nullptr;
        for (const Obs& o : *s.obs) {
            if (o.type != 2) continue;  // 2 == 'Predator'
            if (!nearest || o.distance < nearest->distance) nearest = &o;
        }
        return nearest;
    }

    // Free a claimed fruit before fleeing/facing, or it sits reserved and unreachable
    // by hungrier group-mates for as long as the threat lasts.
    void release_fruit(Mind& m) {
        if (!m.has_fruit) return;
        Group& g = G(m.group);
        if (g.fruits.has(m.fruit)) g.fruits.at(m.fruit)->has_claim = false;
        m.has_fruit = false;
    }

    Plan legacy_act(Mind& m, const AState& s) {
        if (PRED.mode) {
            const Obs* t = threat(s);
            if (t) {
                const double distance = t->distance, angle = t->angle;
                const double tm = PRED.turn_max;
                // Public relative heading only. Hearing remains omnidirectional.
                if (PRED.cone_gate && distance > 60. && t->has_rel_dir &&
                    std::abs(wrap(angle + OPI - t->rel_dir)) > OPI / 6 + PRED.cone_margin)
                    return Policy::act(m, s);
                if (distance <= PRED.r) {
                    double away = wrap(angle + OPI);
                    // A predator outruns an agent in a straight line, so break across
                    // its approach instead of directly away once it is this close.
                    if (distance <= PRED.dodge_r)
                        away = wrap(away + (angle >= 0. ? PRED.dodge_ang : -PRED.dodge_ang));
                    bool sprinting = distance <= PRED.sprint_r;
                    double reach = sprinting ? s.sprint : pmin(s.speed, s.sprint);
                    // Search short reachable headings around OBSERVED edges. Reward
                    // breaking line of sight only beyond the predator's hearing
                    // radius. This is a geometric heuristic, not privileged rollout.
                    if (PRED.wall_escape && !m.edges.empty()) {
                        P2 pos = m.pose->p, pred = polar(*m.pose, *t);
                        double best = -OINF, selected = away;
                        for (int k = 0; k < 24; ++k) {
                            double h = m.pose->theta + away + k * OPI / 12.;
                            P2 q = add(pos, mul(unit(h), PRED.wall_look));
                            bool blocked = false, hidden = false;
                            for (const auto& e : m.edges) {
                                if (time - e.t > 25.) continue;
                                if (segments_cross(pos, q, e.a, e.b) || point_segment(q, e.a, e.b) < 7.) blocked = true;
                                if (segments_cross(pred, q, e.a, e.b)) hidden = true;
                            }
                            if (blocked) continue;
                            double d = dist(q, pred);
                            double value = d - 0.3 * std::abs(wrap(h - m.pose->theta - away));
                            if (distance > 60. && d > 65. && hidden) value += PRED.wall_reward;
                            if (value > best) { best = value; selected = wrap(h - m.pose->theta); }
                        }
                        away = selected;
                    }
                    flee_ticks++;
                    sprint_ticks += sprinting ? 1 : 0;
                    release_fruit(m);
                    double gaze = PRED.gaze ? angle : away;
                    return Plan{reach, away, pmax(-tm, pmin(tm, gaze))};
                }
                if (distance <= PRED.face_r) {
                    face_ticks++;
                    release_fruit(m);
                    return Plan{0., 0., pmax(-tm, pmin(tm, angle))};
                }
            }
        }
        return Policy::act(m, s);
    }
    // Observation-only short-horizon model. Unknown predator energy/terrain are
    // conservatively bounded by a 15-unit direct pursuit step; this is not the engine.
    Plan act(Mind& m, const AState& input) override {
        AState s=input;std::vector<Obs> shared;
        if(P.share_obs) {
            shared=*input.obs;Group& g=G(m.group);
            for(const auto& p:g.predators) {
                bool seen=false;
                for(const auto& o:shared)if(o.type==2&&dist_lt(polar(*m.pose,o),p.p,5.)){seen=true;break;}
                if(seen)continue;
                Obs o{};o.type=2;local_of(*m.pose,p.p,o.distance,o.angle);
                o.has_rel_dir=p.has_heading;
                o.rel_dir=wrap(pm::atan2(m.pose->p.y-p.p.y,m.pose->p.x-p.p.x)-p.heading);
                shared.push_back(o);
            }
            s.obs=&shared;
        }
        Plan base=edge_migration(m,s,legacy_act(m,s));
        bool active=time>=PRED.feature_start && (PRED.feature_population<=0 || states.size()<=PRED.feature_population);
        if (!active) return base;
        const Obs* t=threat(s);
        if (!t && PRED.pulse_degrees!=0 && (!PRED.pulse_idle || base.dist==0)) {
            int64_t period=std::max<int64_t>(1,(int64_t)PRED.pulse_ticks);
            int64_t tick=(int64_t)std::llround(time*10.);
            base.turn=(tick % period==0)?PRED.pulse_degrees*OPI/180.*m.sweep_sign:0.;
        }
        if (!t || t->distance>PRED.look_radius || (PRED.look_steps<=0 && PRED.behind_weight<=0)) return base;
        int horizon=std::max(1,std::min(2,(int)PRED.look_steps));
        double penalty=s.biome==RIVER?.3:s.biome==SWAMP?.5:s.biome==DESERT?.8:1.;
        auto evaluate=[&](const Plan& plan) {
            P2 q={0.,0.}; double clearance=OINF, behind=0.;
            std::vector<P2> threats;
            for(const Obs& o:*s.obs) if(o.type==2) threats.push_back(mul(unit(o.angle),o.distance));
            double distance=pmin(plan.dist,s.sprint);
            if(s.energy<s.max_energy/5.) distance=pmin(distance,s.speed);
            P2 step=mul(unit(plan.direction),distance*penalty);
            for(int k=0;k<horizon;k++) {
                P2 prev=q;q=add(q,step);
                P2 wp=add(m.pose->p,mul(unit(m.pose->theta+pm::atan2(prev.y,prev.x)),dist(prev,P2{0.,0.})));
                P2 wq=add(m.pose->p,mul(unit(m.pose->theta+pm::atan2(q.y,q.x)),dist(q,P2{0.,0.})));
                for(const auto& e:m.edges) if(time-e.t<25. &&
                  (segments_cross(wp,wq,e.a,e.b)||point_segment(wq,e.a,e.b)<5.)) return std::pair<double,double>{-1e9,-1.};
                size_t j=0;
                for(const Obs& o:*s.obs) if(o.type==2) {
                    P2& p=threats[j++];double d=dist(p,q);
                    if(d>0) p=add(p,mul(sub(q,p),pmin(15.,d)/d));
                    clearance=pmin(clearance,dist(p,q));
                    if(o.has_rel_dir && dist(p,q)>60.) {
                        P2 v=sub(q,p);double a=pm::atan2(v.y,v.x);
                        behind-=pm::cos(wrap(a-o.rel_dir));
                    }
                }
            }
            double progress=distance*pm::cos(wrap(plan.direction-base.direction));
            double value=pmin(clearance,90.)+progress*.3+PRED.behind_weight*behind
                         -PRED.energy_weight*cost_now(plan.dist,plan.turn,s)*horizon;
            if(clearance<15.+PRED.risk_margin) value-=10000.+(15.+PRED.risk_margin-clearance)*100.;
            return std::pair<double,double>{value,clearance};
        };
        auto original=evaluate(base);
        if(original.second>=15.+PRED.risk_margin && PRED.behind_weight<=0) return base;
        Plan selected=base;double best=original.first;
        for(int k=0;k<16;k++) for(int speed=0;speed<3;speed++) {
            double heading=-OPI+k*TAU/16.;
            double reach=speed==0?0.:speed==1?pmin(s.speed,s.sprint):s.sprint;
            double turn=PRED.gaze?t->angle:heading;
            Plan candidate={reach,heading,pmax(-PRED.turn_max,pmin(PRED.turn_max,turn))};
            auto score=evaluate(candidate);
            if(score.first>best){best=score.first;selected=candidate;}
        }
        if(selected.dist!=base.dist||selected.direction!=base.direction) release_fruit(m);
        return selected;
    }

};

}  // namespace orchard
