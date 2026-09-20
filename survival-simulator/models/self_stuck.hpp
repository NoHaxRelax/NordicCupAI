// Included inside EvasionPolicy. Public sightings and this agent's observed edges
// only: no engine predator identity, private position, energy, RNG or hidden map.
// Five hypotheses, not a guarantee of confinement. The oracle census is NOT used.
struct StuckMemory { P2 last{}, origin{}; double seen=-1., since=0.; int64_t group=-1; };
std::unordered_map<int64_t,StuckMemory> stuck_memory;
Plan self_stuck(Mind& m,const AState& s,Plan base) {
    const Obs* t=threat(s);
    if(!t || t->distance>PRED.stuck_radius) return base;
    int mode=(int)PRED.stuck_mode;
    P2 pos=m.pose->p,pred=polar(*m.pose,*t);
    auto& mem=stuck_memory[m.aid];
    if(mem.group!=m.group || time-mem.seen>.3 || dist(pred,mem.last)>22. || dist(pred,mem.origin)>20.) {
        mem.since=time;mem.origin=pred;
    }
    mem.group=m.group;mem.last=pred;mem.seen=time;
    bool stationary=time-mem.since>=PRED.stuck_patience;
    // Public rel_dir is bearing FROM predator TO observer minus predator heading.
    double heading=m.pose->theta+t->angle+OPI-t->rel_dir;
    auto exposed=[&](P2 q) {
        P2 v=sub(q,pred);double d=dist(q,pred);
        return d<65. || !t->has_rel_dir || (d<210. && std::abs(wrap(pm::atan2(v.y,v.x)-heading))<OPI/6+.08);
    };
    // A stationary sighting outside detection should not cause a lure/re-engagement.
    bool preserve=(mode==4||mode==5)&&stationary;
    if(preserve&&!exposed(pos)) {
        base=Policy::act(m,s);
        P2 next=add(pos,mul(unit(m.pose->theta+base.direction),base.dist));
        if(!exposed(next)) return base;
    }
    if(mode==4&&!preserve) return base;
    // Luring is abandoned when energy is low. Emergency avoidance still applies.
    if(!preserve && s.energy<s.max_energy*PRED.stuck_energy) return base;
    const EdgeMem* wall=nullptr;P2 target{};double nearest=PRED.stuck_radius;
    for(const auto& e:m.edges) {
        if(time-e.t>25.)continue;
        double len=dist(e.a,e.b);bool boundary=len>1000.;
        if(len<1. || (mode==1&&!boundary) || ((mode==2||mode==3)&&boundary))continue;
        P2 v=mul(sub(e.b,e.a),1./len),w=sub(pred,e.a);
        double along=pmax(0.,pmin(len,w.x*v.x+w.y*v.y));
        P2 foot=add(e.a,mul(v,along));
        if(mode==3)foot=dist(pred,e.a)<dist(pred,e.b)?e.a:e.b;
        double d=dist(pred,foot);
        if(d<nearest){nearest=d;wall=&e;target=foot;}
    }
    if(!wall&&!preserve)return base;
    // Pull towards a visible barrier, then disengage sideways once predator is
    // heading into it. A release angle is tunable; hearing overrides cone escape.
    P2 desired=wall?sub(target,pred):P2{0.,0.};
    double dl=dist(desired,P2{0.,0.});if(dl>0)desired=mul(desired,1./dl);
    bool aligned=wall&&t->has_rel_dir&&dl<65.&&
        std::abs(wrap(heading-pm::atan2(desired.y,desired.x)))<PRED.stuck_release*OPI/180.;
    bool escape=preserve||aligned;
    double terrain=s.biome==RIVER?.3:s.biome==SWAMP?.5:s.biome==DESERT?.8:1.;
    auto score=[&](Plan plan) {
        double step=pmin(plan.dist,s.energy>=s.max_energy/5.?s.sprint:s.speed)*terrain;
        P2 q=add(pos,mul(unit(m.pose->theta+plan.direction),step));
        if(G(m.group).anchored&&(q.x<5.||q.y<5.||q.x>W-5.||q.y>H-5.))return -1e12;
        bool hidden=false;
        for(const auto& e:m.edges)if(time-e.t<25.) {
            if(segments_cross(pos,q,e.a,e.b)||point_segment(q,e.a,e.b)<5.1)return -1e12;
            if(segments_cross(pred,q,e.a,e.b))hidden=true;
        }
        double clearance=OINF;
        for(const Obs& o:*s.obs)if(o.type==2) {
            // Conservative one-step bound; does not read private predator terrain.
            clearance=pmin(clearance,dist(q,polar(*m.pose,o))-15.);
        }
        if(clearance<16.)return -1e8+clearance*1000.;
        double d=dist(q,pred);
        double value=-.15*cost_now(plan.dist,plan.turn,s);
        if(escape) {
            value+=.3*pmin(d,160.);
            if(d>65.&&(!exposed(q)||hidden))value+=PRED.stuck_reward;
            value+=.1*step*pm::cos(wrap(plan.direction-base.direction));
        } else {
            P2 v=sub(q,pred);double aim=(v.x*desired.x+v.y*desired.y)/pmax(1.,d);
            value+=PRED.stuck_reward*aim-.5*std::abs(d-PRED.stuck_gap);
            if(mode==3&&hidden&&d>65.)value+=PRED.stuck_reward;
            // Discourage leaving pursuit before the predator approaches the wall.
            if(!exposed(q))value-=PRED.stuck_reward*.4;
        }
        return value;
    };
    Plan selected=base;double best=score(base);
    for(int k=0;k<16;++k)for(int run=0;run<2;++run) {
        double a=-OPI+k*TAU/16.;
        Plan c={run?s.sprint:pmin(s.speed,s.sprint),a,pmax(-PRED.turn_max,pmin(PRED.turn_max,t->angle))};
        double value=score(c);if(value>best){best=value;selected=c;}
    }
    if(best<=-1e12)return base;
    if(selected.dist!=base.dist||selected.direction!=base.direction)release_fruit(m);
    return selected;
}
