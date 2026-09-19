// Candidate escape actions, using only current observations and recently observed walls.
// Three repeated-action steps approximate a predator's turn-limited pursuit. Hidden
// predator energy, biome and target are unknown: use a conservative 15-unit speed.
// This is a local model, not an exact engine rollout or a survival guarantee.
template<class Threat>
bool escape_search(const AState& s, Plan& pl, const std::vector<Threat>& threats, double gaze) {
    Mind& m=M(s.aid);const PoseObj& ps=*m.pose;
    const double walk=pmin(s.speed,s.sprint);
    const double cap=s.energy>=.2*s.max_energy?s.sprint:walk;
    double best=OINF;Plan win=pl;bool found=false;
    for(double step:{walk,cap})for(int i=0;i<24;i++) {
        double dir=TAU*i/24.;P2 delta=mul(unit(ps.theta+dir),step*MOVE_PENALTY[s.biome]);
        P2 q=ps.p;bool blocked=false;double worst=0.;
        std::vector<P2> pp;std::vector<double> hh;
        for(auto& t:threats){pp.push_back(add(ps.p,mul(unit(ps.theta+t.ang),t.d)));hh.push_back(wrap(ps.theta+t.ang+OPI-t.rel));}
        int horizon=P.evade_search<1.5?3:1;
        for(int k=0;k<horizon;k++) {
            P2 next=add(q,delta);
            for(auto& e:m.edges)if(time-e.t<25. && (segments_cross(q,next,e.a,e.b)||point_segment(next,e.a,e.b)<5.01))blocked=true;
            if(blocked)break;
            for(size_t j=0;j<pp.size();j++) {
                double d=dist(pp[j],q),a=wrap(std::atan2(q.y-pp[j].y,q.x-pp[j].x)-hh[j]);
                bool detects=d<=60. || (d<=250. && std::fabs(a)<=OPI/6.);
                double turn=detects?pmax(-.3,pmin(.3,a*.5)):0.;
                P2 pn=add(pp[j],mul(unit(hh[j]+turn),detects?15.:11.));
                double sep=dist(next,pn);
                worst=pmax(worst,sep<15.?1e6+1e4*(15.-sep):0.);
                pp[j]=pn;hh[j]=wrap(hh[j]+turn);
            }
            q=next;
        }
        if(blocked)continue;
        double detection=0.;
        for(size_t j=0;j<pp.size();j++) {
            double d=dist(q,pp[j]),a=std::fabs(wrap(std::atan2(q.y-pp[j].y,q.x-pp[j].x)-hh[j]));
            detection+=pmax(0.,65.-d)*3.;
            if(d<=250. && a<=OPI/6.)detection+=40.+(OPI/6.-a)*40.;
        }
        double energy=.05*pmin(step,walk)+.5*pmax(0.,step-walk);
        // Energy-aware version increases the sprint price as its reserve runs low.
        double eweight=P.evade_energy*(1.+2.*pmax(0.,1.-s.energy/s.max_energy));
        double goalcost=P.evade_goal*(1.-std::cos(wrap(dir-pl.direction)));
        double rank=worst+detection+eweight*energy+goalcost;
        if(rank<best){best=rank;win=Plan{step,dir,gaze};found=true;}
    }
    if(found)pl=win;
    return found;
}
