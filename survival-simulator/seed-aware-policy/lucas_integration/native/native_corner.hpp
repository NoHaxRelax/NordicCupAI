// Included inside orchard::Policy. Inputs are Orchard's observed map and AState.
// No Engine, true coordinates, predator energy, or hidden fruit are accessible here.
int64_t corner_started=0, corner_aligned=0, corner_missed=0, corner_aborted=0;
P2 corner_for(Group& g, P2 predator) {
    P2 cs[4]={{40.,40.},{W-40.,40.},{40.,H-40.},{W-40.,H-40.}};
    if (P.corner_mode<1.5) {
        int best=0; for(int i=1;i<4;i++) if(dist(cs[i],predator)<dist(cs[best],predator))best=i;
        return cs[best];
    }
    if(g.corner_index<0) {
        double counts[4]={0,0,0,0};
        g.trees.each([&](const int64_t&,TreeP& t){for(int i=0;i<4;i++)if(dist(t->p,cs[i])<400.)counts[i]++;});
        g.corner_index=0;for(int i=1;i<4;i++)if(counts[i]<counts[g.corner_index])g.corner_index=i;
    }
    return cs[g.corner_index];
}
bool corner_steer(const AState& s, Plan& pl) {
    Mind& m=M(s.aid);Group& g=G(m.group);const PoseObj& ps=*m.pose;
    if(!g.anchored || time<m.corner_retry)return false;
    const Obs* o=nullptr;
    double match=OINF,second=OINF;
    for(const Obs& x:*s.obs)if(x.type==2 && x.has_rel_dir) {
        double score=x.distance;
        if(m.corner_phase) {
            P2 q=polar(ps,x);double hh=wrap(ps.theta+x.angle+OPI-x.rel_dir);
            double shift=dist(q,m.corner_pred),turn=std::fabs(wrap(hh-m.corner_heading));
            if(shift>22. || turn>.75)continue;
            score=shift+20.*turn;
        }
        if(score<match){second=match;match=score;o=&x;}else second=pmin(second,score);
    }
    if(m.corner_phase && second-match<8.)o=nullptr; // ambiguous identity is not success
    if(!o) {if(m.corner_phase){corner_missed++;m.corner_phase=0;m.corner_retry=time+8.;}return false;}
    P2 p=polar(ps,*o);double h=wrap(ps.theta+o->angle+OPI-o->rel_dir);
    bool seen=o->distance<=60. || (o->distance<=250. && std::fabs(wrap(o->rel_dir))<=OPI/6.);
    if(m.corner_phase && dist(p,m.corner_pred)>45.) {
        // A different nearest sighting or a localization jump is not an aligned exit.
        corner_missed++;m.corner_phase=0;m.corner_retry=time+8.;return false;
    }
    if(!m.corner_phase) {
        if(!seen || o->distance<90. || s.energy<0.35*s.max_energy || s.sprint<=15.)return false;
        // One nearby observer steers; other gatherers retain Oscar's escape.
        for(const AState& other:states)if(other.aid!=s.aid && M(other.aid).group==m.group) {
            P2 q=M(other.aid).pose->p;double d=dist(q,p),a=wrap(std::atan2(q.y-p.y,q.x-p.x)-h);
            if(d<o->distance-5. && (d<=60. || (d<=250. && std::fabs(a)<=OPI/6.)))return false;
        }
        m.corner_goal=corner_for(g,p);
        double initial_error=std::fabs(wrap(std::atan2(m.corner_goal.y-p.y,m.corner_goal.x-p.x)-h));
        if(initial_error>P.corner_start_angle*OPI/180.)return false;
        m.corner_phase=1;m.corner_start=time;corner_started++;
    }
    double want=std::atan2(m.corner_goal.y-p.y,m.corner_goal.x-p.x);
    double error=std::fabs(wrap(h-want)),tol=P.corner_tolerance*OPI/180.;
    m.corner_pred=p;m.corner_heading=h;m.corner_want=want;m.corner_error=error;
    if(!seen && o->distance>80.) {
        if(error<=tol)corner_aligned++;else corner_missed++;
        m.corner_phase=0;m.corner_retry=time+8.;return false;
    }
    if(time-m.corner_start>P.corner_budget || s.energy<0.25*s.max_energy) {
        corner_aborted++;m.corner_phase=0;m.corner_retry=time+8.;return false;
    }
    m.corner_phase=error<tol*.6?2:1;
    auto clear=[&](P2 a,P2 b,double r){
        for(auto& e:m.edges)if(time-e.t<30.) {
            if(segments_cross(a,b,e.a,e.b) || point_segment(b,e.a,e.b)<r)return false;
        }
        return true;
    };
    auto visible=[&](P2 pp,double hh,P2 q){double d=dist(pp,q);return d<=60. || (d<=250. && std::fabs(wrap(std::atan2(q.y-pp.y,q.x-pp.x)-hh))<=OPI/6. && clear(pp,q,0.));};
    auto pred=[&](P2 pp,double hh,P2 q,double face,double speed){
        double d=dist(pp,q),a=wrap(std::atan2(q.y-pp.y,q.x-pp.x)-hh),dir=0.,turn=0.;
        if(visible(pp,hh,q)) {
            double look=wrap(std::atan2(pp.y-q.y,pp.x-q.x)-face);
            if(std::fabs(look)>OPI/2. || d<90.){turn=std::fabs(a)>.05?pmax(-.3,pmin(.3,a*.5)):0.;dir=std::fabs(a)>.05?turn:a;}
            else {dir=a+(look>0.?-1.:1.)*OPI/4.;turn=std::atan2(d*std::sin(a)-15.*std::sin(dir),d*std::cos(a)-15.*std::cos(dir));}
        } else speed=pmin(speed,11.);
        P2 np=add(pp,mul(unit(hh+dir),speed));
        if(!clear(pp,np,10.))np=pp;
        return std::make_pair(np,wrap(hh+turn));
    };
    double cap=s.energy>=s.max_energy*.2?s.sprint:pmin(s.speed,s.sprint);
    double best=OINF;Plan win=pl;bool found=false;
    for(double length:{pmin(s.speed,cap),cap})for(int i=0;i<48;i++) {
        double direction=TAU*i/48.;P2 q=add(ps.p,mul(unit(direction),length*MOVE_PENALTY[s.biome]));
        if(!clear(ps.p,q,5.01))continue;
        // A small deliberate gaze offset selects the watched-predator strafe side.
        // Centering on a stale sighting left the side sensitive to tiny errors.
        double sign=wrap(want-h)>=0.?1.:-1.;
        double face=std::atan2(p.y-q.y,p.x-q.x)-sign*P.corner_gaze,worst=0.;
        for(double speed:{11.*MOVE_PENALTY[s.biome],15.}) {
            auto lag=pred(p,h,ps.p,ps.theta,speed);auto next=pred(lag.first,lag.second,q,face,speed);
            double d=dist(q,next.first),err=std::fabs(wrap(next.second-want));
            double risk=d<15.?1e6+(15.-d)*1e4:0.;
            for(auto& other:g.pseen)if(dist(other.p,p)>20. && dist(q,other.p)<40.)risk+=1e5;
            double spacing=pmax(0.,95.-d)+pmax(0.,d-125.);
            double objective=m.corner_phase==2 ? (visible(next.first,next.second,q)?100.:0.)+1000.*pmax(0.,err-tol)+err : err*40.+spacing*3.;
            worst=pmax(worst,risk+objective);
        }
        double cost=0.05*pmin(length,s.speed)+.5*pmax(0.,length-s.speed);
        double rank=worst+cost*.1;
        if(rank<best){best=rank;win=Plan{length,wrap(direction-ps.theta),wrap(face-ps.theta)};found=true;}
    }
    if(found)pl=win;
    return found;
}
