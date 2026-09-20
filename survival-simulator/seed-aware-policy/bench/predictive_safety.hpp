#pragma once
// Conditional one-tick model lookahead. Never advances the live world. Births,
// meals and other predators' kills are omitted from this short rollout.
struct PredictiveSafety {
 uint64_t checked=0, searched=0, changed=0, candidates=0;
 struct Frame { Creature pred; std::vector<Obs> edges; std::vector<double> rx,ry; std::vector<int32_t> walls; double minx,maxx,miny,maxy; };
 static double wrap(double x){return py_mod(x+PI,TWO_PI)-PI;}
 Creature move(Engine& e,Creature a,const orchard::Act& act){
  auto walls=e.local_obstacles(a.x,a.y);
  e.update_entity_position(a,act.dist,true,act.direction,walls);
  e.update_entity_direction(a,act.turn);return a;
 }
 void apply(Engine& e,orchard::Policy& policy,std::vector<orchard::Act>& acts,int mode){
  auto original_rng=e.rng;bool original_dirty=e.predators_dirty;
  std::vector<Creature> next=e.agents;std::unordered_map<int64_t,size_t> ids;
  for(size_t i=0;i<next.size();++i)ids[next[i].id]=i;
  for(auto& a:acts)next[ids.at(a.aid)]=move(e,next[ids.at(a.aid)],a);
  std::vector<Frame> frames;
  for(auto& pr:e.predators){
   if(pr.resting&&pr.energy<=pr.max_energy*.5)continue;
   bool near=false;for(auto& a:next)if(np_hypot(a.x-pr.x,a.y-pr.y)<110.){near=true;break;}
   if(!near)continue;
   Frame f;f.pred=pr;f.walls=e.local_obstacles(pr.x,pr.y);
   e.observe(pr,nullptr,nullptr,nullptr,nullptr,f.edges);
   f.rx=e.ring_x;f.ry=e.ring_y;
   f.minx=*std::min_element(f.rx.begin(),f.rx.end());f.maxx=*std::max_element(f.rx.begin(),f.rx.end());
   f.miny=*std::min_element(f.ry.begin(),f.ry.end());f.maxy=*std::max_element(f.ry.begin(),f.ry.end());
   frames.push_back(std::move(f));
  }
  double margin=(mode>=140&&mode<=156)?0.:((mode>=157&&mode<=163)?1.:(mode==174?2.:mode==175?5.:mode==176?10.:mode==35?2.:mode==36?12.:25.));
  for(auto& act:acts){
   size_t ai=ids.at(act.aid);const Creature& agent=e.agents[ai];
   if((mode==171&&agent.energy>0.5*agent.max_energy)||(mode==172&&agent.energy>0.7*agent.max_energy)||(mode==173&&agent.age>60.))continue;
   bool near=false;for(auto& f:frames)if(np_hypot(agent.x-f.pred.x,agent.y-f.pred.y)<100.){near=true;break;}
   if(!near)continue;++checked;
   if(mode==187||mode==188){
    double threshold=mode==187?0.:.5;bool approaching=false;
    for(auto& f:frames){double dx=agent.x-f.pred.x,dy=agent.y-f.pred.y,d=np_hypot(dx,dy);if(d>=100.)continue;double toward=(dx*std::cos(f.pred.direction)+dy*std::sin(f.pred.direction))/std::max(1.,d);if(toward>threshold){approaching=true;break;}}
    if(!approaching)continue;
   }
   auto evaluate=[&](const orchard::Act& candidate){
    ++candidates;next[ai]=move(e,agent,candidate);
    std::vector<Engine::Target> targets;for(auto&a:next)targets.push_back({a.x,a.y,a.direction,a.id});
    double gap=1e6;
    for(auto&f:frames){
     Creature pred=f.pred;auto obs=f.edges;e.ring_x=f.rx;e.ring_y=f.ry;
     int steps=(mode==167?2:mode==168?3:1);
     for(int step=0;step<steps;++step){
      e.process_objects(pred,targets,1,true,true,f.minx,f.maxx,f.miny,f.maxy,obs);
      e.rng=original_rng;e.predator_act(pred,obs,f.walls);
      e.rng=original_rng;e.predators_dirty=original_dirty;
      gap=std::min(gap,np_hypot(next[ai].x-pred.x,next[ai].y-pred.y)-agent.size-pred.size);
     }
    }
    return gap;
   };
   orchard::Act original=act;Creature original_next=next[ai];
   double baseline_gap=evaluate(original);
   auto& mind=policy.M(act.aid);
   double trigger=(mode==177?-5.:mode==178?-10.:mode==179?-15.:mode==184?(mind.has_fruit?-5.:margin):mode==185?(mind.has_fruit?-10.:margin):mode==186?(mind.has_fruit?-1e30:margin):margin);
   if(baseline_gap>=trigger)continue;
   ++searched;double best=-1e30;orchard::Act chosen=original;
   auto consider=[&](const orchard::Act& trial){
    double gap=evaluate(trial);
    // Safety dominates when a collision is forecast. Beyond the margin,
    // preserve task progress and energy rather than maximizing distance forever.
    double displacement=np_hypot(next[ai].x-original_next.x,next[ai].y-original_next.y);
    double cost=agent.energy-next[ai].energy;
    double disp_w=(mode==150?.10:mode==151?.50:mode==152?.10:.25);
    double cost_w=(mode==150?1.50:mode==151?2.50:mode==152?4.00:mode==159?1.50:mode==160?2.50:.80);
    double turn_delta=std::abs(wrap(trial.turn-original.turn));
    double turn_w=(mode==180?0.80:0.);
    double value=100.*std::min(gap,margin)-disp_w*displacement-cost_w*cost-turn_w*turn_delta;
    if(value>best){best=value;chosen=trial;}
   };
   consider(original);
   double nearest=1e9,face=agent.direction;
   for(auto&f:frames){double d=np_hypot(agent.x-f.pred.x,agent.y-f.pred.y);if(d<nearest){nearest=d;face=np_atan2(f.pred.y-agent.y,f.pred.x-agent.x);}}
   for(int i=0;i<16;++i)for(double speed:{agent.speed,agent.sprint_speed})for(int facing=0;facing<2;++facing){
    double heading=TWO_PI*i/16.;
    orchard::Act trial{agent.id,speed,wrap(heading-agent.direction),wrap((facing?face:heading)-agent.direction),original.spawn};
    consider(trial);
   }
   consider({agent.id,0.,0.,wrap(face-agent.direction),original.spawn});
   next[ai]=move(e,agent,chosen);
   if(chosen.dist!=original.dist||chosen.direction!=original.direction||chosen.turn!=original.turn||chosen.spawn!=original.spawn){
    ++changed;if(mode==38){next[ai]=original_next;continue;}act=chosen;auto&m=policy.M(act.aid);
    m.last_action.dist=act.dist;m.last_action.direction=act.direction;m.last_action.turn=act.turn;
    if(!act.spawn){m.spawned_ok=false;auto&sp=policy.last_spawners;sp.erase(std::remove(sp.begin(),sp.end(),act.aid),sp.end());}
   }
  }
  if(e.rng.mti!=original_rng.mti||std::memcmp(e.rng.mt,original_rng.mt,sizeof(e.rng.mt))||e.predators_dirty!=original_dirty)
   throw std::runtime_error("Predictive safety mutated live RNG or predator dirty state");
 }
};
