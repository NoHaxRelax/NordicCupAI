#pragma once
// Forecast only the draws consumed by Engine::spawn_child. It receives a copy
// of the synchronized model RNG, never changes the live generator, and advances
// the copy only for accepted births in actual agent-action order.
struct BirthForecast {
 uint64_t proposed=0, postponed=0, verified=0;
 struct Child {double speed,sprint,max_energy,hearing,vision,cone,direction,max_age;};
 static Child predict(PyRandom& rng,const Creature& p){
  rng.uniform(0,2*PI);rng.uniform(10,30);
  Child c{p.speed,p.sprint_speed,p.max_energy,p.hearing_radius,p.vision_radius,p.cone_angle,0,0};
  for(double* value:{&c.speed,&c.sprint,&c.max_energy,&c.hearing,&c.vision,&c.cone})
   if(rng.random()<.1)*value*=rng.uniform(.5,1.5);
  c.speed=std::min(c.speed,20.);c.sprint=std::min(c.sprint,40.);c.max_energy=std::min(c.max_energy,1000.);
  c.hearing=std::min(c.hearing,100.);c.vision=std::min(c.vision,400.);c.cone=std::min(c.cone,PI/2);
  rng.uniform(0,2*PI);rng.uniform(0,60);
  c.direction=rng.uniform(0,2*PI);c.max_age=60+rng.uniform(0,60);return c;
 }
 std::map<int64_t,Child> expected;
 void apply(Engine& e,orchard::Policy& p,std::vector<orchard::Act>& acts,int mode,const std::map<int64_t,bool>& old_heirs){
  expected.clear();auto rng=e.rng;
  for(auto&act:acts){
   if(!act.spawn)continue;
   auto idx=e.find_agent(act.aid);if(idx<0)continue;Creature parent=e.agents[idx];
   auto walls=e.local_obstacles(parent.x,parent.y);e.update_entity_position(parent,act.dist,true,act.direction,walls);e.update_entity_direction(parent,act.turn);
   if(parent.energy<=100.)continue;
   ++proposed;auto next_rng=rng;Child child=predict(next_rng,parent);
   bool urgent=parent.energy<115.;
   double age_min=mode==44?110.:mode==46?90.:100.;
   bool traits=true;
   if(mode==45)traits=child.speed>=parent.speed&&child.sprint>=parent.sprint_speed;
   if(mode==46)traits=child.speed>=std::max(parent.speed,12.)&&child.sprint>=parent.sprint_speed;
   if(mode==47)traits=child.speed>=parent.speed&&child.sprint>=std::max(parent.sprint_speed,24.);
   if(mode==48)traits=child.speed>=parent.speed&&child.sprint>=parent.sprint_speed&&child.max_energy<=400.;
   bool accept=urgent||(child.max_age>=age_min&&traits);
   if(accept){rng=next_rng;expected[act.aid]=child;continue;}
   ++postponed;act.spawn=false;auto&m=p.M(act.aid);m.spawned_ok=false;
   auto old=old_heirs.find(act.aid);m.heir_done=old==old_heirs.end()?false:old->second;
   auto&sp=p.last_spawners;sp.erase(std::remove(sp.begin(),sp.end(),act.aid),sp.end());
  }
 }
 void verify(int64_t parent,const Creature& child){
  auto it=expected.find(parent);if(it==expected.end())throw std::runtime_error("Unforecast birth");
  auto&c=it->second;
  if(c.speed!=child.speed||c.sprint!=child.sprint_speed||c.max_energy!=child.max_energy||c.hearing!=child.hearing_radius||c.vision!=child.vision_radius||c.cone!=child.cone_angle||c.direction!=child.direction||c.max_age!=child.max_age)
   throw std::runtime_error("Birth forecast differs from actual child");
  ++verified;
 }
};
