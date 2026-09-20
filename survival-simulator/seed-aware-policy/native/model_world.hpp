// Synchronized seed-model snapshot. Policy never reads Engine directly.
struct ModelPose { int64_t id; P2 p; double heading; bool resting=false; };
struct ModelObstacle { double x,y,w,h; };
std::vector<ModelPose> model_agents,model_predators;
std::vector<ModelObstacle> model_obstacles;
bool model_full()const{return resource_mode>=7 && resource_synchronized && std::abs(resource_time-time)<1e-7;}
void model_localize(){
 if(!model_full())return;
 for(auto& a:model_agents){
  if(!minds.has(a.id))continue;auto& m=M(a.id);auto& g=G(m.group);
  if(!g.anchored){double turn=wrap(a.heading-m.pose->theta);transform_group(g,turn,sub(a.p,rot(m.pose->p,turn)));g.anchored=true;}
 }
 for(auto& a:model_agents)if(minds.has(a.id))M(a.id).pose=mkpose(a.p,a.heading);
}
void model_map(){
 if(!model_full()||resource_mode<8)return;
 groups.each([&](const int64_t&,GroupP& g){
  if(!g->anchored)return;
  // Replace noisy accumulated faces with the exact static map. Preserve route caches
  // when the map is already exact; only membership/map transitions invalidate them.
  bool same_map=g->walls.size()==model_obstacles.size()*4;
  if(same_map){size_t i=0;for(auto&o:model_obstacles){auto&w=g->walls[i];if(!w.horiz||w.c!=o.y||w.lo!=o.x||w.hi!=o.x+o.w){same_map=false;break;}i+=4;}}
  if(!same_map){g->walls.clear();for(auto&o:model_obstacles){auto add=[&](bool h,double c,double lo,double hi,double solid){Group::Wall w{h,c,lo,hi,solid,time,100};w.cs={c};w.los={lo};w.his={hi};w.obs1=0;w.obs2=1;w.n_obs=2;g->walls.push_back(w);};add(true,o.y,o.x,o.x+o.w,1);add(true,o.y+o.h,o.x,o.x+o.w,-1);add(false,o.x,o.y,o.y+o.h,1);add(false,o.x+o.w,o.y,o.y+o.h,-1);}g->wall_version++;g->sites_t=-1e9;}
 });
}
void model_predator_map(){
 if(!model_full()||(resource_mode!=9&&resource_mode!=11))return;
 groups.each([&](const int64_t&,GroupP& g){if(!g->anchored)return;g->pseen.clear();g->pmem.clear();for(auto&p:model_predators){g->pseen.push_back({p.p,p.heading});g->pmem.push_back({p.p,time});}});
}
