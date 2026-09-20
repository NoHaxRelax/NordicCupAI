// Structured synchronized-model input. Benchmarks explicitly use engine truth as an upper bound.
struct ResourcePoint { P2 p; double age, energy; int biome=-1; };
int resource_mode=0;
bool harvest_mode()const{return resource_mode==10||resource_mode==11;}
bool routing_mode()const{return harvest_mode()||resource_mode==23||resource_mode==24;}
bool harvest_timing()const{return harvest_mode()||resource_mode==15||resource_mode==17;}
bool harvest_inventory()const{return harvest_mode()||resource_mode==16||resource_mode==17;}
bool resource_synchronized=false;
double resource_time=-1.;
std::vector<ResourcePoint> resource_trees,resource_fruits;
std::unordered_map<const FruitM*,ResourcePoint> matched_fruit;
std::unordered_map<const TreeM*,ResourcePoint> matched_tree;
mutable std::unordered_map<const TreeM*,double> remaining_cache;
bool model_tree_alive(const TreeM& t)const{return resource_synchronized && std::abs(resource_time-time)<1e-7 && matched_tree.count(&t);}
double model_remaining(const TreeM& t,double fallback)const{
 if(!model_tree_alive(t))return fallback;
 auto cached=remaining_cache.find(&t);if(cached!=remaining_cache.end())return cached->second;
 double age=matched_tree.at(&t).age,prob=1.,expected=0.;
 // Conditional expected lifetime from the engine's per-tick age hazard, not a frozen future claim.
 for(double dt=.1;age+dt<100. && prob>1e-5;dt+=.1){double a=age+dt;double hazard=a<=50.?0.:std::pow((a-50.)/50.,2);prob*=1.-hazard;expected+=.1*prob;}
 remaining_cache[&t]=expected;return expected;
}
void ingest_resource_forecast() {
    if(!resource_mode || !resource_synchronized || std::abs(resource_time-time)>1e-7)return;
    matched_fruit.clear();matched_tree.clear();remaining_cache.clear();
    groups.each([&](const int64_t&,GroupP& g){
        if(!g->anchored)return;
        auto reachable=[&](P2 p){bool yes=false;g->agents.each([&](int64_t a){if(resource_mode>=8 || dist_lt(M(a).pose->p,p,300.))yes=true;});return yes;};
        for(auto& r:resource_trees){
            if(resource_mode==3||resource_mode==7)continue;
            TreeP t; double best=4.;for(auto& c:g->near_trees(r.p,4.)){double d=dist(c->p,r.p);if(d<best){best=d;t=c;}}
            if(!t && (resource_mode==2||(resource_mode>=4&&resource_mode!=7)) && reachable(r.p)){t=std::make_shared<TreeM>();t->id=g->next_tree++;t->p=r.p;g->add_tree(t);}
            if(t){if(harvest_inventory()&&r.biome>=0)g->cells.set(cell_of(r.p),CellV{time,r.biome});matched_tree[t.get()]=r;t->first=time-r.age;t->fresh=true;t->last=time;t->dead=false;g->seen_trees.insert(t->id);}
        }
        for(auto& r:resource_fruits){
            FruitP f;double best=4.;for(auto& c:g->near_fruits(r.p,4.)){double d=dist(c->p,r.p);if(d<best){best=d;f=c;}}
            if(!f && (resource_mode==2||(resource_mode>=4&&resource_mode!=7)) && reachable(r.p)){f=std::make_shared<FruitM>();f->id=g->next_fruit++;f->p=r.p;g->add_fruit(f);}
            if(f){matched_fruit[f.get()]=r;g->seen_fruits.insert(f->id);f->born_lo=f->born_hi=time-r.age/2.;f->last=time;f->misses=0;}
        }
        if(harvest_inventory()){
            for(auto id:g->fruits.key_list()){auto f=g->fruits.at(id);if(matched_fruit.count(f.get()))continue;if(f->has_claim&&minds.has(f->claimed)&&M(f->claimed).fruit==id)M(f->claimed).has_fruit=false;g->del_fruit(id);}
            g->trees.each([&](const int64_t&,TreeP&t){if(!matched_tree.count(t.get()))t->dead=true;});
        }
    });
}
bool harvest_arrival_ready(const FruitM& f,const Mind& m,const AState& s,bool fallback)const{
 if(!harvest_timing()||!resource_synchronized)return fallback;
 auto it=matched_fruit.find(&f);if(it==matched_fruit.end())return fallback;
 double speed=pmax(1.,pmin(s.speed,s.sprint)*MOVE_PENALTY[s.biome]*10.);
 double travel=dist(m.pose->p,f.p)/speed;
 double ripe=pmax(0.,(60.-it->second.energy)/2.);
 double rot=(100.-it->second.age)/2.;
 if(travel+0.5>=rot)return false;
 return ripe<=travel+.5 || s.energy<travel+ripe+P.hungry_margin;
}
double harvest_wait_radius(const FruitM& f,const AState& s)const{
 if(!harvest_timing()||!resource_synchronized)return 0.;
 auto it=matched_fruit.find(&f);if(it==matched_fruit.end())return 0.;
 double wait=pmax(0.,(60.-it->second.energy)/2.);
 if(wait<=.05 || s.energy<wait+P.hungry_margin)return 0.;
 return 17.;
}
std::pair<bool,bool> forecast_ready(const FruitM& f,double energy,bool old)const{
    if(!resource_mode||!resource_synchronized||std::abs(resource_time-time)>1e-7)return {false,false};
    auto it=matched_fruit.find(&f);if(it!=matched_fruit.end()){auto& r=it->second;
        double left=std::max(0.,(60.-r.energy)/2.);
        return {true,left<=.05 || (!old && energy<left+P.hungry_margin)};
    }
    return {false,false};
}
double forecast_post_adjustment(const TreeM& t,const Mind& m,const AState&,double travel){
 if(!resource_synchronized||(resource_mode!=21&&resource_mode!=22))return 0.;
 // Short-horizon current-position risk only, not a fixed future path claim.
 // Long journeys require replanning before today's predator state is useful.
 if(travel>8.)return 0.;
 double risk=0.;auto& g=G(m.group);
 for(auto& p:model_predators){
  double d=dist(p.p,t.p);
  if(d>=200.||!path_clear(g,p.p,t.p,10.01))continue;
  double toward=d>1.?((t.p.x-p.p.x)*std::cos(p.heading)+(t.p.y-p.p.y)*std::sin(p.heading))/d:1.;
  risk+=pmax(0.,1.-d/200.)*(toward>0.?.5+.5*toward:.25);
 }
 return -(resource_mode==21?120.:300.)*pmin(2.,risk);
}
