#define SURVIVAL_ENGINE_SOURCE "../lucas_integration/native/_nengine.cpp"
#include "native_common.hpp"
#include "predictive_safety.hpp"
#include "birth_forecast.hpp"
void apply_activation_overrides(orchard::Params& p,const J& config,int mode){
 if(!config.contains("activation_overrides"))return;
 const auto& variants=config.at("activation_overrides");
 auto key=std::to_string(mode);if(!variants.contains(key))return;
 std::map<std::string,double*> fields={{"breed_reserve",&p.breed_reserve},{"breed_reserve_late",&p.breed_reserve_late},
 {"child_prio",&p.child_prio},{"low_pop_reserve",&p.low_pop_reserve},{"cap_mult",&p.cap_mult},{"cap_min",&p.cap_min},{"cap_max",&p.cap_max},
 {"refuge_mode",&p.refuge_mode},{"refuge_r",&p.refuge_r},{"refuge_trigger",&p.refuge_trigger},{"refuge_leave",&p.refuge_leave},{"refuge_slow_only",&p.refuge_slow_only},{"refuge_post_w",&p.refuge_post_w},{"refuge_clear",&p.refuge_clear},{"refuge_sprint",&p.refuge_sprint},{"site_dist_w",&p.site_dist_w},{"fit_speed",&p.fit_speed},{"fit_vision",&p.fit_vision},{"fit_hear",&p.fit_hear},{"fit_energy",&p.fit_energy},{"fit_speed_cap",&p.fit_speed_cap},{"heir_age",&p.heir_age},{"heir_reserve",&p.heir_reserve},{"tree_half",&p.tree_half},{"pred_mode",&p.pred_mode},{"pred_share",&p.pred_share},{"pred_dodge_r",&p.pred_dodge_r},{"pred_dodge_ang",&p.pred_dodge_ang},{"pred_dodge_hold",&p.pred_dodge_hold},{"pred_dodge_hold_face",&p.pred_dodge_hold_face},{"pred_r",&p.pred_r},{"pred_face",&p.pred_face},{"pred_face_r",&p.pred_face_r},{"pred_sprint_r",&p.pred_sprint_r},{"hungry_margin",&p.hungry_margin},{"fruit_min_wait",&p.fruit_min_wait},{"fruit_reach",&p.fruit_reach},{"tree_reach",&p.tree_reach},{"dist_pen",&p.dist_pen},{"cluster_radius",&p.cluster_radius},{"nursery_bonus",&p.nursery_bonus}};
 std::map<std::string,bool*> bool_fields={{"heir_select",&p.heir_select}};
 for(auto& item:variants.at(key).items()){
  if(fields.count(item.key()))*fields.at(item.key())=item.value().get<double>();
  else if(bool_fields.count(item.key()))*bool_fields.at(item.key())=item.value().get<bool>();
  else throw std::runtime_error("Unsupported activation override: "+item.key());
 }
}
int main(int argc,char**argv){
 if(argc!=6&&argc!=7){std::cerr<<"policy_bench CONFIG SEED MODE HORIZON OUT\n";return 2;}
 auto config=read_json(argv[1]);if(config.contains("entrapment_final"))config=config["entrapment_final"];
 auto params=initialize_math(config);uint32_t seed=std::stoull(argv[2]);int mode=std::stoi(argv[3]);double horizon=std::stod(argv[4]);fs::path out=argv[5];double available_at=argc==7?std::stod(argv[6]):(mode==5?180.:(mode==6?600.:0.));
 auto e=make_engine(seed);orchard::Policy p({0},params);p.resource_mode=mode;bool activated=false;
 PredictiveSafety safety;BirthForecast births;
 NativeReplay replay(*e,seed,"Lucas policy resource experiment mode "+std::to_string(mode));
 replay.data["meta"]["policy"]="lucas-6bef2ccd-resource-v3-mode"+std::to_string(mode);
 replay.data["meta"]["scenario"]="Natural seeded world; predators enabled; known-state resource upper bound, not live recovery";
 replay.data["meta"]["notes"]="C++ engine and policy; model inputs are exact current engine resources for this ablation. State-only replay. No evaluation API.";
 uint64_t prefix_hash=1469598103934665603ULL;auto hash_word=[&](uint64_t v){for(int i=0;i<8;i++){prefix_hash^=(v>>(i*8))&255;prefix_hash*=1099511628211ULL;}};auto hash_double=[&](double v){uint64_t x;std::memcpy(&x,&v,8);hash_word(x);};
 auto start=Clock::now();int deaths[4]={0,0,0,0};size_t peak=5,ticks=0;double pop_integral=0; e->on_kill=[&](const Creature&,int cause){if(cause>=0&&cause<4)deaths[cause]++;};
 replay.capture(*e);
 try{
 while(!e->agents.empty() && e->time+1e-6<horizon){
  if(mode){p.resource_time=e->time;p.resource_synchronized=e->time>=available_at;p.resource_trees.clear();p.resource_fruits.clear();for(auto&t:e->trees)p.resource_trees.push_back({{t.x,t.y},t.age,0,e->biome_at(t.x,t.y)});for(auto&f:e->fruits)p.resource_fruits.push_back({{f.x,f.y},f.age,f.energy,e->biome_at(f.x,f.y)});}
  if(mode>=7){p.model_agents.clear();p.model_predators.clear();p.model_obstacles.clear();for(auto&a:e->agents)p.model_agents.push_back({a.id,{a.x,a.y},a.direction,false,a.max_age});for(auto&a:e->predators)p.model_predators.push_back({a.key,{a.x,a.y},a.direction,a.resting});for(auto&o:e->obstacles)p.model_obstacles.push_back({o.x,o.y,o.w,o.h});}
  if(!activated&&p.resource_synchronized){activated=true;apply_activation_overrides(p.P,config,mode);
   if(config.contains("birth_overrides")&&config.at("birth_overrides").contains(std::to_string(mode))){auto&b=config.at("birth_overrides").at(std::to_string(mode));births.age_floor=b.value("age_floor",-1.);births.walking_floor=b.value("walking_floor",-1.);births.urgent_floor=b.value("urgent_floor",-1.);}
   if(mode==12||mode==14){p.P.breed_reserve+=60.;p.P.breed_reserve_late+=60.;}if(mode==24||mode==25){p.P.tree_reach*=2.;p.P.fruit_reach*=2.;}}
  if((mode==197||mode==198)&&p.resource_synchronized&&(p.future_built<0.||e->time-p.future_built>=30.))
   p.forecast_future_fruits(*e,mode==197?60.:180.);
  std::map<int64_t,bool> old_heirs;if(BirthForecast::enabled(mode)&&p.resource_synchronized)p.minds.each([&](const int64_t&id,orchard::MindP&m){old_heirs[id]=m->heir_done;});
  auto acts=p.call(policy_states(e.get()),e->time);
  if(((mode>=35&&mode<=38)||(mode>=140&&mode<=242))&&p.resource_synchronized)safety.apply(*e,p,acts,mode);
  if(BirthForecast::enabled(mode)&&p.resource_synchronized)births.apply(*e,p,acts,mode,old_heirs);
  if(e->time<available_at){for(auto&a:acts){hash_word(a.aid);hash_double(a.dist);hash_double(a.direction);hash_double(a.turn);hash_word(a.spawn);}}
  for(auto&a:acts){size_t before=e->agents.size();e->agent_step({a.aid,a.dist,true,a.direction,a.turn,a.spawn});if(BirthForecast::enabled(mode)&&p.resource_synchronized&&e->agents.size()>before)births.verify(a.aid,e->agents.back());}
  e->non_agent_step();ticks++;peak=std::max(peak,e->agents.size());pop_integral+=e->agents.size()*e->dt;replay.capture(*e);
  if(ticks%1000==0)save_json(out/"progress.json",{{"seed",seed},{"mode",mode},{"time",e->time},{"score",e->score},{"population",e->agents.size()},{"wall_seconds",seconds(start)}});
 }
 size_t meals=0,ripe=0;double food_energy=0.,meal_age=0.;
 for(auto& ev:e->events){if(ev.kind==2){meals++;ripe+=ev.energy>=59.9;food_energy+=ev.energy;meal_age+=ev.age;}}
 replay.data["events"]=J::array();for(auto& ev:e->events)replay.data["events"].push_back({{"kind",ev.kind},{"time",ev.t},{"agent_id",ev.id},{"age",ev.age},{"energy",ev.energy}});
 replay.save(*e,out/"replay.json",e->agents.empty()?"extinction":"horizon");
 save_json(out/"result.json",{{"seed",seed},{"mode",mode},{"score",e->score},{"duration",e->time},{"population",e->agents.size()},{"peak_population",peak},{"population_seconds",pop_integral},{"deaths_by_cause",deaths},{"ticks",ticks},{"wall_seconds",seconds(start)},{"horizon",horizon},{"seed_available_at",available_at},{"pre_activation_action_hash",std::to_string(prefix_hash)},{"meals",meals},{"ripe_meals",ripe},{"food_energy",food_energy},{"mean_meal_age",meals?meal_age/meals:0.},{"refuge_events",p.refuge_events},{"refuge_holds",p.refuge_holds},{"refuge_died_route",p.refuge_died_route},{"refuge_died_hold",p.refuge_died_hold},{"refuge_died_exit",p.refuge_died_exit},{"births_protected",births.protected_births},{"births_proposed",births.proposed},{"births_postponed",births.postponed},{"births_verified",births.verified},{"safety_checked",safety.checked},{"safety_searched",safety.searched},{"safety_changed",safety.changed},{"safety_candidates",safety.candidates},{"source_commit","6bef2ccd3f129feaa9c2913d077987b072711784"},{"resource_provenance",mode?"known current engine state upper bound":"public observations only"}});
 }catch(const std::exception&err){replay.save(*e,out/"replay.json",std::string("failure: ")+err.what());save_json(out/"failure.json",{{"error",err.what()}});return 1;}
}
