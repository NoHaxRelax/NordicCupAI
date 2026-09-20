// Seed recovery plus an exact shadow-world handoff to Oscar's mode 144.
#define SURVIVAL_ENGINE_SOURCE "../../../../seed-aware-policy/lucas_integration/native/_nengine.cpp"
#include "native_common.hpp"
#include "recovery_pacer.hpp"
#include "public_freshness.hpp"
#include "../../../../seed-aware-policy/bench/predictive_safety.hpp"
#include "../../../../seed-aware-policy/bench/birth_forecast.hpp"
#include <fcntl.h>
#include <spawn.h>
extern char** environ;
std::string filter_binary;
orchard::Params policy_config;
std::string config_path;
pid_t launch_stream(const fs::path& dir,const J& samples,uint64_t start,uint64_t count,const std::string& cfg){
    fs::create_directories(dir);{std::ofstream f(dir/"terrain.txt");for(auto p:samples)f<<p[0]<<' '<<p[1]<<' '<<p[2]<<'\n';}
    auto binary=fs::path(filter_binary).parent_path()/"streaming_verification/coordinator";
    std::vector<std::string> args={binary.string(),"stream",filter_binary,(dir/"terrain.txt").string(),(dir/"initial-public.json").string(),cfg,std::to_string(start),std::to_string(count),(dir/"stream").string()};
    std::vector<char*> av;for(auto& a:args)av.push_back(a.data());av.push_back(nullptr);
    posix_spawn_file_actions_t fa;posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa,1,(dir/"search-progress.jsonl").c_str(),O_WRONLY|O_CREAT|O_TRUNC,0600);
    posix_spawn_file_actions_adddup2(&fa,1,2);pid_t pid;
    int result=posix_spawn(&pid,binary.c_str(),&fa,nullptr,av.data(),environ);posix_spawn_file_actions_destroy(&fa);
    if(result)throw std::runtime_error("Could not launch native streaming seed search");return pid;
}
struct Game:std::enable_shared_from_this<Game>{
    fs::path dir;std::string id;orchard::Policy policy;
    std::mutex mu;std::condition_variable changed;std::vector<J> packets;
    std::ofstream packet_log,response_log;Points samples;std::vector<double> walls;
    std::queue<std::pair<uint32_t,std::string>> candidates;J frozen_samples;
    Clock::time_point started=Clock::now(),received=Clock::now();
    bool searching=false,stopped=false,model_failed=false;int64_t recovered=-1,recovery_seq=-1,latest_seq=-1,model_action_seq=-1;J latest,model_actions;
    double last_time=-1;size_t forecast_responses=0;RecoveryPacer pacer;public_freshness::Guard freshness;
    Game(const fs::path& root,const J& body):id(std::to_string(now_ns())),policy({0},policy_config){
        dir=root/id;fs::create_directories(dir);packet_log.open(dir/"packets.jsonl");response_log.open(dir/"responses.jsonl");
        for(auto& a:body["agent_status"])for(auto& o:a["observations"])if(o["type"]=="Edge"){
            double dx=o["coords"][1][0].get<double>()-o["coords"][0][0].get<double>(),dy=o["coords"][1][1].get<double>()-o["coords"][0][1].get<double>();
            double len=std::hypot(dx,dy);if(30.000001<len&&len<99.999999)walls.push_back(len);}
        save_json(dir/"initial-public.json",body);save_json(root/"current.json",{{"game_id",id},{"directory",dir.string()}});
    }
    J status_locked(){J labels=J::object();for(auto [p,b]:samples){auto key=std::to_string(b);labels[key]=labels.value(key,0)+1;}
        return {{"game_id",id},{"sim_time",last_time},{"packets",packets.size()},{"samples",samples.size()},{"labels",labels},{"search_started",searching},
            {"recovered_seed",recovered},{"paced_responses",pacer.paced_responses},{"extra_wait_seconds",pacer.used_extra_wait},{"forecast_responses",forecast_responses},{"last_model",latest},{"model_failed",model_failed},{"elapsed_wall_seconds",seconds(started)},{"last_received_seconds_ago",seconds(received)}};}
    void finish(){std::lock_guard<std::mutex> guard(mu);stopped=true;touch(dir/"stop-search");changed.notify_all();save_json(dir/"server-summary.json",status_locked());}
    void model_loop(std::unique_ptr<Engine> e,uint32_t seed){
        NativeReplay replay(*e,seed,"Native live validation recovered-seed model");replay.capture(*e,J::array(),true);advance(*e,J::array());replay.capture(*e);
        orchard::Policy oracle({0},policy_config);oracle.resource_mode=144;PredictiveSafety safety;BirthForecast births;bool oracle_activated=false;
        std::ofstream checks(dir/"model-checks.jsonl"),predictions(dir/"model-predictions.jsonl");size_t seq=0,mismatches=0;auto begin=Clock::now();
        try{while(true){
            J packet;{std::unique_lock<std::mutex> guard(mu);changed.wait_for(guard,std::chrono::milliseconds(200),[&]{return stopped||packets.size()>seq;});
                if(packets.size()<=seq){if(stopped||seconds(received)>90)break;continue;}packet=packets[seq];}
            bool audit=seq%100==0||int64_t(seq)==recovery_seq;std::string error;J expected,actual;bool full_match=true,match=true;if(audit){expected=canonical(public_state(*e));actual=canonical(packet["before"]);full_match=same(expected,actual);match=same(dynamic_state(expected),dynamic_state(actual),&error);}
            J check={{"seq",seq},{"time",e->time},{"match",match},{"full_dto_match",full_match},{"comparison","All dynamic fields; edge-ray rendering compared separately"},{"checked_ns",now_ns()}};
            if(!match){check["difference"]=error;mismatches++;save_json(dir/"first-mismatch.json",{{"check",check},{"expected",expected},{"actual",actual}});}
            if(!match||seq%100==0){checks<<check.dump()<<'\n';checks.flush();}
            if(!match){std::lock_guard<std::mutex> guard(mu);model_failed=true;latest={{"seq",seq},{"failed",true},{"difference",error}};changed.notify_all();break;}
            if(!packet.value("action_committed",false))oracle_activated=true;bool synchronized=oracle_activated;oracle.resource_time=e->time;oracle.resource_synchronized=synchronized;if(synchronized)oracle.P.child_prio=60.;
            oracle.resource_trees.clear();oracle.resource_fruits.clear();for(auto&t:e->trees)oracle.resource_trees.push_back({{t.x,t.y},t.age,0,e->biome_at(t.x,t.y)});for(auto&f:e->fruits)oracle.resource_fruits.push_back({{f.x,f.y},f.age,f.energy,e->biome_at(f.x,f.y)});
            oracle.model_agents.clear();oracle.model_predators.clear();oracle.model_obstacles.clear();for(auto&a:e->agents)oracle.model_agents.push_back({a.id,{a.x,a.y},a.direction,false,a.max_age});for(auto&a:e->predators)oracle.model_predators.push_back({a.key,{a.x,a.y},a.direction,a.resting});for(auto&o:e->obstacles)oracle.model_obstacles.push_back({o.x,o.y,o.w,o.h});
            Inputs input(packet["before"]);std::map<int64_t,bool> old_heirs;if(synchronized)oracle.minds.each([&](const int64_t&id,orchard::MindP&m){old_heirs[id]=m->heir_done;});auto proposed=oracle.call(std::move(input.states),e->time);if(synchronized){safety.apply(*e,oracle,proposed,144);births.apply(*e,oracle,proposed,144,old_heirs);}
            J proposed_json=J::array();for(auto&a:proposed)proposed_json.push_back({{"agent_id",a.aid},{"move_distance",a.dist},{"move_direction",a.direction},{"turn_angle",a.turn},{"spawn_agent",a.spawn}});
            {std::unique_lock<std::mutex> guard(mu);if(synchronized){model_actions=proposed_json;model_action_seq=seq;changed.notify_all();}changed.wait_for(guard,std::chrono::seconds(10),[&]{return stopped||packets[seq].value("action_committed",false);});packet=packets[seq];}
            std::set<int32_t> old_trees;std::set<int64_t> old_fruits;for(auto& t:e->trees)old_trees.insert(t.key);for(auto& f:e->fruits)old_fruits.insert(f.fruit_id);
            for(auto&a:packet["actions"]){size_t before_agents=e->agents.size();e->agent_step({a.at("agent_id").get<int64_t>(),a.at("move_distance"),true,a.at("move_direction"),a.at("turn_angle"),a.at("spawn_agent")});if(synchronized&&e->agents.size()>before_agents)births.verify(a["agent_id"],e->agents.back());}e->non_agent_step();replay.capture(*e,packet["actions"]);
            J trees=J::array(),fruits=J::array(),poses=J::object();for(auto& t:e->trees)if(!old_trees.count(t.key))trees.push_back({t.x,t.y});
            for(auto& f:e->fruits)if(!old_fruits.count(f.fruit_id))fruits.push_back({f.fruit_id,f.x,f.y});
            for(auto& a:e->agents)poses[std::to_string(a.id)]={a.x,a.y,a.direction};
            int64_t produced=now_ns();J row={{"seq",seq},{"input_time",packet["before"]["sim_time"]},{"predicted_time",e->time},{"produced_ns",produced},
                {"events",{{"time",e->time},{"tree_spawns",trees},{"fruit_spawns",fruits}}},{"expected_agent_poses",poses}};if(seq%100==0)row["expected"]=public_state(*e);
            if(seq%100==0){predictions<<row.dump()<<'\n';predictions.flush();}
            {std::lock_guard<std::mutex> guard(mu);latest_seq=seq;latest={{"seq",seq},{"failed",false},{"time",e->time},{"produced_ns",produced},{"tree_spawns",trees.size()},{"fruit_spawns",fruits.size()}};changed.notify_all();}
            seq++;
        }}catch(const std::exception& error){save_json(dir/"model-exception.json",{{"error",error.what()}});std::lock_guard<std::mutex> guard(mu);model_failed=true;}
        replay.save(*e,dir/"replays/native-model.json","Native model finished; public comparison receipts determine validity");
        save_json(dir/"model-summary.json",{{"seed",seed},{"processed",seq},{"mismatches",mismatches},{"wall_seconds",seconds(begin)},{"final_time",e->time}});
    }
    void search_loop(){
        auto begin=Clock::now();pid_t wp=launch_stream(dir,frozen_samples,0,(1ULL<<32)/6,config_path);
        auto checked=dir/"stream/verified.txt";int status=0;
        std::set<uint32_t> seen;
        while(seconds(begin)<1800){
            {std::lock_guard<std::mutex>g(mu);if(stopped||recovered>=0)break;}
            {std::ifstream f(checked);uint64_t seed;while(f>>seed)if(seen.insert(seed).second){std::lock_guard<std::mutex>g(mu);candidates.push({uint32_t(seed),"master-local-wall-check"});}}
            uint32_t seed=0;std::string worker;bool found=false;
            {std::lock_guard<std::mutex>g(mu);if(!candidates.empty()){auto q=candidates.front();candidates.pop();seed=q.first;worker=q.second;found=true;}}
            if(found){
                auto e=make_engine(seed);bool match=!walls.empty();for(double w:walls){bool hit=false;for(size_t i=4;i<e->obstacles.size();i++)if(std::abs(e->obstacles[i].w-w)<1e-7||std::abs(e->obstacles[i].h-w)<1e-7){hit=true;break;}if(!hit){match=false;break;}}
                if(match){std::lock_guard<std::mutex>g(mu);recovered=seed;recovery_seq=packets.empty()?0:int64_t(packets.size()-1);changed.notify_all();save_json(dir/"recovered.json",{{"seed",seed},{"worker",worker},{"search_seconds",seconds(begin)},{"sim_time",last_time},{"recovery_seq",recovery_seq},{"recovered_ns",now_ns()}});auto self=shared_from_this();std::thread([self,seed,e=std::move(e)]()mutable{self->model_loop(std::move(e),seed);}).detach();}
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        waitpid(wp,&status,0);save_json(dir/"search-summary.json",{{"seconds",seconds(begin)},{"recovered",recovered}});
    }
    J predict(const J& body,int64_t received_ns){
        // One caller holds the HTTP request mutex; background inference has its own lock.
        auto request_start=Clock::now();J actions=policy_actions(policy,body);size_t seq;bool begin=false;
        {std::lock_guard<std::mutex> guard(mu);received=Clock::now();last_time=body["sim_time"];add_samples(samples,freshness.update(body).pose_input);seq=packets.size();
            J packet={{"seq",seq},{"received_ns",received_ns},{"before",body},{"actions",actions},{"action_committed",false}};packets.push_back(packet);if(seq%100==0){packet_log<<packet.dump()<<'\n';packet_log.flush();}
            std::set<int> labels;for(auto [p,b]:samples)labels.insert(b);
            if(!searching&&((samples.size()>=128&&labels.size()>=3)||(samples.size()>=400&&last_time>=180&&labels.size()>=2))){searching=true;frozen_samples=samples_json(samples);begin=true;save_json(dir/"search-input.json",{{"game_id",id},{"sim_time",last_time},{"samples",frozen_samples},{"seed_range",{0,4294967295ULL}}});}
            changed.notify_all();}
        if(begin){auto self=shared_from_this();std::thread([self]{self->search_loop();}).detach();}
        {std::unique_lock<std::mutex> guard(mu);
            if(searching&&recovered<0&&!stopped&&pacer.used_extra_wait<pacer.extra_wait_budget){
                double allowance=std::min(pacer.extra_wait_budget-pacer.used_extra_wait,std::max(0.,pacer.per_response_limit-seconds(request_start)));
                if(allowance>0.){auto pause_start=Clock::now();changed.wait_for(guard,std::chrono::duration<double>(allowance),[&]{return recovered>=0||stopped;});pacer.used_extra_wait+=seconds(pause_start);pacer.paced_responses++;}
            }
        }
        auto wait_start=Clock::now();
        {std::unique_lock<std::mutex> guard(mu);
            if(recovered>=0&&!model_failed)changed.wait_for(guard,std::chrono::seconds(9),[&]{return model_action_seq>=int64_t(seq)||model_failed;});
            bool committed=!model_failed&&model_action_seq==int64_t(seq);if(committed)actions=model_actions;packets[seq]["actions"]=actions;packets[seq]["action_committed"]=true;changed.notify_all();forecast_responses+=committed;
            response_log<<J{{"seq",seq},{"input_time",last_time},{"response_ready_ns",now_ns()},{"forecast_committed",committed},{"model_seq",latest_seq},{"wait_ms",seconds(wait_start)*1000}}.dump()<<'\n';response_log.flush();
            if(seq%100==0)save_json(dir/"server-status.json",status_locked());}
        return {{"actions",actions}};
    }
};

void serve(const fs::path& out,const std::string& token,int port){
    fs::create_directories(out);httplib::Server server;server.set_payload_max_length(64*1024*1024);server.new_task_queue=[](){return new httplib::ThreadPool(4);};
    std::mutex request_mu;std::shared_ptr<Game> game;std::string prefix="/"+token;
    auto reply=[](httplib::Response& res,const J& data){res.set_content(data.dump(),"application/json");};
    server.Get("/health",[&](auto&,auto& res){reply(res,{{"service","native survival validation diagnostic"}});});
    server.Get(prefix+"/status",[&](auto&,auto& res){auto target=std::atomic_load(&game);if(!target){reply(res,{{"waiting",true}});return;}std::lock_guard<std::mutex>state(target->mu);reply(res,target->status_locked());});
    server.Get(prefix+"/worker/([1-5])",[&](auto& req,auto& res){auto target=std::atomic_load(&game);if(!target){reply(res,{{"waiting",true}});return;}std::lock_guard<std::mutex>state(target->mu);
        if(!target->searching){reply(res,{{"waiting",true}});return;}uint64_t shard=std::stoul(req.matches[1]);uint64_t start=(1ULL<<32)*shard/6,end=(1ULL<<32)*(shard+1)/6;reply(res,{{"game_id",target->id},{"initial_public",read_json(target->dir/"initial-public.json")},{"samples",target->frozen_samples},{"start",start},{"count",end-start},{"stop",target->stopped||target->recovered>=0}});});
    server.Post(prefix+"/candidate",[&](auto& req,auto& res){J b=J::parse(req.body);auto target=std::atomic_load(&game);if(!target||b.value("game_id","")!=target->id){reply(res,{{"accepted",false}});return;}
        uint64_t seed=b.at("seed");if(seed>=(1ULL<<32)){res.status=400;return;}std::lock_guard<std::mutex>state(target->mu);target->candidates.push({uint32_t(seed),b.value("worker","remote")});reply(res,{{"accepted",true}});});
    server.Post(prefix+"/finish",[&](auto&,auto& res){auto target=std::atomic_load(&game);if(target)target->finish();reply(res,{{"stopped",true}});});
    server.Post(prefix+"/predict",[&](auto& req,auto& res){int64_t received_ns=now_ns();J b=J::parse(req.body);if(!b.contains("sim_time")){reply(res,{{"actions",J::array()}});return;}
        std::lock_guard<std::mutex> guard(request_mu);
        if(game&&b["sim_time"].get<double>()==game->last_time&&!game->packets.empty()&&game->packets.back()["before"]==b){reply(res,{{"actions",game->packets.back()["actions"]}});return;}
        if(!game||b["sim_time"].get<double>()<game->last_time){if(game)game->finish();std::atomic_store(&game,std::make_shared<Game>(out,b));}
        reply(res,game->predict(b,received_ns));});
    std::cout<<J{{"listening",port},{"runtime","native C++"}}<<std::endl;
    server.listen("0.0.0.0",port);if(game)game->finish();
}

void remote_worker(const fs::path& url_file,const fs::path& out,int threads,int shard){
    auto url=read_text(url_file);while(!url.empty()&&std::isspace(url.back()))url.pop_back();if(url.size()>=8&&url.substr(url.size()-8)=="/predict")url.resize(url.size()-8);
    auto slash=url.find('/',url.find("://")+3);httplib::Client client(url.substr(0,slash));std::string prefix=url.substr(slash);client.set_connection_timeout(10);client.set_read_timeout(20);client.set_follow_location(false);fs::create_directories(out);
    auto start=Clock::now();J task;
    while(seconds(start)<1800){auto r=client.Get((prefix+"/worker/"+std::to_string(shard)).c_str());if(r&&r->status==200){task=J::parse(r->body);if(task.value("stop",false))return;if(!task.value("waiting",false))break;}std::this_thread::sleep_for(std::chrono::seconds(1));}
    if(!task.contains("samples"))return;save_json(out/"task.json",task);save_json(out/"initial-public.json",task["initial_public"]);
    auto native=fs::path(filter_binary).parent_path();auto cfg=native/"../../results/orchard/best-config.json";
    auto wp=launch_stream(out,task["samples"],task["start"],task["count"],cfg.string());int status=0;std::set<uint32_t> sent;
    while(true){std::ifstream f(out/"stream/verified.txt");uint64_t seed;while(f>>seed)if(!sent.count(seed)){auto r=client.Post((prefix+"/candidate").c_str(),J{{"game_id",task["game_id"]},{"seed",seed},{"worker","stream-verified-shard-"+std::to_string(shard)}}.dump(),"application/json");if(r&&r->status==200)sent.insert(seed);}
      if(waitpid(wp,&status,WNOHANG)==wp)break;std::this_thread::sleep_for(std::chrono::milliseconds(200));}
    // A match may have been flushed immediately before child exit.
    std::ifstream f(out/"stream/verified.txt");uint64_t seed;while(f>>seed)if(!sent.count(seed))client.Post((prefix+"/candidate").c_str(),J{{"game_id",task["game_id"]},{"seed",seed},{"worker","stream-verified-shard-"+std::to_string(shard)}}.dump(),"application/json");
}

int main(int argc,char** argv){try{
    if(argc<2)throw std::runtime_error("serve OUT CONFIG FILTER TOKEN_FILE PORT | worker URL_FILE FILTER OUT THREADS SHARD | replay PACKETS SEED OUT CONFIG");
    std::string mode=argv[1];
    if(mode=="worker"){if(argc!=7)throw std::runtime_error("worker arguments");filter_binary=argv[3];remote_worker(argv[2],argv[4],std::stoi(argv[5]),std::stoi(argv[6]));return 0;}
    if(mode=="serve"){if(argc!=7)throw std::runtime_error("serve arguments");config_path=argv[3];policy_config=initialize_math(read_json(argv[3]));filter_binary=argv[4];std::string token=read_text(argv[5]);while(!token.empty()&&std::isspace(token.back()))token.pop_back();serve(argv[2],token,std::stoi(argv[6]));return 0;}
    if(mode=="replay"){if(argc!=6)throw std::runtime_error("replay arguments");policy_config=initialize_math(read_json(argv[5]));uint32_t seed=std::stoul(argv[3]);auto e=make_engine(seed);NativeReplay rec(*e,seed,"Native engine public-history parity check");rec.capture(*e,J::array(),true);advance(*e,J::array());rec.capture(*e);std::ifstream input(argv[2]);std::string line,error;int ticks=0,bad=0,full_bad=0;auto start=Clock::now();
        while(std::getline(input,line)){auto p=J::parse(line);if(!same(public_state(*e),p["before"]))full_bad++;if(!same(dynamic_state(public_state(*e)),dynamic_state(p["before"]),&error)){save_json(fs::path(argv[4])/"first-mismatch.json",{{"expected",public_state(*e)},{"actual",p["before"]}});bad++;break;}advance(*e,p["actions"]);rec.capture(*e,p["actions"]);ticks++;}
        fs::path out=argv[4];rec.save(*e,out/"replays/native-parity.json","C++ comparison to unmodified Python public history");J result={{"ticks",ticks},{"mismatches",bad},{"full_dto_mismatches",full_bad},{"first_difference",error},{"wall_seconds",seconds(start)}};save_json(out/"native-parity.json",result);std::cout<<result<<std::endl;return bad?1:0;}
    throw std::runtime_error("Unknown mode");
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
