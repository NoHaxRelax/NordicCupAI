#define main paced_server_program_main
#include "paced_native_v4.cpp"
#undef main
int main(int argc,char**argv){if(argc!=5)return 2;
 fs::path out=argv[3];fs::create_directories(out);filter_binary=argv[4];config_path=argv[2];policy_config=initialize_math(read_json(argv[2]));
 std::thread([&]{serve(out/"server","stream-integration-test",19126);}).detach();
 httplib::Client c("127.0.0.1",19126);c.set_read_timeout(9);for(int i=0;i<50;i++){if(c.Get("/health"))break;std::this_thread::sleep_for(std::chrono::milliseconds(20));}
 std::ifstream in(argv[1]);std::string line;size_t frames=0;J state;double recovery_response=-1;
 while(std::getline(in,line)){auto p=J::parse(line);auto t=Clock::now();auto r=c.Post("/stream-integration-test/predict",p["before"].dump(),"application/json");double elapsed=seconds(t);frames++;
  if(!r||r->status!=200)break;auto st=c.Get("/stream-integration-test/status");state=J::parse(st->body);
  if(state.value("search_started",false)){recovery_response=elapsed;break;}
 }
 auto st=c.Get("/stream-integration-test/status");state=J::parse(st->body);auto game_dir=out/"server"/state.at("game_id").get<std::string>();
 c.Post("/stream-integration-test/finish","{}","application/json");
 for(int i=0;i<200&&!fs::exists(game_dir/"model-summary.json");i++)std::this_thread::sleep_for(std::chrono::milliseconds(100));
 J model=fs::exists(game_dir/"model-summary.json")?read_json(game_dir/"model-summary.json"):J::object();
 bool passed=state.value("recovered_seed",int64_t(-1))==1854492595&&recovery_response<5.5&&model.value("processed",size_t(0))==frames&&model.value("mismatches",-1)==0;
 J result={{"passed",passed},{"frames",frames},{"recovery_response_seconds",recovery_response},{"state",state},{"model",model},{"scope","Bounded known-seed integration fixture; actual C++ streaming coordinator, HTTP service, early pacing wake and model replay. No competition API; production search remains full uint32."}};
 save_json(out/"result.json",result);std::cout<<result<<std::endl;_exit(passed?0:1);
}
