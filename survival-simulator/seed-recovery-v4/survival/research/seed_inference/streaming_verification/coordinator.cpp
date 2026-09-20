// Seed-only coordinator. No simulation steps or API calls.
#include "../native_common.hpp"
#include "../wall_verification_fast/fast_walls.hpp"
#include <fcntl.h>
#include <signal.h>
#include <cerrno>
struct Child {
 pid_t pid=-1;bool reaped=false;int status=0;
 ~Child(){if(pid>0&&!reaped){kill(pid,SIGTERM);while(waitpid(pid,&status,0)<0&&errno==EINTR){}}}
 void poll(){if(reaped)return;auto r=waitpid(pid,&status,WNOHANG);if(r==pid)reaped=true;else if(r<0&&errno!=EINTR)throw std::runtime_error("waitpid failed");}
 void wait(){while(!reaped){auto r=waitpid(pid,&status,0);if(r==pid)reaped=true;else if(r<0&&errno!=EINTR)throw std::runtime_error("waitpid failed");}}
 int code()const{return WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);}
};
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("coordinator sequential|stream FILTER TERRAIN INITIAL_PUBLIC CONFIG START COUNT OUTPUT_DIR");
 std::string mode=argv[1];if(mode!="stream"&&mode!="sequential")throw std::runtime_error("Unknown mode");
 fs::path dir=argv[8];if(fs::exists(dir/"candidates.txt"))throw std::runtime_error("Output candidates already exist");fs::create_directories(dir);
 initialize_math(read_json(argv[5]));auto body=read_json(argv[4]);std::vector<double> walls;
 for(auto&a:body["agent_status"])for(auto&o:a["observations"])if(o["type"]=="Edge"){
  double dx=o["coords"][1][0].get<double>()-o["coords"][0][0].get<double>(),dy=o["coords"][1][1].get<double>()-o["coords"][0][1].get<double>(),len=std::hypot(dx,dy);
  if(30.000001<len&&len<99.999999&&std::none_of(walls.begin(),walls.end(),[&](double x){return std::abs(x-len)<1e-7;}))walls.push_back(len);
 }
 if(walls.empty())throw std::runtime_error("No public obstacle dimensions");
 const int filter_threads=mode=="stream"?20:24,verify_threads=mode=="stream"?4:24;
 std::ofstream(dir/"candidates.txt").close();
 std::vector<std::string> args={argv[2],argv[3],argv[6],argv[7],std::to_string(filter_threads),(dir/"candidates.txt").string()};
 std::vector<char*> cargs;for(auto&s:args)cargs.push_back(s.data());cargs.push_back(nullptr);
 int logfd=open((dir/"filter.log").c_str(),O_WRONLY|O_CREAT|O_EXCL,0644);if(logfd<0)throw std::runtime_error("Cannot create filter log");
 auto begin=Clock::now();Child child;child.pid=fork();if(child.pid<0){close(logfd);throw std::runtime_error("fork failed");}
 if(child.pid==0){dup2(logfd,STDOUT_FILENO);dup2(logfd,STDERR_FILENO);close(logfd);execv(cargs[0],cargs.data());_exit(127);}close(logfd);
 double filter_finished=-1;if(mode=="sequential"){child.wait();filter_finished=seconds(begin);}
 struct Candidate {uint32_t seed;double arrived;};std::queue<Candidate> q;std::mutex qm,om;std::condition_variable cv;bool producer_done=false;std::atomic<size_t>checked(0);size_t candidates=0,hits=0;double first_match=-1;
 std::ofstream accepted(dir/"verified.txt"),arrivals(dir/"arrivals.jsonl"),verify_log(dir/"verification.jsonl");
 std::vector<std::thread> workers;for(int i=0;i<verify_threads;i++)workers.emplace_back([&]{while(true){
  Candidate c;{std::unique_lock<std::mutex> l(qm);cv.wait(l,[&]{return producer_done||!q.empty();});if(q.empty())return;c=q.front();q.pop();}
  double started=seconds(begin);auto obstacles=fast_walls::generate(c.seed);bool match=true;
  for(double w:walls){bool found=false;for(auto&b:obstacles)if(std::abs(b.w-w)<1e-7||std::abs(b.h-w)<1e-7){found=true;break;}if(!found){match=false;break;}}
  double finished=seconds(begin);checked++;std::lock_guard<std::mutex> l(om);
  verify_log<<J{{"seed",c.seed},{"candidate_arrived_seconds",c.arrived},{"verify_started_seconds",started},{"verified_seconds",finished},{"wall_match",match}}<<'\n';verify_log.flush();
  if(match){hits++;if(first_match<0)first_match=finished;accepted<<c.seed<<'\n';accepted.flush();std::cout<<J{{"seed",c.seed},{"wall_match",true},{"seconds",finished},{"wall_dimensions",walls.size()}}<<std::endl;}
 }});
 int fd=open((dir/"candidates.txt").c_str(),O_RDONLY);std::string pending;char buf[65536];bool read_failed=false;
 while(true){ssize_t n=read(fd,buf,sizeof(buf));if(n>0){pending.append(buf,n);size_t pos;
   while((pos=pending.find('\n'))!=std::string::npos){auto line=pending.substr(0,pos);pending.erase(0,pos+1);if(line.empty())continue;char*end=nullptr;errno=0;auto seed=strtoull(line.c_str(),&end,10);if(errno||*end||seed>UINT32_MAX){read_failed=true;break;}
    Candidate c{uint32_t(seed),seconds(begin)};arrivals<<J{{"seed",c.seed},{"seconds",c.arrived}}<<'\n';arrivals.flush();candidates++;{std::lock_guard<std::mutex>l(qm);q.push(c);}cv.notify_one();
   }if(read_failed)break;continue;
  }if(n<0&&errno!=EINTR){read_failed=true;break;}
  if(child.reaped)break;child.poll();if(child.reaped){filter_finished=seconds(begin);continue;}
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
 }
 close(fd);{std::lock_guard<std::mutex>l(qm);producer_done=true;}cv.notify_all();for(auto&t:workers)t.join();child.wait();if(filter_finished<0)filter_finished=seconds(begin);
 bool complete=!read_failed&&pending.empty()&&child.code()==0&&checked==candidates;
 save_json(dir/"receipt.json",J{{"mode",mode},{"filter_threads",filter_threads},{"verify_threads",verify_threads},{"filter_pid",child.pid},{"filter_cmd",args},{"filter_exit_code",child.code()},{"filter_finished_seconds",filter_finished},{"first_verified_match_seconds",first_match},{"total_seconds",seconds(begin)},{"candidates",candidates},{"checked",checked.load()},{"hits",hits},{"complete",complete},{"read_failed",read_failed},{"partial_line_bytes",pending.size()},{"simulation_ticks",0}});
 return complete?0:2;
}catch(std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
