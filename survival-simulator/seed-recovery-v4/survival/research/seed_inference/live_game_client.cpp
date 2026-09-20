#include "native_common.hpp"

int main(int argc,char**argv){
    if(argc!=7)throw std::runtime_error("live_game_client SEED HOST PORT PATH OUT CONFIG");
    uint32_t seed=std::stoul(argv[1]);std::string host=argv[2],path=argv[4];int port=std::stoi(argv[3]);fs::path out=argv[5];
    initialize_math(read_json(argv[6]));auto e=make_engine(seed);httplib::Client client(host,port);client.set_connection_timeout(10);client.set_read_timeout(10);
    // The competition callback's first actionable packet is the state after the
    // engine's mandatory no-action bootstrap step.
    advance(*e,J::array());
    auto started=Clock::now();double request_seconds=0.,max_request=0.;size_t ticks=0;J body=public_state(*e);
    while(!e->agents.empty()&&e->time<=3000.){
        auto before=Clock::now();auto response=client.Post(path.c_str(),body.dump(),"application/json");double latency=seconds(before);request_seconds+=latency;max_request=std::max(max_request,latency);
        if(!response||response->status!=200)throw std::runtime_error("predict request failed at tick "+std::to_string(ticks));
        auto answer=J::parse(response->body);advance(*e,answer.value("actions",J::array()));ticks++;body=public_state(*e);
        if(ticks%1000==0)save_json(out,{{"seed",seed},{"score",e->score},{"time",e->time},{"population",e->agents.size()},{"ticks",ticks},{"wall_seconds",seconds(started)},{"request_seconds",request_seconds}});
    }
    J result={{"seed",seed},{"score",e->score},{"duration",e->time},{"population",e->agents.size()},{"ticks",ticks},{"wall_seconds",seconds(started)},{"request_seconds",request_seconds},{"max_request_seconds",max_request}};
    save_json(out,result);std::cout<<result<<std::endl;return 0;
}
