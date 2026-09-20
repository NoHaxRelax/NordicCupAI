// Exact CPython seed initialization and first map nuclei/type draws only.
// Skips river generation and rendering. Non-river labels are necessary filters.
#include "py_random.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <algorithm>
#include <array>
#include <filesystem>
#include <sstream>

struct Sample { int x,y,label; };
uint32_t initial[624];
constexpr int LANES=16;
#ifdef __clang__
using U=uint32_t __attribute__((ext_vector_type(LANES)));
#else
typedef uint32_t U __attribute__((vector_size(LANES*sizeof(uint32_t))));
#endif
U vector_initial[624];

void seed32(PyRandom& rng, uint32_t seed) {
    std::memcpy(rng.mt, initial, sizeof(initial));
    int i=1;
    for (int k=0;k<624;k++) {
        rng.mt[i]=(rng.mt[i]^((rng.mt[i-1]^(rng.mt[i-1]>>30))*1664525U))+seed;
        if (++i>=624) {rng.mt[0]=rng.mt[623];i=1;}
    }
    for (int k=0;k<623;k++) {
        rng.mt[i]=(rng.mt[i]^((rng.mt[i-1]^(rng.mt[i-1]>>30))*1566083941U))-uint32_t(i);
        if (++i>=624) {rng.mt[0]=rng.mt[623];i=1;}
    }
    rng.mt[0]=0x80000000U; rng.mti=624;
}

bool map_matches(const int* xs,const int* ys,const int* types,const std::vector<Sample>& samples) {
    for(auto& s:samples){
        int best=0,dist=2147483647;
        for(int i=0;i<10;i++){
            int dx=xs[i]-s.x,dy=ys[i]-s.y,d=dx*dx+dy*dy;
            if(d<dist){dist=d;best=i;}
        }
        if(types[best]!=s.label)return false;
    }
    return true;
}

struct SeedGroup {
    U mt[624];uint32_t cache[256][LANES];uint32_t seed;int generated;
    void init(uint32_t first) {
        seed=first;generated=0;std::memcpy(mt,vector_initial,sizeof(mt));
        U keys;for(int lane=0;lane<LANES;lane++)keys[lane]=first+lane;
        int i=1;
        for(int k=0;k<624;k++){
            mt[i]=(mt[i]^((mt[i-1]^(mt[i-1]>>30))*1664525U))+keys;
            if(++i>=624){mt[0]=mt[623];i=1;}
        }
        for(int k=0;k<623;k++){
            mt[i]=(mt[i]^((mt[i-1]^(mt[i-1]>>30))*1566083941U))-uint32_t(i);
            if(++i>=624){mt[0]=mt[623];i=1;}
        }
        mt[0]=U{}+0x80000000U;
    }
    uint32_t word(int lane,int index){
        if(index>=256){ // Exact rare fallback, never silently truncate rejection sampling.
            PyRandom rng;seed32(rng,seed+lane);uint32_t value=0;
            for(int i=0;i<=index;i++)value=rng.genrand();return value;
        }
        while(generated<=index){
            int i=generated; // Only the first 256 words are required in the fast path.
            U y=(mt[i]&0x80000000U)|(mt[i+1]&0x7fffffffU);
            y=mt[(i+397)%624]^(y>>1)^((U{}-(y&1U))&0x9908b0dfU);
            mt[i]=y;y^=y>>11;y^=(y<<7)&0x9d2c5680U;y^=(y<<15)&0xefc60000U;y^=y>>18;
            std::memcpy(cache[i],&y,sizeof(y));generated++;
        }
        return cache[index][lane];
    }
    void generate(int lane,int* xs,int* ys,int* types){
        int position=0;
        auto below=[&](int limit,int bits){uint32_t r;do{r=word(lane,position++)>>(32-bits);}while(r>=uint32_t(limit));return int(r);};
        for(int i=0;i<10;i++){xs[i]=below(1600,11);ys[i]=below(1200,11);}
        for(int i=0;i<10;i++)types[i]=below(4,3);
    }
};

int main(int argc,char** argv){
    if(argc!=6 && argc!=7){std::cerr<<"terrain_filter SAMPLES START COUNT THREADS CANDIDATES_OUT [STOP_FILE]\n";return 2;}
    std::vector<std::vector<Sample>> datasets;
    std::stringstream paths(argv[1]);std::string path;
    while(std::getline(paths,path,',')){
        std::ifstream file(path);std::vector<Sample> samples;Sample s;
        while(file>>s.x>>s.y>>s.label)samples.push_back(s);
        if(samples.empty()){std::cerr<<"No terrain samples\n";return 2;}
        datasets.push_back(std::move(samples));
    }
    uint64_t start=std::stoull(argv[2]),count=std::stoull(argv[3]),end=start+count;
    if(end>(1ULL<<32)){std::cerr<<"Range outside uint32\n";return 2;}
    int workers=std::stoi(argv[4]);
    PyRandom base;base.init_genrand(19650218U);std::memcpy(initial,base.mt,sizeof(initial));
    for(int i=0;i<624;i++)vector_initial[i]=U{}+initial[i];
    std::atomic<uint64_t> next(start),done(0),hits(0);std::mutex output_lock;
    std::ofstream output(argv[5]);auto begin=std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for(int w=0;w<workers;w++)threads.emplace_back([&]{
        SeedGroup group;
        while(true){
            if(argc==7 && std::filesystem::exists(argv[6]))break;
            auto a=next.fetch_add(4096);if(a>=end)break;auto b=std::min(a+4096,end);
            for(auto seed=a;seed<b;seed+=LANES){group.init(uint32_t(seed));
                for(int lane=0;lane<LANES && seed+lane<b;lane++){
                  int xs[10],ys[10],types[10];group.generate(lane,xs,ys,types);
                  for(size_t d=0;d<datasets.size();d++)if(map_matches(xs,ys,types,datasets[d])){
                    std::lock_guard<std::mutex> lock(output_lock);
                    if(datasets.size()>1)output<<d<<' ';
                    output<<seed+lane<<'\n';output.flush();hits++;
                  }
                }
            }
            auto previous=done.fetch_add(b-a);
            if((previous>>26)!=((previous+b-a)>>26)){
                double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                std::lock_guard<std::mutex> lock(output_lock);
                std::cout<<"{\"progress\":true,\"tested\":"<<done<<",\"hits\":"<<hits<<",\"seconds\":"<<elapsed<<"}"<<std::endl;
            }
        }
    });
    for(auto& thread:threads)thread.join();
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout<<"{\"tested\":"<<done<<",\"hits\":"<<hits<<",\"seconds\":"<<elapsed
             <<",\"seeds_per_second\":"<<double(done)/elapsed<<",\"threads\":"<<workers<<"}\n";
}
