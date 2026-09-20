// Candidate filtering only; a terrain match is NOT a recovered/verified seed.
// PyRandom is extracted at build time from the exact native engine implementation.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "seed_random.inc"
#ifdef __AVX2__
#include "seed_batch.hpp"
#endif
struct Point { int x,y,label; };
template<class Random> bool matches_rng(Random& r,const std::vector<Point>& points) {
    int x[10],y[10],labels[10];
    for(int i=0;i<10;i++){x[i]=r.randbelow(1600);y[i]=r.randbelow(1200);}
    for(auto& label:labels)label=r.randbelow(4);
    for(auto p:points) {
        int best=0,dist=INT32_MAX;
        for(int i=0;i<10;i++) {
            int dx=x[i]-p.x,dy=y[i]-p.y,d=dx*dx+dy*dy;
            if(d<dist){dist=d;best=i;}
        }
        if(labels[best]!=p.label)return false;
    }
    return true;
}
bool matches(uint32_t seed,const std::vector<Point>& points) {
    PyRandom r;r.init_by_array({seed});return matches_rng(r,points);
}
int main(int argc,char** argv) {
    if(argc!=4){std::cerr<<"usage: scan samples.txt start end-exclusive\n";return 2;}
    std::ifstream in(argv[1]);std::vector<Point> points;Point p;
    while(in>>p.x>>p.y>>p.label){
        if(p.x<0||p.x>=1600||p.y<0||p.y>=1200||p.label<0||p.label>3)return 2;
        points.push_back(p);
    }
    if(points.empty()||!in.eof())return 2;
    uint64_t begin=std::stoull(argv[2]),end=std::stoull(argv[3]);
    if(begin>=end||end>(uint64_t(1)<<32))return 2;
    auto start=std::chrono::steady_clock::now();
    uint64_t count=0;
    uint64_t seed=begin;
#ifdef __AVX2__
    SeedBatch8 batch;
    for(;seed+8<=end;seed+=8){
        batch.init((uint32_t)seed);
        for(int lane=0;lane<8;lane++){
            SeedBatch8::Lane r{batch,lane};bool match;
            try{match=matches_rng(r,points);}
            catch(const std::overflow_error&){match=matches((uint32_t)(seed+lane),points);}
            if(match){std::cout<<seed+lane<<'\n';count++;}
        }
    }
#endif
    for(;seed<end;seed++)if(matches(seed,points)){
        std::cout<<seed<<'\n';count++;
    }
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cerr<<"{\"start\":"<<begin<<",\"end\":"<<end<<",\"candidates\":"<<count
             <<",\"seconds\":"<<seconds<<",\"seeds_per_second\":"<<(end-begin)/seconds<<"}\n";
}
