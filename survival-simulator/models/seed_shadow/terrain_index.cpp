// Offline landmark fingerprints. Query input must come from exact public pixels.
// Candidate matches are not verified seeds. No simulation or private state input.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include "seed_random.inc"
#ifdef __AVX2__
#include "seed_batch.hpp"
#endif
constexpr int landmarks[16][2]={{200,200},{1400,1000},{1400,200},{200,1000},
 {800,200},{800,1000},{200,600},{1400,600},{500,400},{1100,800},
 {1100,400},{500,800},{800,400},{800,800},{500,600},{1100,600}};
// Format is explicitly little endian, independent of compiler struct padding.
void put(std::ostream& out,uint64_t n,int bytes){for(int i=0;i<bytes;i++)out.put(char(n>>(i*8)));}
uint64_t get(std::istream& in,int bytes){uint64_t n=0;for(int i=0;i<bytes;i++){int c=in.get();if(c<0)throw std::runtime_error("Truncated index");n|=uint64_t(c)<<(i*8);}return n;}
template<class R> uint32_t fingerprint(R& r){
 int x[10],y[10],label[10];for(int i=0;i<10;i++){x[i]=r.randbelow(1600);y[i]=r.randbelow(1200);}for(auto& v:label)v=r.randbelow(4);
 uint32_t result=0;for(int j=0;j<16;j++){int best=0,dmin=INT32_MAX;for(int i=0;i<10;i++){int dx=x[i]-landmarks[j][0],dy=y[i]-landmarks[j][1],d=dx*dx+dy*dy;if(d<dmin){dmin=d;best=i;}}result|=uint32_t(label[best])<<(2*j);}return result;
}
uint32_t scalar(uint32_t seed){PyRandom r;r.init_by_array({seed});return fingerprint(r);}
int main(int argc,char** argv){try{
 if(argc<2)throw std::runtime_error("build OUTPUT START END | query INDEX CONSTRAINTS");
 auto t=std::chrono::steady_clock::now();std::string mode=argv[1];uint64_t tested=0,hits=0;
 if(mode=="build"){
  if(argc!=5)throw std::runtime_error("build OUTPUT START END");
  std::filesystem::path path=argv[2],tmp=path.string()+".partial";
  if(std::filesystem::exists(path)||std::filesystem::exists(tmp))throw std::runtime_error("Refusing to overwrite index or partial file");
  uint64_t begin=std::stoull(argv[3]),end=std::stoull(argv[4]);if(begin>=end||end>(uint64_t(1)<<32))throw std::runtime_error("Invalid seed range");
  std::ofstream out(tmp,std::ios::binary);out.exceptions(std::ios::failbit|std::ios::badbit);
  out.write("SEEDIDX1",8);put(out,begin,8);put(out,end,8);for(auto& p:landmarks){put(out,p[0],4);put(out,p[1],4);}
  uint64_t seed=begin;
#ifdef __AVX2__
  SeedBatch8 batch;for(;seed+8<=end;seed+=8){batch.init(uint32_t(seed));for(int lane=0;lane<8;lane++){SeedBatch8::Lane r{batch,lane};uint32_t value;try{value=fingerprint(r);}catch(const std::overflow_error&){value=scalar(uint32_t(seed+lane));}put(out,value,4);}}
#endif
  for(;seed<end;seed++)put(out,scalar(uint32_t(seed)),4);
  out.close();std::filesystem::rename(tmp,path);tested=end-begin;
 }else if(mode=="query"){
  if(argc!=4)throw std::runtime_error("query INDEX CONSTRAINTS (landmark_index label, labels 0..3; omitted landmarks are wildcards)");
  uint32_t mask=0,value=0;int id,label;std::ifstream constraints(argv[3]);
  std::string line;while(std::getline(constraints,line)){if(line.find_first_not_of(" \t\r")==std::string::npos)continue;std::istringstream row(line);std::string extra;if(!(row>>id>>label)||(row>>extra))throw std::runtime_error("Malformed constraint line");if(id<0||id>=16||label<0||label>=4)throw std::runtime_error("Invalid landmark constraint");uint32_t bits=3U<<(2*id),v=uint32_t(label)<<(2*id);if((mask&bits)&&((value&bits)!=v))throw std::runtime_error("Contradictory labels");mask|=bits;value|=v;}
  if(!mask||!constraints.eof())throw std::runtime_error("Missing or malformed constraints");
  std::ifstream in(argv[2],std::ios::binary);char magic[8]{};in.read(magic,8);if(std::string(magic,8)!="SEEDIDX1")throw std::runtime_error("Invalid index magic");
  uint64_t begin=get(in,8),end=get(in,8);if(begin>=end||end>(uint64_t(1)<<32))throw std::runtime_error("Invalid index domain");
  for(auto& p:landmarks)if(get(in,4)!=uint32_t(p[0])||get(in,4)!=uint32_t(p[1]))throw std::runtime_error("Index landmark mismatch");
  if(std::filesystem::file_size(argv[2])!=152+4*(end-begin))throw std::runtime_error("Index size mismatch");
  std::array<unsigned char,1<<20> buffer;
  for(uint64_t start=begin;start<end;){size_t count=std::min<uint64_t>(buffer.size()/4,end-start);in.read(reinterpret_cast<char*>(buffer.data()),count*4);if(!in)throw std::runtime_error("Index read failed");for(size_t i=0;i<count;i++){auto* b=&buffer[4*i];uint32_t f=uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);if((f&mask)==value){std::cout<<start+i<<'\n';hits++;}}start+=count;}tested=end-begin;
 }else throw std::runtime_error("Unknown mode");
 double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
 std::cerr<<"{\"mode\":\""<<mode<<"\",\"tested\":"<<tested<<",\"hits\":"<<hits<<",\"seconds\":"<<seconds<<"}\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
