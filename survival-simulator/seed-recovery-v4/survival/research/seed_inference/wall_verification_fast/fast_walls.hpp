#pragma once
#ifdef __AVX2__
#include <immintrin.h>
#endif
// Initial obstacle generation only: no Engine construction, entities or ticks.
namespace fast_walls {
constexpr int W=1600,H=1200,WORDS=(H+63)/64;
inline int64_t floor_div(int64_t a,int64_t b){auto q=a/b,r=a%b;return q-(r<0);}
struct Map {
 int W=1600,H=1200; PyRandom rng;
 std::array<std::array<uint64_t,WORDS>,1600> palette3{};
#include "river_exact.hpp"
 void range(int x,int lo,int hi,bool value){
  if(lo>hi)return;auto& row=palette3[x];int a=lo/64,b=hi/64;
  uint64_t lm=~uint64_t(0)<<(lo%64),rm=~uint64_t(0)>>(63-hi%64);
  if(a==b){auto m=lm&rm;if(value)row[a]|=m;else row[a]&=~m;return;}
  if(value){row[a]|=lm;row[b]|=rm;for(int k=a+1;k<b;k++)row[k]=~uint64_t(0);}
  else{row[a]&=~lm;row[b]&=~rm;for(int k=a+1;k<b;k++)row[k]=0;}
 }
 explicit Map(uint32_t seed){
  rng.init_by_array({seed});int px[10],py[10],types[10];
  for(int i=0;i<10;i++){px[i]=rng.randbelow(W);py[i]=rng.randbelow(H);}
  for(auto& v:types)v=rng.randbelow(4);
  // Integer half-plane intersections give each Voronoi site's interval per column.
  // Strict inequality against an earlier site preserves the original tie rule.
  for(int x=0;x<W;x++)for(int i=0;i<10;i++)if(types[i]==0||types[i]==3){
   int64_t lo=0,hi=H-1;
   for(int j=0;j<10&&lo<=hi;j++)if(i!=j){
    int64_t a=2*(py[j]-py[i]);
    int64_t c=int64_t(x-px[i])*(x-px[i])+py[i]*py[i]-int64_t(x-px[j])*(x-px[j])-py[j]*py[j];
    int64_t bound=(j<i?-1:0)-c;
    if(a>0)hi=std::min(hi,floor_div(bound,a));
    else if(a<0)lo=std::max(lo,-floor_div(bound,-a));
    else if(bound<0)hi=-1;
   }
   range(x,int(lo),int(hi),true);
  }
  int se=rng.randbelow(4),ee=rng.randbelow(4);int64_t sx,sy,ex,ey;edge_point(se,sx,sy);edge_point(ee,ex,ey);
  std::vector<std::pair<int64_t,int64_t>> path;river_path(sx,sy,ex,ey,path);
  int64_t radius=rng.randint(20,100),r2=radius*radius;
  if(path.empty())path.push_back({-1,0});
  std::array<int,101> dy{};for(int d=0;d<=radius;d++){int64_t rem=r2-d*d,v=std::sqrt(double(rem));while(v*v>rem)v--;while((v+1)*(v+1)<=rem)v++;dy[d]=v;}
  for(auto p:path)for(int x=std::max<int64_t>(0,p.first-radius);x<=std::min<int64_t>(W-1,p.first+radius);x++){
   int d=dy[std::abs(x-p.first)];range(x,std::max<int64_t>(0,p.second-d),std::min<int64_t>(H-1,p.second+d),false);
  }
 }
 // Count accepted palette choices in MT blocks, retaining the exact final RNG index.
 // Tempering is independent across words; AVX2 is only an acceleration of uint32 operations.
 struct Choices {
  PyRandom& r;std::array<uint64_t,10> a3{},a4{};
  explicit Choices(PyRandom& rng):r(rng){if(r.mti<624)fill();}
  void fill(){
   a3.fill(0);a4.fill(0);
#ifdef __AVX2__
   for(int j=0;j<624;j+=8){
    auto v=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r.mt+j));
    v=_mm256_xor_si256(v,_mm256_srli_epi32(v,11));
    v=_mm256_xor_si256(v,_mm256_and_si256(_mm256_slli_epi32(v,7),_mm256_set1_epi32(0x9d2c5680U)));
    v=_mm256_xor_si256(v,_mm256_and_si256(_mm256_slli_epi32(v,15),_mm256_set1_epi32(0xefc60000U)));
    v=_mm256_xor_si256(v,_mm256_srli_epi32(v,18));
    unsigned m4=(~_mm256_movemask_ps(_mm256_castsi256_ps(v)))&255;
    unsigned m3=(~_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_and_si256(v,_mm256_slli_epi32(v,1)))))&255;
    a3[j/64]|=uint64_t(m3)<<(j%64);a4[j/64]|=uint64_t(m4)<<(j%64);
   }
#else
   for(int j=0;j<624;j++){uint32_t v=r.mt[j];v^=v>>11;v^=(v<<7)&0x9d2c5680U;v^=(v<<15)&0xefc60000U;v^=v>>18;
    a3[j/64]|=uint64_t((v>>30)<3)<<(j%64);a4[j/64]|=uint64_t((v>>31)==0)<<(j%64);}
#endif
  }
  void consume(bool pal3,int n){while(n){
   if(r.mti==624){r.genrand();r.mti=0;fill();}
   auto bits=(pal3?a3:a4)[r.mti/64]&(~uint64_t(0)<<(r.mti%64));int count=__builtin_popcountll(bits);
   if(count<n){n-=count;r.mti=std::min(624,(r.mti/64+1)*64);}
   else{for(int k=1;k<n;k++)bits&=bits-1;r.mti=(r.mti/64)*64+__builtin_ctzll(bits)+1;n=0;}
  }}
 };
 void render(){Choices choices(rng);for(int x=0;x<W;x++)for(int word=0;word<WORDS;word++){
  auto bits=palette3[x][word];int left=std::min(64,H-word*64);
  while(left){bool pal3=bits&1;auto ends=pal3?~bits:bits;int run=ends?std::min(left,__builtin_ctzll(ends)):left;
   choices.consume(pal3,run);left-=run;bits=run==64?0:bits>>run;}
 }}
 std::vector<Obstacle> walls(){render();std::vector<Obstacle> result;for(int i=0;i<W/20;i++){
  double w=rng.uniform(30,100),h=rng.uniform(30,100),x=rng.uniform(0,W-w),y=rng.uniform(0,H-h);result.push_back({x,y,w,h});
 }return result;}
};
inline std::vector<Obstacle> generate(uint32_t seed){Map m(seed);return m.walls();}
}
