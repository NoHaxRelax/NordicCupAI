#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "seed_random.inc"
#include "seed_batch_fast.hpp"
int main(){SeedBatchFast b; uint32_t seed=0;uint64_t n=0;for(int test=0;test<4096;test++){seed=(seed*1664525U+1013904223U)&0xffffffc0U;b.init(seed);for(int lane=0;lane<b.lanes;lane++){PyRandom r;r.init_by_array({seed+uint32_t(lane)});SeedBatchFast::Lane f{b,lane};for(int i=0;i<b.prefix;i++){if(r.getrandbits(32)!=f.getrandbits(32))return 1;n++;}try{f.getrandbits(32);return 2;}catch(const std::overflow_error&){} }}std::cout<<n<<" exact RNG prefix words; exhaustion guarded\n";}
