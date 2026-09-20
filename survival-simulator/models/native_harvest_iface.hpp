#pragma once
#include "../fastsim/policy_abi.hpp"
namespace native_harvest_api {
struct Handle;
Handle* create(double horizon,double engage,int min_free,int max_ticks);
void destroy(Handle*);
const std::vector<polabi::AState>& observe(Handle*,const std::vector<polabi::AState>&,double,double);
std::vector<polabi::Act> actions(Handle*,const std::vector<polabi::AState>&,double,const std::vector<polabi::Act>&);
struct Metrics { int64_t attempts,confirmed,failed,retries,sacrifices,ignored,dupes; };
Metrics metrics(const Handle*);
}
