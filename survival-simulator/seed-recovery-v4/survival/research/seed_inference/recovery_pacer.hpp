#pragma once
#include <chrono>
#include <thread>
#include <algorithm>
// Deliberate response pacing: buy real search time while advancing only one game tick.
// Production rule: <10s per response, <1200s aggregate response wait.
// This accounts for local elapsed response time only; network overhead needs separate reserve.
struct RecoveryPacer {
 double extra_wait_budget=240., per_response_limit=6., used_extra_wait=0.;
 size_t paced_responses=0;
 double pace(std::chrono::steady_clock::time_point request_start,bool evidence_ready,bool recovering,bool synchronized){
  if(!evidence_ready||!recovering||synchronized||used_extra_wait>=extra_wait_budget)return 0.;
  double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-request_start).count();
  double allowance=std::min(extra_wait_budget-used_extra_wait,std::max(0.,per_response_limit-elapsed));
  if(allowance<=0.)return 0.;
  auto start=std::chrono::steady_clock::now();std::this_thread::sleep_for(std::chrono::duration<double>(allowance));
  double actual=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();used_extra_wait+=actual;paced_responses++;return actual;
 }
};
