#include "native_harvest.hpp"
#include "native_harvest_iface.hpp"
namespace native_harvest_api {
struct Handle { native_harvest::Controller controller; explicit Handle(native_harvest::Config c):controller(c){} };
Handle* create(double horizon,double engage,int min_free,int max_ticks) {
    native_harvest::Config c;c.horizon=horizon;c.engage=engage;c.min_free=min_free;c.max_ticks=max_ticks;
    return new Handle(c);
}
void destroy(Handle* h){delete h;}
const std::vector<polabi::AState>& observe(Handle* h,const std::vector<polabi::AState>& s,double t,double score){return h->controller.observe(s,t,score);}
std::vector<polabi::Act> actions(Handle* h,const std::vector<polabi::AState>& s,double t,const std::vector<polabi::Act>& a){return h->controller.actions(s,t,a);}
Metrics metrics(const Handle* h){auto& c=h->controller;return {c.attempts,c.confirmed,c.failed,c.retries,c.sacrifices,c.memory.ignored,c.total_dupes};}
}
