#include "../models/native_harvest.hpp"
#include <cassert>
using namespace native_harvest;
int main() {
    assert(minimum_drain(100,3000)==180406);
    assert(will_visit({{1,-1},{2,100},{3,100}},3));
    assert(!will_visit({{1,-1},{2,100}},2));
    assert(!will_visit({{1,-1},{2,-1},{3,-1},{4,100}},4));
    std::vector<Obs> obs(1);obs[0].type=2;obs[0].distance=30;obs[0].rel_dir=0;
    std::vector<AState> states{{1,&obs,200,GRASSLAND,.1,10,20,60,1,200,400}};
    Memory memory;memory.update(states,.1);auto key=memory.tracks[1][0].key;
    memory.certify({key},3000,.1);memory.record(states,{});
    states[0].age=.2;memory.update(states,.2);assert(!Memory::sleeping(memory.tracks[1][0],.2));
    memory.record(states,{});states[0].age=.3;memory.update(states,.3);
    assert(Memory::sleeping(memory.tracks[1][0],.3));
    memory.record(states,{});states[0].age=.4;obs[0].distance=15;memory.update(states,.4);
    assert(memory.tracks[1][0].moving && !Memory::sleeping(memory.tracks[1][0],.4));
    // Closing range caused by observer motion is insufficient to commit.
    Memory moving;obs[0].distance=30;states[0].age=.1;moving.update(states,.1);
    moving.record(states,{{1,10,0,0,false}});obs[0].distance=20;states[0].age=.2;moving.update(states,.2);
    assert(!moving.tracks[1][0].moving && !Memory::sleeping(moving.tracks[1][0],.2));
    // Unknown stationary predators must remain visible to the survival policy.
    Controller controller;obs[0].distance=30;states[0].age=.1;
    auto& filtered=controller.observe(states,.1,0);assert(filtered[0].obs->size()==1);
    auto out=controller.actions(states,.1,{});assert(out.empty());
}
