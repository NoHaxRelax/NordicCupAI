#include <cstddef>
#include <cassert>
#include "../models/late_activation.hpp"
int main() {
    for(int logic=0;logic<4;++logic) {
        LateActivation g;g.enabled=true;g.logic=logic;g.after_ticks=2;g.below_population=5;
        assert(g.step(8)==false);assert(g.step(4)==(logic==1||logic==3));
        assert(g.step(8)==(logic==0||logic==3));assert(g.step(4));
    }
    for(int persistence=0;persistence<3;++persistence) {
        LateActivation g;g.enabled=true;g.logic=1;g.persistence=persistence;g.below_population=5;
        assert(!g.step(5));assert(g.step(4));assert(g.step(8)==(persistence==1));
        assert(g.step(4)==(persistence!=2));
    }
    LateActivation off;off.after_ticks=0;assert(!off.step(1));
}
