#include "policy_abi.hpp"
using namespace polabi;
#include "_orchard.hpp"
#include "_evasion.hpp"
#include <cassert>
#include <iostream>
int main(){
 using namespace orchard;
 Params p;p.share_obs=1.;PredParams pp;EvasionPolicy pol({0},p,pp);
 Group a,b;a.walls={{{0,0},{43.2,0},0},{{0,0},{0,67.8},0}};
 double angle=.7;P2 shift{300,400};for(auto e:a.walls){e.a=add(rot(e.a,angle),shift);e.b=add(rot(e.b,angle),shift);b.walls.push_back(e);}
 double th=0;P2 off{};assert(pol.stone_transform(a,b,th,off));assert(std::abs(th-angle)<1e-8&&dist(off,shift)<1e-8);
 Group single;single.walls={a.walls[0]};assert(!pol.stone_transform(single,b,th,off));
 auto copy=b.walls;for(auto e:copy){e.a.x+=500;e.b.x+=500;b.walls.push_back(e);}assert(!pol.stone_transform(a,b,th,off));
 auto g=pol.new_group();
 for(int64_t id:{1,2}){auto m=std::make_shared<Mind>();m->aid=id;m->group=g->id;m->pose=mkpose({(id-1)*100.,0.},0.);pol.minds.set(id,m);g->agents.add(id);}
 Obs predator{};predator.type=2;predator.distance=150;predator.angle=0;predator.rel_dir=0;predator.has_rel_dir=true;
 std::vector<Obs> own{predator},empty;
 AState one{1,&own,500,0,0,10,20,50,OPI/3,200,500},two{2,&empty,500,0,0,10,20,50,OPI/3,200,500};
 pol.states={one,two};pol.sidx={{1,0},{2,1}};pol.shared_reports();assert(g->predators.size()==1);assert(std::abs(g->predators[0].heading-OPI)<1e-8||std::abs(g->predators[0].heading+OPI)<1e-8);
 pol.act(pol.M(2),two);assert(pol.flee_ticks==1);assert(empty.empty());
 std::cout<<"PASS: rigid map transform, reject one-wall/ambiguous matches, unseen-by-self predator shared, original observations unchanged\n";
}
