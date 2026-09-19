"""Shared-information/economic policies from expanded-local-food checkpoints."""
import json
import numpy as np
import tune_late250 as run
run.BASE=json.loads((run.ROOT/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config']
WEIGHTS={'fit_vision':(.1,2.),'fit_hear':(0,1.5),'fit_energy':(0,2.),'fit_speed':(0,2.)}
IDEAS=[
 ('shared_alerts',{}, {'pred_r':(55,110),'pred_sprint_r':(25,70)}),
 ('shared_one_tick',{'look_steps':1}, {'look_radius':(70,180),'risk_margin':(0,8),'energy_weight':(0,2)}),
 ('food_capacity',{'cap_budget':.5}, {'cap_budget':(.1,1.),'econ_horizon':(15,70),'cap_max':(15,60)}),
 ('late_food_capacity',{'cap_budget':.5,'econ_start':900.}, {'cap_budget':(.1,1.),'econ_horizon':(15,70),'econ_start':(300,1500)}),
 ('fruit_auction',{'fruit_auction':1.}, {'auction_cost':(.2,3.),'fruit_reach':(150,450)}),
 ('net_energy_fruit',{'fruit_net':0.}, {'fruit_net':(-10,25),'fruit_reach':(150,500),'rock_penalty':(0,1)}),
 ('congestion_pricing',{'crowd_weight':.3}, {'crowd_weight':(.05,.8),'econ_radius':(70,250),'tree_slots':(1,3)}),
 ('safe_food',{'food_risk':100.,'fruit_auction':1.}, {'food_risk':(20,300),'auction_cost':(.2,3.),'tree_reach':(350,800)}),
 ('rock_aware_posts',{'post_opt':1.,'rock_penalty':1.}, {'rock_penalty':(.1,3.),'min_stay':(0,25),'repost_every':(2,20)}),
 ('fruit_centroid_posts',{'post_opt':1.}, {'fruit_reach':(100,350),'tree_reach':(350,800)}),
 ('leave_empty_area',{'relocate_after':15.}, {'relocate_after':(1,45),'relocate_energy':(30,160),'explore_radius':(150,650)}),
 ('renewal_sites',{'renewal_weight':2.}, {'renewal_weight':(.2,8.),'watch_refresh':(5,60),'watch_reach':(150,650)}),
 ('budget_breeding',{'budget_reserve':1.}, {'budget_reserve':(.1,2.),'econ_horizon':(15,70),'econ_radius':(80,250)}),
 ('aging_efficiency',{'fruit_auction':1.,'aging_food':5.}, {'aging_food':(.1,15),'auction_cost':(.2,3.),'heir_age':(40,85)}),
 ('late_auction',{'fruit_auction':1.,'econ_start':900.}, {'econ_start':(300,1500),'auction_cost':(.2,3.),'fruit_net':(-10,20)}),
 ('late_positioning',{'post_opt':1.,'rock_penalty':1.,'econ_start':900.}, {'econ_start':(300,1500),'rock_penalty':(.1,3.),'watch_patience':(5,60)}),
 ('cooperative_harvest',{'fruit_auction':1.,'crowd_weight':.3}, {'crowd_weight':(.05,.8),'auction_cost':(.2,3.),'econ_radius':(70,220)}),
 ('safe_harvest',{'fruit_auction':1.,'look_steps':1,'food_risk':100.}, {'food_risk':(20,250),'look_radius':(90,180),'auction_cost':(.2,3.)}),
 ('mobile_colony',{'relocate_after':15.,'renewal_weight':2.,'rock_penalty':.5}, {'relocate_after':(1,40),'renewal_weight':(.2,6.),'watch_patience':(5,60)}),
 ('scarcity_economy',{'cap_budget':.4,'budget_reserve':.5,'fruit_auction':1.}, {'cap_budget':(.1,.9),'budget_reserve':(.1,1.5),'econ_horizon':(15,70)}),
]
run.FAMILIES=[(name,{'share_obs':1.,**fixed},{**space,**WEIGHTS}) for name,fixed,space in IDEAS]
run.TRAIN=list(range(17001,17201))
run.ALLOW_SHORT=True  # keep prespecified seeds; checkpoint at t=0 if lifetime <250s

def configuration(i,x):
    _,fixed,ranges=run.FAMILIES[i];c={**run.BASE,**fixed}
    for (k,(lo,hi)),v in zip(ranges.items(),x):
        value=float(lo+(hi-lo)*v);c[k]=int(round(value)) if k in run.INTEGER else value
    return c
run.configuration=configuration
# Missing optional knobs have explicit initial values for reproducible seeded trials.
DEFAULTS=dict(share_obs=0.,econ_start=0.,econ_radius=180.,econ_horizon=40.,cap_budget=0.,crowd_weight=0.,fruit_auction=0.,auction_cost=1.,fruit_net=-1e9,food_risk=0.,post_opt=0.,rock_penalty=0.,relocate_after=0.,relocate_energy=70.,renewal_weight=0.,budget_reserve=0.,aging_food=0.)
run.BASE.update(DEFAULTS)
original_continuation=run.continuation
def continuation(sim,cfg,meta):
    sim.pop_events()
    r=original_continuation(sim,cfg,meta)
    events=sim.pop_events();r['predation_deaths']=sum(e[0]=='predator' for e in events);r['energy_deaths']=sum(e[0]=='starvation' for e in events)
    return r
run.continuation=continuation
if __name__=='__main__':run.main()
