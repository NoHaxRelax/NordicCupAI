"""Second checkpoint search: 20 ideas seeded from local_food, 32 trials × 200 maps."""
import json
import numpy as np
import tune_late250 as run
run.BASE=json.loads((run.ROOT/'docs/late250/pod-0/local_food-winner.json').read_text())['config']
run.BASE.update(pulse_degrees=0.,pulse_ticks=8,pulse_idle=0,feature_start=0.,feature_population=0.,look_steps=0,look_radius=130.,risk_margin=3.,behind_weight=0.,energy_weight=.3,phase_start=900.,phase_population=0.,late_cap=-1.,late_retire=-1.,late_reach=-1.,late_reserve=-1.,pred_gaze=0.,pred_cone_gate=0.,pred_cone_margin=.1,pred_wall_escape=0.,pred_wall_look=30.,pred_wall_reward=80.)
# Breeding trait weights remain tunable in EVERY family, including future searches.
WEIGHTS={'fit_vision':(.1,2.),'fit_hear':(0,1.5),'fit_energy':(0,2.),'fit_speed':(0,2.)}
IDEAS=[
 ('pulse_scan',{'pulse_degrees':60.},{'pulse_degrees':(10,180),'pulse_ticks':(1,30)}),
 ('idle_pulse',{'pulse_degrees':90.,'pulse_idle':1},{'pulse_degrees':(10,180),'pulse_ticks':(1,40)}),
 ('fast_travel',{}, {'travel_turn':(.25,3.14159),'sweep_rate':(.01,.5)}),
 ('late_pulse',{'pulse_degrees':90.,'feature_start':900.},{'pulse_degrees':(10,180),'pulse_ticks':(1,30),'feature_start':(300,1400)}),
 ('sparse_pulse',{'pulse_degrees':90.,'feature_population':12.},{'pulse_degrees':(10,180),'pulse_ticks':(1,30),'feature_population':(2,25)}),
 ('late_retirement',{'late_retire':80.},{'late_retire':(55,140),'phase_start':(200,1400)}),
 ('late_small_colony',{'late_cap':.3},{'late_cap':(.1,.65),'phase_start':(300,1400),'phase_population':(4,30)}),
 ('late_food_radius',{'late_reach':160.},{'late_reach':(60,500),'phase_start':(300,1400)}),
 ('late_breed_reserve',{'late_reserve':180.},{'late_reserve':(105,300),'phase_start':(300,1400)}),
 ('small_colony_rescue',{'late_reserve':140.,'phase_population':8.},{'phase_population':(2,18),'late_reserve':(105,250),'phase_start':(0,1200)}),
 ('risk_one_tick',{'look_steps':1},{'look_radius':(60,180),'risk_margin':(0,15),'energy_weight':(0,3)}),
 ('risk_two_ticks',{'look_steps':2},{'look_radius':(60,180),'risk_margin':(0,15),'energy_weight':(0,3)}),
 ('behind_predator',{'look_steps':1,'behind_weight':30.,'pred_cone_gate':1},{'behind_weight':(5,100),'look_radius':(70,220),'pred_cone_margin':(0,.5)}),
 ('behind_two_tick',{'look_steps':2,'behind_weight':30.,'pred_gaze':1},{'behind_weight':(5,100),'look_radius':(70,220),'risk_margin':(0,10)}),
 ('wall_risk',{'look_steps':2,'pred_wall_escape':1},{'pred_wall_look':(15,70),'pred_wall_reward':(10,150),'risk_margin':(0,10)}),
 ('late_risk',{'look_steps':2,'feature_start':900.},{'feature_start':(200,1400),'look_radius':(70,180),'risk_margin':(0,10)}),
 ('fast_scan_risk',{'pulse_degrees':90.,'look_steps':2},{'pulse_degrees':(20,180),'pulse_ticks':(2,25),'travel_turn':(.25,3.14159)}),
 ('expanded_local_food',{}, {'fruit_reach':(160,450),'tree_reach':(400,850),'post_radius':(40,120)}),
 ('cluster_local_food',{'cluster_radius':70.},{'cluster_radius':(0,180),'tree_slots':(1,3),'dist_pen':(.1,1)}),
 ('trait_selection',{}, {'heir_slack':(0,.5),'heir_age':(35,80),'select_min_young':(0,12)}),
]
run.FAMILIES=[(name,fixed,{**space,**WEIGHTS}) for name,fixed,space in IDEAS]
run.TRAIN=list(range(14001,14201))
run.INTEGER|={'pulse_ticks','feature_population','phase_population','select_min_young'}
def configuration(i,x):
    _,fixed,ranges=run.FAMILIES[i];c={**run.BASE,**fixed}
    for (k,(lo,hi)),v in zip(ranges.items(),x):
        value=float(lo+(hi-lo)*v);c[k]=int(round(value)) if k in run.INTEGER else value
    return c
run.configuration=configuration
original_continuation=run.continuation
def continuation(sim,cfg,meta):
    sim.pop_events()  # discard events before the training checkpoint
    row=original_continuation(sim,cfg,meta)
    events=sim.pop_events()
    row['predation_deaths']=sum(e[0]=='predator' for e in events)
    row['energy_deaths']=sum(e[0]=='starvation' for e in events)
    return row
run.continuation=continuation
if __name__=='__main__':run.main()
