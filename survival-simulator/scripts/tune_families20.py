"""Second campaign: ten new economy combinations, 60 trials, fresh 64/200 split."""
import json,pathlib
import numpy as np
import tune_families10 as run

ROOT=run.ROOT
ORIGINAL=dict(run.BASE)
def previous(name):
    return json.loads((ROOT/f'docs/families10/{name}/winner.json').read_text())['config']
POP=previous('population'); HARVEST=previous('harvest'); EXPLORE=previous('exploration')
run.BASE=dict(POP)
run.FAMILIES=[
 ('expanded_population','Extend population beyond previous search boundary',{},
  {'cap_mult':(.35,1.2),'cap_max':(25,70),'cap_min':(2,8),'tree_half':(350,1400),'cap_tree_slack':(0,6)}),
 ('population_harvest','Combine population and efficient harvesting',{},
  {'cap_mult':(.3,1.),'tree_half':(350,1100),'ripen_wait':(8,30),'fruit_reach':(100,300),'dist_pen':(.1,.8),'rot_margin':(34,49)}),
 ('population_explore','Combine population and exploration',{},
  {'cap_mult':(.3,1.),'explore_energy':(70,240),'explore_radius':(180,600),'watch_patience':(15,90),'switch_gain':(20,150),'min_stay':(5,50)}),
 ('clustered_orchards','Prefer clusters with room for multiple residents',{'cluster_radius':80.,'tree_slots':2},
  {'cluster_radius':(20,160),'tree_slots':(1,3),'post_radius':(15,65),'dist_pen':(.1,.7),'cap_mult':(.3,1.)}),
 ('distributed_orchards','Spread population among productive patches',{'spread_weight':.5},
  {'spread_weight':(.05,2.),'tree_reach':(200,700),'lone_reach_mult':(1,2.5),'repost_every':(3,25),'switch_gain':(20,180)}),
 ('ripeness_reserve','Wait for ripe fruit with a hunger escape threshold',{},
  {'ripen_wait':(10,38),'fruit_min_wait':(0,6),'hungry_margin':(2,45),'rot_margin':(35,49),'fruit_reach':(100,300)}),
 ('mobile_harvest','Relocate quickly when orchard yield falls',{},
  {'repost_every':(1,15),'switch_gain':(0,100),'min_stay':(0,30),'tree_reach':(180,650),'dist_pen':(.1,.8),'watch_refresh':(10,90)}),
 ('scheduled_breeding','Population plus time-varying reproductive reserves',{},
  {'breed_reserve':(120,320),'breed_reserve_late':(110,300),'reserve_t0':(100,800),'reserve_t1':(900,2000),'low_pop_reserve':(105,220),'births_per_tick':(1,5)}),
 ('trait_succession','Population plus young-agent trait selection and nursery food',{'nursery_bonus':20.},
  {'heir_age':(40,80),'heir_reserve':(150,320),'fit_vision':(.3,2.),'fit_energy':(0,1.5),'fit_speed':(0,1.5),'nursery_bonus':(0,80)}),
 ('economy_gaze','Population and local harvesting with watched-predator escape',{'pred_gaze':1},
  {'pred_r':(65,150),'pred_sprint_r':(30,100),'pred_dodge_ang':(.7,1.7),'pred_turn_max':(.2,1.5),'dist_pen':(.1,.8),'fruit_reach':(100,300)}),
]
run.TRAIN=list(range(8001,8065));run.TEST=list(range(9001,9201))
run.ITERATIONS=60;run.STARTUP=10;run.RNG_SEED=192000;run.DEADLINE=5400
run.EXTRA_SOURCES=[pathlib.Path(__file__)]+[ROOT/f'docs/families10/{n}/winner.json'for n in ('population','harvest','exploration')]
run.CONTROLS={0:{'baseline':POP},1:{'original_baseline':ORIGINAL}}
INTEGERS={'tree_slots','births_per_tick','cap_max','cap_min','cap_tree_slack'}

def seed_config(i):
    c={**POP,**run.FAMILIES[i][2]}
    if i in (1,5,6):
        c.update({k:HARVEST[k]for k in ('ripen_wait','fruit_reach','dist_pen','rot_margin')})
    if i==2:
        c.update({k:EXPLORE[k]for k in run.FAMILIES[i][3]if k!='cap_mult'})
    return c

def initial(i):
    c=seed_config(i)
    return np.array([np.clip((c[k]-lo)/(hi-lo),0,1)for k,(lo,hi)in run.FAMILIES[i][3].items()])

def config(i,x):
    c=seed_config(i)
    for (k,(lo,hi)),v in zip(run.FAMILIES[i][3].items(),x):
        value=float(lo+(hi-lo)*v);c[k]=int(round(value))if k in INTEGERS else value
    if i==9:c['pred_dodge_r']=c['pred_r']
    return c

run.initial=initial;run.config=config
if __name__=='__main__':run.main()
