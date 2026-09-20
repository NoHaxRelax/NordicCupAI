"""Full-game late activation experiment with an average runtime constraint."""
import pathlib,json,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=json.loads((ROOT/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config']
BASE={**BASE,'share_obs':0.,'stuck_mode':0.,'gate_enabled':0.}
SOURCES=json.loads((ROOT/'docs/late-gated5/source-presets.json').read_text())
NAMES=['congestion_net','cooperative_auction','renewal_relocation','capacity_budget','auction_aging']
TRAIN=list(range(41001,41101));TEST=list(range(43001,45001));PILOT=list(range(42001,42033));ITERATIONS=16
GATE={'gate_ticks':(5000.,24000.),'gate_population':(3.,24.),'gate_logic':(0.,3.),'gate_persistence':(0.,2.)}
EXTRA=[{'crowd_weight':(.05,.8),'fruit_net':(-10.,25.)},{'crowd_weight':(.05,.8),'auction_cost':(.2,3.)},{'renewal_weight':(.2,8.),'relocate_after':(1.,45.)},{'cap_budget':(.1,1.),'budget_reserve':(.1,2.)},{'auction_cost':(.2,3.),'aging_food':(.1,15.)}]
SPACES=[{**GATE,**{'late_'+k:v for k,v in x.items()}}for x in EXTRA]
def preset(*parts):
 c={'share_obs':1.,'econ_start':0.}
 for name,keys in parts:
  c.update({k:SOURCES[name][k] for k in keys if k in SOURCES[name]})
 return c
PRESETS=[preset(('congestion_pricing',['crowd_weight','econ_radius','tree_slots']),('net_energy_fruit',['fruit_net','fruit_reach','rock_penalty'])),preset(('cooperative_harvest',['fruit_auction','crowd_weight','auction_cost','econ_radius'])),preset(('renewal_sites',['renewal_weight','watch_refresh','watch_reach']),('leave_empty_area',['relocate_after','relocate_energy','explore_radius'])),preset(('food_capacity',['cap_budget','econ_horizon']),('budget_breeding',['budget_reserve','econ_radius'])),preset(('aging_efficiency',['fruit_auction','aging_food','auction_cost']),('late_auction',['fruit_net']))]
DEFAULTS={'gate_ticks':14000.,'gate_population':12.,'gate_logic':2.,'gate_persistence':1.}
def initial(i):
 c={**DEFAULTS,**{'late_'+k:v for k,v in PRESETS[i].items()}}
 return [float(np.clip((c.get(k,(lo+hi)/2)-lo)/(hi-lo),0,1))for k,(lo,hi)in SPACES[i].items()]
def config(i,x):
 c={**BASE,'gate_enabled':1.,**DEFAULTS,**{'late_'+k:v for k,v in PRESETS[i].items()}}
 for (k,(lo,hi)),v in zip(SPACES[i].items(),x):
  value=float(lo+(hi-lo)*v);c[k]=round(value)if k in GATE else value
 return c
