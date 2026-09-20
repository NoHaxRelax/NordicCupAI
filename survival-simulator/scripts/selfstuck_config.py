"""Prespecified full-game self-stuck campaign; no confinement oracle inputs."""
import json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=json.loads((ROOT/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config']
BASE={**BASE,'share_obs':0.,'stuck_mode':0.}
NAMES=['boundary_steer','rock_face_steer','rock_corner_escape','preserve_stationary','adaptive_stuck']
SPACE={'stuck_radius':(100.,240.),'stuck_gap':(75.,140.),'stuck_reward':(10.,150.),'stuck_release':(5.,40.),'stuck_patience':(5.,60.),'stuck_energy':(.2,.75)}
DEFAULT={'stuck_radius':180.,'stuck_gap':105.,'stuck_reward':70.,'stuck_release':20.,'stuck_patience':20.,'stuck_energy':.4}
TRAIN=list(range(21001,21151));TEST=list(range(22001,24001));PILOT=list(range(25001,25017))
ITERATIONS=50

SPACES=[{k:v for k,v in SPACE.items() if k!='stuck_patience'} for _ in range(3)]+[{k:SPACE[k] for k in ('stuck_radius','stuck_reward','stuck_patience')},SPACE]

def config(i,x):
 return {**BASE,'stuck_mode':i+1,**DEFAULT,**{k:float(lo+(hi-lo)*v) for (k,(lo,hi)),v in zip(SPACES[i].items(),x)}}
def initial(i):return [(DEFAULT[k]-lo)/(hi-lo) for k,(lo,hi) in SPACES[i].items()]
