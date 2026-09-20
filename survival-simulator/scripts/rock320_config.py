"""Rock-face steering retune: frozen protocol, 320 full maps per trial."""
import json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=json.loads((ROOT/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config']
BASE={**BASE,'share_obs':0.,'stuck_mode':0.}
PREVIOUS=json.loads((ROOT/'docs/selfstuck5/run/frozen-winners.json').read_text())['rock_face_steer']['config']
NAMES=['rock_face_steer']
SPACES=[{'stuck_radius':(100.,240.),'stuck_gap':(75.,140.),'stuck_reward':(10.,150.),'stuck_release':(5.,40.),'stuck_energy':(.2,.75)}]
TRAIN=list(range(21001,21321));TEST=list(range(22001,24001));PILOT=list(range(25001,25017));ITERATIONS=50
def config(i,x):return {**PREVIOUS,**{k:float(lo+(hi-lo)*v)for(k,(lo,hi)),v in zip(SPACES[i].items(),x)}}
def initial(i):return [(PREVIOUS[k]-lo)/(hi-lo) for k,(lo,hi) in SPACES[i].items()]
