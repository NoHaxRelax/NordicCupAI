"""Frozen final-controller validation, new seed/geometry split, no tuning."""
from experiment import run,save,compact
import random,math,json

def main():
    rng=random.Random(403971);cases=[];rows=[]
    for k in range(12):
        cases.append(dict(seed=401+k,position=[rng.uniform(400,1150),rng.uniform(350,850)],rotation=rng.uniform(-math.pi,math.pi),heading=rng.uniform(-2,2),gap=rng.uniform(50,85),pred_energy=rng.uniform(40,200),energy=rng.choice([150,250,400]),food='orchard',max_age=rng.uniform(60,120),terrain='split' if k%4==0 else 'desert' if k%4==1 else 'forest',obstacle=k%3==0))
    for c in cases:
        for policy in ('adaptive','nofood','predictive'):
            r=run(c,policy,140);rows.append(r);print(json.dumps(compact(r)),flush=True)
    save('validation-final',rows)
    rows=[]
    for walk,capacity in ((10,500),(10,300),(12,300),(16,500)):
        for energy in (75,150):
            for food in ('none','orchard'):
                c=dict(energy=energy,walk=walk,capacity=capacity,food=food)
                r=run(c,seconds=140);rows.append(r);print(json.dumps(compact(r)),flush=True)
    save('traits-final',rows)
if __name__=='__main__':main()
