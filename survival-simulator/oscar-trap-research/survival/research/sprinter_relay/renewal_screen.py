from run import run,save
import json
runs=[]
for food in ('trees','renewing'):
    for energy in (150,500):
        r=run(policy='orbit',energy=energy,food=food,seconds=180,relay=True,forage=True,newborn_ready=140,switch_energy=120,dwell=10,renew=True)
        runs.append(r);save('renewal-screen',runs)
        print(json.dumps({k:v for k,v in r.items() if k not in ('trace','events')}),flush=True)
