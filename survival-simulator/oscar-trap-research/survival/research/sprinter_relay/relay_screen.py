import run_exploratory as runner
from controller_stage2 import RelayController
runner.RelayController=RelayController
run,save=runner.run,runner.save
import json
runs=[]
for food in ('none','finite','trees'):
    for energy in (150,500):
        for relay in (False,True):
            r=run(policy='orbit',energy=energy,food=food,seconds=120,relay=relay,forage=True,newborn_ready=140,switch_energy=120,dwell=10,renew=True)
            runs.append(r);save('relay-screen',runs)
            print(json.dumps({k:v for k,v in r.items() if k not in ('trace','events')}),flush=True)
