import run_exploratory as runner
from controller_before_relay import RelayController
runner.RelayController=RelayController
run,save=runner.run,runner.save
import math,json
runs=[]
for heading in (0,-math.pi/2,math.pi/2,.6):
    for energy in (150,500):
        r=run(policy='orbit',energy=energy,heading=heading,seconds=60)
        runs.append(r);save('orbit-screen',runs)
        print(json.dumps({k:v for k,v in r.items() if k not in ('trace','events')}),flush=True)
