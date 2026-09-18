import run_exploratory as runner
from controller_before_relay import RelayController
runner.RelayController=RelayController
run,save=runner.run,runner.save
import json,math
runs=[]
for hw in (.01,.1,1.):
    for angle in (math.pi,math.pi/2,-math.pi/2):
        for aw in (0,100):
            row=run(energy=500,seconds=30,home_weight=hw,reserve_angle=angle,assignment_weight=aw,newborn_ready=130,switch_energy=430,dwell=.5)
            runs.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('trace','events')}),flush=True)
            save('iterate',runs)
