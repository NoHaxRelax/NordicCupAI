"""Optimistic one-lineage search using original Environment.spawn_agent.
No game time, food search, predation or mortality: optimistic mutation effort estimate,
not a full-game preparation result. Every tested birth still costs 100 energy.
"""
from experiment import fixture,add_agent,save
import statistics,json

def study(trials=1000,limit=500):
    rows=[]
    for target in ('capacity375','walk12','walk12_capacity375','walk15_1','walk12_capacity250'):
        births=[];fail=0;examples=[]
        for seed in range(trials):
            e=fixture(seed,width=100,height=100);parent=add_agent(e,50,50,500)
            def quality(a):
                speed=min(a.speed,a.sprint_speed)
                if a.sprint_speed<18:return -1e6+a.sprint_speed
                if target=='capacity375':return -max(0,a.max_energy-375)
                if target=='walk12':return min(12,speed)
                if target=='walk15_1':return min(15.1,speed)
                cap=250 if target.endswith('250') else 375
                return min(12,speed)/12 + min(1,cap/a.max_energy)
            def reached(a):
                return a.sprint_speed>=18 and (a.max_energy<=375 if target=='capacity375' else min(a.speed,a.sprint_speed)>=12 if target=='walk12' else min(a.speed,a.sprint_speed)>=15.1 if target=='walk15_1' else min(a.speed,a.sprint_speed)>=12 and a.max_energy<=(250 if target.endswith('250') else 375))
            for n in range(1,limit+1):
                # Ideal nursery continually feeds the sole selected parent.
                parent.energy=101
                child=e.spawn_agent(parent=parent);parent.energy-=100
                if quality(child)>quality(parent):
                    e.kill_agent(parent);parent=child
                else:e.kill_agent(child)
                if reached(parent):
                    births.append(n)
                    if seed<3:examples.append(dict(seed=seed,births=n,walk=parent.speed,sprint=parent.sprint_speed,capacity=parent.max_energy,newborn_energy=75,sprint_cutoff=parent.max_energy/5))
                    break
            else:fail+=1
        b=sorted(births)
        rows.append(dict(target=target,trials=trials,censor_births=limit,failed=fail,p10=b[int(.1*(len(b)-1))],median=statistics.median(b),p90=b[int(.9*(len(b)-1))],median_parent_birth_energy=100*statistics.median(b),examples=examples))
        print(json.dumps(rows[-1]),flush=True)
    save('genetics',rows)
if __name__=='__main__':study()
