"""Same arranged native starts and fixed horizon, including water fallback."""
from common import *
from simple_policies import SimplePolicy
from types import SimpleNamespace

class ComparisonPolicy:
    def __init__(self,pair,baseline=False):
        self.pair=pair;self.baseline=SimplePolicy('nursery',0)
        self.fallback_t=0 if baseline else None
        self.decisions={};self.events=pair.events;self.births=pair.births
        self.base=pair.base if not baseline else SimpleNamespace(acquisition_complete=None)

    @property
    def active(self):return self.pair.active

    @property
    def positions(self):return self.pair.positions if self.fallback_t is None else {}

    def __call__(self,states,t):
        alive={s['agent_id'] for s in states}
        if self.fallback_t is None and not set(self.active)<=alive:self.fallback_t=t
        if self.fallback_t is None:
            actions=self.pair(states,t);self.decisions=self.pair.decisions
            return actions
        self.decisions={s['agent_id']:dict(rule='Observation-only nursery after water retirement') for s in states}
        return self.baseline(states,t)

    def integrate(self,actions,states):
        if self.fallback_t is None:self.pair.integrate(actions,states)

if __name__=='__main__':
    from run import run
    maps={m['seed']:m['sites'] for m in json.loads((OUT/'sites-1-40.json').read_text())['data']}
    rows=[]
    for index in [8,10]:
        for baseline in [True,False]:
            c=dict(native=True,seed=37,site=maps[37][index],site_index=index,seconds=150,
                crew=4,relay='reserve',collision_model=True,comparison='nursery' if baseline else 'water',
                stop_on_loss=False)
            r=run(c);rows.append(r)
            print(index,c['comparison'],r.get('elapsed_s'),r.get('score'),r.get('worker_captures'),r.get('alive'),flush=True)
            save('protection-comparison',rows)
