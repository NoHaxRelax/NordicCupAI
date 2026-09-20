"""Action journal and observation-checked shadow reconstruction.

Only commands and public responses enter the journal. Hidden evaluator state is
never an input. Matching public responses validates a hypothesis, not unseen state.
Any later visible disagreement revokes access to the shadow immediately.
"""
import copy
import json
import math
import time
from pathlib import Path

from .public_terrain import TerrainSamples, agent_states


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k,v in sorted(value.items())}
    if isinstance(value, (list,tuple)):
        rows = [canonical(v) for v in value]
        if rows and all(isinstance(v,dict) for v in rows):
            # Python uses identity-hashed sets to emit observations. Order itself
            # is not evidence of a different world. Coordinates stay ordered.
            rows.sort(key=lambda v: json.dumps(v,sort_keys=True))
        return rows
    return value


def close(a,b,tolerance=1e-8):
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(close(a[k],b[k],tolerance) for k in a)
    if isinstance(a,(tuple,list)):
        if not isinstance(b,(tuple,list)) or len(a)!=len(b):
            return False
        if a and all(isinstance(x,dict) for x in a) and all(isinstance(y,dict) for y in b):
            # Public entity/observation collections are unordered. Sorting by
            # floats is unstable under tolerated roundoff. Find a one-to-one
            # matching instead; preserve multiplicity and every field check.
            if all(close(x,y,tolerance) for x,y in zip(a,b)):
                return True
            def identity(row):
                return (row.get('type'),row.get('id'),row.get('agent_id'))
            buckets={}
            for j,y in enumerate(b):
                buckets.setdefault(identity(y),[]).append(j)
            edges=[]
            for x in a:
                options=[j for j in buckets.get(identity(x),[]) if close(x,b[j],tolerance)]
                if not options:
                    return False
                edges.append(options)
            owners={}
            def assign(i,seen):
                for j in edges[i]:
                    if j in seen:
                        continue
                    seen.add(j)
                    if j not in owners or assign(owners[j],seen):
                        owners[j]=i
                        return True
                return False
            return all(assign(i,set()) for i in range(len(a)))
        return all(close(x,y,tolerance) for x,y in zip(a,b))
    if isinstance(a,float) or isinstance(b,float):
        return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=0,abs_tol=tolerance)
    return a==b


def public_frame(body):
    # Accept both official agent_status and local observations envelope names.
    return canonical({'agents':agent_states(body),
                      **{k:body[k] for k in ('score','sim_time','num_agents') if k in body}})


def commands(actions):
    result=[]
    for aid,action in actions:
        if not isinstance(action,dict):
            action={k:getattr(action,k) for k in ('move_distance','move_direction','turn_angle','spawn_agent')}
        result.append((aid,{k:action[k] for k in ('move_distance','move_direction','turn_angle','spawn_agent')}))
    return result


def step(sim,actions):
    from src.utils.DTOs import ActionRequest
    return sim.step([(aid,ActionRequest(agent_id=aid,**action)) for aid,action in actions])


class ShadowJournal:
    def __init__(self, path=None, deadline_seconds=600):
        self.frames=[]
        self.terrain=TerrainSamples()
        self.path=Path(path) if path else None
        if self.path and self.path.exists():
            raise FileExistsError('Refusing to overwrite an action journal')
        self._shadow=None
        self.seed=None
        self.status='searching'
        self.failure=None
        self.compute_budget=deadline_seconds
        self.started=None
        self.deadline=None

    def begin_search(self):
        """Start the compute clock once observations are ready, before scanning."""
        if self.started is None:
            self.started=time.monotonic()
            self.deadline=self.started+self.compute_budget

    def expired(self):
        if self.seed is None and self.deadline is not None and time.monotonic()>=self.deadline:
            self._shadow=None
            self.status='deadline_exceeded'
            return True
        return False

    def record(self,actions,response):
        """Call once after each real action batch succeeds, including empty batches."""
        frame={'actions':commands(actions),'public':copy.deepcopy(response)}
        self.frames.append(frame)
        self.terrain.observe(response)
        if self.path:
            with self.path.open('a') as f:
                f.write(json.dumps(frame,separators=(',',':'))+'\n')
        if self.expired():
            return
        if self._shadow is not None:
            prediction=step(self._shadow,frame['actions'])
            if not close(public_frame(prediction),public_frame(response)):
                self.failure={'tick':len(self.frames),'reason':'public observation mismatch'}
                self._shadow=None
                self.status='desynchronized'

    def replay_candidate(self,seed,factory):
        """factory creates an independent simulator with the standard game setup."""
        if not self.frames:
            return None
        if self.expired():
            return None
        sim=factory(seed=seed)
        for frame in self.frames:
            if self.expired():
                return None
            actual=public_frame(step(sim,frame['actions']))
            if not close(actual,public_frame(frame['public'])):
                return None
        return sim

    def recover(self,candidates,factory,*,complete_search=False):
        """Accept exactly one replay survivor after the supplied search finishes.

        The caller must document the searched domain. complete_search means that
        domain was exhausted, not that a bounded test covered all uint32 seeds.
        """
        self.begin_search()
        self._shadow=None
        self.seed=None
        survivors=[]
        for seed in dict.fromkeys(candidates):
            if self.expired():
                return None
            sim=self.replay_candidate(seed,factory)
            if sim is not None:
                survivors.append((seed,sim))
                if len(survivors)>1:
                    break
        if self.expired():
            return None
        self.status='ambiguous' if survivors else 'no_match'
        if len(survivors)==1 and complete_search:
            self.seed,self._shadow=survivors[0]
            self.status='public_consistent'
        return self.seed

    @property
    def shadow(self):
        """Hypothesized full state; unavailable after any observable disagreement."""
        if self._shadow is None:
            raise RuntimeError('No public-consistent shadow is available')
        return self._shadow
