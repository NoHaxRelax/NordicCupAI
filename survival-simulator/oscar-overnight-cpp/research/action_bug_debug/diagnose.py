"""Burst failure diagnosis. All simulation and policy physics run in native C++.

Python supplies the unchanged Harvester and read-only replay/log capture. Run with
--runtime pointing to the isolated, instrumented nightsim build, never a server.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace
import uuid
import zlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--budget', type=int, default=200000)
    ap.add_argument('--horizon', type=float, default=3000)
    ap.add_argument('--selector', default='contact_nearest')
    ap.add_argument('--margin', type=float, default=1.0)
    ap.add_argument('--trace', action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.runtime))
    import nightsim
    from nightsim.serve.harvest import Harvester
    from debugger.recorder import ReplayRecorder
    source_sha256={p:hashlib.sha256((args.runtime/p).read_bytes()).hexdigest() for p in
                   ('nightsim/serve/harvest.py','nightsim/serve/contact_predictor.py')}

    class Biomes:
        def __init__(self, data, h):
            self.data, self.h = data, h
            self.values = [SimpleNamespace(type=n) for n in nightsim.BIOME_NAMES]
        def __getitem__(self, xy):
            x, y = xy
            return self.values[self.data[x*self.h+y]]

    class View:
        def __init__(self, sim):
            self.sim = sim
            self.width, self.height = sim.env_width, sim.env_height
            self.obstacles = sim.env.obstacles
            self.biome_map = Biomes(sim._engine.biome_map(), self.height)
            self.tree_ids = {}
            self.refresh(sim.state())
        def refresh(self, state):
            env = self.sim.env
            self.time, self.score = env.time, env.score
            self.agents, self.predators = env.agents, env.predators
            self.fruits, self.trees = env.fruits, env.trees
            self.agent_observations = {s['agent_id']:s['observations'] for s in state['observations']}
            for i,p in enumerate(self.predators):
                p.replay_id=i; p.max_energy=200; p.speed=11; p.sprint_speed=15
                p.hearing_radius=60; p.vision_radius=250; p.cone_angle=math.pi/3
            for t in self.trees:
                key=(t.x,t.y)
                if key not in self.tree_ids: self.tree_ids[key]=len(self.tree_ids)
                t.replay_id=self.tree_ids[key]

    class NativeRecorder(ReplayRecorder):
        def _id(self, kind, entity):
            return entity.replay_id
        def _background(self):
            # Exact C++ biome grid with a simple PNG encoder; no simulation draws.
            import numpy as np
            palette=np.array([(38,72,53),(72,81,68),(136,117,80),(75,99,64),(44,92,119)],dtype=np.uint8)
            data=np.frombuffer(self.env.biome_map.data,dtype=np.uint8).reshape(self.env.width,self.env.height)
            rgb=palette[data.T]
            raw=b''.join(b'\0'+row.tobytes() for row in rgb)
            def chunk(name,data):
                return struct.pack('!I',len(data))+name+data+struct.pack('!I',zlib.crc32(name+data)&0xffffffff)
            png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',self.env.width,self.env.height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
            return 'data:image/png;base64,'+base64.b64encode(png).decode()

    sim=nightsim.SimulationCore(seed=args.seed,predators=True)
    eng=sim._engine
    cfg_path=args.runtime/'nightsim/serve/pred_best.json'
    cfg=json.loads(cfg_path.read_text())
    eng.policy_init(nightsim.seed_key(args.seed),cfg)
    hv=Harvester(budget=args.budget,max_harvests=100,sacrifice_mode=args.selector,contact_margin=args.margin)
    args.out.mkdir(parents=True,exist_ok=True)
    name=f'{args.selector}-{args.budget}-s{args.seed}-{uuid.uuid4().hex[:8]}'
    view=View(sim)
    rec=NativeRecorder(view,title=f'Burst diagnosis: {name}',policy='unchanged pred_best + '+args.selector,
        seed=args.seed,every=50,native_render=False,scenario='generated; local C++ diagnostic reproduction',
        notes='State-only fallback for diagnostic batch. C++ physics and policy; Python harvester. Hidden state is logged after decisions, never supplied to policy. Action multiplicity is in burst decisions and traces; replay action fields show only the last action per agent.',
        policy_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest())
    rec.capture(force=True)
    state=sim.step([]); view.refresh(state); rec.capture()
    bursts=[]; active=None; trace_until=-1; admission_ms=[]; progress_at=100.
    t0=time.perf_counter()
    while state['observations'] and state['sim_time'] < args.horizon:
        inputs=state['observations']; action_t=state['sim_time']
        base=[dict(agent_id=a,move_distance=d,move_direction=di,turn_angle=t,spawn_agent=sp)
              for a,d,di,t,sp in eng.policy_act()]
        count=len(hv.log)
        admission_started=time.perf_counter()
        out=hv.apply(inputs,action_t,base,score=state['score'])
        admission_ms.append(1000*(time.perf_counter()-admission_started))
        new=len(hv.log)>count
        if new:
            active=dict(hv.log[-1],decision_t=action_t,seed=args.seed,trace=[],outcome='pending',
                agents_before=eng.agents(),predators_before=eng.predators(),
                farm_input=next(s for s in inputs if s['agent_id']==hv.log[-1]['farm']))
            bursts.append(active)
            trace_until=action_t+.21
            if args.trace: eng.dbg_trace_burst(active['farm'])
        force=new or action_t <= trace_until
        if new: rec.capture(force=True)
        state=sim.step([(a['agent_id'],a) for a in out])
        if active and args.trace:
            active['trace'].extend(json.loads(s) for s in eng.dbg_pop_trace())
        for kind,t,aid,age,en in eng.pop_events():
            if active and aid==active['farm'] and kind!='fruit' and active['outcome']=='pending':
                active.update(outcome=kind,death_t=t,death_energy=en)
        if args.trace and state['sim_time'] > trace_until: eng.dbg_trace_burst(-1)
        view.refresh(state)
        decisions={}
        if force and active:
            decisions[active['farm']]=dict(burst=active['actions'],outcome=active['outcome'],doomed=active['doomed'])
        # The recorder takes a per-agent map; duplicate count stays in decisions.
        last={a['agent_id']:a for a in out}
        rec.capture(list(last.items()),decisions=decisions,inputs=inputs,action_t=action_t,force=force)
        if state['sim_time']>=progress_at:
            print(json.dumps(dict(progress=round(state['sim_time'],1),seed=args.seed,bursts=len(bursts),
                                  outcomes=dict(Counter(b['outcome'] for b in bursts)))),flush=True)
            progress_at+=100.
    summary=dict(seed=args.seed,budget=args.budget,selector=args.selector,
        score=eng.info()['score'],survival=eng.info()['time'],bursts=len(bursts),
        outcomes=dict(Counter(b['outcome'] for b in bursts)),wall=time.perf_counter()-t0,
        trace_enabled=args.trace,run=name,margin=args.margin,
        admission_stats=dict(hv.contact_predictor.stats) if hv.contact_predictor else {},
        public_confirmed=hv.contact_predictor.confirmed if hv.contact_predictor else None,
        admission_ms=dict(mean=sum(admission_ms)/max(1,len(admission_ms)),
                          p99=sorted(admission_ms)[min(len(admission_ms)-1,int(len(admission_ms)*.99))],
                          maximum=max(admission_ms,default=0.)),
        source_sha256=source_sha256)
    rec.save(args.out/'replays'/f'{name}.json.gz',reason='extinction' if not state['observations'] else 'diagnostic horizon')
    (args.out/f'{name}.json').write_text(json.dumps(dict(summary=summary,bursts=bursts),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
