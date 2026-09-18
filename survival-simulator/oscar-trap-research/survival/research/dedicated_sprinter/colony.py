"""Small-caste deployment prototype with one observed bait per predator cluster.
Builds temporary common frames from Agent DTO distance/angle/rel_dir links.
Disconnected components can see the same predator without recognising it; this
is an explicit observability limit. No hidden IDs, map, energy or positions.
Workers retain existing nursery foraging and reproduction. No proactive handoff.
"""
import math
import numpy as np
from controller import unit,wrap
from importlib import import_module
import os
BaitController=import_module(os.environ.get("DEDICATED_CONTROLLER","controller")).BaitController
from simple_policies import SimplePolicy

class CastePolicy:
    def __init__(self,seed=0,breeding=False):
        self.base=SimplePolicy('nursery',seed);self.baits={};self.last_seen={}
        self.stats=dict(assignments=0,bait_action_ticks=0,conflict_ticks=0,peak_baits=0)
        self.breeding=breeding;self.last_decisions={}
    def __call__(self,states,t):
        byid={s['agent_id']:s for s in states}
        base=dict(self.base(states,t));self.last_decisions={}
        for aid in list(self.baits):
            if aid not in byid:self.baits.pop(aid);self.last_seen.pop(aid,None)
        poses={};components={};clusterlist=[]
        for root in sorted(byid):
            if root in poses:continue
            poses[root]=(np.zeros(2),0.);queue=[root];components[root]=root
            while queue:
                aid=queue.pop();xy,theta=poses[aid]
                for o in byid[aid]['observations']:
                    if o['type']!='Agent' or o['id'] not in byid or o['id'] in poses:continue
                    other=o['id'];pos=xy+o['distance']*unit(theta+o['angle'])
                    facing=wrap(theta+o['angle']+math.pi-o['rel_dir'])
                    poses[other]=(pos,facing);components[other]=root;queue.append(other)
        for aid,s in byid.items():
            xy,theta=poses[aid]
            for p in [o for o in s['observations'] if o['type']=='Predator']:
                pt=xy+p['distance']*unit(theta+p['angle'])
                existing=next((c for c in clusterlist if c['component']==components[aid] and np.linalg.norm(c['point']-pt)<12),None)
                if existing is None:
                    existing=dict(component=components[aid],point=pt,candidates={});clusterlist.append(existing)
                existing['candidates'][aid]=p['distance']
        chosen=set()
        for cluster in clusterlist:
            if len(chosen)>=min(2,len(states)//3):break
            candidates=cluster['candidates']
            old=[aid for aid in candidates if aid in self.baits and aid not in chosen]
            if old:aid=min(old,key=lambda q:candidates[q])
            else:
                eligible=[aid for aid,d in candidates.items() if aid not in chosen and d<120 and byid[aid]['sprint_speed']>=18 and byid[aid]['energy']>byid[aid]['max_energy']/5+70 and byid[aid]['biome'] in ('forest','grassland','desert')]
                if not eligible:continue
                # A specialist is local and prepared. Faster cheap walking helps;
                # do not conscript a remote worker merely because it has a mutation.
                aid=min(eligible,key=lambda q:candidates[q]-2*min(byid[q]['speed'],15))
                self.baits[aid]=BaitController();self.stats['assignments']+=1
                self.stats.setdefault('preparation',[]).append(dict(t=t,agent_id=aid,energy=byid[aid]['energy'],age=byid[aid]['age'],walk=byid[aid]['speed'],sprint=byid[aid]['sprint_speed'],capacity=byid[aid]['max_energy']))
            chosen.add(aid);self.last_seen[aid]=t
        for aid in list(self.baits):
            if t-self.last_seen.get(aid,t)>2:
                self.baits.pop(aid);continue
            if len(chosen)<min(2,len(states)//3):chosen.add(aid)
            elif aid not in chosen:self.baits.pop(aid)
        self.stats['peak_baits']=max(self.stats['peak_baits'],len(chosen))
        for aid in chosen:
            action,why=self.baits[aid].step(byid[aid],t);base[aid]=action
            self.last_decisions[aid]=why;self.stats['bait_action_ticks']+=1
        # Optional spare-slot specialist bias. Retain emergency births, worker
        # access to food, and unchanged ordinary foraging. Disabled by default.
        if self.breeding and not self.baits and len(states)>=3:
            births=[s for s in states if base[s['agent_id']].spawn_agent]
            if births:
                best=max(states,key=lambda s:min(s['speed'],s['sprint_speed'])+.002*(500-s['max_energy']))
                if best in births:
                    for s in births:
                        if s is not best:base[s['agent_id']].spawn_agent=False
        return list(base.items())
