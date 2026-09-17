"""Mapped-site controller. Predator inputs are cached public observations.

Initial shared poses and river frame are provided by the fixture. Optional
collision-aware dead reckoning uses the supplied static obstacle map, not live
agent/predator coordinates. Newborn poses are obtained from Agent observations.
"""
from common import *
from base_snapshot import Config, Scenario, Controller
from types import SimpleNamespace
import numpy as np
from sites import world, local
from src.utils.DTOs import ActionRequest

PENALTY={'river':.3,'swamp':.5,'desert':.8}

class PairPolicy:
    def __init__(self,site,poses,*,outside=0,relay='none',mapped_obstacles=(),food=True,safety_launch=False,active_ids=(0,1),anchor_window=None,association_gate=False):
        self.site=site
        self.positions={k:np.array(v[:2],float) for k,v in poses.items()}
        self.directions={k:v[2] for k,v in poses.items()}
        self.active=list(active_ids)
        self.sides={k:(-1 if v[0]<800 else 1) for k,v in poses.items()}
        self.sides.update(dict(zip(self.active,[-1,1])))
        self.relay=relay;self.food=food
        self.obstacles=mapped_obstacles
        self.pending={};self.events=[];self.decisions={};self.spawned=set()
        self.cooldown={};self.births=[]
        self.safety_launch=safety_launch;self.dogleg=False
        self.c=Config(mode='cross',reserve=18,switch=9,max_move=9,
            lookahead=3,margin=2,track_y=.7,escape=22,face='fixed',observed=True,
            hold_ticks=4,prediction='model')
        self.base=Controller(self.c,Scenario(width=site['width'],outside_distance=outside,
            shore_launch=True),[SimpleNamespace(agent_id=i,x=self.positions[aid][0],
            y=self.positions[aid][1],direction=self.directions[aid]) for i,aid in enumerate(self.active)])
        self.base.anchor_window=anchor_window
        self.base.association_gate=association_gate

    def discover_births(self,states,t):
        # Agent DTO includes id and relative heading. One existing observer
        # suffices to reconstruct the child's local pose with no hidden state.
        for s in states:
            aid=s['agent_id']
            if aid not in self.positions:continue
            for o in s['observations']:
                if o['type']!='Agent' or o['id'] in self.positions:continue
                theta=self.directions[aid]+o['angle']
                p=self.positions[aid]+o['distance']*np.array([math.cos(theta),math.sin(theta)])
                child=o['id'];self.positions[child]=p
                self.directions[child]=wrap(theta+math.pi-o['rel_dir'])
                self.sides[child]=-1 if p[0]<800 else 1
                self.births.append(dict(t=t,id=child,localized_by=aid))

    def action_to(self,s,goal,move=10,turn=None,spawn=False):
        aid=s['agent_id'];d=np.array(goal)-self.positions[aid]
        angle=math.atan2(d[1],d[0]);facing=self.directions[aid]
        request=min(move,float(np.linalg.norm(d))/PENALTY.get(s['biome'],1))
        return ActionRequest(agent_id=aid,move_distance=request,
            move_direction=wrap(angle-facing),turn_angle=0 if turn is None else wrap(turn-facing),spawn_agent=spawn)

    def __call__(self,states,t):
        states={s['agent_id']:s for s in states};self.discover_births(list(states.values()),t)
        self.decisions={}
        B=self.site['width']/2+2
        # Request a replacement early; hand over only when physically close.
        if self.relay!='none' and self.base.acquisition_complete is not None:
            for i,old in enumerate(self.active):
                if old not in states:continue
                s=states[old];side=-1 if i==0 else 1
                needs=s['energy']<60 or s['age']>52
                candidates=[a for a,q in states.items() if a not in self.active
                    and a in self.positions and self.sides[a]==side and
                    q['energy']>45 and min(q['speed'],q['sprint_speed'])>=9
                    and q['age']<55 and self.cooldown.get(a,0)<t]
                if needs and i not in self.pending and candidates:
                    eligible=[a for a in candidates if self.relay!='newborn' or a>=4]
                    if eligible:self.pending[i]=max(eligible,key=lambda a:states[a]['energy'])
                replacement=self.pending.get(i)
                if replacement in states:
                    gap=np.linalg.norm(self.positions[replacement]-self.positions[old])
                    if gap<17 and (s['energy']<50 or s['age']>52):
                        self.active[i]=replacement
                        self.pending.pop(i)
                        self.cooldown[old]=t+10
                        self.events.append(dict(t=t,side=side,old=old,new=replacement,
                            old_energy=s['energy'],new_energy=states[replacement]['energy'],gap=float(gap)))
        actions=[]
        if all(a in states for a in self.active):
            fake={i:dict(states[aid],agent_id=i) for i,aid in enumerate(self.active)}
            proxy=SimpleNamespace(agents=[SimpleNamespace(agent_id=i,sprint_speed=fake[i]['sprint_speed']) for i in [0,1]],
                get_agent_state=lambda i:fake[i])
            for i,aid in enumerate(self.active):
                self.base.positions[i]=self.positions[aid].copy()
                self.base.directions[i]=self.directions[aid]
            aa=self.base.actions(proxy,None,round(t*10))
            if self.safety_launch and t<.01 and self.base.last_pred is not None:
                self.dogleg=np.linalg.norm(self.base.last_pred[:2]-self.positions[self.active[0]])<30
                if self.dogleg:self.base.anchor_y=640
            if self.dogleg and t<.2-.001:
                # Gain clearance by turning across the pursuer's initial
                # heading while still on land, then use dry-shore launch.
                aid=self.active[0]
                aa[0]=(0,ActionRequest(agent_id=0,move_distance=20,
                    move_direction=wrap(math.pi/2-self.directions[aid]),turn_angle=0,spawn_agent=False))
                self.base.last_decisions[0]=dict(rule='Two-step land dogleg before close-gap water entry')
            for i,a in aa:
                aid=self.active[i];actions.append((aid,a.model_copy(update={'agent_id':aid})))
                self.decisions[aid]=self.base.last_decisions.get(i,{})
        for aid,s in states.items():
            if aid in self.active or aid not in self.positions:continue
            side=self.sides[aid];pos=self.positions[aid]
            pending=[i for i,a in self.pending.items() if a==aid]
            if pending:
                old=self.active[pending[0]]
                goal=self.positions[old]+np.array([side*12,0])
                action=self.action_to(s,goal,move=9)
                rule='Approach behind retiring bait'
            else:
                home=np.array([800+side*(B+120),600.])
                fruit=[]
                for o in s['observations']:
                    if o['type']!='Fruit' or not self.food:continue
                    theta=self.directions[aid]+o['angle']
                    fp=pos+o['distance']*np.array([math.cos(theta),math.sin(theta)])
                    if side*(fp[0]-800)>B+70 and abs(fp[1]-600)<130:fruit.append((o['distance'],fp))
                goal=min(fruit,key=lambda x:x[0])[1] if fruit else home
                # One real birth per original shore donor at a scheduled need.
                bait=self.active[0 if side<0 else 1]
                need_birth=bait in states and states[bait]['energy']<70
                spawn=(self.relay=='newborn' and aid in [2,3] and aid not in self.spawned
                    and need_birth and s['energy']>106)
                if self.relay in ['renewal','renewal-urgent']:
                    pool=sum(self.sides.get(k)==side for k in states)
                    young_spare=any(k!=aid and k not in self.active and self.sides.get(k)==side and
                        q['age']<50 and q['energy']>45 for k,q in states.items())
                    needs_young=bait in states and (states[bait]['age']>45 or states[bait]['energy']<70)
                    # An old donor need not preserve a large personal reserve
                    # when the only sustainable successor is a new generation.
                    threshold=112 if self.relay=='renewal-urgent' and needs_young and not young_spare else 185
                    spawn=(pool<(4 if self.relay=='renewal-urgent' else 3) and not young_spare and s['energy']>threshold and
                        self.cooldown.get(('birth',aid),0)<t)
                    if spawn:self.cooldown[('birth',aid)]=t+25
                if spawn:self.spawned.add(aid)
                action=self.action_to(s,goal,turn=self.directions[aid]+.1,spawn=spawn)
                rule='Native food nursery; legal 100-energy birth' if spawn else 'Recover at native shore food'
            actions.append((aid,action));self.decisions[aid]=dict(rule=rule)
        # Unknown newborns must still receive a legal stationary action.
        for aid in states:
            if aid not in self.positions:
                actions.append((aid,ActionRequest(agent_id=aid,move_distance=0,move_direction=0,turn_angle=.2,spawn_agent=False)))
        assert len(actions)==len(set(a for a,_ in actions))==len(states)
        return actions

    def integrate(self,actions,states):
        states={s['agent_id']:s for s in states}
        for aid,a in actions:
            if aid not in self.positions:continue
            s=states[aid];move=min(a.move_distance,s['sprint_speed'])
            if s['energy']<s['max_energy']/5:move=min(move,s['speed'])
            move*=PENALTY.get(s['biome'],1)
            angle=self.directions[aid]+a.move_direction
            old=self.positions[aid].copy()
            def attempt(heading):
                p=old+move*np.array([math.cos(heading),math.sin(heading)])
                wp=world(self.site,p-[800,600])
                for o in self.obstacles:
                    # Upstream uses an expanded box, including square corners.
                    if o[0]-5<wp[0]<o[0]+o[2]+5 and o[1]-5<wp[1]<o[1]+o[3]+5:return None
                wp=np.clip(wp,[5,5],[1595,1195])
                return local(self.site,wp)+[800,600]
            p=attempt(angle)
            if p is None:
                for k in range(36):
                    p=attempt(angle+math.pi/18*((k+1)//2)*(-1)**k)
                    if p is not None:break
            if p is not None:self.positions[aid]=p
            self.directions[aid]+=a.turn_angle

class TwinPolicy:
    """One independent pair per predator station; arranged allocation only."""
    def __init__(self,sites,poses,obstacles,anchor_window=None,association_gate=False):
        self.sites=sites
        self.pairs=[PairPolicy(site,pose,mapped_obstacles=obstacles,active_ids=tuple(pose),anchor_window=anchor_window,association_gate=association_gate)
                    for site,pose in zip(sites,poses)]
        self.base=SimpleNamespace(acquisition_complete=0)
        self.events=[];self.births=[];self.decisions={}
        self.active=[a for p in self.pairs for a in p.active]
        self.refresh()

    def refresh(self):
        self.positions={}
        for site,p in zip(self.sites,self.pairs):
            for aid,uv in p.positions.items():
                self.positions[aid]=local(self.sites[0],world(site,uv-[800,600]))+[800,600]

    def __call__(self,states,t):
        actions=[];self.decisions={}
        for p in self.pairs:
            subset=[s for s in states if s['agent_id'] in p.active]
            actions.extend(p(subset,t));self.decisions.update(p.decisions)
        return actions

    def integrate(self,actions,states):
        for p in self.pairs:
            p.integrate([(aid,a) for aid,a in actions if aid in p.active],
                        [s for s in states if s['agent_id'] in p.active])
        self.refresh()
