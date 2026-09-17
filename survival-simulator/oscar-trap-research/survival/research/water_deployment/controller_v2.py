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
    def __init__(self,site,poses,*,outside=0,relay='none',mapped_obstacles=(),food=True):
        self.site=site
        self.positions={k:np.array(v[:2],float) for k,v in poses.items()}
        self.directions={k:v[2] for k,v in poses.items()}
        self.active=[0,1]
        self.sides={k:(-1 if v[0]<800 else 1) for k,v in poses.items()}
        self.sides.update({0:-1,1:1})
        self.relay=relay;self.food=food
        self.obstacles=mapped_obstacles
        self.pending={};self.events=[];self.decisions={};self.spawned=set()
        self.cooldown={};self.births=[]
        self.c=Config(mode='cross',reserve=18,switch=9,max_move=9,
            lookahead=3,margin=2,track_y=.7,escape=22,face='fixed',observed=True,
            hold_ticks=4,prediction='model')
        self.base=Controller(self.c,Scenario(width=site['width'],outside_distance=outside,
            shore_launch=True),[SimpleNamespace(agent_id=i,x=self.positions[i][0],
            y=self.positions[i][1],direction=self.directions[i]) for i in [0,1]])

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
                if self.relay=='renewal':
                    pool=sum(self.sides.get(k)==side for k in states)
                    spawn=(pool<3 and s['energy']>185 and self.cooldown.get(('birth',aid),0)<t)
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
