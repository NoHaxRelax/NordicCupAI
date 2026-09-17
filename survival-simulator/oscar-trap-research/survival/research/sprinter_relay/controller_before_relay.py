"""Observation-only coordinated bait controller. No imports from simulator.

All coordinates live in an arbitrary frame established from agent sightings.
Odometry is exact only on clear, flat ground; edges reject unsafe commands.
Predator position/heading come only from cached observations, never live state.
"""
from __future__ import annotations
import math
import numpy as np


def wrap(x): return (x+math.pi)%(2*math.pi)-math.pi

def unit(a): return np.array([math.cos(a),math.sin(a)])

def bearing(v): return math.atan2(v[1],v[0])


def pred_move(pos, heading, points, directions, speed=15.):
    """Public predator rule, clear-ground prediction. points shape (...,n,2)."""
    delta=points-pos
    dist=np.linalg.norm(delta,axis=-1)
    angle=(np.arctan2(delta[...,1],delta[...,0])-heading+np.pi)%(2*np.pi)-np.pi
    seen=(dist<=60)|((dist<=250)&(np.abs(angle)<=np.pi/6))
    which=np.argmin(np.where(seen,dist,1e6),axis=-1)
    selected=np.take_along_axis(delta,which[...,None,None],axis=-2)[...,0,:]
    d=np.take_along_axis(dist,which[...,None],axis=-1)[...,0]
    a=(np.arctan2(selected[...,1],selected[...,0])-heading+np.pi)%(2*np.pi)-np.pi
    facing=np.take_along_axis(np.broadcast_to(directions,dist.shape),which[...,None],axis=-1)[...,0]
    rel=(np.arctan2(-selected[...,1],-selected[...,0])-facing+np.pi)%(2*np.pi)-np.pi
    chasing=(np.abs(rel)>np.pi/2)|(d<90)
    turn=np.where(np.abs(a)>.05,np.clip(a*.5,-.3,.3),0.)
    md=np.where(chasing,np.where(np.abs(a)>.05,turn,a),a-np.sign(rel)*np.pi/4)
    step=np.minimum(speed,d)
    dx=step*np.cos(heading+md);dy=step*np.sin(heading+md)
    new=pos+np.stack([dx,dy],axis=-1)
    nh=heading+turn
    return new,nh,which,np.any(seen,axis=-1)


class RelayController:
    def __init__(self, bait_ids, variant='mpc', switch_energy=260, dwell=2., target_radius=53,
                 recovery=True, newborn_ready=180, pair_period=0., home_weight=.0003, assignment_weight=12, **unused):
        self.baits=set(bait_ids);self.variant=variant;self.switch_energy=switch_energy
        self.dwell=dwell;self.radius=target_radius;self.recovery=recovery;self.newborn_ready=newborn_ready
        self.pair_period=pair_period;self.home_weight=home_weight;self.assignment_weight=assignment_weight
        self.pose={};self.pred=None;self.prev_pred=None;self.pred_speed=15.
        self.active=min(self.baits) if self.baits else None;self.incoming=None;self.retiring=set()
        self.events=[];self.last_switch=-10.;self.orbit_sign=None;self.turns=0;self.home=None;self.food={};self.pending_parent=None

    def localize(self,states,t):
        rows={s['agent_id']:s for s in states}
        if not self.pose and states:self.pose[states[0]['agent_id']]=(np.zeros(2),0.)
        for _ in range(3):
            for aid,s in rows.items():
                if aid not in self.pose:continue
                pos,theta=self.pose[aid]
                for o in s['observations']:
                    if o['type']=='Agent' and o['id'] not in self.pose:
                        b=theta+o['angle']
                        self.pose[o['id']]=(pos+o['distance']*unit(b),wrap(b+math.pi-o['rel_dir']))
                        if self.pending_parent is not None:
                            self.baits.add(o['id']);self.events.append(dict(t=t,event='newborn_localized',agent=o['id']))
                            self.pending_parent=None
        sightings=[]
        for aid,s in rows.items():
            if aid not in self.pose:continue
            pos,theta=self.pose[aid]
            for o in s['observations']:
                if o['type']=='Predator':
                    b=theta+o['angle'];p=pos+o['distance']*unit(b)
                    sightings.append((o['distance'],p,wrap(b+math.pi-o['rel_dir'])))
                if o['type']=='Fruit':
                    p=pos+o['distance']*unit(theta+o['angle']);key=tuple(np.round(p/8).astype(int))
                    self.food[key]=(p,t)
        # Identityless predators: track closest sighting to previous track.
        if sightings:
            if self.pred is None:best=min(sightings,key=lambda x:x[0])
            else:best=min(sightings,key=lambda x:np.linalg.norm(x[1]-self.pred[0]))
            _,pos,heading=best
            if self.prev_pred is not None:
                self.pred_speed=float(np.clip(np.linalg.norm(pos-self.prev_pred),0,15))
            self.prev_pred=pos.copy();self.pred=(pos,heading);self.last_seen=t
            if self.home is None:self.home=pos.copy()
        self.food={k:v for k,v in self.food.items() if t-v[1]<3}
        return rows

    def candidates(self,s):
        aid=s['agent_id'];pos,theta=self.pose[aid]
        walk=min(s['speed'],s['sprint_speed']);cap=s['sprint_speed'] if s['energy']>=s['max_energy']/5+3 else walk
        angles=np.arange(16)*math.pi/8
        steps=sorted(set([0.,5.,walk,min(15.2,cap),cap]))
        vec=np.array([d*unit(a) for d in steps for a in angles] if cap>0 else [np.zeros(2)])
        # Reject edge-crossing and close endpoint candidates in observed local geometry.
        valid=np.ones(len(vec),dtype=bool)
        for o in s['observations']:
            if o['type']!='Edge':continue
            aa,bb=map(np.array,o['coords']);rot=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
            aa=rot@aa;bb=rot@bb;line=bb-aa;den=np.dot(line,line)
            u=np.clip(((vec-aa)@line)/max(den,1e-9),0,1)
            near=np.linalg.norm(vec-(aa+u[:,None]*line),axis=-1)<18
            valid&=~near
        vec=vec[valid]
        if len(vec)==0:vec=np.zeros((1,2))
        # Deduplicate stationary action.
        vec=np.unique(np.round(vec,5),axis=0)
        d=np.linalg.norm(vec,axis=-1);cost=np.minimum(d,walk)*.05+np.maximum(d-walk,0)*.5
        return vec,cost

    def choose_pair(self,ss,t):
        p,heading=self.pred
        positions=np.array([self.pose[s['agent_id']][0] for s in ss])
        dirs=np.array([bearing(p-x) for x in positions])
        # Cached observation is pre-predator step. Estimate its already executed
        # step using observed speed. Also guard sudden waking with speed15 below.
        pc,hc,_,_=pred_move(p,heading,positions,dirs,self.pred_speed)
        vecs=[];costs=[]
        for s in ss:
            v,c=self.candidates(s);vecs.append(v);costs.append(c)
        if len(ss)==1:
            points=(positions[0]+vecs[0])[:,None,:];cost=costs[0];moves=vecs[0][:,None,:]
        else:
            va,vb=vecs
            pa=np.broadcast_to((positions[0]+va)[:,None,:],(len(va),len(vb),2))
            pb=np.broadcast_to((positions[1]+vb)[None,:,:],(len(va),len(vb),2))
            points=np.stack([pa,pb],axis=-2);cost=costs[0][:,None]+costs[1][None,:]
            moves=np.stack([np.broadcast_to(va[:,None,:],pa.shape),np.broadcast_to(vb[None,:,:],pb.shape)],axis=-2)
        newdirs=np.arctan2(pc[1]-points[...,1],pc[0]-points[...,0])
        pn,hn,target,seen=pred_move(pc,hc,points,newdirs,15 if self.pred_speed>1 else 0)
        gap=np.linalg.norm(points-pn[...,None,:],axis=-1)
        cached_gap=np.linalg.norm(points-pc,axis=-1)
        # Safety on both motion and no-motion possibilities, including wake.
        pw,_,_,_=pred_move(pc,hc,points,newdirs,15)
        worst=np.minimum(gap,np.linalg.norm(points-pw[...,None,:],axis=-1))
        score=cost*.8+np.sum(np.maximum(33-worst,0)**2,axis=-1)*30
        score+=np.maximum(np.min(gap,axis=-1)-self.radius,0)**2*.13
        score+=np.maximum(self.radius-8-np.min(gap,axis=-1),0)**2*.05
        score+=(~seen)*2500
        # Keep baits on different sides: avoids both fleeing the same trajectory.
        if len(ss)==2:
            v=points-pn[...,None,:]
            cosine=np.sum(v[...,0,:]*v[...,1,:],axis=-1)/np.maximum(gap[...,0]*gap[...,1],1)
            score+=np.maximum(cosine+.35,0)**2*80
            score+=np.sum(np.maximum(gap-90,0)**2,axis=-1)*.05
            # Explicit assignment: incoming target gets priority only after safe approach.
            ids=[s['agent_id'] for s in ss]
            desired=self.incoming if self.incoming in ids else self.active
            ix=ids.index(desired) if desired in ids else 0
            score+=(target!=ix)*self.assignment_weight
            # Ready reserve is held farther away; retiring agent opens separation.
            for j,s in enumerate(ss):
                if s['agent_id'] in self.retiring:
                    score+=np.maximum(90-gap[...,j],0)**2*.1
        # Confine geometry by softly minimizing displacement from acquired center.
        if self.home is not None:score+=np.sum((pn-self.home)**2,axis=-1)*self.home_weight
        index=np.unravel_index(np.argmin(score),score.shape)
        selected=points[index];target_id=ss[int(target[index])]['agent_id'] if seen[index] else None
        if self.incoming is not None and target_id==self.incoming:
            old=self.active;self.active=self.incoming;self.incoming=None;self.last_switch=t
            self.events.append(dict(t=t,event='predicted_handoff',outgoing=old,incoming=self.active))
        return {s['agent_id']:moves[index][j] for j,s in enumerate(ss)}

    def choose_orbit(self,ss,t):
        p,heading=self.pred
        primary=next((s for s in ss if s['agent_id']==self.active),ss[0])
        pos,_=self.pose[primary['agent_id']]
        if self.orbit_sign is None:
            self.orbit_sign=1 if wrap(bearing(pos-p)-heading)>=0 else -1
        sign=self.orbit_sign
        speed=self.pred_speed if self.pred_speed>1 else 15.
        radius=max(30,speed/(2*math.sin(.15)))
        center=p+radius*unit(heading+sign*(math.pi/2+.15))
        moves={}
        for s in ss:
            aid=s['agent_id'];pos,theta=self.pose[aid]
            v,c=self.candidates(s);points=pos+v
            # Predict source's previous move using that bait only. Capture guard
            # covers full-speed and stationary alternatives independently.
            pc,hc,_,_=pred_move(p,heading,np.array([pos]),np.array([theta]),self.pred_speed)
            newdirs=np.arctan2(pc[1]-points[:,1],pc[0]-points[:,0])
            pn,_,_,seen=pred_move(pc,hc,points[:,None,:],newdirs[:,None],15.)
            dist=np.minimum(np.linalg.norm(points-pn,axis=1),np.linalg.norm(points-pc,axis=1))
            if aid==self.active:
                score=np.sum((points-center)**2,axis=1)*.5+c*.2+np.maximum(25-dist,0)**2*100+(~seen)*1000
            else:
                # Reserve remains outside the active orbit until called.
                staging=center+(pos-center)/max(np.linalg.norm(pos-center),1)*145
                score=np.sum((points-staging)**2,axis=1)*.1+c+np.maximum(60-dist,0)**2*30
            moves[aid]=v[np.argmin(score)]
        return moves

    def __call__(self,states,t):
        rows=self.localize(states,t);live=[s for s in states if s['agent_id'] in self.baits and s['agent_id'] in self.pose]
        if self.active not in rows:self.active=live[0]['agent_id'] if live else None
        if self.incoming not in rows:self.incoming=None
        if self.active is not None and len(live)>1:
            current=rows[self.active]
            candidates=[s for s in live if s['agent_id']!=self.active and s['energy']>=self.newborn_ready and s['energy']>current['energy']+30]
            urgent=current['energy']<self.switch_energy
            timed=self.pair_period and t-self.last_switch>self.pair_period
            if candidates and (urgent or timed) and t-self.last_switch>self.dwell and self.incoming is None:
                self.incoming=max(candidates,key=lambda s:s['energy'])['agent_id']
                self.events.append(dict(t=t,event='request_handoff',outgoing=self.active,incoming=self.incoming))
        moves={}
        if self.pred and t-self.last_seen<.5 and live:
            chosen=sorted(live,key=lambda s:(s['agent_id']!=self.active,s['agent_id']!=self.incoming,-s['energy']))[:2]
            if self.variant=='mpc':moves=self.choose_pair(chosen,t)
            elif self.variant=='orbit':
                moves=self.choose_orbit(chosen,t)
            else:
                p,_=self.pred
                for s in chosen:
                    aid=s['agent_id'];pos,_=self.pose[aid];d=np.linalg.norm(pos-p)
                    target=self.radius if aid==self.active else self.radius+25
                    v=(pos-p)/max(d,.1)
                    moves[aid]=v*np.clip(target-d,-10,20 if s['energy']>=100 else 10)
        actions=[];decisions={}
        for s in states:
            aid=s['agent_id'];turn=.25;v=np.zeros(2);spawn=False;why='scan and localize'
            if aid in self.pose:
                pos,theta=self.pose[aid]
                if aid in moves:
                    v=moves[aid];turn=wrap(bearing(self.pred[0]-pos-v)-theta);why='active' if aid==self.active else 'staged incoming'
                    # Opportunistically take food only if nearby and outside danger.
                else:
                    fruit=[o for o in s['observations'] if o['type']=='Fruit']
                    predators=[o for o in s['observations'] if o['type']=='Predator']
                    if predators and min(o['distance'] for o in predators)<130:
                        o=min(predators,key=lambda o:o['distance']);v=-unit(theta+o['angle'])*min(s['sprint_speed'],20 if s['energy']>=s['max_energy']/5 else s['speed']);turn=o['angle'];why='worker escape'
                    elif fruit:
                        f=min(fruit,key=lambda o:o['distance']);v=unit(theta+f['angle'])*min(s['speed'],max(0,f['distance']-4));turn=f['angle'];why='forage'
                    else:turn=.35;why='scan food'
                distance=float(np.linalg.norm(v));direction=wrap(bearing(v)-theta) if distance else 0.
                actual=distance
                if s['energy']<s['max_energy']/5:actual=min(actual,s['speed'])
                penalty={'swamp':.5,'river':.3,'desert':.8}.get(s['biome'],1.)
                self.pose[aid]=(pos+v*(actual/max(distance,1e-9))*penalty,wrap(theta+turn))
            else:distance=direction=0.
            actions.append(dict(agent_id=aid,move_distance=distance,move_direction=direction,turn_angle=float(turn),spawn_agent=spawn));decisions[aid]=why
        return actions,decisions
