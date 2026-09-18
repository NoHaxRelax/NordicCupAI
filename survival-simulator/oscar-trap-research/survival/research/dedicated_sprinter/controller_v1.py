"""Observation-only single bait controller. No environment imports or world pose.

All distances are engine units per 0.1 s action. Predators are unlabelled in DTOs;
nearest-neighbour association is only trusted for one nearby predator. Unknown
terrain is treated as current terrain, with worst-case predator motion at wake.
"""
import math
import numpy as np
from src.utils.DTOs import ActionRequest

TAU=2*math.pi
PENALTY={'forest':1.,'grassland':1.,'desert':.8,'swamp':.5,'river':.3}
def wrap(x):return (x+math.pi)%TAU-math.pi
def unit(a):return np.array([np.cos(a),np.sin(a)])
def segment_distance(points,a,b):
    v=b-a
    t=np.clip(((points-a)*v).sum(axis=-1)/(v@v or 1),0,1)
    return np.linalg.norm(points-(a+t[...,None]*v),axis=-1)


def chase(p,heading,agent,face,speed):
    """Source-faithful unobstructed predator pursuit; arrays broadcast."""
    diff=agent-p; dist=np.linalg.norm(diff,axis=-1)
    bearing=np.arctan2(diff[...,1],diff[...,0]); angle=wrap(bearing-heading)
    rel=wrap(bearing+math.pi-face)
    direct=(np.abs(rel)>math.pi/2)|(dist<90)
    turn=np.where(np.abs(angle)>.05,np.clip(angle*.5,-.3,.3),0.)
    md=np.where(np.abs(angle)>.05,turn,angle)
    pivot=angle-np.sign(rel)*math.pi/4
    md=np.where(direct,md,pivot)
    move=np.minimum(speed,dist)
    delta=move[...,None]*np.stack([np.cos(heading+md),np.sin(heading+md)],axis=-1)
    nextp=p+delta
    pivotturn=np.arctan2(diff[...,1]-delta[...,1],diff[...,0]-delta[...,0])-heading
    nh=heading+np.where(direct,turn,wrap(pivotturn))
    return nextp,nh


class Tracker:
    def __init__(self):
        self.xy=np.zeros(2);self.theta=0.;self.landmarks=[];self.edges=[]
        self.food=[];self.lastpred=None;self.lasttime=None;self.still=0
        self.corrected=0;self.error=0.;self.terrain={}
    def update(self,s,t):
        obs=s['observations']; current=[]
        for o in obs:
            if o['type'] in ('Fruit','Tree'):
                current.append((o['type'],self.xy+o['distance']*unit(self.theta+o['angle'])))
            elif o['type']=='Edge':
                for pt in o['coords']:
                    current.append(('Edge',self.xy+np.array([[math.cos(self.theta),-math.sin(self.theta)],[math.sin(self.theta),math.cos(self.theta)]])@np.array(pt)))
        corrections=[]
        for typ,point in current:
            same=[old for ot,old in self.landmarks if ot==typ]
            if same:
                closest=min(same,key=lambda old:np.linalg.norm(point-old))
                if np.linalg.norm(closest-point)<25:corrections.append(closest-point)
        if corrections:
            correction=np.median(corrections,axis=0)
            agree=sum(np.linalg.norm(c-correction)<1 for c in corrections)
            if agree>=2 or (len(corrections)==1 and any(o['type']=='Tree' for o in obs)):
                self.xy+=correction;self.corrected+=1;self.error=float(np.linalg.norm(correction))
                current=[(typ,pt+correction) for typ,pt in current]
        self.landmarks=current
        rotation=np.array([[math.cos(self.theta),-math.sin(self.theta)],[math.sin(self.theta),math.cos(self.theta)]])
        for o in obs:
            if o['type']=='Edge':
                a,b=[self.xy+rotation@np.array(pt) for pt in o['coords']]
                if not any(np.linalg.norm(a-aa)+np.linalg.norm(b-bb)<2 for aa,bb in self.edges):self.edges.append((a,b))
        # Food memory expires; delete reached positions even if the DTO precedes eating.
        self.food=[(pt,seen) for pt,seen in self.food if t-seen<12 and np.linalg.norm(pt-self.xy)>11]
        for typ,pt in current:
            if typ=='Fruit' and np.linalg.norm(pt-self.xy)>11:
                near=[i for i,(q,_) in enumerate(self.food) if np.linalg.norm(pt-q)<8]
                if near:self.food[near[0]]=(pt,t)
                else:self.food.append((pt,t))
        self.terrain[tuple(np.round(self.xy/20).astype(int))]=PENALTY[s['biome']]
        pp=[o for o in obs if o['type']=='Predator']
        if not pp:return None
        p=min(pp,key=lambda o:o['distance']);pos=self.xy+p['distance']*unit(self.theta+p['angle'])
        heading=wrap(self.theta+p['angle']+math.pi-p['rel_dir'])
        speed=15.; still=False
        if self.lastpred is not None and t-self.lasttime<.15:
            speed=float(np.linalg.norm(pos-self.lastpred));still=speed<.15
        self.still=self.still+1 if still else 0
        self.lastpred=pos.copy();self.lasttime=t
        return dict(pos=pos,heading=heading,speed=min(15.,speed),raw=p)
    def advance(self,s,d,angle,turn):
        actual=min(d,s['sprint_speed'])
        if s['energy']<s['max_energy']/5:actual=min(actual,s['speed'])
        self.xy+=actual*PENALTY[s['biome']]*unit(self.theta+angle)
        self.theta=wrap(self.theta+turn)


class BaitController:
    def __init__(self, *, food=True, margin=24., fuel_weight=1.4, orbit_weight=10., rest=True):
        self.track=Tracker();self.food=food;self.margin=margin
        self.fuel_weight=fuel_weight;self.orbit_weight=orbit_weight;self.rest=rest
        self.stats={'inferred_rest_ticks':0,'food_target_ticks':0,'missing_ticks':0}
        self.last_decision={}
    def step(self,s,t):
        tr=self.track;p=tr.update(s,t);walk=min(s['speed'],s['sprint_speed'])
        cap=s['sprint_speed'] if s['energy']>=s['max_energy']/5 else walk
        multiplier=PENALTY[s['biome']]
        if p is None:
            self.stats['missing_ticks']+=1
            # Scan, allowing movement to remembered food only if contact was absent >1s.
            d=0.;angle=0.;turn=.45
            if self.food and tr.food and (tr.lasttime is None or t-tr.lasttime>1):
                point=min(tr.food,key=lambda f:np.linalg.norm(f[0]-tr.xy))[0]-tr.xy
                angle=wrap(math.atan2(point[1],point[0])-tr.theta);d=min(walk,np.linalg.norm(point)/multiplier);turn=angle
            why='scan missing predator'
        else:
            # Cached predator pose precedes one predator move. Advance it before
            # evaluating the candidate's own move and the following predator move.
            speed=p['speed'];resting=self.rest and tr.still>=2
            if resting:self.stats['inferred_rest_ticks']+=1
            nominal=0. if resting else max(3.3,speed)
            # Motion/rest is inferred, never read. At every rest tick also guard a
            # fast wake-up; speed 15 is an upper bound on any unknown next biome.
            speeds=sorted(set([nominal,15.]))
            directions=tr.theta+np.arange(48)*TAU/48
            foodtarget=None
            if self.food and tr.food and s['energy']<s['max_energy']-40:
                foodtarget=min(tr.food,key=lambda f:np.linalg.norm(f[0]-tr.xy))[0]
                da=math.atan2(*(foodtarget-tr.xy)[::-1]);directions=np.r_[directions,da]
                self.stats['food_target_ticks']+=1
            lengths=np.unique(np.clip([0.,walk,11.,12.,13.,15.,16.,18.,cap],0,cap))
            dirs=np.tile(directions,len(lengths));ds=np.repeat(lengths,len(directions))
            endpoints=tr.xy+ds[:,None]*multiplier*np.stack([np.cos(dirs),np.sin(dirs)],axis=-1)
            face=np.arctan2(p['pos'][1]-endpoints[:,1],p['pos'][0]-endpoints[:,0])
            gaps=[];finals=[];heads=[]
            for sp in speeds:
                q,h=chase(p['pos'],p['heading'],tr.xy,tr.theta,np.array(sp))
                qp,hp=chase(q,h,endpoints,face,np.full(len(ds),sp))
                gaps.append(np.linalg.norm(endpoints-qp,axis=1));finals.append(qp);heads.append(hp)
            safe=np.min(gaps,axis=0)
            # Avoid observed obstacle segments before collision rotation corrupts odometry.
            wall=np.full(len(ds),1000.)
            for a,b in tr.edges:
                if segment_distance(tr.xy,a,b)<80:
                    wall=np.minimum(wall,segment_distance(endpoints,a,b))
            energy=.05*np.minimum(ds,s['speed'])+.5*np.maximum(0,ds-s['speed'])
            gap=gaps[0]
            cost=self.fuel_weight*energy+.004*(gap-44.)**2
            cost+=np.maximum(0,self.margin-safe)**2*15
            cost+=np.maximum(0,10-wall)**2*15
            # Turning around the pursuer makes its bounded steering work in our
            # favour; retain hearing contact rather than straight-line separation.
            rel=endpoints-finals[0]; bearing=np.arctan2(rel[:,1],rel[:,0])
            cost+=self.orbit_weight*np.cos(wrap(bearing-heads[0]))
            cost+=np.maximum(0,gap-58)**2*.2
            if foodtarget is not None:
                progress=np.linalg.norm(endpoints-foodtarget,axis=1)-np.linalg.norm(tr.xy-foodtarget)
                cost+=.40*progress
            # Known slow terrain is expensive while pursued. No true map input.
            for i,pt in enumerate(endpoints):
                m=tr.terrain.get(tuple(np.round(pt/20).astype(int)),multiplier)
                if m<multiplier:cost[i]+=15*(multiplier-m)
            i=int(np.argmin(cost));d=float(ds[i]);angle=wrap(float(dirs[i])-tr.theta)
            # Keep predator in view. With close hearing contact, scan food periodically.
            turn=wrap(float(face[i])-tr.theta)
            if p['raw']['distance']<48 and int(round(t*10))%16==0:turn=wrap(turn+math.pi)
            why='rest forage' if resting else 'predictive food pursuit' if foodtarget is not None else 'predictive occupation'
            self.last_decision=dict(predicted_min_gap=round(float(safe[i]),2),inferred_speed=round(speed,2),inferred_rest=resting,food_target=None if foodtarget is None else foodtarget.tolist())
        tr.advance(s,d,angle,turn)
        return ActionRequest(agent_id=s['agent_id'],move_distance=d,move_direction=angle,turn_angle=turn,spawn_agent=False),why
