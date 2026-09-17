# Frozen containment/acquisition controller snapshot from water_tuning.py, 2026-09-17.
"""Water bait tuning against unchanged upstream simulation mechanics.

All scenarios and map knowledge are explicitly synthetic diagnostic fixtures.
Default controllers know current predator geometry; `observed=True` replaces
that input with cached public observations and estimated pose, not hidden energy.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
import sys
import time

from common import ROOT, wrap
import numpy as np
from src.utils.DTOs import ActionRequest
from src.elements.fruit import Fruit
from src.elements.obstacle import Obstacle


@dataclass(frozen=True)
class Config:
    mode: str = "lead"
    lead: float = 55
    reserve: float = 12
    switch: float = 3
    max_move: float = 7
    lookahead: float = 3
    margin: float = 2
    track_y: float = 0
    escape: float = 27
    face: str = "back"
    turn_y: float = 920
    observed: bool = False
    rest_wait: bool = False
    hold_ticks: int = 3
    y_bias: float = 0
    prediction: str = "none"
    arc_cos: float = .7


@dataclass(frozen=True)
class Scenario:
    width: int = 60
    seconds: float = 60
    energy: float = 150
    heading: float = math.pi/2
    offset_x: float = 0
    offset_y: float = 0
    n_agents: int = 2
    max_age: float = 120
    food_spacing: float = 0
    seed: int = 173
    outside_distance: float = 0
    secondary_offset: float = 60
    shore_launch: bool = False


class Controller:
    def __init__(self, config, scenario, agents):
        self.c=config;self.s=scenario
        self.role=0
        self.direction=1
        self.switches=0
        self.last_switch=-999
        self.positions={a.agent_id:np.array([a.x,a.y],float) for a in agents}
        self.directions={a.agent_id:a.direction for a in agents}
        self.last_pred=None
        self.last_raw_pred=None
        self.estimate_age=0
        self.last_decisions={}
        self.acquisition_complete=0 if not scenario.outside_distance else None
        self.predator_entered=False
        self.acquisition_energy=None
        self.anchor_y=600
        self.anchor_window=None
        self.association_gate=False

    def acquisition_actions(self, env, pose, tick, states):
        """Cross one bait through the river, then hand pursuit to shore two.

        Both agents start on the predator's land side. Entry detection and all
        motion use the same cached-observation pose estimate as containment.
        """
        px,py,_=pose
        c,s=self.c,self.s
        edge=800+s.width/2
        if px<edge-.5:self.predator_entered=True
        by_id={x['agent_id']:x for x in states}
        if 0 not in by_id or 1 not in by_id:return []
        left=800-s.width/2-c.margin
        a0=self.positions[0];a1=self.positions[1]
        if a0[0]<=left-15 and abs(a1[1]-self.anchor_y)<=25 and 800-s.width/2<px<edge:
            self.acquisition_complete=tick/10
            self.acquisition_energy={aid:by_id[aid]['energy'] for aid in [0,1]}
            self.role=1;self.last_switch=tick
            return None
        actions=[];self.last_decisions={}
        penalties={'river':.3,'swamp':.5,'desert':.8}
        for aid in [0,1]:
            apos=self.positions[aid];facing=self.directions[aid]
            if aid==0:
                if s.shore_launch and apos[0]>edge+.001:
                    move=min(20,apos[0]-edge)
                    rule='Step onto dry shoreline launch point'
                elif apos[0]>left:
                    move=min(20,(apos[0]-left)/penalties.get(by_id[aid]['biome'],1))
                    rule='Cross river to draw predator into water'
                else:
                    move=20
                    rule='Retreat to transfer pursuit to the other bank'
                absolute=math.pi
            else:
                dy=self.anchor_y-apos[1]
                move=min(10,abs(dy)) if self.predator_entered else 0
                absolute=math.pi/2 if dy>0 else -math.pi/2
                rule='Approach second shore after predator enters water' if self.predator_entered else 'Wait clear of initial chase'
            to_pred=math.atan2(py-apos[1],px-apos[0])
            turn=wrap(to_pred-facing)
            if abs(turn)<.05:turn=0
            actions.append((aid,ActionRequest(agent_id=aid,move_distance=move,move_direction=wrap(absolute-facing),turn_angle=turn,spawn_agent=False)))
            self.last_decisions[aid]=dict(rule=rule,detail='Acquisition uses mapped straight-river geometry and cached public predator observations.')
        return actions

    def estimate(self, states):
        estimates=[]
        for s in states:
            aid=s['agent_id'];o=[o for o in s['observations'] if o['type']=='Predator']
            if self.association_gate:
                reference=self.last_raw_pred[:2] if self.last_raw_pred is not None else np.array([800.,self.anchor_y])
                radius=min(90,30+15*self.estimate_age) if self.last_raw_pred is not None else 90
                matched=[]
                for item in o:
                    theta=self.directions[aid]+item['angle']
                    q=self.positions[aid]+item['distance']*np.array([math.cos(theta),math.sin(theta)])
                    error=float(np.linalg.norm(q-reference))
                    if error<radius:matched.append((error,item))
                o=[min(matched,key=lambda x:x[0])[1]] if matched else []
            if not o:continue
            obs=min(o,key=lambda x:x['distance'])
            theta=self.directions[aid]+obs['angle']
            pos=self.positions[aid]+obs['distance']*np.array([math.cos(theta),math.sin(theta)])
            heading=wrap(theta+math.pi-obs['rel_dir'])
            estimates.append((pos,heading))
        if estimates:
            pos=np.mean([x[0] for x in estimates],axis=0)
            angle=math.atan2(sum(math.sin(x[1]) for x in estimates),sum(math.cos(x[1]) for x in estimates))
            raw=np.array([pos[0],pos[1],angle])
            predicted=raw.copy()
            if self.last_raw_pred is not None and self.c.prediction!='none':
                velocity=raw[:2]-self.last_raw_pred[:2]
                if self.c.prediction=='velocity':predicted[:2]+=velocity
                else:
                    # Model the deterministic chase turn from observed pose
                    # and our own tracked bait positions. Speed/rest is inferred
                    # from consecutive observed positions, never read from p.
                    speed=float(np.linalg.norm(velocity))
                    possible=[]
                    for aid,apos in self.positions.items():
                        rel=apos-raw[:2];d=float(np.linalg.norm(rel));q=wrap(math.atan2(rel[1],rel[0])-angle)
                        if d<=60 or (d<=250 and abs(q)<=math.pi/6):possible.append((d,q))
                    if possible and speed>.01:
                        d,q=min(possible)
                        turn=max(-.3,min(.3,q*.5)) if abs(q)>.05 else q
                        predicted[:2]+=speed*np.array([math.cos(angle+turn),math.sin(angle+turn)])
                        predicted[2]=wrap(angle+(turn if abs(q)>.05 else 0))
            self.last_raw_pred=raw
            self.last_pred=predicted;self.estimate_age=0
        else:
            self.estimate_age+=1
        return self.last_pred

    def actions(self, env, p, tick):
        c,s=self.c,self.s
        states=[env.get_agent_state(a.agent_id) for a in env.agents]
        observed=self.estimate(states)
        if c.observed:
            if observed is None:
                return [(a.agent_id,ActionRequest(agent_id=a.agent_id,move_distance=0,move_direction=0,turn_angle=.2,spawn_agent=False)) for a in env.agents]
            px,py,pheading=observed
        else:px,py,pheading=p.x,p.y,p.direction
        if self.acquisition_complete is None:
            acquiring=self.acquisition_actions(env,(px,py,pheading),tick,states)
            if acquiring is not None:return acquiring
        if py>c.turn_y:self.direction=-1
        elif py<1200-c.turn_y:self.direction=1
        pred_x=px+c.lookahead*4.5*math.cos(pheading)
        desired=self.role
        if c.mode=='arc':
            side=-1 if self.role==0 else 1
            if side*math.cos(pheading)>c.arc_cos:desired=1-self.role
        else:
            if pred_x>800+c.switch:desired=0
            elif pred_x<800-c.switch:desired=1
        if desired!=self.role and tick-self.last_switch>=c.hold_ticks:
            self.role=desired;self.switches+=1;self.last_switch=tick
        B=s.width/2+c.margin
        goals={}
        if c.mode=="lead":
            for a in env.agents:
                side=-1 if a.agent_id%2==0 else 1
                goals[a.agent_id]=np.array([800+side*B,py+self.direction*(c.lead+(0 if a.agent_id%2==self.role else c.reserve))])
        elif c.mode in ["cross","cross_side","arc"]:
            # Keep chosen bait at its shore, put the rival just farther away
            # from the predator. The formula uses the actual other-bait
            # distance, rather than a fixed large retreat on every switch.
            role_side=-1 if self.role==0 else 1
            ry=self.anchor_y+c.track_y*(py-self.anchor_y)
            if self.anchor_window is not None:
                ry=self.anchor_y+max(-self.anchor_window,min(self.anchor_window,ry-self.anchor_y))
            if c.y_bias:
                ry=py+max(-c.y_bias,min(c.y_bias,600-py))
            active=np.array([800+role_side*B,ry])
            d=math.hypot(active[0]-px,active[1]-py)+c.reserve
            other_side=-role_side
            if c.mode in ['cross','arc']:
                retreat_x=px+other_side*math.sqrt(max(0,d*d-(ry-py)**2))
                reserve=np.array([800+other_side*max(B,other_side*(retreat_x-800)),ry])
            else:
                bx=800+other_side*B
                dy=math.sqrt(max(0,d*d-(bx-px)**2))
                reserve=np.array([bx,py+self.direction*dy])
            for a in env.agents:goals[a.agent_id]=active if a.agent_id%2==self.role else reserve
        elif c.mode=="single_bank":
            for a in env.agents:goals[a.agent_id]=np.array([800+B,py+self.direction*c.lead])
        else:raise ValueError(c.mode)
        actions=[]
        self.last_decisions={}
        for a in env.agents:
            aid=a.agent_id
            pos=self.positions[aid] if c.observed else np.array([a.x,a.y])
            facing=self.directions[aid] if c.observed else a.direction
            delta=goals[aid]-pos
            requested=min(c.max_move,float(np.linalg.norm(delta)))
            rule="Lead predator along bank" if c.mode=="lead" else "Shift nearest bait"
            to_pred=np.array([px,py])-pos
            distance=float(np.linalg.norm(to_pred))
            # Failed-switch escape is allowed to continue beyond the nominal
            # goal. It uses actual legal sprinting and low-energy restrictions.
            if distance<c.escape:
                away=-to_pred/max(.001,distance)
                if c.mode in ["lead","cross_side","single_bank"]:
                    # Prefer along-bank escape while retaining dry ground.
                    away=np.array([0,self.direction],float)
                    if abs(to_pred[0])<10:away=-to_pred/max(.001,distance)
                requested=min(a.sprint_speed,max(c.max_move,c.escape-distance+4.6))
                delta=away;rule="Escape failed handoff"
            if c.rest_wait and self.last_pred is not None:
                # Oracle rest is intentionally absent. Settling at the fixed
                # requested lead already pauses bait movement during sleep.
                pass
            move_angle=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>1e-8 else facing
            if c.face=="back":desired_face=math.atan2(to_pred[1],to_pred[0])
            elif c.face=="away":desired_face=math.atan2(-to_pred[1],-to_pred[0])
            elif c.face=="relay":
                desired_face=math.atan2(-to_pred[1],-to_pred[0]) if aid%2==self.role else math.atan2(to_pred[1],to_pred[0])
            else:desired_face=facing
            turn=wrap(desired_face-facing)
            # Avoid paying for microscopic turns when already seeing the target.
            if abs(turn)<.05:turn=0
            action=ActionRequest(agent_id=aid,move_distance=requested,move_direction=wrap(move_angle-facing),turn_angle=turn,spawn_agent=False)
            actions.append((aid,action))
            self.last_decisions[aid]=dict(rule=rule,detail=f"Family {c.mode}; target role {self.role}; estimated predator distance {distance:.1f}; move request {requested:.2f}.")
        return actions

    def integrate(self, actions, states):
        """Dead reckoning only, with biome modifier from public own state."""
        state={s['agent_id']:s for s in states}
        penalties={'river':.3,'swamp':.5,'desert':.8}
        for aid,action in actions:
            s=state[aid]
            move=min(action.move_distance,s['sprint_speed'])
            if s['energy']<s['max_energy']/5:move=min(move,s['speed'])
            move*=penalties.get(s['biome'],1)
            heading=self.directions[aid]+action.move_direction
            self.positions[aid]+=move*np.array([math.cos(heading),math.sin(heading)])
            self.directions[aid]+=action.turn_angle
