"""Observation-only burst admission. No engine state, seed, or map access.

Predator observations precede its move: predict that missing move and the pending
move. Track motion in the observer's frame, correcting ego motion with static
edges. Uncertain associations and resting targets are deliberately rejected.
"""
import math
from collections import Counter

PI = math.pi
PENALTY = {'forest': 1., 'grassland': 1., 'desert': .8, 'swamp': .5, 'river': .3}


def wrap(a):
    return (a + PI) % (2 * PI) - PI


def rotate(p, a):
    c, s = math.cos(a), math.sin(a)
    return (c*p[0]-s*p[1], s*p[0]+c*p[1])


def sub(a, b):
    return a[0]-b[0], a[1]-b[1]


def norm(p):
    return math.hypot(*p)


def point(o):
    return rotate((float(o['distance']), 0.), float(o['angle']))


def heading(o):
    return wrap(float(o['angle']) + PI - float(o['rel_dir']))


def move(state, action):
    """Requested translation before rotation, in the agent's current frame."""
    d = max(0., min(float(action.get('move_distance', 0)), state['sprint_speed']))
    if state['energy'] < state['max_energy']/5:
        d = min(d, state['speed'])
    return rotate((d*PENALTY[state['biome']], 0.), float(action.get('move_direction', 0)))


def edge_distance(p, a, b):
    v = sub(b, a); w = sub(p, a)
    n = v[0]*v[0]+v[1]*v[1]
    t = max(0., min(1., (w[0]*v[0]+w[1]*v[1])/n)) if n else 0.
    return norm((w[0]-t*v[0], w[1]-t*v[1]))


class ContactPredictor:
    def __init__(self, margin=1.0):
        self.margin = margin
        self.previous = {}
        self.frozen = {}
        self.pending = None
        self.stats = Counter()
        self.confirmed = 0

    def _motion(self, old, state, edges):
        turn = wrap(old['action'].get('turn_angle', 0.))
        guess = rotate(move(old['state'], old['action']), -turn)
        candidates = []
        indexed = {}
        for c,d in edges:
            v=sub(d,c)
            indexed.setdefault((round(v[0],2),round(v[1],2)),[]).append((c,d))
        # Equal directed static edges identify translation independently of
        # collisions or terrain changes. Match both endpoints, not just length.
        for a, b in old['edges']:
            a, b = rotate(a, -turn), rotate(b, -turn)
            v = sub(b, a)
            for c, d in indexed.get((round(v[0],2),round(v[1],2)),[]):
                if norm(sub(v, sub(d, c))) > .015:
                    continue
                delta = sub(a, c)
                if norm(delta) <= norm(guess)+.1:
                    candidates.append(delta)
        if candidates:
            counts=Counter((round(p[0],2),round(p[1],2)) for p in candidates)
            clusters = [(counts[(round(p[0],2),round(p[1],2))], -norm(sub(p,guess)), p)
                        for p in candidates]
            votes, _, best = max(clusters)
            # One unique matching segment is sufficient if it agrees with
            # odometry; otherwise require two independently agreeing edges.
            if votes >= 2 or norm(sub(best, guess)) < .1:
                return turn, best, True
        return turn, guess, False

    def _update(self, states, actions, t, score):
        live = {s['agent_id'] for s in states}
        if self.pending and t > self.pending['t']:
            p = self.pending
            confirmed = (score is not None and p['score'] is not None and
                         score-p['score'] > p['payout']*.8 and p['farm'] not in live)
            if confirmed:
                self.confirmed += 1
                for aid, point_ in p['marks'].items():
                    self.frozen.setdefault(aid, []).append((point_, p['until'], 0.))
            self.pending = None
        by_action = {a['agent_id']:a for a in actions}
        current = {}
        for s in states:
            aid = s['agent_id']
            obs = s.get('observations', [])
            edges = [o['coords'] for o in obs if o['type'].lower() == 'edge']
            preds = [dict(p=point(o), h=heading(o), observation=o) for o in obs
                     if o['type'].lower() == 'predator' and 'rel_dir' in o and float(o['distance'])<=60.]
            walls=list(edges)
            old = self.previous.get(aid)
            if old and (preds and old['preds'] or self.frozen.get(aid)) and abs(t-old['t']-.1) < .001 and abs(s['age']-old['state']['age']-.1) < .001:
                turn, delta, reliable = self._motion(old, s, edges)
                if reliable:
                    # Walls persist when the agent turns its vision away. Use
                    # only registered ego motion; uncertain motion drops memory.
                    seen={tuple(round(v,2) for p in edge for v in p) for edge in walls}
                    for a,b in old.get('walls',old['edges']):
                        edge=(sub(rotate(a,-turn),delta),sub(rotate(b,-turn),delta))
                        key=tuple(round(v,2) for p in edge for v in p)
                        if key not in seen and edge_distance((0.,0.),*edge)<180.:
                            walls.append(edge); seen.add(key)
                    walls=sorted(walls,key=lambda e:edge_distance((0.,0.),*e))[:64]
                transported = [dict(p=sub(rotate(p['p'],-turn),delta), h=wrap(p['h']-turn))
                               for p in old['preds']]
                used = set()
                for pred in sorted(preds,key=lambda p:norm(p['p'])):
                    matches = sorted((norm(sub(pred['p'],p['p'])) + abs(wrap(pred['h']-p['h'])), j)
                                     for j,p in enumerate(transported) if j not in used)
                    if not matches or matches[0][0] > 17.:
                        continue
                    if len(matches)>1 and matches[1][0]-matches[0][0] < 2.:
                        self.stats['ambiguous_track'] += 1
                        continue
                    _, j = matches[0]; used.add(j)
                    pred['speed'] = norm(sub(pred['p'],transported[j]['p']))
                    pred['reliable'] = reliable
                marks=[]
                for p, until, uncertainty in self.frozen.get(aid, []):
                    uncertainty += 0. if reliable else .5
                    if until>t and uncertainty<=5.:
                        marks.append((sub(rotate(p,-turn),delta),until,uncertainty))
                self.frozen[aid] = marks
            else:
                self.frozen.pop(aid, None)
            current[aid] = dict(state=s, preds=preds, edges=edges, walls=walls, t=t,
                                action=by_action.get(aid, {}))
        self.previous = current
        self.frozen = {k:v for k,v in self.frozen.items() if k in live}
        return current

    def _step(self, p, h, speed, peers, edges, farm):
        # Close-range candidates are within the predator's hearing radius.
        visible = [(norm(sub(q,p)),aid,q) for aid,q in peers.items() if norm(sub(q,p))<=60.]
        if not visible:
            return None
        d, target, q = min(visible)
        angle = wrap(math.atan2(q[1]-p[1],q[0]-p[0])-h)
        turn = max(-.3,min(.3,angle*.5)) if abs(angle)>.05 else 0.
        direction = h + (turn if abs(angle)>.05 else angle)
        delta = rotate((min(speed,d),0.),direction)
        out = (p[0]+delta[0],p[1]+delta[1])
        # Known walls: admit only an unobstructed forecast; do not guess the
        # engine's collision-redirection choice from an incomplete local map.
        if any(edge_distance(out,a,b)<10.5 for a,b in edges):
            return None
        return out, wrap(h+turn), target

    def apply(self, hv, states, t, actions, score=None, payout_limit=None):
        current = self._update(states, actions, t, score)
        if not hv.enabled or hv.harvests>=hv.max_harvests or t-hv.last_t<hv.cooldown or len(states)<hv.min_free+2:
            return actions
        by_id = {s['agent_id']:s for s in states}
        by_action = {a['agent_id']:a for a in actions}
        choices=[]
        for i,s in enumerate(states):
            if i==0: continue
            aid, doomed = s['agent_id'], states[i-1]['agent_id']
            c = current[aid]
            # Public peer positions are in this agent's local frame. Their
            # ordinary actions still run during the burst; the doomed one dies.
            peers={aid:(0.,0.)}; future={aid:(0.,0.)}
            for o in s.get('observations',[]):
                if o['type'].lower()!='agent' or 'id' not in o: continue
                peer=o['id']; p=point(o); peers[peer]=p
                if peer==doomed: continue
                if peer in by_id:
                    v=rotate(move(by_id[peer],by_action.get(peer,{})),heading(o))
                    future[peer]=(p[0]+v[0],p[1]+v[1])
                else: future[peer]=p
            for pred in c['preds']:
                d=norm(pred['p'])
                if d>hv.trigger: continue
                if any(norm(sub(pred['p'],p))<18.+u for p,until,u in self.frozen.get(aid,[])):
                    self.stats['confirmed_frozen']+=1; continue
                speed=pred.get('speed')
                if speed is None or not pred.get('reliable'):
                    self.stats['untracked']+=1; continue
                if speed<1.:
                    self.stats['stationary']+=1; continue
                # Observed displacement incorporates terrain and low-energy
                # speed. Allow a fast predator to drop to walking this tick.
                slow=speed*11./15. if any(abs(speed-v)<.3 for v in (4.5,7.5,12.,15.)) else speed
                # The target's biome is public even though the predator's is
                # not. Cover a crossing into that biome during either move.
                slow=min(slow,11.*PENALTY[s['biome']])
                distances=[]; ends=[]
                for v in (slow,speed):
                    first=self._step(pred['p'],pred['h'],v,peers,c['walls'],aid)
                    if first is None: break
                    second=self._step(first[0],first[1],v,future,c['walls'],aid)
                    if second is None or second[2]!=aid: break
                    distances.append(norm(second[0])); ends.append(second[0])
                if len(distances)!=2 or max(distances)>=15.-self.margin:
                    self.stats['no_contact']+=1; continue
                # A low-energy predecessor of the sacrifice can itself die,
                # skip the sacrifice and leave the farm's death check active.
                if i>=2 and states[i-2]['energy']<15.:
                    self.stats['unsafe_predecessor']+=1; continue
                choices.append((max(distances),d,aid,doomed,ends[-1],speed))
        if not choices: return actions
        expected_d,d,farm,doomed,end,speed=min(choices)
        s=by_id[farm]
        budget=hv.budget
        if payout_limit is not None:
            payout_limit=float(payout_limit)
            if not math.isfinite(payout_limit) or payout_limit<=0.:
                return actions
            budget=min(budget,max(0,int(math.floor((100.*payout_limit+s['energy'])/.5))))
            while budget>0 and (budget*.5-s['energy'])/100.>payout_limit:
                budget-=1
        if budget*.5<=s['energy']+1.:
            return actions
        n=max(0,int(math.ceil((by_id[doomed]['energy']+1.)/.5)))
        def drain(aid):
            return dict(agent_id=aid,move_distance=0.,move_direction=0.,turn_angle=PI,spawn_agent=False)
        out=[a for a in actions if a['agent_id'] not in (farm,doomed)]
        out += [drain(doomed)]*n+[drain(farm)]*budget
        hv.harvests+=1; hv.last_t=t
        payout=(budget*.5-s['energy'])/100.
        hv.log.append(dict(t=round(t,1),farm=farm,doomed=doomed,pred_d=round(d,1),farm_e=s['energy'],
                           budget=budget,expected_score=payout,actions=len(out),predicted_contact=expected_d,
                           observed_speed=speed))
        # Transfer confirmation uses only the next public score and disappearance.
        # Project the predicted resting position into surviving observers' frames.
        marks={}
        for o in s.get('observations',[]):
            if o['type'].lower()=='agent' and o.get('id') not in (farm,doomed) and 'rel_dir' in o:
                marks[o['id']]=rotate(sub(end,point(o)),-heading(o))
        reserve=200.+sum(max(0.,v['energy']) for v in states if v['agent_id'] not in (farm,doomed))
        until=t+max(0.,(budget*.5-s['energy']-reserve)/30.)
        self.pending=dict(t=t,score=score,payout=payout,farm=farm,marks=marks,until=until)
        for aid in (farm,doomed):
            if aid in self.previous:
                self.previous[aid]['action']=dict(move_distance=0.,turn_angle=0.)
        return out
