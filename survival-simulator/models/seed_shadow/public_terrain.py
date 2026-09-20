"""Terrain constraints obtained exclusively from ordinary agent observations.

Boundary anchoring is adapted from Oscar's native_common.hpp at 5638ee7.
The official 1600 x 1200 map and 30-unit boundary thickness are game constants.
No simulator object, hidden position, or evaluation seed enters this module.
"""
import math

LABELS = {name: i for i, name in enumerate(('forest', 'swamp', 'desert', 'grassland'))}


def agent_states(body):
    return body.get('agent_status', body.get('observations', []))


def frame_poses(body, fresh_ids=None):
    agents = {a['agent_id']: a for a in agent_states(body)}
    poses = {}
    for aid, agent in agents.items():
        if fresh_ids is not None and aid not in fresh_ids:
            continue
        for obs in agent['observations']:
            if obs['type'] != 'Edge':
                continue
            (ax, ay), (bx, by) = obs['coords']
            length = math.hypot(bx-ax, by-ay)
            vertical = abs(length-1200) < 1e-6
            horizontal = abs(length-1600) < 1e-6
            if not (vertical or horizontal):
                continue
            h = (math.pi/2 if vertical else 0) - math.atan2(by-ay, bx-ax)
            rx = math.cos(h)*ax - math.sin(h)*ay
            ry = math.sin(h)*ax + math.cos(h)*ay
            candidates = []
            for wall in ((30,1570) if vertical else (30,1170)):
                x, y = (wall-rx, -ry) if vertical else (-rx, wall-ry)
                if 30+1e-6 < x < 1570-1e-6 and 30+1e-6 < y < 1170-1e-6:
                    candidates.append((x,y,h))
            if len(candidates) == 1:
                poses[aid] = candidates[0]
                break
    todo = list(poses)
    for aid in todo:
        if fresh_ids is not None and aid not in fresh_ids:
            continue
        x,y,h = poses[aid]
        for obs in agents[aid]['observations']:
            other = obs.get('id')
            if obs['type'] != 'Agent' or other not in agents or other in poses:
                continue
            # atan2(0,0) is zero in both directions, not bearings separated by pi.
            # A newborn can share its parent's exact position after a blocked birth.
            # Its heading cannot be inferred using the ordinary reverse-bearing rule.
            if obs['distance']<1e-7:
                continue
            angle = h + obs['angle']
            poses[other] = (x+obs['distance']*math.cos(angle),
                            y+obs['distance']*math.sin(angle),
                            angle+math.pi-obs['rel_dir'])
            todo.append(other)
    return poses


class TerrainSamples:
    def __init__(self):
        self.points = {}
        self.ages = {}
        self.last_poses = {}
        self.last_fresh = set()

    def observe(self, body):
        # non_agent_step mutates the agents list while iterating it. A skipped
        # agent keeps its previous observations even after this tick's movement.
        # Its age does not advance either, exposing that stale observation cache.
        fresh = {a['agent_id'] for a in agent_states(body)
                 if a['age'] > self.ages.get(a['agent_id'],0.)}
        self.last_fresh = fresh
        poses = frame_poses(body, fresh)
        self.ages = {a['agent_id']: a['age'] for a in agent_states(body)}
        self.last_poses = poses
        for agent in agent_states(body):
            pose = poses.get(agent['agent_id'])
            label = LABELS.get(agent['biome'])
            if pose is None or label is None:  # River overrides the underlying label.
                continue
            x,y,_ = pose
            if not (30<x<1570 and 30<y<1170):
                continue
            # Float error must not change which integer pixel is sampled.
            if min(abs(x-round(x)),abs(y-round(y))) < 1e-7:
                continue
            key = (int(x),int(y))
            if key in self.points and self.points[key] != label:
                raise ValueError('Contradictory public terrain constraints')
            self.points[key] = label

    def rows(self):
        # Spread checks across labels and positions before checking dense clusters.
        groups = {label: [] for label in range(4)}
        for (x,y),label in sorted(self.points.items()):
            groups[label].append((x,y,label))
        out = []
        while any(groups.values()):
            for rows in groups.values():
                if rows:
                    out.append(rows.pop())
        return out
