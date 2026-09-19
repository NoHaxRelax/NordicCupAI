"""Walking ETA from visited terrain samples; unknown route remains river-slow.

Samples are points, not biome polygons. Nearby interpolation is an estimate,
so the caller must retain its handoff overlap and re-evaluate during travel.
"""
import math

TERRAIN = {'forest':1., 'grassland':1., 'desert':.8, 'swamp':.5, 'river':.3}


class ObservedBaitTravel:
    def __init__(self):
        self.frame = None
        self.next_update = 0.
        self.cells = {}

    def update(self, group, now):
        frame = (group.group_id,group.frame_revision)
        if frame == self.frame and now < self.next_update: return
        self.frame,self.next_update = frame,now+.5
        self.cells = {}
        for sample in group.biomes.values():
            if sample.uncertainty > 4.: continue
            key = tuple(math.floor(v/24.) for v in sample.position)
            self.cells.setdefault(key,[]).append((tuple(sample.position),TERRAIN[sample.biome],sample.uncertainty))

    def estimate(self, points, state):
        distance = requested = known = 0.
        for start,end in zip(points,points[1:]):
            length = math.dist(start,end)
            steps = max(1,math.ceil(length/8.))
            for step in range(steps):
                t = (step+.5)/steps
                p = (start[0]+t*(end[0]-start[0]),start[1]+t*(end[1]-start[1]))
                cx,cy = (math.floor(v/24.) for v in p)
                samples = [sample for x in range(cx-1,cx+2) for y in range(cy-1,cy+2)
                           for sample in self.cells.get((x,y),()) if math.dist(sample[0],p)+sample[2]<=12.]
                # Conflicting nearby labels use the slower terrain.
                factor = min((s[1] for s in samples),default=.3)
                part = length/steps
                distance += part; requested += part/factor; known += part*bool(samples)
        walk = max(.1,min(state['speed'],state['sprint_speed']))
        return dict(seconds=requested/(walk*10.),walk_cost=.05*requested+6.,
                    distance=distance,known_fraction=known/distance if distance else 1.,
                    method='nearby_observed_samples_unknown_river')
