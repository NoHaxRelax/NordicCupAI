"""A* using only the supplied static edges. No simulator state is imported.

Route cost is geometric length: no global biome map is part of this API, so
this does not claim minimum travel time across different terrain types.
"""
import heapq
import math

from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep

PREDATOR_CLEARANCE = 11.0  # Native radius 10, plus one unit of margin.


def fixed_frame(bait, edges):
    """Use bait as origin and an invariant map edge as the fixed x axis.

    Length and distance to the fixed bait identify the same edge after the
    guide translates/rotates, even if the edge list is reordered.
    """
    candidates = [e for e in edges if math.dist(*e) > 1e-8]
    if not candidates:
        raise ValueError('At least one nonzero map edge is required')
    def key(edge):
        distances = sorted(math.dist(p, bait) for p in edge)
        return (round(math.dist(*edge), 5), *(round(d, 5) for d in distances))
    a, b = max(candidates, key=key)
    if math.dist(a, bait) < math.dist(b, bait):
        a, b = b, a
    length = math.dist(a, b)
    c, s = (b[0]-a[0])/length, (b[1]-a[1])/length

    def to_fixed(point):
        x, y = point[0]-bait[0], point[1]-bait[1]
        return x*c+y*s, -x*s+y*c

    def to_local(point):
        return bait[0]+point[0]*c-point[1]*s, bait[1]+point[0]*s+point[1]*c

    return to_fixed, to_local


class RoutePlanner:
    def __init__(self, edges, clearance=PREDATOR_CLEARANCE):
        self.clearance = clearance
        # Sweeping a square-capped buffer around each wall matches the native
        # rectangular collision bounds conservatively, including wall corners.
        walls = unary_union([LineString(e).buffer(clearance, cap_style=3, join_style=2)
                             for e in edges if math.dist(*e) > 1e-8])
        self.blocked = prep(walls)
        points = [p for e in edges for p in e]
        self.bounds = (min(p[0] for p in points), min(p[1] for p in points),
                       max(p[0] for p in points), max(p[1] for p in points))
        self.grids = {}

    def free(self, point):
        x0,y0,x1,y1 = self.bounds
        return x0 <= point[0] <= x1 and y0 <= point[1] <= y1 and not self.blocked.intersects(Point(point))

    def clear(self, a, b):
        return self.free(a) and self.free(b) and not self.blocked.intersects(LineString([a,b]))

    def _grid(self, step):
        if step not in self.grids:
            x0,y0,x1,y1 = self.bounds
            nodes = {}
            for i in range(math.ceil((x1-x0)/step)):
                for j in range(math.ceil((y1-y0)/step)):
                    p = x0+(i+.5)*step, y0+(j+.5)*step
                    if self.free(p): nodes[i,j] = p
            self.grids[step] = nodes
        return self.grids[step]

    def plan(self, start, goal):
        if not self.free(start) or not self.free(goal):
            return []
        if self.clear(start, goal):
            return [goal]
        # Retry at finer resolution so a coarse grid doesn't hide a valid lane.
        for step in (20., 10.):
            route = self._astar(start, goal, step)
            if route:
                return self.smooth(start, route)
        return []

    def _astar(self, start, goal, step):
        nodes = self._grid(step)
        start_links = [k for k,p in nodes.items() if math.dist(p,start)<=step*3 and self.clear(start,p)]
        goal_links = {k:math.dist(p,goal) for k,p in nodes.items()
                      if math.dist(p,goal)<=step*3 and self.clear(p,goal)}
        if not start_links or not goal_links:
            return []
        distances = {k:math.dist(start,nodes[k]) for k in start_links}
        parent = {k:None for k in start_links}
        queue = [(g+math.dist(nodes[k],goal),g,k) for k,g in distances.items()]
        heapq.heapify(queue)
        best, winner = math.inf, None
        while queue:
            f,g,key = heapq.heappop(queue)
            if g != distances[key]: continue
            if f >= best: break
            if key in goal_links and g+goal_links[key] < best:
                best,winner = g+goal_links[key],key
            i,j = key
            for di,dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                other = i+di,j+dj
                if other not in nodes or not self.clear(nodes[key],nodes[other]): continue
                cost = g+math.dist(nodes[key],nodes[other])
                if cost < distances.get(other,math.inf):
                    distances[other],parent[other] = cost,key
                    heapq.heappush(queue,(cost+math.dist(nodes[other],goal),cost,other))
        if winner is None: return []
        route = [goal]
        while winner is not None:
            route.append(nodes[winner])
            winner = parent[winner]
        return route[::-1]

    def smooth(self, start, route):
        """Remove grid zigzags only when the entire predator-width segment fits."""
        result=[]
        while route:
            reachable=[i for i,p in enumerate(route) if self.clear(start,p)]
            if not reachable: return []
            i=max(reachable)
            start=route[i]
            result.append(start)
            route=route[i+1:]
        return result


def navigation_plan(bait, edges, target, memory):
    to_fixed,to_local = fixed_frame(bait,edges)
    start,goal = to_fixed((0.,0.)),to_fixed(target)
    if '_route_planner' not in memory:
        memory['_route_planner'] = RoutePlanner([[to_fixed(a),to_fixed(b)] for a,b in edges])
    planner=memory['_route_planner']
    if not planner.free(start):
        # The local survival controller uses an agent-width corridor. Native
        # deflection can also leave us near a wall. Rejoin an open predator lane
        # instead of asking a radius-11 planner to start inside its own buffer.
        if '_lane_rejoin_planner' not in memory:
            memory['_lane_rejoin_planner'] = RoutePlanner(
                [[to_fixed(a), to_fixed(b)] for a, b in edges], clearance=5.01)
        agent_planner = memory['_lane_rejoin_planner']
        candidates = []
        for radius in (8., 12., 18., 25., 35.):
            for i in range(24):
                angle = i*math.tau/24
                point = (start[0]+radius*math.cos(angle), start[1]+radius*math.sin(angle))
                if planner.free(point) and agent_planner.clear(start, point):
                    candidates.append((radius+math.dist(point, goal), point))
        for _, point in sorted(candidates)[:8]:
            rest = planner.plan(point, goal)
            if not rest:
                continue
            waypoint = to_local(point)
            memory['_route'] = [point]+rest
            memory['_route_goal'] = goal
            memory['_route_replans'] = memory.get('_route_replans', 0)+1
            return dict(dist=math.dist(start, point)+sum(math.dist(a,b) for a,b in zip([point]+rest,rest)),
                        dist_to_point=math.hypot(*waypoint),dir=math.atan2(waypoint[1],waypoint[0]),
                        waypoints=[to_local(p) for p in [point]+rest],clearance=5.01,
                        mode='rejoin_predator_lane',replans=memory['_route_replans'])
        return None
    route=memory.get('_route',[])
    target_changed=math.dist(goal,memory.get('_route_goal',goal))>0.01
    # Keep an existing turn until actually reached; shortcut only along a
    # collision-clear segment. Replan if the engine has displaced the guide.
    while route and math.dist(start,route[0])<1.:
        route=route[1:]
    if target_changed or not route or not planner.clear(start,route[0]):
        route=planner.plan(start,goal)
        memory['_route_replans']=memory.get('_route_replans',0)+1
    elif len(route)>1:
        reachable=[i for i,p in enumerate(route) if planner.clear(start,p)]
        if reachable: route=route[max(reachable):]
    memory['_route'],memory['_route_goal']=route,goal
    if not route:
        return None
    waypoint=to_local(route[0])
    points=[start]+route
    return dict(dist=sum(math.dist(a,b) for a,b in zip(points,points[1:])),
                dist_to_point=math.hypot(*waypoint),dir=math.atan2(waypoint[1],waypoint[0]),
                waypoints=[to_local(p) for p in route],clearance=planner.clearance,
                replans=memory['_route_replans'])
