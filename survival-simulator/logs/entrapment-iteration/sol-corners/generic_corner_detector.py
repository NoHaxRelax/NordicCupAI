"""Static full-map detector for generic predator-contact-safe corner pockets."""
from __future__ import annotations
import math
from shapely.geometry import Point,LineString,box
from shapely.ops import unary_union,nearest_points

def _parts(g,min_area=1.):
    xs=list(g.geoms) if g.geom_type in ('MultiPolygon','GeometryCollection') else [g]
    return [x for x in xs if x.geom_type=='Polygon' and x.area>=min_area]

def _space(width,height,rects,radius):
    return box(radius,radius,width-radius,height-radius).difference(unary_union(
        [box(x-radius,y-radius,x+w+radius,y+h+radius) for x,y,w,h in rects]))

def detect(static):
    """Return site-contract dictionaries without map seeds or native state."""
    width,height=map(float,(static['width'],static['height']));rects=[tuple(map(float,r)) for r in static['obstacles']]
    agent=_space(width,height,rects,5.01);predator=_space(width,height,rects,10.01);route=_space(width,height,rects,11.01)
    pred_parts=_parts(predator);route_parts=_parts(route)
    if not pred_parts or not route_parts:return []
    front_component=max(pred_parts,key=lambda x:x.area);front_route=max(route_parts,key=lambda x:x.area)
    safe=agent.difference(predator.buffer(15.05,resolution=24));sites=[]
    for region in _parts(safe,.25):
        # Choose the contact-safe point closest to the main predator component;
        # this maximizes bait preference over an approaching replacement.
        boundary=region.boundary;_,near=nearest_points(front_component,boundary)
        candidates=[region.representative_point(),near]
        minx,miny,maxx,maxy=region.bounds
        for x in range(math.floor(minx),math.ceil(maxx)+1):
            for y in range(math.floor(miny),math.ceil(maxy)+1):
                p=Point(x,y)
                if region.covers(p):candidates.append(p)
        candidates=[p for p in candidates if p.distance(predator)>=15.049]
        if not candidates:continue
        bait=min(candidates,key=lambda p:(p.distance(front_component),-p.distance(region.boundary)))
        # A normal guide needs a radius-11 reachable handoff in the main open
        # component, 18--38 units from bait and with useful local clearance.
        handoffs=[]
        for radius in (28.,32.,35.,25.,38.):
            for k in range(72):
                q=Point(bait.x+radius*math.cos(2*math.pi*k/72),bait.y+radius*math.sin(2*math.pi*k/72))
                if front_route.covers(q):handoffs.append(q)
        if not handoffs:continue
        handoff=max(handoffs,key=lambda q:q.distance(front_route.boundary))
        # Rear must be agent-reachable by a straight segment, outside the main
        # predator component, and generally opposite the handoff.
        vx,vy=bait.x-handoff.x,bait.y-handoff.y;n=math.hypot(vx,vy) or 1.;rear=None
        for distance in (70.,55.,40.,30.,25.):
            for angle in (0.,.2,-.2,.4,-.4,.7,-.7):
                c,s=math.cos(angle),math.sin(angle);ux=(vx*c-vy*s)/n;uy=(vx*s+vy*c)/n
                q=Point(bait.x+ux*distance,bait.y+uy*distance)
                if agent.covers(q) and agent.covers(LineString([bait,q])) and not front_component.covers(q):rear=q;break
            if rear:break
        if rear is None:continue
        inward=((bait.x-handoff.x)/math.dist((bait.x,bait.y),(handoff.x,handoff.y)),
                (bait.y-handoff.y)/math.dist((bait.x,bait.y),(handoff.x,handoff.y)))
        sites.append(dict(site_kind='generic_corner_safe_region',goal=[bait.x,bait.y],handoff=[handoff.x,handoff.y],
            mouth=[bait.x,bait.y],inward=list(inward),cross=[-inward[1],inward[0]],gap=None,overlap=0.,
            far=[handoff.x,handoff.y],hold=[handoff.x,handoff.y],runup=[handoff.x,handoff.y],approach_lane_offset=0.,
            offset_approach=False,obstacle_indices=[],axis=None,other_mouth=[rear.x,rear.y],replacement_entry=[rear.x,rear.y],
            second_access_clear=True,second_access_radius=5.01,bait_depth=None,boundary_indices=[],
            geometric_replacement_access_only=True,minimum_predator_distance=bait.distance(predator),safe_area=region.area,
            rear_contract='straight radius-5.01 access outside main/front radius-10.01 component'))
    sites.sort(key=lambda s:(s['minimum_predator_distance'],s['safe_area']))
    return sites
