"""Extract possible RNG leaks from public, globally anchored geometry.

The four uniform draws inside a rectangle have known order: width, height, x, y.
The ORDER BETWEEN RECTANGLES is not observed. Never feed the sorted collection to
an RNG solver as though it were generation order. Public fruit observations have
no IDs, and fruit may come from tree-relative spawning rather than uniform draws.
"""
import math
from .public_terrain import agent_states


class PublicGeometry:
    def __init__(self):
        self.horizontal={}
        self.vertical={}
        self.fruit_positions={}

    def observe(self,body,terrain):
        for agent in agent_states(body):
            aid=agent['agent_id']
            if aid not in terrain.last_fresh or aid not in terrain.last_poses:
                continue
            x,y,h=terrain.last_poses[aid];c,s=math.cos(h),math.sin(h)
            for obs in agent['observations']:
                if obs['type']=='Edge':
                    (ax,ay),(bx,by)=obs['coords']
                    ax,ay=x+c*ax-s*ay,y+s*ax+c*ay
                    bx,by=x+c*bx-s*by,y+s*bx+c*by
                    if abs(ay-by)<1e-7 and 30-1e-7<=abs(ax-bx)<=100+1e-7:
                        row=(min(ax,bx),max(ax,bx),(ay+by)/2)
                        self.horizontal.setdefault(tuple(round(z,6) for z in row),row)
                    elif abs(ax-bx)<1e-7 and 30-1e-7<=abs(ay-by)<=100+1e-7:
                        row=((ax+bx)/2,min(ay,by),max(ay,by))
                        self.vertical.setdefault(tuple(round(z,6) for z in row),row)
                elif obs['type']=='Fruit':
                    angle=h+obs['angle'];distance=obs['distance']
                    point=(x+distance*math.cos(angle),y+distance*math.sin(angle))
                    self.fruit_positions.setdefault(tuple(round(z,6) for z in point),
                        dict(x=point[0],y=point[1],first_seen=body.get('sim_time')))

    def rectangles(self):
        result={}
        for (x0,x1,y),hrow in self.horizontal.items():
            for (x,y0,y1),vrow in self.vertical.items():
                if x!=x0 or y!=y0:
                    continue
                if (x0,x1,y1) not in self.horizontal or (x1,y0,y1) not in self.vertical:
                    continue
                # Use original doubles; quantized values only match endpoints.
                left,right,top=hrow
                _,low,high=vrow
                result[(x0,y0,x1,y1)]=(left,top,right-left,high-low)
        return list(result.values())

    def evidence(self):
        return dict(rectangles=[dict(x=x,y=y,width=w,height=h,
            draw_order=['width','height','x','y'],generation_index=None,
            absolute_error_bound=1e-7) for x,y,w,h in self.rectangles()],
            fruit_positions=list(self.fruit_positions.values()),
            ready_for_linear_solver=False,
            blocker='Rectangle generation order and missing RNG draws are unknown')
