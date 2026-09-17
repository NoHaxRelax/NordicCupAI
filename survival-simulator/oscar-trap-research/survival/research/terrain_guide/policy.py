"""Static terrain-cost routing variant; all dynamic control remains observation-only."""
import math,heapq
from simple_chase.policy_v21_replaceable import SimpleChase as Base

class TerrainGuide(Base):
    GRID=20.
    def __init__(self,*a,**k):
        super().__init__(*a,**k)
        self.nav_cache={}

    def _cost(self,a,b):
        distance=math.dist(a,b)
        n=max(1,math.ceil(distance/5.))
        return distance*sum(1/self._terrain_modifier((a[0]+(b[0]-a[0])*(i+.5)/n,a[1]+(b[1]-a[1])*(i+.5)/n))**2 for i in range(n))/n

    def _grid(self,radius):
        radius=round(radius,2)
        if radius in self.nav_cache:return self.nav_cache[radius]
        step=self.GRID;nodes={}
        for i in range(math.ceil(self.width/step)):
            for j in range(math.ceil(self.height/step)):
                p=(i*step+step/2,j*step+step/2)
                if self._free(p,radius):nodes[i,j]=p
        adjacency={key:[] for key in nodes}
        for key,p in nodes.items():
            i,j=key
            for di,dj in ((1,0),(0,1),(1,1),(1,-1)):
                other=i+di,j+dj
                if other in nodes and self._clear(p,nodes[other],radius):
                    cost=self._cost(p,nodes[other]);adjacency[key].append((other,cost));adjacency[other].append((key,cost))
        self.nav_cache[radius]=nodes,adjacency
        return nodes,adjacency

    def _path(self,start,target,radius=11.):
        direct=math.dist(start,target)
        if self._clear(start,target,radius) and self._cost(start,target)<=direct*1.2:return [target]
        nodes,edges=self._grid(radius)
        sr=radius if self._free(start,radius) else 5.01
        near_start=[key for key,p in sorted(nodes.items(),key=lambda row:math.dist(start,row[1]))[:32] if math.dist(start,nodes[key])<150 and self._clear(start,nodes[key],sr)]
        near_end={key:self._cost(p,target) for key,p in sorted(nodes.items(),key=lambda row:math.dist(target,row[1]))[:32] if math.dist(target,p)<150 and self._clear(p,target,radius)}
        if not near_start or not near_end:return super()._path(start,target,radius)
        distance={key:self._cost(start,nodes[key]) for key in near_start}
        previous={key:None for key in near_start}
        queue=[(cost+math.dist(nodes[key],target),cost,key) for key,cost in distance.items()];heapq.heapify(queue)
        winner=None;best=math.inf
        while queue:
            bound,cost,key=heapq.heappop(queue)
            if cost!=distance[key]:continue
            if bound>=best:break
            if key in near_end and cost+near_end[key]<best:best=cost+near_end[key];winner=key
            for other,step in edges[key]:
                value=cost+step
                if value<distance.get(other,math.inf):
                    distance[other]=value;previous[other]=key
                    heapq.heappush(queue,(value+math.dist(nodes[other],target),value,other))
        if winner is None:return super()._path(start,target,radius)
        path=[target];key=winner
        while key is not None:path.append(nodes[key]);key=previous[key]
        path.reverse()
        # Preserve terrain-aware detours while removing unnecessary grid turns.
        simplified=[];current=start;index=0
        while index<len(path):
            best_index=index;cost=0.;last=current
            for j in range(index,len(path)):
                cost+=self._cost(last,path[j]);last=path[j]
                if self._clear(current,path[j],sr if not simplified else radius) and self._cost(current,path[j])<=cost*1.05:best_index=j
            simplified.append(path[best_index]);current=path[best_index];index=best_index+1
        return simplified
