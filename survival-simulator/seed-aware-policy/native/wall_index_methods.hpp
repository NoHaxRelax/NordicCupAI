// Included inside orchard::Policy after confirmed(). This is a broad phase only:
// original exact collision predicates and original wall order remain unchanged.
static int wi_cell(double value,int n) {
    if(value<=0)return 0;if(value>=50.*(n-1))return n-1;
    return int(std::floor(value/50.));
}
void wi_prepare(Group& g) {
    g.wi_nx=int(std::floor(W/50.))+1;g.wi_ny=int(std::floor(H/50.))+1;
    g.wi_cells.clear();g.wi_cells.resize(g.wi_nx*g.wi_ny);g.wi_confirmed.clear();g.wi_boxes.clear();
    for(size_t i=0;i<g.walls.size();i++) {
        const auto& w=g.walls[i];if(!confirmed(w))continue;g.wi_confirmed.push_back(int(i));
        double x0=w.horiz?pmin(w.lo,w.hi):w.c,x1=w.horiz?pmax(w.lo,w.hi):w.c;
        double y0=w.horiz?w.c:pmin(w.lo,w.hi),y1=w.horiz?w.c:pmax(w.lo,w.hi);
        for(int x=wi_cell(x0,g.wi_nx);x<=wi_cell(x1,g.wi_nx);x++)
            for(int y=wi_cell(y0,g.wi_ny);y<=wi_cell(y1,g.wi_ny);y++)g.wi_cells[x*g.wi_ny+y].push_back(int(i));
    }
    g.wi_ready=true;
}
const std::vector<int>& wi_query(const Group& g,double ax,double ay,double bx,double by) const {
    if(!g.wi_ready) {
        if(g.wi_all.size()!=g.walls.size()){g.wi_all.resize(g.walls.size());for(size_t i=0;i<g.walls.size();i++)g.wi_all[i]=int(i);}
        return g.wi_all;
    }
    // Padding only expands the candidate set, guarding grid-boundary rounding.
    int x0=wi_cell(ax-1e-7,g.wi_nx),x1=wi_cell(bx+1e-7,g.wi_nx);
    int y0=wi_cell(ay-1e-7,g.wi_ny),y1=wi_cell(by+1e-7,g.wi_ny);
    if((x1-x0+1)*(y1-y0+1)>64)return g.wi_confirmed;
    uint64_t key=(uint64_t(x0)<<48)|(uint64_t(x1)<<32)|(uint64_t(y0)<<16)|uint64_t(y1);
    auto old=g.wi_boxes.find(key);if(old!=g.wi_boxes.end())return old->second;
    if(g.wi_boxes.size()>=2048)g.wi_boxes.clear();
    std::vector<int> result;
    for(int x=x0;x<=x1;x++)for(int y=y0;y<=y1;y++) {
        const auto& cell=g.wi_cells[x*g.wi_ny+y];result.insert(result.end(),cell.begin(),cell.end());
    }
    std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end()),result.end());
    return g.wi_boxes.emplace(key,std::move(result)).first->second;
}
