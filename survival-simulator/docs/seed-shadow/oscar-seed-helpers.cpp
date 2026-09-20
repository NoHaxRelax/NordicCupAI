using Points=std::map<std::pair<int,int>,int>;
struct Pose{double x,y,h;};
std::map<int64_t,Pose> frame_poses(const J& body){
    std::map<int64_t,Pose> poses;std::map<int64_t,const J*> agents;std::vector<int64_t> todo;
    for(auto& a:body.at("agent_status")){int64_t id=a.at("agent_id");agents[id]=&a;
        for(auto& o:a.at("observations")){if(o["type"]!="Edge")continue;double ax=o["coords"][0][0],ay=o["coords"][0][1],bx=o["coords"][1][0],by=o["coords"][1][1];
            double length=std::hypot(bx-ax,by-ay);bool vertical=std::abs(length-1200)<1e-6,horizontal=std::abs(length-1600)<1e-6;if(!vertical&&!horizontal)continue;
            double h=(vertical?PI/2:0.)-std::atan2(by-ay,bx-ax),rx=std::cos(h)*ax-std::sin(h)*ay,ry=std::sin(h)*ax+std::cos(h)*ay;
            std::vector<Pose> candidates;
            for(double wall:vertical?std::vector<double>{30.,1570.}:std::vector<double>{30.,1170.}){
                double x=vertical?wall-rx:-rx,y=vertical?-ry:wall-ry;
                if(30+1e-6<x&&x<1570-1e-6&&30+1e-6<y&&y<1170-1e-6)candidates.push_back({x,y,h});
            }
            if(candidates.size()==1){poses[id]=candidates[0];todo.push_back(id);break;}}
    }
    for(size_t i=0;i<todo.size();i++){auto p=poses[todo[i]];for(auto& o:agents[todo[i]]->at("observations")){
        if(o["type"]!="Agent"||!o.contains("id"))continue;int64_t id=o["id"];if(!agents.count(id)||poses.count(id))continue;
        double angle=p.h+o["angle"].get<double>(),d=o["distance"];poses[id]={p.x+d*std::cos(angle),p.y+d*std::sin(angle),angle+PI-o["rel_dir"].get<double>()};todo.push_back(id);}}
    return poses;
}
void add_samples(Points& points,const J& body){auto poses=frame_poses(body);for(auto& a:body.at("agent_status")){
    int64_t id=a["agent_id"];int label=biome_index(a["biome"]);if(label<0||label>=4||!poses.count(id))continue;auto p=poses[id];
    if(!(30<p.x&&p.x<1570&&30<p.y&&p.y<1170)||std::min(std::abs(p.x-std::round(p.x)),std::abs(p.y-std::round(p.y)))<1e-7)continue;
    auto key=std::make_pair(int(p.x),int(p.y));if(points.count(key)&&points[key]!=label)throw std::runtime_error("Contradictory public terrain samples");points[key]=label;}}
J samples_json(const Points& points){std::map<int,std::vector<J>> groups;for(auto [p,label]:points)groups[label].push_back({p.first,p.second,label});J out=J::array();bool added=true;while(added){added=false;for(auto& [label,v]:groups)if(!v.empty()){out.push_back(v.back());v.pop_back();added=true;}}return out;}
bool terrain_matches(uint32_t seed,const Points& points){PyRandom rng;rng.init_by_array({seed});int x[10],y[10],labels[10];for(int i=0;i<10;i++){x[i]=rng.randbelow(1600);y[i]=rng.randbelow(1200);}for(auto& label:labels)label=rng.randbelow(4);for(auto [p,label]:points){int best=0,d=INT_MAX;for(int i=0;i<10;i++){int dx=x[i]-p.first,dy=y[i]-p.second,dd=dx*dx+dy*dy;if(dd<d){d=dd;best=i;}}if(labels[best]!=label)return false;}return true;}
