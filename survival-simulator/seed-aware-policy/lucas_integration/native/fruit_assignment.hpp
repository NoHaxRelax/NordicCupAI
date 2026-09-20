// Maximize estimated useful food minus travel cost across mutually exclusive
// agent/fruit pairs. Existing claims are retained; each agent may also do nothing.
int fruit_variant()const{return resource_mode>=146&&resource_mode<=149?resource_mode-88:resource_mode;}
bool economic_fruit_mode()const{return model_full()&&((resource_mode>=58&&resource_mode<=61)||(resource_mode>=146&&resource_mode<=149));}
double fruit_pair_value(const FPair& pr,Group& g){
 const auto&s=st(pr.a);const auto&m=M(pr.a);auto f=g.fruits.at(pr.fid);
 auto it=matched_fruit.find(f.get());if(it==matched_fruit.end())return -1e6;
 double penalty=pmax(.1,MOVE_PENALTY[s.biome]);
 double travel=pr.d/pmax(1.,pmin(s.speed,s.sprint)*penalty*10.);
 if(travel+.3>=(100.-it->second.age)/2.)return -1e6;
 double move_cost=pr.d*.05/penalty;
 double drain=m.old?1.+.1*s.age:1.;
 if(s.energy<move_cost+drain*travel+2.)return -1e6;
 double energy=pmin(60.,it->second.energy+2.*travel);
 double useful=pmin(energy,pmax(0.,s.max_energy-s.energy+move_cost+drain*travel));
 if(fruit_variant()==60)return 300.-20.*pr.bucket-.2*pr.d;
 double value=(useful-move_cost-drain*travel)*(1.+pmax(0.,180.-s.energy)/90.);
 if(!m.heir_done&&s.age>=heir_age_for(s.aid)-5.&&s.energy<P.heir_reserve)value+=25.;
 if(fruit_variant()==59&&m.old)value*=.25;
 return value;
}
void economic_assign(Group& g,const std::vector<FPair>& pairs){
 if(pairs.empty())return;
 std::map<int64_t,int> row_index,col_index;
 for(auto&pr:pairs){row_index[pr.a]=0;col_index[pr.fid]=0;}
 int n=0,m=0;for(auto&v:row_index)v.second=n++;for(auto&v:col_index)v.second=m++;
 std::vector<int64_t> agents(n),fruits(m);for(auto&v:row_index)agents[v.second]=v.first;for(auto&v:col_index)fruits[v.second]=v.first;
 std::vector<std::vector<double>> cost(n,std::vector<double>(m+n,0.));
 for(auto&row:cost)std::fill(row.begin(),row.begin()+m,1e6);
 for(auto&pr:pairs)cost[row_index.at(pr.a)][col_index.at(pr.fid)]=-fruit_pair_value(pr,g);
 std::vector<int> chosen(n,-1);
 if(fruit_variant()==61){
  struct Edge {double cost;int r,c;};std::vector<Edge> edges;
  for(int r=0;r<n;++r)for(int c=0;c<m;++c)if(cost[r][c]<0.)edges.push_back({cost[r][c],r,c});
  std::sort(edges.begin(),edges.end(),[](const Edge&a,const Edge&b){if(a.cost!=b.cost)return a.cost<b.cost;if(a.r!=b.r)return a.r<b.r;return a.c<b.c;});
  std::vector<bool> used(m,false);for(auto&e:edges)if(chosen[e.r]<0&&!used[e.c]){chosen[e.r]=e.c;used[e.c]=true;}
 }else{
  int cols=m+n;std::vector<double> u(n+1),v(cols+1);std::vector<int> p(cols+1),way(cols+1);
  for(int i=1;i<=n;++i){
   p[0]=i;int j0=0;std::vector<double> minv(cols+1,1e30);std::vector<bool> used(cols+1,false);
   do{
    used[j0]=true;int i0=p[j0],j1=0;double delta=1e30;
    for(int j=1;j<=cols;++j)if(!used[j]){
     double cur=cost[i0-1][j-1]-u[i0]-v[j];if(cur<minv[j]){minv[j]=cur;way[j]=j0;}
     if(minv[j]<delta){delta=minv[j];j1=j;}
    }
    for(int j=0;j<=cols;++j)if(used[j]){u[p[j]]+=delta;v[j]-=delta;}else minv[j]-=delta;
    j0=j1;
   }while(p[j0]!=0);
   do{int j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0);
  }
  for(int j=1;j<=m;++j)if(p[j]&&cost[p[j]-1][j-1]<0.)chosen[p[j]-1]=j-1;
 }
 for(int i=0;i<n;++i)if(chosen[i]>=0){auto f=g.fruits.at(fruits[chosen[i]]);f->has_claim=true;f->claimed=agents[i];M(agents[i]).has_fruit=true;M(agents[i]).fruit=f->id;}
}
