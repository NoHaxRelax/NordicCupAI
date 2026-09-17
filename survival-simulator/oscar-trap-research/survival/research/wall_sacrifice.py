"""Whether a depleted guide can establish a wall trap at low resource cost."""
import json
from wall_handoff import run,ROOT
if __name__=='__main__':
 rows=[run(energy=e,width=w,height=80,gap=90,peel=0,offset=o,heading=h) for e in [20,75,150] for w in [30,32,35,38] for o in [-15,15] for h in [-.3,.3]]
 (ROOT/'results/wall-sacrifice.json').write_text(json.dumps(dict(scope='Arranged geometry,150energy hidden bait, depleted/young guide deliberately waits at near face; no food/new predators; original physics. Guide is expendable, no successful safe escape implied.',runs=rows),indent=2,default=lambda v:v.item())+'\n')
 for e in [20,75,150]:
  rs=[r for r in rows if r['energy']==e]
  print(e,sum(r['success'] for r in rs),'/',len(rs),'guide-deaths',sum(not r['guide_alive'] for r in rs),flush=True)
