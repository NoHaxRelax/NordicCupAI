from concurrent.futures import ProcessPoolExecutor
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE),str(ROOT)]
from generic_corner_detector import detect
def one(seed):
 from src.core import SimulationCore
 from models.entrapment.observed_trap_sites import our_sites
 e=SimulationCore(seed=seed,starting_agents=0,starting_trees=0,starting_fruits=1).env
 static=dict(width=e.width,height=e.height,obstacles=[(o.x,o.y,o.width,o.height) for o in e.obstacles]);sites=detect(static)
 return dict(seed=seed,ordinary=len(our_sites(static)),generic=sites,static=static if sites else None)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
 seeds=[1214781215,1036448012,1989373803,1035850866]+list(range(96))
 with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(one,seeds))
 a.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(maps=100,with_generic=sum(bool(x['generic']) for x in rows),extra=sum(not x['ordinary'] and bool(x['generic']) for x in rows),known=[(x['seed'],x['ordinary'],len(x['generic'])) for x in rows[:4]]),indent=2))
if __name__=='__main__':main()
