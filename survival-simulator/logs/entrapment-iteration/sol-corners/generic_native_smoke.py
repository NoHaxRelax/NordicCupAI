"""One native normal-guide encounter for each newly covered detector map."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
ENCOUNTERS=[1382370471,854619562,1643999355,2085883922,1374628056,772838390]
def worker(job):
 index,row,encounter_seed,root=job
 import guide_multi_corner_override as o
 site=row['generic'][0];o.MAP_SEED=row['seed'];o.GOAL=tuple(site['goal']);o.FRONT=tuple(site['handoff']);o.REAR=tuple(site['replacement_entry'])
 # Preserve original site metadata while using the shared runner contract.
 o.site_contract=lambda:site
 out=Path(root)/f'case-{index:02d}-map-{row["seed"]}-enc-{encounter_seed}';out.mkdir(parents=True)
 folder=o.run_one(encounter_seed,out,True);s=json.loads((folder/'summary.json').read_text())
 return dict(map_seed=row['seed'],encounter_seed=encounter_seed,folder=str(folder.relative_to(Path(root))),outcome=s['outcome'],
  initial_min_held=s['initial_min_held'],final_hold_min=s['final_hold_min'],delivery=s['deliveries'][0] if s['deliveries'] else None,replacement=s.get('replacement'),rear=s['replacement_side_final_period'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--survey',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--aggregate-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rows=[x for x in json.loads(a.survey.read_text()) if not x['ordinary'] and x['generic']]
 (a.output/'manifest.json').write_text(json.dumps(dict(map_seeds=[x['seed'] for x in rows],encounter_seeds=ENCOUNTERS,
  selection='first generic candidate unchanged; six fixed encounters per map'),indent=2)+'\n')
 out=[]
 if a.aggregate_only:
  for path in a.output.glob('case-*/multi-*/summary.json'):
   s=json.loads(path.read_text());out.append(dict(map_seed=s['seed'],encounter_seed=s['encounter_seed'],folder=str(path.parent.relative_to(a.output)),outcome=s['outcome'],initial_min_held=s['initial_min_held'],final_hold_min=s['final_hold_min'],delivery=s['deliveries'][0] if s['deliveries'] else None,replacement=s.get('replacement'),rear=s['replacement_side_final_period']))
  out.sort(key=lambda x:(x['map_seed'],x['encounter_seed']));(a.output/'results.json').write_text(json.dumps(out,indent=2)+'\n');print('passes',sum(x['outcome']=='delivery_pass' for x in out),'/',len(out));return
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  jobs=[(x,e) for x in rows for e in ENCOUNTERS]
  fs=[pool.submit(worker,(i,x,e,str(a.output))) for i,(x,e) in enumerate(jobs)]
  for f in as_completed(fs):r=f.result();out.append(r);print(r,flush=True)
 out.sort(key=lambda x:x['map_seed']);(a.output/'results.json').write_text(json.dumps(out,indent=2)+'\n');print('passes',sum(x['outcome']=='delivery_pass' for x in out),'/',len(out))
if __name__=='__main__':main()
