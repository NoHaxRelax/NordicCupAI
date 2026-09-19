"""No tuning: frozen winners on 200 fresh equivalent late-game checkpoints."""
import argparse,hashlib,json,multiprocessing as mp,pathlib,time
import tune_sharedfood20 as setup
from evaluate_sharedfood20 import configs
run=setup.run

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=20);a=ap.parse_args()
    assert 0<=a.shard<10
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if(out/'manifest.json').exists():raise RuntimeError('Refusing to overwrite validation')
    cfg=configs();seeds=list(range(18001+a.shard*20,18021+a.shard*20))
    files=list((run.ROOT/'fastsim').glob('*.cpp'))+list((run.ROOT/'fastsim').glob('*.hpp'))+[pathlib.Path(__file__),pathlib.Path(setup.__file__),pathlib.Path(run.__file__)]
    manifest=dict(configs=cfg,seeds=seeds,baseline=run.BASE,window_seconds=250,policy_seed=0,horizon=3000.,short_games='checkpoint at time zero; no seed replacement',sources={str(p.relative_to(run.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},build=json.loads((run.ROOT/'fastsim/build-info-policy.json').read_text()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.monotonic();ctx=mp.get_context('fork');actors=[];pipes=[]
    try:
        for k in range(min(a.workers,len(seeds))):
            parent,child=ctx.Pipe();p=ctx.Process(target=run.actor,args=(child,seeds[k::min(a.workers,len(seeds))]));p.start();child.close();pipes.append(parent);actors.append(p)
        def receive():
            results=[]
            for c in pipes:
                if not c.poll(3600):raise TimeoutError('Validation actor timeout')
                r=c.recv()
                if 'error'in r:raise RuntimeError(r['error'])
                results.append(r)
            return results
        metas=sorted([m for r in receive()for m in r['ready']],key=lambda r:r['seed'])
        (out/'checkpoints.json').write_text(json.dumps(metas,indent=2)+'\n')
        count=0
        with (out/'games.jsonl').open('w',buffering=1) as f:
            for name,cfg0 in cfg.items():
                for c in pipes:c.send(cfg0)
                rows=[r for batch in receive()for r in batch['rows']]
                assert sorted(r['seed']for r in rows)==seeds
                for r in rows:r['model']=name;f.write(json.dumps(r)+'\n');count+=1
                print(json.dumps(dict(model=name,games=count,total=len(cfg)*len(seeds),elapsed=time.monotonic()-start)),flush=True)
        (out/'complete.json').write_text(json.dumps(dict(games=count,elapsed_seconds=time.monotonic()-start))+'\n')
        print('COMPLETE',flush=True)
    finally:
        for c in pipes:
            try:c.send(None)
            except (BrokenPipeError,EOFError):pass
        for p in actors:p.join(5)
        for p in actors:
            if p.is_alive():p.terminate();p.join()
if __name__=='__main__':main()
