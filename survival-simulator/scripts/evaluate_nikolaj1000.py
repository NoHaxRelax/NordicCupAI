"""Evaluate frozen Nikolaj rank1 on seeds 19001..20000; deploy beside evaluation kernel."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,pathlib,time,hashlib,subprocess
from multiprocessing import Pool
import evaluate_frozen1000 as kernel

def main():
    root=kernel.ROOT;out=root/'results/nikolaj1000';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Run already started')
    source=root/'models/best_policies/rank1_pod-03.json'
    nested=json.loads(source.read_text())
    cfg={k:float('inf') if v is None else v for k,v in nested['orchard'].items()}
    cfg.update(nested['evasion'])
    manifest=dict(commit='e8c8628862066d3e59226a3d35bd365b635be81d',
        config_path=str(source.relative_to(root)),config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        config=cfg,seeds=list(range(19001,20001)),policy_seed=0,horizon=3000,
        predators=True,workers=12,profile=True,
        cpu=subprocess.check_output(['lscpu'],text=True),
        build=json.loads((root/'fastsim/build-info-policy.json').read_text()),
        numpy=kernel.np.__version__,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.monotonic()
    with Pool(12) as pool,(out/'games.jsonl').open('w',buffering=1) as f:
        for n,row in enumerate(pool.imap_unordered(kernel.one,[('nikolaj_rank1',s,cfg) for s in range(19001,20001)]),1):
            f.write(json.dumps(row)+'\n')
            if n%10==0:
                os.fsync(f.fileno())
                print(json.dumps(dict(games=n,total=1000,elapsed_seconds=time.monotonic()-start)),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(games=1000,elapsed_seconds=time.monotonic()-start))+'\n')
if __name__=='__main__':main()
