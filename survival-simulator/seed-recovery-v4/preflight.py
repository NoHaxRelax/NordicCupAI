"""Check package/fixture integrity; optionally exercise compiled search and wall verification."""
from pathlib import Path
import argparse, hashlib, json, random, subprocess, tempfile
p=argparse.ArgumentParser(); p.add_argument('--native', action='store_true'); a=p.parse_args()
root=Path(__file__).resolve().parent
for name,digest in json.loads((root/'SHA256SUMS.json').read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise SystemExit('Checkpoint file changed: '+name)
seed=1854492595
r=random.Random(seed); sites=[(r.randrange(1600),r.randrange(1200)) for _ in range(10)]; labels=[r.randrange(4) for _ in range(10)]
samples=[tuple(map(int,line.split())) for line in (root/'fixtures/terrain.txt').read_text().splitlines() if line.strip()]
for x,y,label in samples:
    nearest=min(range(10),key=lambda i:(sites[i][0]-x)**2+(sites[i][1]-y)**2)
    if labels[nearest]!=label: raise SystemExit('Fixture disagrees with independent CPython RNG')
coverage=json.loads((root/'evidence/coverage-audit.json').read_text()); end=0
for shard in coverage['shards']:
    assert shard['start']==end and shard['tested']==shard['count'];end+=shard['count']
assert end==2**32 and coverage['verified_seeds']==[614466944]
print(f'PASS: hashes, {len(samples)} public terrain samples, and historical full uint32 coverage')
if a.native:
    src=root/'survival/research/seed_inference'
    with tempfile.TemporaryDirectory(prefix='seed-v4-check-') as tmp:
        for label,start,expected in [('positive',seed,[seed]),('negative',0,[])]:
            out=Path(tmp)/label
            subprocess.run([str(src/'streaming_verification/coordinator'),'stream',str(src/'terrain_filter'),str(root/'fixtures/terrain.txt'),str(root/'fixtures/initial-public.json'),str(root/'survival/results/orchard/best-config.json'),str(start),'1',str(out)],check=True)
            receipt=json.loads((out/'receipt.json').read_text())
            hits=[int(s) for s in (out/'verified.txt').read_text().split()]
            assert receipt['complete'] and receipt['filter_exit_code']==0 and hits==expected,(label,hits,receipt)
            print('PASS:',label,'native terrain + wall check')
    print('Bounded known-seed diagnostics only; this is not blind recovery or live parity validation.')
