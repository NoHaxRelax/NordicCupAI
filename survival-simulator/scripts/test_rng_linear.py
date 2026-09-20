"""Synthetic test ONLY: demonstrates inversion, not acquisition of game outputs."""
import pathlib,sys,random,time,json,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.seed_shadow.rng_linear import MTLinear
p=argparse.ArgumentParser();p.add_argument('--rounded',action='store_true');args=p.parse_args()
r=random.Random(0x913ab249)
# Discard an unknown prefix: reconstruction does not need the seed or draw count.
for _ in range(12347):r.random()
solver=MTLinear();start=time.monotonic()
for _ in range(1600 if args.rounded else 700):
    value=r.random()
    if args.rounded:solver.observe_uniform(30,100,30+70*value,1e-10)
    else:solver.observe_random(value)
clone=solver.clone()
matches=sum(clone.getrandbits(32)==r.getrandbits(32) for _ in range(1000))
print(json.dumps(dict(synthetic_only=True,rounded=args.rounded,rank=len(solver.basis),
    elapsed_seconds=time.monotonic()-start,heldout_matches=matches,heldout_total=1000)),flush=True)
assert matches==1000
