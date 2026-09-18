"""Generate perturbation configs around a base: gen.py BASE_LABEL OUT_PREFIX N P RNG > configs json (base included as BASE_LABEL)."""
import sys, json, random
sys.path.insert(0, 'survival/research/orchard')
from opt import perturb
base_label, prefix, n, p, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
space = json.load(open('survival/research/orchard/space_v14.json'))
pool = json.load(open('artifacts/overnight/configs-all.json'))
base = pool[base_label]; rng = random.Random(seed)
out = {base_label: base}
for i in range(n): out[f'{prefix}{i}'] = perturb(base, space, rng, p=p)
print(json.dumps(out))
