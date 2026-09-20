"""Regenerate independent CPython RNG and existing-filter parity fixtures."""

import json
from pathlib import Path
import platform
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seed_biome_prefix_probe import candidate_prefix, LAND_TYPES
from seed_survey_probe import compatible


seeds = [0, 1, 3, 11, 78431, 20260919, 2**31, 2**32 - 1]
control = random.Random(672419)
seeds += [control.getrandbits(32) for _ in range(24)]
positions = set(range(12)) | {226, 227, 228, 622, 623, 624, 625, 1246, 1247, 1248, 1999}
result = dict(python=platform.python_version(), rng_vectors=[], prefix_vectors=[], search_vectors=[])
for seed in seeds:
    rng = random.Random(seed)
    outputs = [(index, rng.getrandbits(32)) for index in range(2000)]
    result['rng_vectors'].append(dict(seed=seed, outputs=[pair for pair in outputs if pair[0] in positions]))
    sites, types = candidate_prefix(seed)
    result['prefix_vectors'].append(dict(seed=seed, sites=sites, types=[LAND_TYPES.index(kind) for kind in types]))
for start, count, samples in [
    (0, 101, [dict(x=300., y=500., biome='forest', radius=4000.)]),
    (0, 409, [dict(x=2., y=1198., biome='swamp', radius=7.), dict(x=1250., y=70., biome='grassland', radius=9.)]),
    (2**32 - 67, 67, [dict(x=1560.5, y=1050.2, biome='desert', radius=12.)]),
]:
    expected = [seed for seed in range(start, start + count) if compatible(*candidate_prefix(seed), samples)]
    result['search_vectors'].append(dict(start=start, count=count, samples=samples, expected=expected))
output = Path(__file__).with_name('python_vectors.json')
output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(f'Wrote {len(seeds)} CPython stream/prefix vectors and 3 search vectors to {output}')
