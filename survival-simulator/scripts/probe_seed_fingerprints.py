"""Measure an offline terrain-index idea, NOT live/public-observation recovery.

Each seed supplies the underlying non-river terrain at fixed landmarks. A live
query would have to visit and observe these exact pixels; river observations are
wildcards. This pilot does not model navigation, acquisition, or river coverage.
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--seeds', type=int, default=200000)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if not 2 <= a.seeds <= 2**32:
    p.error('seeds must be between 2 and 2**32')
# First eight spread around the perimeter, then eight interior landmarks.
landmarks = [(200, 200), (1400, 1000), (1400, 200), (200, 1000),
             (800, 200), (800, 1000), (200, 600), (1400, 600),
             (500, 400), (1100, 800), (1100, 400), (500, 800),
             (800, 400), (800, 800), (500, 600), (1100, 600)]
seeds = random.Random(2026092008).sample(range(2**32), a.seeds)
fingerprints = np.zeros(a.seeds, dtype=np.uint32)
started = time.perf_counter()
landmarks_array = np.asarray(landmarks, dtype=np.int32)
for start in range(0, a.seeds, 4096):
    batch = seeds[start:start + 4096]
    sites, labels = [], []
    for seed in batch:
        r = random.Random(seed)
        sites.append([(r.randrange(1600), r.randrange(1200)) for _ in range(10)])
        labels.append([r.randrange(4) for _ in range(10)])
    delta = np.asarray(sites, dtype=np.int32)[:, None, :, :] - landmarks_array[None, :, None, :]
    nearest = (delta * delta).sum(axis=-1).argmin(axis=-1)
    values = np.asarray(labels, dtype=np.uint32)[np.arange(len(batch))[:, None], nearest]
    fingerprints[start:start + len(batch)] = (values << (2 * np.arange(16, dtype=np.uint32))).sum(axis=1, dtype=np.uint32)
rows = []
for known in [4, 6, 8, 12, 16]:
    keys = fingerprints & np.uint32((1 << (2 * known)) - 1)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    # Leave-one-out frequency estimates. Zero observed peers are not proof that
    # no other seeds share a fingerprint; no false precision for rare buckets.
    expected_other_seeds = (counts[inverse] - 1) / (a.seeds - 1) * (2**32 - 1)
    rows.append({'known_landmarks': known, 'observed_buckets': len(counts),
                 'singleton_query_fraction': float(np.mean(counts[inverse] == 1)),
                 'estimated_other_candidates_mean': float(expected_other_seeds.mean()),
                 'estimated_other_candidates_p50_p90_p95_p99': np.quantile(expected_other_seeds, [.5, .9, .95, .99]).tolist(),
                 'largest_observed_bucket_fraction': float(counts.max() / a.seeds)})
result = {'scope': 'Offline index selectivity pilot only; labels generated from seeds, NOT public observations. No seed recovery or latency claim.',
          'seeds': a.seeds, 'sampling_seed': 2026092008, 'landmarks': landmarks,
          'seconds': time.perf_counter() - started, 'rows': rows,
          'index_storage_note': '8-landmark counting-sort seed lists require 16 GiB plus offsets for the entire uint32 domain; an unrestricted 16-landmark key+seed representation requires 32 GiB before overhead.',
          'limitations': ['No river masking or navigation simulated.', 'Rare bucket estimates are noisy; zero sampled peers does not mean unique.', 'Homogeneous maps can remain expensive even with many known landmarks.', 'Offline construction and download costs must be reported separately from online query.']}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
