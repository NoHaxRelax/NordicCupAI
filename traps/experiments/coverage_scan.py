"""Map-failure rate for each trap_sites preset, using the delivered script."""

import argparse
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from trap_sites import find_trap_sites

W, H = 1600, 1200
PRESETS = ("measured", "safe", "paranoid")


def sampled(seed):
    rng = random.Random(seed)
    r = [(0.0, 0.0, float(W), 30.0), (0.0, H - 30.0, float(W), 30.0),
         (0.0, 0.0, 30.0, float(H)), (W - 30.0, 0.0, 30.0, float(H))]
    for _ in range(W // 20):
        w = rng.uniform(30, 100)
        h = rng.uniform(30, 100)
        r.append((rng.uniform(0, W - w), rng.uniform(0, H - h), w, h))
    return r


def work(seed):
    rects = sampled(seed)
    out = []
    for p in PRESETS:
        ss = find_trap_sites(rects, W, H, safety=p)
        out.append((len(ss), sum(x.kind == "slot" for x in ss)))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    counts = {p: [] for p in PRESETS}
    slots = {p: [] for p in PRESETS}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, row in enumerate(pool.map(work, range(a.maps), chunksize=8), 1):
            for p, (n, ns) in zip(PRESETS, row):
                counts[p].append(n)
                slots[p].append(ns)
            if i % 500 == 0:
                print(f"  {i}/{a.maps}", flush=True)

    print(f"\n{'preset':<10} {'sites/map':>10} {'median':>7} {'maps with 0':>13} "
          f"{'rate':>9} {'1 in':>8}")
    for p in PRESETS:
        c = np.array(counts[p])
        zero = int((c == 0).sum())
        rate = zero / len(c)
        sl = np.array(slots[p])
        print(f"{p:<10} {c.mean():>10.2f} {sl.mean():>10.2f} {np.median(c):>7.0f} "
              f"{zero:>13} {100 * rate:>8.3f}% "
              f"{('%.0f' % (1 / rate)) if rate else 'never':>8}")
        dist = Counter(c.tolist())
        print("           " + "  ".join(
            f"{k}:{100 * dist[k] / len(c):.1f}%" for k in range(0, 7)))


if __name__ == "__main__":
    main()
