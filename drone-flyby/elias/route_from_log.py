"""Per-class routing table from the same-day one-class portal runs in elias/out/logs/measure_f5.log.

Every "=== <time> <MODEL> class <cls> (<host>)" line is followed by its "RESULT score <s>" line; the class AP is
score x 13 (13 classes present in validation). The deployed F3 answers every class unless another model beats it
by more than the margin on a same-day run; a class F3 was never measured on today stays with F3, except ta-ta,
which F3 cannot see at all (no walker sprites in its training). Prints the table and the ELIAS_ROUTE JSON.

    python elias/route_from_log.py [--margin 0.03] [--json-only]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {'F3_both_m1280': 0, 'F5': 1, 'F5_fixed_m1280': 1, 'F8_terrain_m1280': 2}
WEIGHTS = ['/root/out/F3_both_m1280.last.pt', '/root/out/F5_fixed_m1280.last.pt', '/root/out/F8_terrain_m1280.last.pt']
CLASSES = ['small_launcher', 'medium_launcher', 'large_launcher', 'small_plane', 'medium_plane', 'jet_plane', 'small_tower',
           'large_tower', 'tank', 'mine_roller', 'hangar', 'helicopter', 'ta-ta', 'condor', 'jammer', 'spacecraft']


def read(log):
    aps = defaultdict(dict)          # cls -> model index -> best AP seen today
    pending = None
    for line in Path(log).read_text(encoding='utf-8', errors='replace').splitlines():
        m = re.match(r'=== \S+ (\S+) class (\S+) \(', line)
        if m:
            pending = (m.group(1), m.group(2)); continue
        m = re.match(r'RESULT score ([0-9.eE+-]+)', line)
        if m and pending:
            model, cls = pending; pending = None
            if model in MODELS:
                k = MODELS[model]; ap = float(m.group(1))*13
                aps[cls][k] = max(ap, aps[cls].get(k, 0.))
    return aps


def route(aps, margin):
    out = {}
    for cls in CLASSES:
        base = aps[cls].get(0)
        if cls == 'ta-ta' and base is None:
            base = 0.
        best_k, best = 0, base if base is not None else None
        if best is not None:
            for k, ap in aps[cls].items():
                if k != 0 and ap > best+margin:
                    best_k, best = k, ap
        out[cls] = best_k
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--log', default=str(HERE/'out'/'logs'/'measure_f5.log'))
    ap.add_argument('--margin', type=float, default=0.03); ap.add_argument('--json-only', action='store_true')
    a = ap.parse_args()
    aps = read(a.log); r = route(aps, a.margin)
    if not a.json_only:
        print(f"{'class':<16}{'F3':>7}{'F5':>7}{'F8':>7}  route")
        for cls in CLASSES:
            row = ''.join(f"{aps[cls][k]:>7.2f}" if k in aps[cls] else f"{'-':>7}" for k in (0, 1, 2))
            print(f"{cls:<16}{row}  {['F3', 'F5', 'F8'][r[cls]]}")
        used = sorted({k for k in r.values()})
        print('models used:', [['F3', 'F5', 'F8'][k] for k in used])
    used = sorted({k for k in r.values()})
    remap = {k: i for i, k in enumerate(used)}
    print(json.dumps({'ELIAS_WEIGHTS': ','.join(WEIGHTS[k] for k in used), 'ELIAS_ROUTE': {c: remap[k] for c, k in r.items()}}))


if __name__ == '__main__':
    main()
