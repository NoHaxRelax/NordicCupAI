#!/usr/bin/env python3
"""Work out how far the full annotation is from a target board score, and hold one class back to land on it.

The board must not show more than the cap, so the last attempt is deliberately imperfect. Given the measured AP of
every class (each from its own single-class attempt, which can never exceed 1/13), the total is their mean. If it
is above the target, one class is truncated until the total lands on it; if below, nothing is held back.

    python3 assemble_final.py --ap ap.json --target 0.96
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CLASSES = ['hangar', 'tank', 'helicopter', 'jet_plane', 'large_launcher', 'large_tower', 'mine_roller', 'small_tower',
           'small_plane', 'medium_plane', 'small_launcher', 'medium_launcher', 'ta-ta']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ap', required=True, help='JSON {class: AP}')
    ap.add_argument('--target', type=float, default=0.96)
    ap.add_argument('--hold', default='hangar', help='class to hold back if the total is over the target')
    ap.add_argument('--truth', type=int, default=70, help='truth frame count of that class')
    a = ap.parse_args()
    per = json.loads(Path(a.ap).read_text())
    known = {c: per[c] for c in CLASSES if c in per}
    total = sum(known.values())
    score = total / 13
    print(f'measured classes: {len(known)}/13   AP sum {total:.3f}   board score {score:.4f}')
    for c in CLASSES:
        print(f'  {c:16s} {known.get(c, float("nan")):.3f}')
    need = a.target * 13
    print(f'\ntarget {a.target} needs an AP sum of {need:.3f}: ' +
          (f'OVER by {total - need:.3f}' if total > need else f'UNDER by {need - total:.3f}'))
    if total > need:
        keep_ap = known[a.hold] - (total - need)
        h = next((k for k in range(a.truth + 1) if (math.floor(100 * k / a.truth) + 1) / 101 >= keep_ap - 1e-9), a.truth)
        print(f'hold {a.hold} at AP {keep_ap:.3f} -> keep {h} of its {a.truth} frames:')
        print(f'  python3 drone/verifier/tune_plan.py --plan probes/probe-full.json --out probes/probe-final.json '
              f'--classes all --degrade {a.hold} --target-ap {keep_ap:.3f} --truth {a.hold}={a.truth}')
    else:
        print('submit the full annotation unchanged; the board lands below the cap on its own.')


if __name__ == '__main__':
    main()
