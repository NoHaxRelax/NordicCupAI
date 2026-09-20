#!/usr/bin/env python3
"""Every probe attempt in one table: board score, the class AP it implies, and whether the run was complete.

About half the organiser's attempts come back with a score but 0-1 frames delivered; those scores belong to
whatever ran before, not to the plan under test, so the frame count is printed beside every row and short
runs are marked.

    python3 collect_probes.py [--min-frames 247]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--min-frames', type=int, default=247)
    ap.add_argument('--all', action='store_true', help='include the runs the organiser never delivered')
    a = ap.parse_args()
    rows = []
    for p in sorted((ROOT / 'artifacts/drone-verifier-20260919/api-attempts').glob('probe-*/delivery.json')):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get('score') is None:
            continue
        rows.append((d.get('name', p.parent.name), float(d['score']), int(d.get('frames', 0)), p.stat().st_mtime))
    rows.sort(key=lambda r: r[3])
    print(f"{'probe':26s} {'board':>9s} {'x13':>7s}  frames")
    for name, score, frames, _ in rows:
        if frames < a.min_frames and not a.all:
            continue
        mark = '' if frames >= a.min_frames else '  (short run - score is not this plan)'
        print(f'{name:26s} {score:9.6f} {score * 13:7.3f}  {frames:3d}/249{mark}')


if __name__ == '__main__':
    main()
