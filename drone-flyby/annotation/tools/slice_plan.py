#!/usr/bin/env python3
"""Cut a full plan down to one class and a confidence floor, to measure what the lower tiers are worth.

The plan answers each class in five confidence bands: the tracker's own box first, then alternates and
speculative candidates. A candidate is only free if it ranks below every hit - COCO sorts the whole class by
confidence, so a band that misses puts its boxes ahead of the hits in the bands under it and drags their
precision down. This slices the plan so that cost can be measured instead of assumed.

    python3 slice_plan.py --plan probe-full.json --cls large_tower --floors 0.8,0.6,0.4,0.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--floors', default='0.8,0.6,0.4,0.0')
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    short = a.cls.replace('_', '').replace('-', '')
    for floor in (float(v) for v in a.floors.split(',')):
        pbf = {}
        for f, boxes in plan['predictions_by_frame'].items():
            keep = [b for b in boxes if b['object_id'] == a.cls and b['confidence'] >= floor]
            if keep:
                pbf[f] = keep
        name = f"probe-{short}-f{floor:g}".replace('.', '')
        Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270],
                                                               'predictions_by_frame': pbf}))
        print(f'  {name}: {sum(len(v) for v in pbf.values())} boxes over {len(pbf)} frames')


if __name__ == '__main__':
    main()
