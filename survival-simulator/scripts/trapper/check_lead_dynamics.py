"""Model-only check of the leading mechanics (no engine): a guide walks straight
away at 10/tick while facing the predator with an alternating gaze offset.
Prints the gap and the predator's lateral offset every 5 ticks for sprinting
and walking predators, and the same for a guide with its back turned."""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.trapper import predator_model as pm      # noqa: E402
from models.trapper.geometry import Rect             # noqa: E402

W, H = 1600, 1200
rects = [Rect(0, 0, W, 30), Rect(0, H - 30, W, 30), Rect(0, 0, 30, H), Rect(W - 30, 0, 30, H)]


def run(energy, facing, offset=0.12, gap0=115.0, guide_speed=10.0, ticks=60, alternate=True):
    gx, gy = 500.0, 600.0
    pred = pm.PredState(gx - gap0, gy, 0.0, energy=energy)
    gaze = 1
    rows = []
    for t in range(ticks):
        # guide moves +x
        gx += guide_speed
        bearing = math.atan2(pred.y - gy, pred.x - gx)
        if facing:
            gaze = -gaze if alternate else 1
            gh = bearing + offset * gaze
        else:
            gh = 0.0
        pred, info = pm.step(pred, [(1, gx, gy, gh)], rects, W, H, lambda p: 1.0)
        if t % 5 == 0 or info['kills']:
            rows.append(f"t={t:2d} gap={math.hypot(gx - pred.x, gy - pred.y):6.1f} lat={pred.y - gy:6.1f} mode={info['mode']:6s} e={pred.energy:6.1f} kills={info['kills']}")
        if info['kills']:
            break
    return rows


if __name__ == '__main__':
    for energy in (102.0, 39.0):
        for facing in (True, False):
            print(f'--- predator energy {energy}, guide facing={facing} ---')
            print('\n'.join(run(energy, facing)))
    print('--- facing, constant gaze offset (no alternation) ---')
    print('\n'.join(run(102.0, True, alternate=False)))
