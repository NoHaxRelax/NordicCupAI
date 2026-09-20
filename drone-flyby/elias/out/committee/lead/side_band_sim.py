"""CPU check of the sweep's band variants that no served run has exercised: a vertical band at the left or right edge
(sideways flight) and the bottom band (flight in the opposite direction). Simulates the camera loop: the view the
server would deliver is the last LEGAL request, exactly as the organisers' evaluator applies it (level change of at
most one step, centre move within the level's limit, centre inside the level's bounds). Reports illegal requests."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tracking.workflow import LevelOneSweep

W, H = 3840, 2160
LIMITS = {0: 2203.0, 1: 1102.0, 2: 551.0}
BOUNDS = [{'resolution_level': 0, 'width': 960, 'height': 540, 'minimum_center_x': 1920, 'maximum_center_x': 1920, 'minimum_center_y': 1080, 'maximum_center_y': 1080},
          {'resolution_level': 1, 'width': 960, 'height': 540, 'minimum_center_x': 960, 'maximum_center_x': 2880, 'minimum_center_y': 540, 'maximum_center_y': 1620},
          {'resolution_level': 2, 'width': 960, 'height': 540, 'minimum_center_x': 480, 'maximum_center_x': 3360, 'minimum_center_y': 270, 'maximum_center_y': 1890}]


def request(level, cx, cy):
    return {'original_width': W, 'original_height': H,
            'view': {'resolution_level': level, 'center_x': cx, 'center_y': cy},
            'camera_constraints': {'maximum_center_delta': LIMITS[level], 'allowed_resolution_levels': [0, 1, 2],
                                   'full_view_reset_exempt_from_delta': True, 'center_bounds': BOUNDS}}


def run(side, vertical_fraction, waypoints, frames=40):
    cam = LevelOneSweep(vertical_fraction, waypoints=waypoints); cam.side = side
    level, cx, cy = 0, 1920., 1080.
    illegal, seen = [], []
    for k in range(frames):
        out = cam.next_view(request(level, cx, cy))
        if out is None:
            illegal.append((k, 'None')); continue
        nl, nx, ny = int(out['resolution_level']), float(out['center_x']), float(out['center_y'])
        b = BOUNDS[nl]
        ok = abs(nl-level) <= 1 and b['minimum_center_x']-1e-6 <= nx <= b['maximum_center_x']+1e-6 and b['minimum_center_y']-1e-6 <= ny <= b['maximum_center_y']+1e-6
        ok = ok and (np.hypot(nx-cx, ny-cy) <= LIMITS[level]+1e-6 or (nl == 0))
        if ok:
            level, cx, cy = nl, nx, ny
        else:
            illegal.append((k, (level, cx, cy), (nl, nx, ny)))
        seen.append((level, round(cx), round(cy)))
    return illegal, seen


for side, vf in (('left', 0.), ('right', 0.), (None, 1.0), (None, 0.)):
    for wp in (4, 2):
        illegal, seen = run(side, vf, wp)
        cyc = seen[4:12]
        print(f"side={side} vertical_fraction={vf} waypoints={wp}: illegal requests {len(illegal)} {illegal[:2]}; views {cyc}")
