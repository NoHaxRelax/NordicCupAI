"""Fast offline simulator of camera policies for the drone flyby task.

Replays a scene's annotations (no images, no server) through the evaluator's own camera rules
(`local_evaluator.Camera`) and either Oscar's sweep (`tracking.workflow.LevelOneSweep`) or a policy
defined here, with a recognition model that only "sees" an object when its longer side in the
DELIVERED image reaches a threshold. The tracker is a proxy for `tracking/revisit.py`: a whole,
recognised object births a track, a born track is forecast (and scored as a hit) on every later
frame, and a track is retired after `miss_retire` full opportunities to see it that produced nothing.

Score = recall per class over all object-frames, macro-averaged. With no false positives and
forecasts inside IoU 0.5 that is what COCO AP@0.5 measures, so it tracks the real harness; the
`calibrate` command checks exactly that against harness runs.

    python elias/policy_sim.py table      --scene validation
    python elias/policy_sim.py calibrate  # compare with the harness numbers recorded below
    python elias/policy_sim.py classes    --scene validation --policy cued --min-px 24
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from local_evaluator import Camera, CameraRejection  # noqa: E402
from tracking.workflow import LevelOneSweep  # noqa: E402

W, H = 3840, 2160
FACTOR = {0: 4, 1: 2, 2: 1}
MOVE_LIMIT = {0: 2203.0, 1: 1102.0, 2: 551.0}
BOUNDS = {0: (1920, 1920, 1080, 1080), 1: (960, 2880, 540, 1620), 2: (480, 3360, 270, 1890)}

# Harness runs of 2026-09-18 (oracle detector through run_local_eval.py, DRONE_ORACLE_MIN_PX).
HARNESS = {
    ('helsinki', 'l1', 0): 0.978, ('helsinki', 'l1', 16): 0.553, ('helsinki', 'l1', 24): 0.481,
    ('helsinki', 'l1', 32): 0.303, ('helsinki', 'l2_top', 0): 0.753, ('helsinki', 'l2_top', 16): 0.470,
    ('helsinki', 'l2_top', 24): 0.468, ('helsinki', 'l2_top', 32): 0.293,
    ('validation', 'l1', 0): 0.923, ('validation', 'l1', 16): 0.750, ('validation', 'l1', 24): 0.517,
    ('validation', 'l1', 32): 0.380, ('validation', 'l2_top', 0): 0.729, ('validation', 'l2_top', 16): 0.727,
    ('validation', 'l2_top', 24): 0.726, ('validation', 'l2_top', 32): 0.679,
}


# ----------------------------------------------------------------------------- scene
@dataclass
class Obj:
    track: str
    cls: str
    box: np.ndarray  # source xyxy


def load_scene(name: str):
    """[(frame_number, [Obj, ...]), ...] in frame order."""
    frames = []
    for path in sorted(glob.glob(str(ROOT/'src'/name/'annotations'/'*.json'))):
        d = json.loads(Path(path).read_text())
        prov = d.get('provenance') or [{}]*len(d['annotations'])
        objs = [Obj(p.get('track_id') or a['object_id'], a['object_id'], np.array(a['bbox'], float))
                for a, p in zip(d['annotations'], prov)]
        frames.append((int(d['frame']), objs))
    return frames


# ----------------------------------------------------------------------------- geometry
def region(level, cx, cy):
    f = FACTOR[level]
    return np.array([cx-480*f, cy-270*f, cx+480*f, cy+270*f], float)


def clamp_center(level, x, y):
    x0, x1, y0, y1 = BOUNDS[level]
    return int(round(min(max(x, x0), x1))), int(round(min(max(y, y0), y1)))


def legal(cam: Camera, level, cx, cy) -> bool:
    if abs(level-cam.resolution_level) > 1:
        return False
    if (cx, cy) != clamp_center(level, cx, cy):
        return False
    if level == 0:
        return True
    return math.hypot(cx-cam.center_x, cy-cam.center_y) <= MOVE_LIMIT[cam.resolution_level]


def toward(cam: Camera, level, tx, ty):
    """A legal (level, cx, cy) as close to the target as one move allows, or None."""
    if abs(level-cam.resolution_level) > 1:
        return None
    if level == 0:
        return 0, 1920, 1080
    tx, ty = clamp_center(level, tx, ty)
    limit = MOVE_LIMIT[cam.resolution_level]-1
    sx, sy = clamp_center(level, cam.center_x, cam.center_y)
    if math.hypot(sx-cam.center_x, sy-cam.center_y) > limit:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(30):
        m = (lo+hi)/2
        px, py = sx+m*(tx-sx), sy+m*(ty-sy)
        if math.hypot(px-cam.center_x, py-cam.center_y) <= limit:
            lo = m
        else:
            hi = m
    cx, cy = clamp_center(level, sx+lo*(tx-sx), sy+lo*(ty-sy))
    return (level, cx, cy) if legal(cam, level, cx, cy) else None


# ----------------------------------------------------------------------------- recognition
@dataclass
class Recognition:
    """An object is recognised when its longer delivered side reaches min_px (per class or scalar).
    `notice_px`: a cheaper class-agnostic cue ("something is there") that a policy may act on."""
    min_px: float | dict = 0.0
    notice_px: float = 8.0

    def need(self, cls):
        return self.min_px.get(cls, self.min_px.get('*', 0.0)) if isinstance(self.min_px, dict) else self.min_px

    def sees(self, cls, size_px):
        return size_px >= self.need(cls)

    def level_needed(self, cls, box):
        side = max(box[2]-box[0], box[3]-box[1])
        for level in (0, 1, 2):
            if side/FACTOR[level] >= self.need(cls):
                return level
        return None  # too small even at native resolution


# ----------------------------------------------------------------------------- policies
class OscarSweep:
    """Oscar's deployed sweep, driven the way tracking/workflow.py drives it."""
    def __init__(self, mode='l1', overview_between_sides=True, vertical_fraction=0.0):
        self.sweep = LevelOneSweep(vertical_fraction, overview_between_sides=overview_between_sides, mode=mode)

    def next_view(self, i, cam, state):
        request = {'original_width': W, 'original_height': H, 'camera_constraints': cam.constraints(),
                   'view': {'center_x': cam.center_x, 'center_y': cam.center_y, 'resolution_level': cam.resolution_level}}
        v = self.sweep.next_view(request, overview=(i == 0))
        return None if v is None else (v['resolution_level'], v['center_x'], v['center_y'])


class Cued(OscarSweep):
    """Base sweep, interrupted to give a noticed but unrecognised object the pixels it needs.

    A candidate is an object the recogniser has noticed (or that an earlier look failed to name)
    and that is not tracked. The most urgent candidate (closest to leaving the frame) whose next
    position can be put whole inside a view at the level it needs gets the camera; if that level is
    two steps away the camera takes the intermediate step towards it."""
    def __init__(self, base='l1', overview_between_sides=True, max_detour=3):
        super().__init__(base, overview_between_sides)
        self.detour = 0
        self.max_detour = max_detour

    def next_view(self, i, cam, state):
        best = None
        for c in state['candidates']:
            level = c['level']
            if level is None or level <= 0:
                continue
            box = c['next_box']
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            step_level = level if abs(level-cam.resolution_level) <= 1 else cam.resolution_level+(1 if level > cam.resolution_level else -1)
            v = toward(cam, step_level, cx, cy)
            if v is None:
                continue
            r = region(*v)
            whole = box[0] > r[0]+2 and box[1] > r[1]+2 and box[2] < r[2]-2 and box[3] < r[3]-2
            if step_level == level and not whole:
                continue
            urgency = box[3]  # lower in the frame = leaves sooner
            if best is None or urgency > best[0]:
                best = (urgency, v)
        if best is not None and self.detour < self.max_detour:
            self.detour += 1
            return best[1]
        self.detour = 0
        return super().next_view(i, cam, state)


POLICIES = {
    'l1': lambda: OscarSweep('l1', True),
    'l1_lcr': lambda: OscarSweep('l1', False),
    'l2_top': lambda: OscarSweep('l2_top', True),
    'cued': lambda: Cued('l1', True),
    'cued_lcr': lambda: Cued('l1', False),
}


# ----------------------------------------------------------------------------- simulation
@dataclass
class Track:
    born: bool = False
    provisional: bool = False
    misses: int = 0
    noticed: bool = False
    age: int = 0  # frames since the object was last seen whole


@dataclass
class Result:
    hits: dict = field(default_factory=dict)
    total: dict = field(default_factory=dict)
    views: list = field(default_factory=list)

    def score(self):
        return float(np.mean([self.hits.get(c, 0)/n for c, n in self.total.items()])) if self.total else 0.0


def simulate(frames, policy, rec: Recognition, *, miss_retire=3, miss_rule='any', partial_hit=0.3,
             entry_tracks=True, drift=0.0, tol=0.3) -> Result:
    """miss_rule: 'any' = Oscar's rule (every full opportunity counts, whatever the zoom);
    'sized' = only when the view gives the class the pixels it needs; 'never' = no retirement.
    drift: forecast error in source px per frame since the last whole sighting; a forecast scores
    only while drift*age <= tol*min(box side), which is roughly where IoU falls under 0.5."""
    cam = Camera()
    tracks: dict[str, Track] = {}
    res = Result()
    for i, (frame_no, objs) in enumerate(frames):
        r = region(cam.resolution_level, cam.center_x, cam.center_y)
        f = FACTOR[cam.resolution_level]
        res.views.append((cam.resolution_level, cam.center_x, cam.center_y))
        candidates = []
        for o in objs:
            t = tracks.setdefault(o.track, Track())
            b = o.box
            side = max(b[2]-b[0], b[3]-b[1])/f
            ix = max(0.0, min(b[2], r[2])-max(b[0], r[0]))
            iy = max(0.0, min(b[3], r[3])-max(b[1], r[1]))
            vis = ix*iy/max(1.0, (b[2]-b[0])*(b[3]-b[1]))
            whole = b[0] > r[0]+f and b[1] > r[1]+f and b[2] < r[2]-f and b[3] < r[3]-f
            seen = vis > 0 and rec.sees(o.cls, side)
            hit = False
            t.age += 1
            if seen and whole:
                t.born, t.provisional, t.misses, t.age = True, False, 0, 0
            elif seen and vis >= partial_hit:
                hit = True  # emit_partials: reported for this frame, extended to the class size
                if entry_tracks and b[1] <= 1 and not t.born:
                    t.born, t.provisional, t.misses = True, True, 0
            elif t.born and whole and not seen:
                counts = miss_rule == 'any' or (miss_rule == 'sized' and rec.sees(o.cls, side))
                if counts:
                    t.misses += 1
                    if t.misses >= (1 if t.provisional else miss_retire):
                        t.born, t.provisional, t.misses = False, False, 0
            if vis > 0 and side >= rec.notice_px:
                t.noticed = True
            hit = hit or (t.born and drift*t.age <= tol*min(b[2]-b[0], b[3]-b[1]))
            res.total[o.cls] = res.total.get(o.cls, 0)+1
            res.hits[o.cls] = res.hits.get(o.cls, 0)+int(hit)
            if t.noticed and not t.born:
                candidates.append(o)
        nxt = {o.track: o.box for o in frames[i+1][1]} if i+1 < len(frames) else {}
        state = {'candidates': [{'cls': o.cls, 'next_box': nxt[o.track], 'level': rec.level_needed(o.cls, o.box)}
                                for o in candidates if o.track in nxt]}
        v = policy.next_view(i, cam, state)
        if v is not None:
            try:
                cam.apply(*v)
            except CameraRejection:
                pass
    return res


# ----------------------------------------------------------------------------- commands
def cmd_calibrate(a):
    rows, errs = [], []
    scenes = {s: load_scene(s) for s in ('helsinki', 'validation')}
    for (scene, pol, px), real in sorted(HARNESS.items()):
        sim = simulate(scenes[scene], POLICIES[pol](), Recognition(px), miss_retire=a.miss_retire,
                       miss_rule=a.miss_rule, partial_hit=a.partial_hit, drift=a.drift).score()
        rows.append((scene, pol, px, real, sim)); errs.append(sim-real)
    print(f"{'scene':<11} {'policy':<7} {'px':>3} {'harness':>8} {'sim':>7} {'diff':>7}")
    for scene, pol, px, real, sim in rows:
        print(f'{scene:<11} {pol:<7} {px:>3} {real:>8.3f} {sim:>7.3f} {sim-real:>+7.3f}')
    e = np.array(errs); real = np.array([r[3] for r in rows]); sim = np.array([r[4] for r in rows])
    print(f'\nmean diff {e.mean():+.3f}  mean |diff| {np.abs(e).mean():.3f}  max |diff| {np.abs(e).max():.3f}  '
          f'pearson r {np.corrcoef(real, sim)[0, 1]:.3f}')


def cmd_table(a):
    frames = load_scene(a.scene)
    pxs = [0, 8, 12, 16, 20, 24, 32]
    print(f"scene {a.scene}, miss rule {a.miss_rule} (retire after {a.miss_retire})\n{'policy':<10}" + ''.join(f'{p:>7}' for p in pxs))
    for name, make in POLICIES.items():
        print(f'{name:<10}' + ''.join(
            f'{simulate(frames, make(), Recognition(p, a.notice_px), miss_retire=a.miss_retire, miss_rule=a.miss_rule, partial_hit=a.partial_hit, drift=a.drift).score():>7.3f}'
            for p in pxs))


def cmd_classes(a):
    frames = load_scene(a.scene)
    res = simulate(frames, POLICIES[a.policy](), Recognition(a.min_px, a.notice_px), miss_retire=a.miss_retire,
                   miss_rule=a.miss_rule, partial_hit=a.partial_hit, drift=a.drift)
    for c in sorted(res.total, key=lambda c: res.hits.get(c, 0)/res.total[c]):
        print(f'  {c:<16} {res.hits.get(c, 0):>4}/{res.total[c]:<4} {res.hits.get(c, 0)/res.total[c]:.3f}')
    lv = np.bincount([v[0] for v in res.views], minlength=3)
    print(f'score {res.score():.3f}   frames at L0/L1/L2: {lv.tolist()}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['calibrate', 'table', 'classes'])
    ap.add_argument('--scene', default='validation')
    ap.add_argument('--policy', default='l1', choices=list(POLICIES))
    ap.add_argument('--min-px', type=float, default=24)
    ap.add_argument('--notice-px', type=float, default=8)
    ap.add_argument('--miss-retire', type=int, default=3)
    ap.add_argument('--miss-rule', default='any', choices=['any', 'sized', 'never'])
    ap.add_argument('--partial-hit', type=float, default=1.1)
    ap.add_argument('--drift', type=float, default=0.5)
    a = ap.parse_args()
    {'calibrate': cmd_calibrate, 'table': cmd_table, 'classes': cmd_classes}[a.cmd](a)


if __name__ == '__main__':
    main()
