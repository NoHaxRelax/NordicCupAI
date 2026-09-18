"""Streaming, conservative audit of prepared crowd retention and bait coverage."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from debugger.replay_stream import iter_frames


def audit(receipt_path):
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    site = receipt['site']
    tracked = None
    first_loss = None
    first_target_failure = None
    first_uncovered = None
    frames = native = 0
    minimum_held = 33
    last_time = None
    for frame in iter_frames(ROOT / receipt['replay']):
        frames += 1
        native += bool(frame.get('native_image'))
        last_time = frame['t']
        if tracked is None:
            tracked = {p['id'] for p in frame['predators'][:33]}
        occupants = {a['id'] for a in frame['agents']
                     if math.dist((a['x'], a['y']), site['goal']) < 3.001}
        if not occupants and first_uncovered is None:
            first_uncovered = frame['t']
        held = 0
        for predator in frame['predators']:
            if predator['id'] not in tracked:
                continue
            p = (predator['x'], predator['y'])
            delta = [p[i] - site['mouth'][i] for i in range(2)]
            physical = (math.hypot(*delta) < 75.001 and
                        sum(delta[i]*site['inward'][i] for i in range(2)) <= 10.001)
            held += physical
            # Inside native hearing range, walls and gaze do not obscure bait.
            # Requiring it to be closest among ALL agents makes this sufficient
            # for the closest-observed-agent rule without reconstructing vision.
            if not predator['resting'] and frame['t'] >= 30:
                distances = sorted((math.dist(p, (a['x'], a['y'])), a['id'])
                                   for a in frame['agents'])
                target_ok = bool(distances and distances[0][0] < predator['hearing_radius']-.001
                                 and distances[0][1] in occupants)
                if not target_ok and first_target_failure is None:
                    first_target_failure = dict(time=frame['t'], predator_id=predator['id'],
                                                nearest=distances[0] if distances else None)
        minimum_held = min(minimum_held, held)
        if held != 33 and first_loss is None:
            first_loss = dict(time=frame['t'], held=held)
    result = dict(schema='prepared-crowd-retention-audit-v1', receipt=str(receipt_path),
                  receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                  frames=frames, native_frames=native, seconds=last_time,
                  minimum_physically_held=minimum_held, first_physical_loss=first_loss,
                  first_bait_coverage_gap=first_uncovered,
                  first_active_target_check_failure_after_30s=first_target_failure,
                  target_check='sufficient closest-agent inside hearing; uncertain cases fail',
                  pass_value=bool(frames == native == round(receipt['requested_seconds']*10)+1
                      and first_loss is None and first_uncovered is None
                      and first_target_failure is None),
                  limitations='Prepared 33 near-mouth starts; no transport reliability inference.')
    out = receipt_path.with_suffix('.retention-audit.json')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipt')
    audit(p.parse_args().receipt)
