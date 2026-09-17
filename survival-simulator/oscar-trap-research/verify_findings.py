#!/usr/bin/env python3
"""Check headline handoff numbers against saved results; does not run simulations."""
from pathlib import Path
import json
import statistics

BASE = Path(__file__).resolve().parent / 'survival/results'

def read(path):
    return json.loads((BASE / path).read_text())

renewal = read('wall_deployment/natural-renewal-v7.json')['runs']
assert len(renewal) == 2
for r, seed, food, generation in zip(renewal, [7, 8], [130, 85], [5, 6]):
    assert (r['seed'], r['food_count'], r['max_generation']) == (seed, food, generation)
    assert r['seconds'] == 300 and r['captures'] == 0 and r['first_hold_loss'] is None
    assert len(r['births']) == 10 and r['alive'] == 4
    assert r['held_fraction'] == 1
survey = read('wall_deployment/scout-wall-survey.json')['maps']
assert len(survey) == 50 and sum(x['usable_walls'] > 0 for x in survey) == 49
assert statistics.median(x['usable_walls'] for x in survey) == 3
for name, expected in [('protocol-delivery-v1', {'first': 4, 'second_straight': 2, 'corner': 0, 'corner_track': 0}), ('protocol-guard-v1', {'guard_active': 2, 'guard_stationary': 0}), ('protocol-guard-v2', {'guard_active': 1, 'guard_stationary': 0})]:
    rows = read(f'wall_deployment/{name}.json')['runs']
    assert len(rows) == 16
    for kind, successes in expected.items():
        assert sum(r['success'] for r in rows if r['kind'] == kind) == successes
native = []
for name, count in [('alternative-gap-native-all-terrain', 18), ('alternative-gap-native-21-40-all-terrain', 20), ('alternative-gap-native-multiple', 4)]:
    rows = read(f'water_deployment/{name}.json')['data']
    assert len(rows) == count
    assert all(r['alive'] and r['capture_s'] is None and abs(r['elapsed_s'] - 60) < 1e-6 and r['joint_fraction'] == 1 for r in rows)
    if name != 'alternative-gap-native-multiple':
        native.extend(rows)
assert len(native) == 38 and len({r['case']['seed'] for r in native}) == 19
maps = read('water_deployment/alternative-gap-survey.json')['data'] + read('water_deployment/alternative-gap-survey-21-40.json')['data']
assert len(maps) == 40 and sum(bool(x['sites']) for x in maps) == 19
assert sum(len(x['sites']) for x in maps) == 36
control = read('water_deployment/alternative-gap-controlled.json')['data']
wide = [r for r in control if r['case']['gap'] == 22]
assert len(wide) == 4 and all(abs(r['capture_s'] - .5) < 1e-6 for r in wide)
withdrawal = read('water_deployment/alternative-gap-withdrawal.json')['data']
assert len(withdrawal) == 2 and all(13.29 < r['first_escape_s'] < 13.41 for r in withdrawal)
print('Verified wall renewal, 50-map wall census, delivery/guard counts, gap availability, all 38 native gap holds, four multi-predator holds, too-wide failures and withdrawal escapes.')
