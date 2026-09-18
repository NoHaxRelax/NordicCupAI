"""Read-only audit of a completed API batch; never schedules further queries."""
from collections import Counter
from common import ROOT, DATA, atomic, now, read


def audit():
    state = read(DATA / 'state.json')
    status = read(DATA / 'status.json')
    roots = list(state['roots'].values())
    query_shapes = Counter()
    for path in sorted((DATA / 'queries').glob('q*/plan.json')):
        plan = read(path)['predictions_by_frame']
        count = sum(map(len, plan.values()))
        query_shapes['single_box' if count == 1 else 'multiple_frames' if len(plan) > 1 else 'multiple_boxes_one_frame'] += 1
    ledger = read(ROOT / 'artifacts/drone-validation-coverage/coverage-ledger.json')
    report = {
        'created_at': now(), 'status': status['status'],
        'runs_used': state['runs_reserved'], 'runs_completed': state['runs_completed'],
        'initial_boxes': state['initial_seed_count'], 'final_boxes': len(state['seeds']),
        'new_boxes': len(state['seeds']) - state['initial_seed_count'],
        'new_boxes_from_visual_candidates': sum(r.get('seeds_found', 0) for r in roots if r['kind'] in ('visual_seed', 'local_grid')),
        'new_boxes_from_entry_searches': sum(r.get('seeds_found', 0) for r in roots if r['kind'] == 'entry_scan'),
        'confirmed_classes': sorted({s['class'] for s in state['seeds']}),
        'query_shapes': dict(query_shapes), 'incomplete_attempts': status['incomplete_attempts'],
        'unscreened_groups': len(state['screen']), 'unfinished_groups': len(state['work']),
        'entry_searches': [{k: r.get(k) for k in ('id', 'status', 'seeds_found')} for r in roots if r['kind'] == 'entry_scan'],
        'native_review_completion': ledger['completion'],
        'limitations': [
            'Box counts include repeated frames of the same physical object.',
            'A confirmed box has IoU >= .5 to a hidden label; it is not an exact label.',
            'No-match roots rule out only the tested hypotheses, not the class or region.',
            'The prior dense-grid benchmark supplied centers close to ground truth, so it did not measure discovery recall.',
            'The recorded API cap is a stopping condition, not completion of data acquisition.'
        ]}
    return report


if __name__ == '__main__':
    import json
    report = audit()
    atomic(ROOT / 'artifacts/drone-api-tests/discovery-audit/batch-audit.json', report)
    print(json.dumps(report, indent=2))
