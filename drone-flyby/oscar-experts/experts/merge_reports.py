"""Merge evaluate_all shard directories into one report directory (same per-class report format)."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarise(rows, empties, families, zooms):
    summary = {}
    for family in families:
        t, e = Counter(), Counter()
        for row in rows:
            res = row['results'].get(family, {})
            matched = set(int(k) for k in res.get('matched', {}))
            for i, target in enumerate(row['targets']):
                k = 'complete' if target['complete'] else 'partial'
                t[f'{k}_targets'] += 1; t[f"{k}_targets_zoom{row['zoom']}"] += 1
                if i in matched:
                    t[f'{k}_matched'] += 1; t[f"{k}_matched_zoom{row['zoom']}"] += 1
            t['proposals'] += len(res.get('proposals', [])); t['unmatched_proposals'] += res.get('unmatched', 0)
        for emp in empties:
            e['tiles'] += 1; e['false_alarms'] += emp['alarms'].get(family, 0); e['tiles_with_alarm'] += bool(emp['alarms'].get(family, 0))
        summary[family] = dict(complete=f"{t['complete_matched']}/{t['complete_targets']}", partial=f"{t['partial_matched']}/{t['partial_targets']}",
                               per_zoom={z: f"{t[f'complete_matched_zoom{z}']}/{t[f'complete_targets_zoom{z}']}" for z in zooms},
                               proposals=t['proposals'], unmatched_proposals=t['unmatched_proposals'], empty_tiles=e['tiles'], false_alarms=e['false_alarms'], tiles_with_alarm=e['tiles_with_alarm'])
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('shards', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    classes = sorted({d.name for s in a.shards for d in s.iterdir() if (d / 'report.json').exists()})
    seconds = 0.
    for c in classes:
        rows, empties, base = [], [], None
        for s in a.shards:
            rp = s / c / 'report.json'
            if not rp.exists():
                continue
            r = json.loads(rp.read_text()); base = base or r
            rows += r['crops']; empties += r['empty']; seconds += r.get('seconds', 0)
        families = list(base['summary'].keys())
        report = dict(base, crops=rows, empty=empties, positives=len(rows), seconds=seconds, summary=summarise(rows, empties, families, base['zooms']), note='merged shards: ' + ', '.join(str(s) for s in a.shards))
        (a.output / c).mkdir()
        (a.output / c / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(classes=len(classes), output=str(a.output))))


if __name__ == '__main__':
    main()
