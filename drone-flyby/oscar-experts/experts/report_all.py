"""Summarise one evaluate.py output directory per class into a single table (markdown + json)."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    rows = []
    for report_path in sorted(a.runs.glob('*/report.json')):
        r = json.loads(report_path.read_text())
        main_family = [f for f in r['summary'] if f != 'sift'][0]
        s = r['summary'][main_family]
        sift = r['summary'].get('sift', {})
        n_pos = r['positives']
        cands = sum(len([c for c in c_['candidates'] if c.get('rejected_by') is None]) for c_ in r['crops'])
        rows.append(dict(class_name=r['class_name'], complete=s['complete'], partial=s['partial'], per_zoom=s['per_zoom'], candidates_per_tile=round(cands / max(1, n_pos), 2),
                         empty_tiles=s['empty_tiles'], false_alarms=s['false_alarms'], tiles_with_alarm=s['tiles_with_alarm'], sift_complete=sift.get('complete'),
                         seconds_per_tile=round(r['seconds'] / max(1, n_pos + s['empty_tiles']), 2)))
    lines = ['| class | complete found | partial | L0/L1/L2 | cand/tile | empty tiles | false alarms | tiles alarmed | SIFT | s/tile |', '|---|---|---|---|---|---|---|---|---|---|']
    for x in rows:
        lines.append(f"| {x['class_name']} | {x['complete']} | {x['partial']} | {'/'.join(str(x['per_zoom'].get(str(z), x['per_zoom'].get(z, '-'))) for z in (0, 1, 2))} | {x['candidates_per_tile']} | {x['empty_tiles']} | {x['false_alarms']} | {x['tiles_with_alarm']} | {x['sift_complete']} | {x['seconds_per_tile']} |")
    text = '\n'.join(lines)
    print(text)
    if a.output:
        a.output.write_text(text + '\n')
        a.output.with_suffix('.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
