"""Compare saved predictions at fixed thresholds without running inference again."""
import argparse
import json
from pathlib import Path

from .evaluate import measure


def summarize(report, thresholds=(.65,.7,.75,.8,.85,.9)):
    thresholds = tuple(t for t in thresholds if t >= report['settings']['score_threshold'])
    rows=[]
    for threshold in thresholds:
        row={'threshold':threshold}
        for split in sorted({r['split'] for r in report['results']}):
            results=[measure([p for p in r['predictions'] if p['score']>=threshold],r['truth'])
                     for r in report['results'] if r['split']==split]
            targets=sum(r['targets'] for r in results)
            hits=sum(r['matched'] for r in results)
            proposals=sum(r['proposals'] for r in results)
            row[split]={'targets':targets,'matched':hits,'proposals':proposals,
                        'recall':hits/targets if targets else None,
                        'unmatched_proposals':proposals-hits}
        rows.append(row)
    return {'bank_sha256':report['bank_sha256'],'zoom':report['zoom'],
            'top_band':report.get('top_band',False),'thresholds':rows,
            'warning':'Validation annotations are incomplete; unmatched proposals are not confirmed false positives.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=json.loads(a.report.read_text())
    result=summarize(report)
    with a.output.open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
