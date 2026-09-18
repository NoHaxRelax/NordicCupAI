"""Evaluate proposal fusion using frozen blind-search predictions."""
import argparse,json
from pathlib import Path
from .evaluate import measure
from .hybrid import merge_proposals
from .prepare import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cnn-threshold',type=float,default=.25);p.add_argument('--background-threshold',type=float,default=.5)
    a=p.parse_args();report=json.loads(a.report.read_text());features=json.loads(a.features.read_text())
    if report['zoom']!=2 or features['zoom']!=2:raise ValueError('Native benchmark required')
    lookup={(r['split'],r['frame']):r for r in features['results']}
    result=dict(cnn_report_sha256=sha(a.report),feature_report_sha256=sha(a.features),cnn_threshold=a.cnn_threshold,
                background_threshold=a.background_threshold,merger='feature priority, class-aware IoU .35 suppression',scope='Offline fusion diagnostic. Per-view verification before global suppression may differ in the runtime.',results=[])
    for row in report['results']:
        key=(row['split'],row['frame'])
        if key not in lookup:raise ValueError('Frame cohorts differ')
        truth=[t for t in row['truth'] if 0<t['bbox'][0]<t['bbox'][2]<3839 and 0<t['bbox'][1]<t['bbox'][3]<2159]
        cnn=[p for p in row['predictions'] if p['score']>=a.cnn_threshold and p['background_probability']<a.background_threshold]
        predictions=merge_proposals(cnn,lookup[key]['predictions'])
        result['results'].append(dict(split=row['split'],frame=row['frame'],predictions=predictions,truth=truth,**measure(predictions,truth)))
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    for split in ['reference','validation']:
        rows=[r for r in result['results'] if r['split']==split]
        print(json.dumps(dict(split=split,**{k:sum(r[k] for r in rows) for k in ['matched','targets','proposals']})))


if __name__=='__main__':main()
