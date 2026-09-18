"""Compact read-only status for this experiment's isolated Windows directory."""
import argparse,json
from pathlib import Path


def read(path):return json.loads(path.read_text()) if path.exists() else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    result={}
    for name,path in [('initial',a.root/'run-amp-v3/progress.json'),('stage',a.root/'followup-v5-results/status.json'),
                      ('adapted',a.root/'followup-v5-results/adapted/progress.json'),
                      ('continuation_verified',a.root/'followup-v5-results/adapted/continuation-verified.json'),
                      ('complete',a.root/'followup-v5-results/complete.json')]:
        result[name]=read(path)
    failure=a.root/'followup-v5-results/failure.txt'
    if failure.exists():result['failure']=failure.read_text()[-4000:]
    result['evaluations']={}
    for path in sorted((a.root/'followup-v5-results').glob('*-L*/report.json')):
        report=read(path);summary={}
        for split in ['reference','validation']:
            for threshold in ['0.1','0.25','0.5']:
                rows=[r['thresholds'][threshold]['class_aware'] for r in report['results'] if r['split']==split]
                summary[split+'@'+threshold]={key:sum(r[key] for r in rows) for key in ['targets','matched','proposals']}
        result['evaluations'][path.parent.name]=dict(frames=len(report['results']),original_annotation_cohort=summary)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
