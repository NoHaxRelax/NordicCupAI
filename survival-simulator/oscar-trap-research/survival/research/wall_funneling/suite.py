"""Reproduce a bounded funnel experiment suite; every case saves a unique replay."""
import argparse
from datetime import datetime, timezone
import json
from run import run, OUT

GEOMETRIES = [(30,30),(30,50),(30,60),(30,70),(30,85),(30,100),
              (32,70),(32,85),(32,100),(35,70),(35,85),(35,100),
              (38,70),(38,100),(40,100),(45,100),(60,100),(100,100)]


def cases(name):
    if name == 'geometry':
        for w,l in GEOMETRIES:
            for heading,offset in [(-.3,-15),(.3,15),(0,0)]:
                yield dict(width=w,length=l,heading=heading,offset=offset,seconds=90)
    elif name == 'escape':
        for mode in ['loop','peel']:
            for peel in [25,35,45,55,65]:
                yield dict(mode=mode,peel=peel,seconds=45)
    elif name == 'gated':
        for w in [30,32,35]:
            for l in [70,85,100]:
                yield dict(width=w,length=l,mode='rest_gate',seconds=180)
    elif name == 'stress':
        for w,l in [(30,70),(30,85),(30,100),(32,70),(32,100)]:
            yield dict(width=w,length=l,waves=10,interval=15,seconds=300)
        for w in [30,32,35]:
            yield dict(width=w,length=100,mode='rest_gate',waves=10,seconds=600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',choices=['geometry','escape','gated','stress'])
    args = parser.parse_args()
    rows = []
    for case in cases(args.suite):
        r = run(**case)
        rows.append(r)
        print({k:r[k] for k in ['mode','width','length','seconds','final_success',
              'physical_continuous_success']},flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    path = OUT / f'suite-{args.suite}-{stamp}.json'
    path.write_text(json.dumps(rows,indent=2)+'\n')
    print(path)
