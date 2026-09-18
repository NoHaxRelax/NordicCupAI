"""Adapt annotations to the existing inspector; never alter simulation frames.

Run only after recording processes finish. The recorder accepts arbitrary
decision values; the existing inspector expects {rule, detail} objects.
"""
import gzip
import json
import sys
from pathlib import Path
from run import OUT


paths=[Path(p) for p in sys.argv[1:]] if len(sys.argv)>1 else sorted(OUT.glob('*.json.gz'))
for path in paths:
    assert path.resolve().parent==OUT.resolve(),'Only owned replay outputs may be edited'
    with gzip.open(path,'rt') as f:data=json.load(f)
    changed=0
    for frame in data['frames']:
        for agent in frame['agents']:
            if isinstance(agent.get('decision'),str):
                agent['decision']={'rule':agent['decision'],'detail':''};changed+=1
    if changed:
        data['meta']['annotation_format_note']='Recorded decision strings wrapped in existing viewer rule/detail objects after recording; no actions, observations or physics data changed.'
        temporary=path.with_name(path.name+'.tmp')
        with gzip.open(temporary,'wt',encoding='utf8') as f:json.dump(data,f,separators=(',',':'),allow_nan=False)
        temporary.replace(path)
    print(f'{path.name}: {changed} annotations formatted')
