"""Audit original-case coverage, immutable backfill receipts and live discovery."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

BASE=Path(__file__).resolve().parent
OUT=BASE.parents[1]/'results/sprinter_relay'
SURVIVAL=BASE.parents[1]
PHASES=['pilot','iterate','orbit-screen','relay-screen','heldout-invalid-single-baseline',
        'heldout','workers','multiple','renewal-screen','replays']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser()
parser.add_argument('--catalog',default='http://127.0.0.1:9053/recordings/manifest.json')
args=parser.parse_args()
coverage=[]
for phase in PHASES:
    source=OUT/(phase+'.json');source_hash=sha(source)
    for index,row in enumerate(json.loads(source.read_text())['runs']):
        original=row.get('replay')
        if original and (OUT/original).exists():
            coverage.append(dict(source_case=f'{phase}[{index}]',status='original_recording',replay=original))
        else:
            receipt=OUT/'backfill-receipts'/f'{phase}-{index:03d}.json'
            item=json.loads(receipt.read_text())
            assert item['source_file_sha256']==source_hash,'Original evidence was altered'
            assert (OUT/item['replay']).exists()
            coverage.append(dict(source_case=f'{phase}[{index}]',status='new_reproduction',
                replay=item['replay'],limitations=item['limitations'],
                original_retention=item['original_retention'],reproduced_retention=item['reproduced_retention']))
paths=sorted(OUT.rglob('*.json.gz'))
recordings=[]
for p in paths:
    with gzip.open(p,'rt') as f:data=json.load(f)
    assert data['format']=='survival-replay' and data['version']==1
    assert data['summary']['frames']==len(data['frames'])
    assert abs(data['summary']['duration']-data['frames'][-1]['t'])<1e-4
    assert data['frames'][0]['t']==0
    assert all(a['t']<b['t'] for a,b in zip(data['frames'],data['frames'][1:]))
    recordings.append(dict(source=str(p.relative_to(SURVIVAL)),title=data['meta']['title'],
        frames=len(data['frames']),duration=data['summary']['duration'],
        native=any('native_image' in frame for frame in data['frames'])))
with urllib.request.urlopen(args.catalog,timeout=30) as response:catalog=json.load(response)
listed={source for row in catalog.get('recordings',[]) for source in row.get('sources',[row.get('source')])}
missing=[r['source'] for r in recordings if r['source'] not in listed]
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),catalog_url=args.catalog,
    catalog_stats=catalog.get('stats'),original_cases=len(coverage),
    original_recordings=sum(c['status']=='original_recording' for c in coverage),
    new_reproductions=sum(c['status']=='new_reproduction' for c in coverage),
    strategy_replay_files=len(recordings),native_replays=sum(r['native'] for r in recordings),
    missing_from_live_catalog=missing,coverage=coverage,recordings=recordings)
(OUT/'replay-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('coverage','recordings')},indent=2))
if missing:raise SystemExit('Replay files are complete but live catalog has not indexed every file yet.')
