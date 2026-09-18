"""Read-only verification against the live local Survival Lab catalog."""
import argparse,datetime,json,urllib.request,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/dedicated_sprinter'
def check(require_complete=False):
    endpoint='http://127.0.0.1:9053/recordings/manifest.json'
    data=json.load(urllib.request.urlopen(endpoint,timeout=60))
    entries=data['recordings'];by_source={}
    for entry in entries:
        for source in entry.get('sources',[entry.get('source')]):
            if source:by_source[source]=entry
    local=[str(p.relative_to(ROOT)) for p in OUT.rglob('*.json.gz')]
    missing=[p for p in local if p not in by_source]
    audit=json.loads((OUT/'replay-backfill-audit.json').read_text())
    pending=[c['id'] for c in audit['cases'] if c['status']=='missing recording']
    result=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),endpoint=endpoint,
        total_catalog_entries=len(entries),local_sprinter_replays=len(local),matched=len(local)-len(missing),missing=missing,
        pending_backfill=pending,records=[dict(source=p,id=by_source[p]['id'],title=by_source[p]['title']) for p in local if p in by_source])
    path=OUT/f'catalog-verification-{uuid.uuid4().hex[:12]}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','pending_backfill')}))
    print('Pending historical run records:',len(pending));print(path)
    if require_complete:assert not missing and not pending,'Catalog or backfill is not complete yet.'
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--require-complete',action='store_true');a=ap.parse_args();check(a.require_complete)
