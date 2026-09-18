"""Verify every complete colony replay is selectable in the running local lab."""
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/colony_banishment'
URL='http://127.0.0.1:9053/recordings/manifest.json'


def main():
    expected={str(p.relative_to(ROOT)) for p in OUT.rglob('*.json.gz')}
    with urllib.request.urlopen(URL,timeout=60) as response:
        manifest=json.load(response)
    found={}
    for row in manifest['recordings']:
        for source in row.get('sources',[row.get('source','')]):
            if source in expected:
                found[source]={'id':row['id'],'title':row['title'],'file':row['file']}
    missing=sorted(expected-set(found))
    result=dict(verified_at=datetime.now(timezone.utc).isoformat(),url=URL,
        expected_files=len(expected),catalogued_files=len(found),missing=missing,
        catalogued_recordings=len({v['id'] for v in found.values()}),
        catalog_scanning=manifest.get('stats',{}).get('scanning',False),recordings=found)
    path=OUT/'live-catalog-verification.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(path)
    print(json.dumps({k:v for k,v in result.items() if k not in ('recordings','missing')}|
        {'missing_count':len(missing),'missing_sample':missing[:3]},indent=2))
    if missing:raise SystemExit(1)


if __name__=='__main__':main()
