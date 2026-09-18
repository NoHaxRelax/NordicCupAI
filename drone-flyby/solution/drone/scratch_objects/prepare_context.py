"""Add hash-verified reviewed early camera views without consuming late evaluation."""
import argparse, hashlib, json, os
from pathlib import Path
from drone.overnight.common import covered


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(base, source, fixture, output, repeats=3):
    original=json.loads((base/'manifest.json').read_text())
    reviewed=json.loads((source/'manifest.json').read_text())
    evaluation=json.loads((fixture/'fixture.json').read_text())
    assert original['classes']==reviewed['classes']
    ledger_rel='artifacts/drone-validation-coverage/coverage-ledger.json'
    ledger_path=source/'sources'/ledger_rel
    assert sha(ledger_path)==reviewed['source_hashes'][ledger_rel]
    ledger=json.loads(ledger_path.read_text())
    assert ledger['completion']['safe_for_reviewed_region_negative_training']
    forbidden={t['track'] for e in evaluation['examples'] for t in e['truth'] if t.get('track')}
    forbidden.update(t for r in reviewed['records'] if r['source']=='validation' and r['frame']>90 for t in r.get('tracks',[]))
    eval_hashes={r['sha256'] for r in evaluation['examples']}
    eligible=[r for r in reviewed['records'] if r['task']=='detector' and r['split']=='train' and r['source']=='validation' and r['frame']<=90 and r['zoom'] in (1,2) and not forbidden.intersection(r['tracks'])]
    regions={}
    for sheet in ledger['sheets']:
        if all(sheet.get(k,{}).get('status') in ('reviewed_empty','reviewed_objects_resolved') for k in ('primary_review','small_object_review')):
            regions.setdefault(sheet['frame'],[]).extend(sheet['source_regions_xyxy'])
    output.mkdir(parents=True,exist_ok=False)
    records=[]; additions=[]
    def link(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
    for row in original['records']:
        for key,h in [('file','sha256'),('label','label_sha256')]:
            assert sha(base/row[key])==row[h]
            link(base/row[key],output/row[key])
        records.append(row)
    for index,row in enumerate(eligible):
        x1,y1,x2,y2=row['source_region']
        assert covered([x1,y1,x2,y2],regions[row['frame']])
        assert row['sha256'] not in eval_hashes
        for key,h in [('file','sha256'),('label_file','label_sha256')]:assert sha(source/row[key])==row[h]
        assert len((source/row['label_file']).read_text().splitlines())==len(row['classes'])
        for repeat in range(repeats):
            image=f'images/train/context-{index:04d}-{repeat}.png';label=f'labels/train/context-{index:04d}-{repeat}.txt'
            link(source/row['file'],output/image);link(source/row['label_file'],output/label)
            records.append(dict(split='train',file=image,sha256=row['sha256'],label=label,label_sha256=row['label_sha256'],reviewed_source=row))
            additions.extend([image,label])
    manifest={**original,'records':records,'context_followup':dict(base_manifest_sha256=sha(base/'manifest.json'),reviewed_manifest_sha256=sha(source/'manifest.json'),review_ledger_sha256=sha(ledger_path),fixture_sha256=sha(fixture/'fixture.json'),frames=sorted({r['frame'] for r in eligible}),unique_views=len(eligible),repeats=repeats,excluded_tracks=sorted(forbidden),validation_max_frame=90,late_pixels_read=False)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'context-additions.json').write_text(json.dumps(additions))
    print(json.dumps(manifest['context_followup'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.base,a.source,a.fixture,a.output)
