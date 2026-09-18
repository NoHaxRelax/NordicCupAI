"""Apply Oscar's hash-bound empty-square review without changing frozen inputs."""
import argparse
import json
from pathlib import Path
from .build import digest, write

def approve(dataset, receipt):
    mpath=dataset/'manifest.json';m=json.loads(mpath.read_text());r=json.loads(receipt.read_text())
    if r.get('reviewer')!='Oscar Svendsen' or r.get('attested_personal_visual_review') is not True:raise ValueError('Personal review attestation required')
    if r.get('dataset_manifest_sha256')!=digest(mpath):raise ValueError('Review refers to a different dataset')
    candidates={v['tile_id']:v for v in json.loads((dataset/'review-candidates.json').read_text())['candidates']}
    accepted=set();seen=set()
    for row in r['decisions']:
        tid=row['tile_id']
        if tid in seen:raise ValueError('Duplicate decision')
        seen.add(tid)
        source=candidates[tid]
        for k in ['frame','source_rect_xyxy','source_sha256','native_image','native_image_sha256']:
            if row[k]!=source[k]:raise ValueError('Candidate identity mismatch: '+tid)
        if digest(dataset/source['native_image'])!=source['native_image_sha256']:raise ValueError('Reviewed pixels changed')
        if row['status'] not in {'verified_empty','contains_target','unsure'}:raise ValueError('Unknown decision')
        if row['status']=='verified_empty':accepted.add(tid)
    for v in m['records']:
        if v['tile_id'] in accepted:
            if v['split']!='pending_review' or v['annotations']:raise ValueError('Not an unlabelled pending candidate')
            if digest(dataset/v['file'])!=v['sha256']:raise ValueError('Paired pixels changed')
            target_split=v.get('intended_split','train')
            if target_split not in {'train','dev'}:raise ValueError('Invalid intended split')
            v.update(split=target_split,kind='background',background_target=1,supervision='human_verified_empty',annotation_complete=True,eligible_for_training=target_split=='train',reviewer=r['reviewer'])
    m['human_review']={'receipt_sha256':digest(receipt),'accepted_squares':len(accepted),'reviewer':r['reviewer']}
    dest=dataset/'manifest-approved.json'
    if dest.exists():raise ValueError('An approved manifest already exists; preserve it and create a new version explicitly')
    write(dest,m)
    sampler=json.loads((dataset/'sampler.json').read_text()) if (dataset/'sampler.json').exists() else {'addendum_only':True};sampler['background_ids']=[v['id'] for v in m['records'] if v['eligible_for_training'] and v['kind']=='background']
    sampler['manifest']='manifest-approved.json';sampler['manifest_sha256']=digest(dest)
    write(dataset/'sampler-approved.json',sampler);write(dataset/'human-review-receipt.json',r)
    print(f'Approved {len(accepted)} squares / {len(accepted)*3} images. Original manifest preserved.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args();approve(a.dataset,a.review)
