"""Build 384-source-pixel grid squares, with exact 96/192/384 zoom views.

Validation positives provide positive supervision only. Unlabelled classes and
locations remain unknown. Validation negatives require a separate human receipt.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIDE, STRIDE = 384, 192
GOOD = {'reviewed_positive', 'score_confirmed_match'}
ALIASES = {
    'large-tower-a-extension-037': 'large-tower-038-067',
    'helicopter-b-early-077-104': 'helicopter-b-105-149',
    'small-launcher-b-early-118': 'small-launcher-b-119-140',
    'small-tower-b-early-238-239': 'small-tower-b-240-249',
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')

def positions(length):
    return sorted(set(range(0, length-SIDE+1, STRIDE)) | {length-SIDE})

def intersection(a,b):
    x,y,u,v=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    return [x,y,u,v] if u>x and v>y else None

def contains(a,b):
    return a[0]<=b[0] and a[1]<=b[1] and a[2]>=b[2] and a[3]>=b[3]

def group(a):
    t=a['track_id']; return ALIASES.get(t,t)

def sampled(values, n):
    values=sorted(set(values))
    return [values[i] for i in sorted(set(np.linspace(0,len(values)-1,min(n,len(values))).round().astype(int)))] if values else []

def render(square, zoom):
    """Integer-aligned INTER_AREA equals cropping the officially reduced view."""
    side=SIDE//(4//(2**zoom))
    low=cv2.resize(square,(side,side),interpolation=cv2.INTER_AREA)
    return cv2.resize(low,(SIDE,SIDE),interpolation=cv2.INTER_LINEAR) if side!=SIDE else low

def build(out):
    if out.exists(): raise ValueError('Use a new output directory; never overwrite a frozen dataset')
    out.mkdir(parents=True)
    ref=ROOT/'data/drone/reference/helsinki'
    val=ROOT/'data/drone/training/score-anchored-validation-v8/finetune-metadata/annotations'
    sources={}; frames={}; tracks=defaultdict(list)
    def remember(p): sources[str(p.relative_to(ROOT))]=digest(p)
    for source,folder in [('reference',ref/'annotations'),('validation',val)]:
        for p in sorted(folder.glob('*.json')):
            d=json.loads(p.read_text()); f=d['frame']; remember(p)
            rows=[]
            for a in d['annotations']:
                a=dict(a)
                if source=='reference': a.update(track_id='reference-'+a['object_id'],provenance='organizer_reference')
                a['group']=group(a); rows.append(a)
                if source=='validation': tracks[a['group']].append(f)
            frames[source,f]=rows
    # Full annotation metadata, including projected tails, determines track splits.
    split={t:('train' if max(fs)<=90 else 'dev' if min(fs)>=92 and max(fs)<=180 else 'reserved') for t,fs in tracks.items()}
    classes=sorted({a['object_id'] for (s,f),aa in frames.items() if s=='reference' for a in aa})
    capture=ROOT/'data/drone/reconstructed-validation/manifest.json';remember(capture)
    complete={r['frame'] for r in json.loads(capture.read_text()) if r.get('complete') and r.get('native_coverage')==1}
    chosen=defaultdict(list)
    for (s,f),aa in frames.items():
        for a in aa:
            if s=='reference' or (split[a['group']]!='reserved' and a['provenance'] in GOOD and f in complete and (f<=90 or 100<=f<=180)):
                b=a['bbox']
                if 0<b[0]<b[2]<3839 and 0<b[1]<b[3]<2159: chosen[s,a['group']].append(f)
    selected={k:set(sampled(v,8)) for k,v in chosen.items()}
    grid=[[x,y,x+SIDE,y+SIDE] for y in positions(2160) for x in positions(3840)]
    planned=defaultdict(dict); excluded=Counter()
    for (s,f),aa in frames.items():
        for a in aa:
            if f not in selected.get((s,a['group']),set()):continue
            candidates=[r for r in grid if contains(r,a['bbox'])]
            if not candidates:excluded['object_not_contained_by_grid']+=1
            # Deterministic off-centre examples from the real grid, not centred crops.
            candidates.sort(key=lambda r:hashlib.sha256(f'{s}:{f}:{a["group"]}:{r}'.encode()).hexdigest())
            for r in candidates[:3]:
                hits=[b for b in aa if intersection(r,b['bbox'])]
                if s=='validation' and any(split[b['group']]!=split[a['group']] for b in hits):
                    excluded['mixed_split_tile']+=1;continue
                if s=='validation' and any(b['provenance'] not in GOOD for b in hits):
                    excluded['tile_intersects_projected_label']+=1;continue
                planned[s,f][tuple(r)]={'split':'train' if s=='reference' else split[a['group']], 'kind':'positive'}
    # Reference negatives are licensed by organizer annotations; validation negatives
    # are candidates only, never inferred to be empty from missing annotations.
    for s,fs,n in [('reference',sampled([f for ss,f in frames if ss=='reference'],8),12),('validation',[5,13,25,37,49,61,73,85],8)]:
        for f in fs:
            if s=='validation' and f not in complete:continue
            aa=frames[s,f]
            candidates=[r for r in grid if not any(intersection([r[0]-32,r[1]-32,r[2]+32,r[3]+32],a['bbox']) for a in aa)]
            # Spread candidates spatially; keep all validation examples pending Oscar.
            candidates.sort(key=lambda r:hashlib.sha256(f'negative:{s}:{f}:{r}'.encode()).hexdigest())
            for r in candidates[:n]: planned[s,f][tuple(r)]={'split':'train' if s=='reference' else 'pending_review','kind':'background' if s=='reference' else 'unknown'}
    records=[]; candidates=[]
    for (s,f),regions in sorted(planned.items()):
        p=ref/'images'/f'frame_{f:06d}.png' if s=='reference' else ROOT/'data/drone/reconstructed-validation'/f'frame_{f:06d}.png'
        remember(p); im=cv2.imread(str(p)); assert im is not None and im.shape[:2]==(2160,3840)
        for rect,meta in sorted(regions.items()):
            x,y,u,v=rect; tile_id=f'{s}-f{f:06d}-x{x:04d}-y{y:04d}'
            labels=[]
            for a in frames[s,f]:
                hit=intersection(rect,a['bbox'])
                if hit:
                    labels.append(dict(class_name=a['object_id'],class_id=classes.index(a['object_id']),track_id=a['track_id'],group=a['group'],bbox_xyxy=[hit[0]-x,hit[1]-y,hit[2]-x,hit[3]-y],source_bbox_xyxy=a['bbox'],fully_contained=contains(rect,a['bbox']),visible_fraction=(hit[2]-hit[0])*(hit[3]-hit[1])/((a['bbox'][2]-a['bbox'][0])*(a['bbox'][3]-a['bbox'][1])),provenance=a['provenance']))
            target_classes=sorted({a['class_id'] for a in labels if a['fully_contained']})
            for z in range(3):
                rel=f'images/{meta["split"]}/{tile_id}-L{z}.png'; dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                assert cv2.imwrite(str(dest),render(im[y:v,x:u],z))
                rec=dict(id=tile_id+f'-L{z}',tile_id=tile_id,source=s,frame=f,source_file=str(p.relative_to(ROOT)),source_sha256=sources[str(p.relative_to(ROOT))],source_rect_xyxy=list(rect),zoom=z,native_crop_size=SIDE//(4//2**z),input_size=SIDE,file=rel,sha256=digest(dest),**meta,annotations=labels,target_class_ids=target_classes,annotation_complete=s=='reference',supervision='complete_boxes' if s=='reference' else 'known_positives_only' if meta['kind']=='positive' else 'none',background_target=1 if meta['kind']=='background' else 0 if meta['kind']=='positive' else None,eligible_for_training=meta['split']=='train')
                records.append(rec)
            if meta['split']=='pending_review':candidates.append(dict(tile_id=tile_id,frame=f,source_rect_xyxy=list(rect),source_sha256=records[-1]['source_sha256'],native_image=records[-1]['file'],native_image_sha256=records[-1]['sha256'],status='unreviewed'))
    # Explicit class/track/frame/zoom hierarchy; no duplicated files for balancing.
    buckets={c:defaultdict(lambda:defaultdict(list)) for c in classes};neg=[]
    for r in records:
        if not r['eligible_for_training']:continue
        if r['kind']=='background':neg.append(r['id'])
        for a in r['annotations']:
            if a['fully_contained']:buckets[a['class_name']][a['group']][str(r['frame'])].append(r['id'])
    counts=Counter((r['split'],r['kind'],r['zoom']) for r in records)
    summary=dict(total_images=len(records),unique_source_squares=len(records)//3,counts=[dict(split=k[0],kind=k[1],zoom=k[2],images=v) for k,v in sorted(counts.items())],human_negative_candidates=len(candidates),classes=classes,train_class_groups={c:len(buckets[c]) for c in classes},dev_classes=sorted({a['class_name'] for r in records if r['split']=='dev' for a in r['annotations'] if a['fully_contained']}),excluded=dict(excluded))
    manifest=dict(schema=1,config=dict(source_square=SIDE,source_stride=STRIDE,input_size=SIDE,native_sizes=[96,192,384],downsample='INTER_AREA',upsample='INTER_LINEAR',grid_origin=[0,0],edge_policy='append final edge-aligned window',seed='sha256 deterministic ordering',max_frames_per_track=8,max_containing_windows_per_target=3),classes=classes,source_hashes=sources,track_splits=split,track_aliases=ALIASES,records=records,limitations=['Participant validation positives are not organizer ground truth.','Validation annotation completeness is unknown; never use unlabelled pixels/classes as negative targets.','Reference objects and backgrounds recur across reference frames: all are training only.','Dev tracks are held out from this dataset only; other experiments may have used them.','No validation pixels after frame 180 are consumed. No claim of project-wide untouched test data.','Partial-only grid squares excluded from positive sampling; partial intersecting boxes retained explicitly.','Pending validation negatives require Oscar review before inclusion.'])
    write(out/'manifest.json',manifest);write(out/'summary.json',summary)
    write(out/'sampler.json',dict(positive_probability=.5,background_probability=.5,positive_strategy='uniform class, then uniform physical group, then uniform frame, then uniform zoom, then uniform eligible tile',positive_buckets=buckets,background_ids=neg,background_strategy='uniform source, then frame, then tile, then zoom among approved background records',augmentation='No offline augmentation. Apply joint image/box transforms only after split; all paired zooms share a split.'))
    write(out/'review-candidates.json',dict(dataset_manifest_sha256=digest(out/'manifest.json'),candidates=candidates))
    (out/'review-data.js').write_text('window.REVIEW_DATA='+json.dumps(dict(dataset_manifest_sha256=digest(out/'manifest.json'),candidates=candidates))+';')
    shutil.copyfile(Path(__file__).with_name('review.html'),out/'review.html')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.output)
