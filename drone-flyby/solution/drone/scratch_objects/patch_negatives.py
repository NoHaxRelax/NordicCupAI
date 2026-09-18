"""Add reviewed empty views from already-consumed early validation frames only."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
import torch
from .patch_cnn import normalize_crop,sha
from .prepare import overlap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args()
    from ultralytics import YOLO
    bank=json.loads((a.bank/'manifest.json').read_text());snapshot=json.loads((a.snapshot/'manifest.json').read_text())
    frames={r['frame'] for r in bank['templates'] if r['source']=='reviewed_validation'}
    ledger='artifacts/drone-validation-coverage/coverage-ledger.json'
    if sha(a.root/ledger)!=snapshot['source_hashes'][ledger]:raise ValueError('Completeness ledger changed since negative snapshot')
    annotations={f:[] for f in frames}
    for path in (a.root/'data/drone/training/algorithmic-full-validation').glob('*.json'):
        for row in json.loads(path.read_text()).get('annotations',[]):
            if row['frame'] in frames:annotations[row['frame']].append(row['bbox_source_xyxy'])
    manifest=json.loads((a.base/'manifest.json').read_text())
    if sha(a.base/'crops.npz')!=manifest['crops_sha256']:raise ValueError('Base crop data changed')
    base=np.load(a.base/'crops.npz');extra=[];records=manifest['records'].copy();rng=np.random.default_rng(1731)
    model=YOLO(str(a.weights));consumed=[];skipped=[]
    for row in snapshot['records']:
        if not(row['task']=='detector' and row['split']=='train' and row['source']=='validation' and row['frame'] in frames and not row['classes']):continue
        path=a.snapshot/row['file'];label=a.snapshot/row['label_file']
        if sha(path)!=row['sha256'] or sha(label)!=row['label_sha256'] or label.read_text().strip():raise ValueError('Negative snapshot changed')
        if any(overlap(row['source_region'],b)>0 for b in annotations[row['frame']]):skipped.append(row['file']);continue
        image=cv2.imread(str(path));h,w=image.shape[:2]
        result=model.predict(image,imgsz=960,conf=.05,max_det=6,device='cpu',verbose=False)[0]
        boxes=result.boxes.xyxy.cpu().tolist()
        for _ in range(3):
            bw=int(rng.integers(12,min(250,w)));bh=int(rng.integers(12,min(250,h)))
            x=int(rng.integers(0,w-bw));y=int(rng.integers(0,h-bh));boxes.append([x,y,x+bw,y+bh])
        for box in boxes:
            if min(box[2]-box[0],box[3]-box[1])<3:continue
            extra.append(normalize_crop(image,box));records.append(dict(file=row['file'],source_image_sha256=row['sha256'],bbox=box,
                 frame=row['frame'],source_region=row['source_region'],class_name='background',kind='reviewed_empty_early_validation_crop'))
        consumed.append(row)
    if not extra:raise ValueError('No valid additional backgrounds')
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/'crops.npz',images=np.concatenate([base['images'],np.array(extra)]),labels=np.concatenate([base['labels'],np.zeros(len(extra),np.int64)]))
    manifest.update(records=records,crops_sha256=sha(a.output/'crops.npz'),base_manifest_sha256=sha(a.base/'manifest.json'),
                    reviewed_negative_snapshot_sha256=sha(a.snapshot/'manifest.json'),code_sha256=sha(__file__),
                    extra_negative_views=consumed,extra_negative_crops=len(extra),extra_negative_frames=sorted({r['frame'] for r in consumed}),
                    skipped_current_annotation_overlap=skipped,additional_validation_frames=0,validation_pixels_added=True)
    manifest['counts']['background']+=len(extra)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(extra_negative_views=len(consumed),extra_negative_crops=len(extra),frames=manifest['extra_negative_frames'],skipped=len(skipped))),flush=True)


if __name__=='__main__':torch.set_num_threads(4);cv2.setNumThreads(1);main()
