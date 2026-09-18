"""Build reproducible copy/paste detection data from a frozen template bank.

Only bank crops and annotated reference training frames provide training pixels.
Backgrounds exclude all organizer annotation boxes with a 24 pixel guard.
Synthetic validation is a fitting diagnostic, never a transfer accuracy claim.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def transform(image,mask,angle,scale):
    h,w=image.shape[:2]
    matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,scale)
    width=int(np.ceil(scale*(w*abs(np.cos(np.deg2rad(angle)))+h*abs(np.sin(np.deg2rad(angle))))))
    height=int(np.ceil(scale*(h*abs(np.cos(np.deg2rad(angle)))+w*abs(np.sin(np.deg2rad(angle))))))
    matrix[:,2]+=np.array([width/2-w/2,height/2-h/2])
    return (cv2.warpAffine(image,matrix,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101),
            cv2.warpAffine(mask,matrix,(width,height),flags=cv2.INTER_LINEAR))


def build(root,bank,output,count=768,seed=1731):
    rng=np.random.default_rng(seed)
    manifest=json.loads((bank/'manifest.json').read_text())
    classes=sorted(manifest['classes'])
    templates=defaultdict(list)
    for row in manifest['templates']:
        for key,digest in [('file','sha256'),('mask_file','mask_sha256')]:
            if row.get(key) and sha(bank/row[key])!=row[digest]:
                raise ValueError('Bank changed: '+row[key])
        image=cv2.imread(str(bank/row['file']))
        mask=cv2.imread(str(bank/row['mask_file']),0)
        # Preserve failed masks as rectangular crops; do not silently lose a class.
        # These context-bearing samples are recorded in the embedded bank manifest.
        templates[row['class']].append((row,image,mask))
    if set(templates)!=set(classes):
        raise ValueError('Mask quality filtering removed a class')
    output.mkdir(parents=True,exist_ok=False)
    backgrounds=[]
    hashes={}
    for frame in sorted({r['frame'] for r in manifest['templates'] if r['source']=='organizer_reference'}):
        if frame in manifest['reference_holdout_frames']:
            raise ValueError('Reference split overlap')
        stem=f'frame_{frame:06d}'
        source=root/'data/drone/reference/helsinki'
        annotation=source/'annotations'/f'{stem}.json'
        path=source/'images'/f'{stem}.png'
        for p in [annotation,path]:hashes[str(p.relative_to(root))]=sha(p)
        image=cv2.imread(str(path))
        boxes=[a['bbox'] for a in json.loads(annotation.read_text())['annotations']]
        accepted=0
        for attempt in range(1000):
            x=int(rng.integers(0,3840-640));y=int(rng.integers(0,2160-640))
            if any(overlap([x-24,y-24,x+664,y+664],b)>0 for b in boxes):continue
            backgrounds.append((stem,[x,y,x+640,y+640],image[y:y+640,x:x+640].copy()))
            accepted+=1
            if accepted==8:break
    if not backgrounds:raise ValueError('No known reference backgrounds')
    records=[]
    for split,n in [('train',count),('synthetic_dev',96)]:
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
        for index in range(n):
            bg_index=int(rng.integers(len(backgrounds)))
            stem,region,background=backgrounds[bg_index]
            canvas=background.copy()
            if rng.random()<.5:canvas=canvas[:,::-1].copy()
            canvas=np.rot90(canvas,int(rng.integers(4))).copy()
            labels=[];boxes=[];sources=[]
            # Every eighth training image is an explicitly empty scene.
            objects=0 if index%8==0 else int(rng.integers(3,10))
            for object_index in range(objects):
                label=classes[(index*7+object_index)%len(classes)]
                candidates=templates[label]
                row,patch,mask=candidates[int(rng.integers(len(candidates)))]
                scale=float(np.exp(rng.uniform(np.log(.4),np.log(1.45))))
                angle=float(rng.uniform(-180,180))
                patch,alpha=transform(patch,mask,angle,scale)
                h,w=patch.shape[:2]
                if min(h,w)<8 or max(h,w)>420:continue
                for _ in range(30):
                    x=int(rng.integers(2,638-w));y=int(rng.integers(2,638-h))
                    box=[x,y,x+w,y+h]
                    if not any(overlap(box,b)>0 for b in boxes):break
                else:continue
                alpha=cv2.GaussianBlur(alpha,(3,3),.5).astype(np.float32)/255
                gain=float(rng.uniform(.85,1.15))
                adjusted=np.clip(patch.astype(np.float32)*gain,0,255)
                canvas[y:y+h,x:x+w]=np.uint8(adjusted*alpha[:,:,None]+canvas[y:y+h,x:x+w]*(1-alpha[:,:,None]))
                boxes.append(box)
                labels.append(f'{classes.index(label)} {(x+w/2)/640:.7f} {(y+h/2)/640:.7f} {w/640:.7f} {h/640:.7f}')
                sources.append(dict(template=row['id'],angle=angle,scale=scale,bbox=box))
            filename=f'{index:05d}'
            image_path=output/'images'/split/(filename+'.jpg')
            label_path=output/'labels'/split/(filename+'.txt')
            cv2.imwrite(str(image_path),canvas,[cv2.IMWRITE_JPEG_QUALITY,95])
            label_path.write_text('\n'.join(labels)+'\n' if labels else '')
            records.append(dict(split=split,file=str(image_path.relative_to(output)),sha256=sha(image_path),
                                label=str(label_path.relative_to(output)),label_sha256=sha(label_path),
                                background=dict(frame=stem,region=region),objects=sources))
    report=dict(schema=1,seed=seed,bank_sha256=sha(bank/'manifest.json'),bank=manifest,
                classes=classes,source_hashes=hashes,records=records,
                usable_templates={k:len(v) for k,v in templates.items()},
                policy='Training crops exactly from bank; reference-only background pixels; no extra validation pixels. Synthetic dev shares training assets and is not an independent test.')
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(images=Counter(r['split'] for r in records),usable_templates=report['usable_templates'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=768)
    a=p.parse_args();cv2.setNumThreads(4);build(a.root,a.bank,a.output,a.count)
