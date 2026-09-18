"""A scratch crop CNN used only as an optional blind-proposal verifier.

Training positives come from the frozen training scenes. Negatives are crops
from explicitly empty training images, including detector-mined false alarms.
Crop fitting accuracy is never reported as object discovery performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch import nn


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_crop(image,box):
    x1,y1,x2,y2=box
    x1=max(0,int(np.floor(x1)));y1=max(0,int(np.floor(y1)))
    x2=min(image.shape[1],int(np.ceil(x2)));y2=min(image.shape[0],int(np.ceil(y2)))
    patch=image[y1:y2,x1:x2,:3]
    if min(patch.shape[:2])<2:raise ValueError('Empty proposal crop')
    h,w=patch.shape[:2];factor=48/max(h,w)
    patch=cv2.resize(patch,(max(1,round(w*factor)),max(1,round(h*factor))),interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR)
    canvas=np.full((64,64,3),127,np.uint8);h,w=patch.shape[:2]
    canvas[(64-h)//2:(64-h)//2+h,(64-w)//2:(64-w)//2+w]=patch
    return canvas


class PatchCNN(nn.Module):
    def __init__(self,classes=17):
        super().__init__()
        blocks=[];channels=3
        for out,stride in [(24,1),(32,2),(64,2),(96,2),(128,2)]:
            blocks.extend([nn.Conv2d(channels,out,3,stride,1,bias=False),nn.BatchNorm2d(out),nn.SiLU()]);channels=out
        self.features=nn.Sequential(*blocks)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(128*4*4,128),nn.SiLU(),nn.Dropout(.15),nn.Linear(128,classes))
    def forward(self,x):return self.head(self.features(x))


def prepare(a):
    from ultralytics import YOLO
    manifest=json.loads((a.data/'manifest.json').read_text());classes=['background',*manifest['classes']]
    template_class={r['id']:r['class'] for r in manifest['bank']['templates']}
    model=YOLO(str(a.weights));rng=np.random.default_rng(1731)
    images=[];labels=[];sources=[];negative_images=0
    for row in manifest['records']:
        if row['split']!='train':continue
        path=a.data/row['file']
        if sha(path)!=row['sha256']:raise ValueError('Training image hash mismatch')
        im=cv2.imread(str(path));h,w=im.shape[:2]
        if row['objects']:
            for obj in row['objects']:
                label=classes.index(template_class[obj['template']])
                images.append(normalize_crop(im,obj['bbox']));labels.append(label)
                sources.append(dict(file=row['file'],bbox=obj['bbox'],class_name=classes[label],kind='training_positive'))
        else:
            labelpath=a.data/row['label']
            if sha(labelpath)!=row['label_sha256'] or labelpath.read_text().strip():raise ValueError('Unverified negative')
            result=model.predict(im,imgsz=960,conf=.05,max_det=6,device='cpu',verbose=False)[0]
            boxes=result.boxes.xyxy.cpu().tolist()
            for _ in range(3):
                bw=int(rng.integers(12,min(250,w)));bh=int(rng.integers(12,min(250,h)))
                x=int(rng.integers(0,w-bw));y=int(rng.integers(0,h-bh));boxes.append([x,y,x+bw,y+bh])
            for box in boxes:
                if min(box[2]-box[0],box[3]-box[1])<3:continue
                images.append(normalize_crop(im,box));labels.append(0)
                sources.append(dict(file=row['file'],bbox=box,class_name='background',kind='verified_empty_training_crop'))
            negative_images+=1
        if len(sources)%500<10:print(json.dumps(dict(crops=len(sources),negative_images=negative_images)),flush=True)
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/'crops.npz',images=np.array(images,np.uint8),labels=np.array(labels,np.int64))
    report=dict(classes=classes,records=sources,data_manifest_sha256=sha(a.data/'manifest.json'),detector_sha256=sha(a.weights),
                crops_sha256=sha(a.output/'crops.npz'),code_sha256=sha(__file__),negative_images=negative_images,
                counts={c:labels.count(i) for i,c in enumerate(classes)},validation_pixels_added=0)
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['counts']),flush=True)


def train(a):
    if a.steps<1:raise ValueError('Positive training step count required')
    rng=np.random.default_rng(1731);torch.manual_seed(1731)
    manifest=json.loads((a.data/'manifest.json').read_text())
    if sha(a.data/'crops.npz')!=manifest['crops_sha256']:raise ValueError('Crop data changed')
    dataset=np.load(a.data/'crops.npz');images=dataset['images'];labels=dataset['labels']
    groups=[np.flatnonzero(labels==i) for i in range(len(manifest['classes']))]
    if any(not len(g) for g in groups):raise ValueError('Class missing from crop training')
    a.output.mkdir(parents=True,exist_ok=False)
    model=PatchCNN(len(groups));optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps,eta_min=.0001)
    started=time.time()
    for step in range(a.steps):
        chosen=rng.integers(1,len(groups),64);chosen[:32]=0;rng.shuffle(chosen)
        indices=[int(rng.choice(groups[label])) for label in chosen];batch=[]
        for index in indices:
            patch=images[index]
            matrix=cv2.getRotationMatrix2D((32,32),float(rng.uniform(-180,180)),float(rng.uniform(.85,1.15)))
            patch=cv2.warpAffine(patch,matrix,(64,64),borderValue=(127,127,127))
            if rng.random()<.5:patch=patch[:,::-1]
            patch=np.clip(patch.astype(np.float32)*rng.uniform(.85,1.15)+rng.uniform(-8,8),0,255)
            batch.append(patch)
        x=torch.from_numpy(np.stack(batch).transpose(0,3,1,2)/255).float();y=torch.tensor(chosen)
        model.train();optimizer.zero_grad(set_to_none=True);logits=model(x)
        loss=nn.functional.cross_entropy(logits,y,label_smoothing=.02)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();optimizer.step();scheduler.step()
        if step%50==0 or step+1==a.steps:
            row=dict(step=step+1,steps=a.steps,seconds=time.time()-started,loss=float(loss.detach()),batch_accuracy=float((logits.argmax(1)==y).float().mean()))
            (a.output/'progress.json').write_text(json.dumps(row));print(json.dumps(row),flush=True)
    checkpoint=a.output/'last.pt'
    torch.save(dict(state_dict=model.state_dict(),classes=manifest['classes'],seed=1731,initialization='random',steps=a.steps,
                    data_manifest_sha256=sha(a.data/'manifest.json'),code_sha256=sha(__file__)),checkpoint)
    (a.output/'complete.json').write_text(json.dumps(dict(seconds=time.time()-started,checkpoint_sha256=sha(checkpoint))))


def rerank(a):
    checkpoint=torch.load(a.weights,map_location='cpu',weights_only=True);classes=checkpoint['classes']
    model=PatchCNN(len(classes));model.load_state_dict(checkpoint['state_dict']);model.eval()
    report=json.loads(a.report.read_text())
    if report['zoom']!=2:raise ValueError('Full-source crop verifier diagnostics require native-resolution proposals')
    fixture=json.loads((a.fixture/'fixture.json').read_text())
    lookup={(r['split'],r['frame']):r for r in fixture['examples']}
    for row in report['results']:
        src=lookup[(row['split'],row['frame'])];path=a.fixture/src['file']
        if sha(path)!=src['sha256']:raise ValueError('Fixture changed')
        image=cv2.imread(str(path));preds=row['predictions'];output=[]
        for first in range(0,len(preds),64):
            batch=preds[first:first+64];crops=np.stack([normalize_crop(image,r['bbox']) for r in batch])
            with torch.inference_mode():probabilities=model(torch.from_numpy(crops.transpose(0,3,1,2).copy()).float()/255).softmax(1).numpy()
            for p,probs in zip(batch,probabilities):
                # Preserve the proposal's class and score. Verifier probabilities
                # and alternate class are separate; thresholding remains auditable.
                output.append({**p,'verifier_probability':float(probs[classes.index(p['class'])]),
                               'background_probability':float(probs[0]),'verifier_class':classes[int(probs.argmax())],
                               'verifier_top_probability':float(probs.max())})
        row['predictions']=output
    report.update(verifier_sha256=sha(a.weights),raw_report_sha256=sha(a.report),verifier_policy='Raw detector scores/classes preserved; independent verifier probabilities attached.')
    with a.output.open('x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','train','rerank'])
    p.add_argument('--data',type=Path);p.add_argument('--weights',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=800);p.add_argument('--report',type=Path);p.add_argument('--fixture',type=Path)
    a=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1);globals()[a.mode](a)
