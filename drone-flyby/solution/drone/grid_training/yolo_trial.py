"""YOLO comparison restricted to complete training tiles; partial dev recall only."""
import argparse,json,os,shutil,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
from .train import sha,write

def iou(a,b):
 x,y,u,v=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]);inter=max(0,u-x)*max(0,v-y);return inter/max(1e-6,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--epochs',type=int,default=40);a=p.parse_args();root=Path(os.environ['NORDIC_RUN_DIR']);m=json.loads((a.data/'manifest.json').read_text());training=[r for r in m['records'] if r['split']=='train' and r['annotation_complete']];dev=[r for r in m['records'] if r['split']=='dev'];side=training[0]['input_size']
 dataset=root/'dataset';images=dataset/'images/train';labels=dataset/'labels/train';images.mkdir(parents=True);labels.mkdir(parents=True)
 for r in training:
  src=a.data/r['file'];assert sha(src)==r['sha256'];os.link(src,images/(r['id']+'.png'))
  lines=[]
  for ann in r['annotations']:
   x,y,u,v=ann['bbox_xyxy'];lines.append(f'{ann["class_id"]} {(x+u)/2/side} {(y+v)/2/side} {(u-x)/side} {(v-y)/side}')
  (labels/(r['id']+'.txt')).write_text('\n'.join(lines))
 import yaml
 (dataset/'data.yaml').write_text(yaml.safe_dump(dict(path=str(dataset.resolve()),train='images/train',val='images/train',names=m['classes'])))
 write(root/'contract.json',dict(training_images=len(training),excluded_partial_training_images=sum(r['split']=='train' and not r['annotation_complete'] for r in m['records']),manifest_sha256=sha(a.data/'manifest.json'),note='Trainer validation uses training tiles only and is not reported as generalization. Fixed last checkpoint; true dev known-positive recall and reviewed-background false positives below. Standard YOLO sampler, not class-balanced classifier sampler.'))
 model=YOLO('yolo26s.pt');model.train(data=str(dataset/'data.yaml'),epochs=a.epochs,imgsz=side,batch=32,device=0,workers=4,project=str(root),name='fit',exist_ok=False,pretrained=True,optimizer='AdamW',lr0=.001,patience=0,val=False,plots=False,save=True,seed=1731,mosaic=0,mixup=0,translate=.1,scale=.15,degrees=15)
 model=YOLO(str(root/'fit/weights/last.pt'));metrics=defaultdict(list);background=[];predictions=[]
 for r in dev:
  result=model.predict(str(a.data/r['file']),imgsz=side,conf=.25,verbose=False,device=0)[0]
  preds=[dict(class_id=int(c),box=b,score=float(s)) for b,c,s in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist())];used=set()
  if r['kind']=='background':background.append(float(bool(preds)))
  for ann in r['annotations']:
   if not ann['fully_contained']:continue
   eligible=[(iou(ann['bbox_xyxy'],v['box']),j) for j,v in enumerate(preds) if j not in used and v['class_id']==ann['class_id']];best=max(eligible,default=(0,-1));ok=best[0]>=.5
   if ok:used.add(best[1])
   metrics[ann['class_name'],r['zoom'],ann['group']].append(float(ok))
  predictions.append(dict(id=r['id'],kind=r['kind'],predictions=preds))
 per={}
 for c,z,g in metrics:
  per[f'{c}/L{z}']=float(np.mean([np.mean(v) for (cc,zz,gg),v in metrics.items() if cc==c and zz==z]))
 write(root/'dev-predictions.json',predictions);write(root/'complete.json',dict(known_positive_class_and_iou50_recall=float(np.mean(list(per.values()))),per_class_zoom=per,background_false_positive_rate=float(np.mean(background)),checkpoint_sha256=sha(root/'fit/weights/last.pt'),limitations='Partial development labels: unmatched positive-scene predictions cannot be scored as false positives. Different supervision from CNNs; not an architecture-only comparison.',completed=time.time()))
if __name__=='__main__':main()
