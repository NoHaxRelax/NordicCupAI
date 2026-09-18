"""Blind legal-view CNN search on the identical template development fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def nms(rows,threshold=.35):
    keep=[]
    for row in sorted(rows,key=lambda r:-r['score']):
        if not any(row['class']==k['class'] and iou(row['bbox'],k['bbox'])>threshold for k in keep):keep.append(row)
    return keep[:300]


def measure(rows,truth,class_aware=True):
    used=set()
    matches=[]
    for p in sorted(rows,key=lambda r:-r['score']):
        score,index=max(((iou(p['bbox'],t['bbox']),i) for i,t in enumerate(truth) if i not in used and (not class_aware or p['class']==t['class'])),default=(0,-1))
        if score>=.5:used.add(index);matches.append(dict(target=index,iou=score,score=p['score'],predicted_class=p['class']))
    return dict(targets=len(truth),matched=len(used),proposals=len(rows),unmatched_proposals=len(rows)-len(used),matches=matches)


def starts(length,span):return sorted(set(range(0,length-span+1,span*3//4))|{length-span})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--zoom',type=int,choices=[0,1,2],default=2)
    a=p.parse_args()
    import torch
    torch.set_num_threads(a.threads);cv2.setNumThreads(2)
    from ultralytics import YOLO
    model=YOLO(str(a.weights))
    fixture=json.loads((a.fixture/'fixture.json').read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),zoom=a.zoom,
                fixture_sha256=hashlib.sha256((a.fixture/'fixture.json').read_bytes()).hexdigest(),results=[],
                limitations=['Incomplete manual validation labels: unmatched proposals are not necessarily false positives.',
                             'Reference holdout is the same physical objects/backgrounds, not independent asset generalization.',
                             'Late frames 181-249 not accessed by this experiment.'])
    for example in fixture['examples']:
        image_path=a.fixture/example['file']
        if hashlib.sha256(image_path.read_bytes()).hexdigest()!=example['sha256']:raise ValueError('Fixture hash mismatch')
        image=cv2.imread(str(image_path),cv2.IMREAD_UNCHANGED)
        if image.shape[2]==4 and np.any(image[:,:,3]!=255):raise ValueError('Unobserved image pixels')
        source_w,source_h=3840//2**a.zoom,2160//2**a.zoom
        locations=[(x,y) for y in starts(2160,source_h) for x in starts(3840,source_w)]
        views=[cv2.resize(image[y:y+source_h,x:x+source_w,:3],(960,540),interpolation=cv2.INTER_AREA) for x,y in locations]
        start=time.perf_counter();rows=[]
        for first in range(0,len(views),4):
            outputs=model.predict(views[first:first+4],imgsz=960,conf=.01,iou=.35,max_det=300,device=a.device,verbose=False)
            for result,(x,y) in zip(outputs,locations[first:first+4]):
                for b,c,s in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                    rows.append(dict(bbox=[x+b[0]*source_w/960,y+b[1]*source_h/540,x+b[2]*source_w/960,y+b[3]*source_h/540],
                                     **{'class':model.names[int(c)]},score=s))
        rows=nms(rows)
        row=dict(split=example['split'],frame=example['frame'],seconds=time.perf_counter()-start,predictions=rows,truth=example['truth'],thresholds={})
        for threshold in [.01,.05,.1,.25,.5,.75]:
            filtered=[p for p in rows if p['score']>=threshold]
            row['thresholds'][str(threshold)]=dict(class_aware=measure(filtered,example['truth']),class_agnostic=measure(filtered,example['truth'],False))
        report['results'].append(row)
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        overlay=cv2.resize(image[:,:,:3],(1920,1080))
        for target in example['truth']:
            x1,y1,x2,y2=[round(v/2) for v in target['bbox']];cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,180,0),1)
        for pred in rows:
            if pred['score']<.25:continue
            x1,y1,x2,y2=[round(v/2) for v in pred['bbox']];cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.putText(overlay,f"{pred['class']} {pred['score']:.2f}",(x1,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,255,0),1)
        cv2.imwrite(str(a.output/f"{example['split']}-{example['frame']:06d}.jpg"),overlay)
        print(json.dumps(dict(split=row['split'],frame=row['frame'],seconds=row['seconds'],at_025={kind:{k:v for k,v in score.items() if k!='matches'} for kind,score in row['thresholds']['0.25'].items()})),flush=True)


if __name__=='__main__':main()
