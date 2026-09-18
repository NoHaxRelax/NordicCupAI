"""Blind detection and optional crop re-ranking on complete reviewed dev views.

This is a per-view diagnostic with overlapping views, not competition scoring.
It never centers search on labels. Saves all predictions for later fusion.
"""
import argparse
import json
import math
from pathlib import Path
import time

from common import read, sha, verify_dataset, write


def iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union > 0 else 0



def tile_starts(length, size):
    return sorted(set(range(0,max(1,length-size+1),max(1,size//2))) | {max(0,length-size)})


def merge_predictions(predictions, threshold=.5, limit=300):
    selected=[]
    for pred in sorted(predictions,key=lambda x:-x['score']):
        if any(pred['class']==other['class'] and iou(pred['bbox'],other['bbox'])>threshold for other in selected):continue
        selected.append(pred)
        if len(selected)>=limit:break
    return selected


def predict_view(model, image, device, tiled=False):
    # Search geometry depends only on image dimensions, never on target labels.
    w,h=image.size;windows=[(0,0,w,h)]
    if tiled:
        tw,th=min(w,480),min(h,270)
        windows += [(x,y,x+tw,y+th) for x in tile_starts(w,tw) for y in tile_starts(h,th)
                    if (x,y,x+tw,y+th)!=(0,0,w,h)]
    predictions=[]
    for x,y,x2,y2 in windows:
        out=model.predict(image.crop((x,y,x2,y2)),imgsz=960,conf=.001,iou=.5,device=device,verbose=False)[0]
        for box,cls,score in zip(out.boxes.xyxy.cpu().tolist(),out.boxes.cls.cpu().tolist(),out.boxes.conf.cpu().tolist()):
            a,b,c,d=map(float,box)
            # Discard clipped tile-edge proposals; full-view proposals are retained.
            if (x>0 and a<=2) or (y>0 and b<=2) or (x2<w and c>=x2-x-2) or (y2<h and d>=y2-y-2):continue
            predictions.append({'class':model.names[int(cls)],'score':float(score),'bbox':[a+x,b+y,c+x,d+y]})
    return merge_predictions(predictions) if tiled else predictions


def classify_proposal(pred, prob, classes, foreground=False):
    if foreground:
        index=max(range(len(prob)),key=prob.__getitem__)
        if index==len(classes):return None
        label=classes[index]
    else:
        label=pred['class'];index=classes.index(label)
    return dict(pred, **{'class':label, 'score':math.sqrt(pred['score']*prob[index]),
                         'detector_score':pred['score'],'background_probability':prob[-1]})


def metrics(rows, classes, threshold, class_agnostic=False):
    tp=fp=targets=0
    counts={}
    for row in rows:
        used=set();targets+=len(row['truth'])
        for t in row['truth']:
            counts.setdefault(t['class'],dict(targets=0,matched=0))['targets']+=1
        for p in sorted(row['predictions'],key=lambda x:-x['score']):
            if p['score']<threshold:continue
            candidates=[(iou(p['bbox'],t['bbox']),i) for i,t in enumerate(row['truth'])
                        if i not in used and (class_agnostic or p['class']==t['class'])]
            overlap,index=max(candidates,default=(0,-1))
            if overlap>=.5:
                used.add(index);tp+=1;counts[row['truth'][index]['class']]['matched']+=1
            else:fp+=1
    return dict(threshold=threshold,targets=targets,tp=tp,fp=fp,
                precision=tp/(tp+fp) if tp+fp else 0,recall=tp/targets if targets else None,
                per_class=counts,missing_classes=[c for c in classes if c not in counts])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--detector',type=Path,required=True)
    p.add_argument('--classifier',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--native-tiles',action='store_true',help='Full view plus overlapping 480x270 tiles on zoom 2 only')
    p.add_argument('--foreground',action='store_true',help='Binary object proposals, scored against original multiclass labels')
    args=p.parse_args()
    if args.foreground and not args.classifier:raise SystemExit('Foreground evaluation requires a classifier')
    if args.output.exists():raise SystemExit('Use a new output directory')
    manifest=read(args.data/'manifest.json');verify_dataset(args.data,manifest)
    import torch
    torch.set_num_threads(4)
    from ultralytics import YOLO
    from PIL import Image,ImageOps
    from torchvision.transforms import functional as F
    model=YOLO(str(args.detector));recognizer=None
    expected=['object'] if args.foreground else manifest['classes']
    if [model.names[i] for i in range(len(model.names))]!=expected:raise ValueError('Detector class mapping differs')
    device=torch.device('cpu' if args.device=='cpu' else 'cuda:'+args.device)
    if args.classifier:
        from torchvision.models import resnet50
        saved=torch.load(args.classifier,map_location='cpu',weights_only=False)
        classifier_size=int(saved.get('spec',{}).get('input_size',224))
        if saved['classes']!=manifest['classifier_classes']:raise ValueError('Class ordering differs')
        if saved.get('spec',{}).get('architecture','resnet50') in ['convnext_tiny','convnext_small']:
            from torchvision.models import convnext_tiny,convnext_small
            recognizer={'convnext_tiny':convnext_tiny,'convnext_small':convnext_small}[saved['spec']['architecture']](weights=None)
            recognizer.classifier[2]=torch.nn.Linear(recognizer.classifier[2].in_features,len(saved['classes']))
        else:
            recognizer=resnet50(weights=None)
            recognizer.fc=torch.nn.Linear(2048,len(saved['classes']))
        recognizer.load_state_dict(saved['model']);recognizer.to(device).eval()
    reports={'detector':[]}
    if recognizer:reports['detector_classifier']=[]
    started=time.monotonic()
    for item in manifest['records']:
        if item['task']!='detector' or item['split']!='dev':continue
        path=args.data/item['file'];im=Image.open(path).convert('RGB');w,h=im.size
        truth=[]
        for line in (args.data/item['label_file']).read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split())
            truth.append(dict(class_name=manifest['classes'][int(c)],bbox=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
        truth=[{'class':t['class_name'],'bbox':t['bbox']} for t in truth]
        preds=predict_view(model,im,args.device,args.native_tiles and item['zoom']==2)
        base=dict(file=item['file'],zoom=item['zoom'],frame=item['frame'],source_region=item['source_region'],truth=truth)
        reports['detector'].append(dict(base,predictions=preds))
        if recognizer:
            rescored=[]
            for offset in range(0,len(preds),64):
                group=preds[offset:offset+64];batch=[]
                for pred in group:
                    x1,y1,x2,y2=pred['bbox'];margin=max(2,.12*max(x2-x1,y2-y1))
                    crop=im.crop((max(0,math.floor(x1-margin)),max(0,math.floor(y1-margin)),min(w,math.ceil(x2+margin)),min(h,math.ceil(y2+margin))))
                    crop=ImageOps.pad(crop,(classifier_size,classifier_size),method=Image.Resampling.BILINEAR,color=(114,114,114))
                    batch.append(F.normalize(F.to_tensor(crop),[.485,.456,.406],[.229,.224,.225]))
                with torch.inference_mode():probs=recognizer(torch.stack(batch).to(device)).softmax(1).cpu().tolist()
                for pred,prob in zip(group,probs):
                    scored=classify_proposal(pred,prob,manifest['classes'],args.foreground)
                    if scored is not None:rescored.append(scored)
            reports['detector_classifier'].append(dict(base,predictions=rescored))
    args.output.mkdir(parents=True)
    for name,rows in reports.items():
        write(args.output/(name+'.json'),dict(manifest_sha256=sha(args.data/'manifest.json'),results=rows))
    summary=dict(native_tiles=args.native_tiles,foreground_proposals=args.foreground,detector_sha256=sha(args.detector),classifier_sha256=sha(args.classifier) if args.classifier else None,
                 elapsed_seconds=time.monotonic()-started,metrics={})
    for name,rows in reports.items():
        summary['metrics'][name]={f'L{z}':[metrics([r for r in rows if r['zoom']==z],manifest['classes'],t)
                                            for t in [.01,.05,.1,.25,.5,.75]] for z in range(3)}
        summary['metrics'][name+'_class_agnostic']={f'L{z}':[metrics([r for r in rows if r['zoom']==z],manifest['classes'],t,True)
                                            for t in [.01,.05,.1,.25,.5,.75]] for z in range(3)}
    summary['limitations']=manifest['limitations']+['Metrics count complete reviewed views, including overlapping views of the same object. Thresholds and fusion are development diagnostics.']
    write(args.output/'summary.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':main()
