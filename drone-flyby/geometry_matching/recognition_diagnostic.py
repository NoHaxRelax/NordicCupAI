"""Conditional recognition diagnostic: reviewed validation boxes supply object locations.
NOT an autonomous detector, and NOT a validation-accuracy claim.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,math
import cv2,numpy as np
from experiment import Camera,save

def normalized(crop,mask):
    valid=mask>0
    if valid.sum()<10:return None
    ys,xs=np.where(valid); crop=crop[ys.min():ys.max()+1,xs.min():xs.max()+1];mask=mask[ys.min():ys.max()+1,xs.min():xs.max()+1]
    scale=54/max(crop.shape[:2]);w=max(1,round(crop.shape[1]*scale));h=max(1,round(crop.shape[0]*scale))
    rgb=np.zeros((80,80,3),np.float32);alpha=np.zeros((80,80),np.float32);x=(80-w)//2;y=(80-h)//2
    a=cv2.resize(mask.astype(np.float32)/255,(w,h));im=cv2.resize(crop,(w,h)).astype(np.float32)/255
    # Normalize foreground luminance while retaining colour ratios and silhouette.
    mean=np.mean(im[a>.5]);im=np.clip(im/(mean+1e-5),0,3)/3
    rgb[y:y+h,x:x+w]=im*a[:,:,None];alpha[y:y+h,x:x+w]=a
    feat=np.concatenate([rgb,alpha[:,:,None]*.5],axis=2)
    return feat

def extract(data,frame,box,scene):
    if scene=='helsinki':path=data/f'src/helsinki/images/frame_{frame:06d}.png'
    else:path=data/f'validation/reconstructed-validation/frame_{frame:06d}.png'
    im=cv2.imread(str(path));x1,y1,x2,y2=np.round(box).astype(int)
    xa,ya=max(0,x1-7),max(0,y1-7);xb,yb=min(3840,x2+7),min(2160,y2+7);crop=im[ya:yb,xa:xb]
    gc=np.zeros(crop.shape[:2],np.uint8)
    try:cv2.grabCut(crop,gc,(x1-xa,y1-ya,x2-x1,y2-y1),np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_RECT)
    except cv2.error:return None
    mask=np.uint8(np.isin(gc,[1,3]))*255
    return normalized(crop,mask)

def main(args):
    cv2.setNumThreads(4);cv2.setRNGSeed(19);data,out=args.data,args.out
    bank=json.loads((data/'elias/sprites/bank.json').read_text())['sprites'];templates=[];names=[];sources=[]
    for s in bank:
        if s['source']!='helsinki':continue
        spr=cv2.imread(str(data/'elias/sprites'/s['file']),-1);f=normalized(spr[:,:,:3],spr[:,:,3])
        if f is not None:templates.append(f);names.append(s['class_name']);sources.append(dict(frame=s['frame'],file=s['file'],box=s['box_src'],name=s['class_name']))
    ref=defaultdict(list)
    for p in (data/'src/helsinki/annotations').glob('*.json'):
        j=json.loads(p.read_text())
        for a in j['annotations']:
            b=a['bbox']
            if min(b[:2])>2 and b[2]<3837 and b[3]<2157:ref[a['object_id']].append((j['frame'],b))
    for name,rows in ref.items():
        if name in names:continue
        for frame,b in sorted(rows)[::max(1,len(rows)//5)]:
            f=extract(data,frame,b,'helsinki')
            if f is not None:templates.append(f);names.append(name);sources.append(dict(frame=frame,box=b,name=name))
    variants=[];labels=[];baseids=[];angles=[]
    for ti,t in enumerate(templates):
        for angle in range(0,360,10):
            m=cv2.getRotationMatrix2D((39.5,39.5),angle,1)
            f=cv2.warpAffine(t,m,(80,80)).reshape(-1)
            variants.append(f/(np.linalg.norm(f)+1e-6));labels.append(names[ti]);baseids.append(ti);angles.append(angle)
    # Matrix-vector cosine matching is normalized for brightness and includes silhouette support.
    matrix=np.array(variants,np.float32);
    traincam=Camera(**json.loads((out/'camera_training.json').read_text())['parameters']);valcam=Camera(**json.loads((out/'camera_validation.json').read_text())['parameters'])
    def pixels_per_metre(cam,box):
        bb=np.array(box);g=cam.unproject((bb[:2]+bb[2:])/2,0)[0];return cam.f/((g-cam.centre(0))@cam.R.T)[2]
    source_wh=np.array([np.array(s['box'][2:])-s['box'][:2] for s in sources]);source_scale=np.array([pixels_per_metre(traincam,s['box']) for s in sources])
    ca=np.abs(np.cos(np.radians(angles)));sa=np.abs(np.sin(np.radians(angles)));bi=np.array(baseids);expected_wh=np.c_[ca*source_wh[bi,0]+sa*source_wh[bi,1],sa*source_wh[bi,0]+ca*source_wh[bi,1]]/source_scale[bi,None]
    classes=sorted(set(labels));classidx={n:np.where(np.array(labels)==n)[0] for n in classes}
    tracks=defaultdict(list)
    for p in sorted((data/'src/validation/annotations').glob('*.json')):
        j=json.loads(p.read_text())
        for a,prov in zip(j['annotations'],j['provenance']):
            b=a['bbox']
            if prov['review_status']=='directly_reviewed_track' and min(b[:2])>2 and b[2]<3837 and b[3]<2157:tracks[prov['track_id']].append((j['frame'],a))
    result=[];queryarrays=[]
    for track,rows in tracks.items():
        select=[rows[i] for i in np.unique(np.linspace(0,len(rows)-1,min(5,len(rows))).astype(int))];per=[]
        for frame,a in select:
            q=extract(data,frame,a['bbox'],'validation')
            if q is None:continue
            best=np.full(len(classes),-100.);winning={}
            query_wh=np.array(a['bbox'][2:])-a['bbox'][:2];pred_wh=expected_wh*pixels_per_metre(valcam,a['bbox']);size_penalty=.18*np.sum(np.log(query_wh/pred_wh)**2,axis=1)
            for dy,dx in [(0,0),(-3,0),(3,0),(0,-3),(0,3)]:
                f=cv2.warpAffine(q,np.float32([[1,0,dx],[0,1,dy]]),(80,80)).reshape(-1);f/=np.linalg.norm(f)+1e-6;scores=matrix@f-size_penalty
                for ci,n in enumerate(classes):
                    vi=int(classidx[n][np.argmax(scores[classidx[n]])])
                    if scores[vi]>best[ci]:best[ci]=scores[vi];winning[n]=dict(source=sources[baseids[vi]],angle=angles[vi])
            per.append(dict(frame=frame,box=a['bbox'],scores=dict(zip(classes,map(float,best))),best_match=winning[classes[int(np.argmax(best))]]))
        if not per:continue
        aggregate={n:float(np.median([p['scores'][n] for p in per])) for n in classes};ranking=sorted(aggregate,key=aggregate.get,reverse=True);truth=rows[0][1]['object_id']
        row=dict(track=track,reviewed_class=truth,predicted_class=ranking[0],correct=ranking[0]==truth,top3=ranking[:3],scores=aggregate,observations=per)
        result.append(row);print(track,truth,'=>',ranking[:3],flush=True)
    save(out/'recognition_diagnostic.json',dict(warning='Conditional on participant-reviewed validation boxes, with GrabCut masks, normalized appearance, and a camera-derived physical size penalty (coefficient 0.18, exploratory). This tests appearance retrieval given object locations; not detection. Full rotation optimized per frame, so not a rigid 3D yaw estimate. Participant labels can be wrong. No validation examples in reference bank.',reference_templates=len(templates),reference_classes=classes,tracks=len(result),top1_correct=sum(r['correct'] for r in result),top3_correct=sum(r['reviewed_class'] in r['top3'] for r in result),results=result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
