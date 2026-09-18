"""Measure appearance tracking against labelled reference and one visual seed."""
import json
import time
from pathlib import Path
import cv2
from pilot_feature_search import iou

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'data/drone/reference/helsinki'
OUT=ROOT/'artifacts/drone-api-tests/planning-pilot'

def track(directory,seed,box,lo,hi):
    def image(f):return cv2.imread(str(directory/f'frame_{f:06d}.png'))
    output={seed:list(box)}
    for direction,limit in [(-1,lo),(1,hi)]:
        tracker=cv2.TrackerCSRT_create()
        tracker.init(image(seed),tuple(map(int,[box[0],box[1],box[2]-box[0],box[3]-box[1]])))
        for f in range(seed+direction,limit+direction,direction):
            success,b=tracker.update(image(f))
            if not success:break
            x,y,w,h=b
            b=[max(0,x),max(0,y),min(3840,x+w),min(2160,y+h)]
            if b[2]<=b[0] or b[3]<=b[1]:break
            output[f]=list(map(float,b))
    return output

def main():
    cv2.setNumThreads(4);cv2.setRNGSeed(0)
    docs={int(p.stem.split('_')[-1]):json.loads(p.read_text()) for p in (REF/'annotations').glob('*.json')}
    results=[]
    for label in ['helicopter','small_launcher','large_launcher']:
        truth={f:a['bbox'] for f,d in docs.items() for a in d['annotations'] if a['object_id']==label}
        seed=sorted(truth)[len(truth)//2]; t=time.monotonic()
        predictions=track(REF/'images',seed,truth[seed],0,24)
        measured=[{'frame':f,'iou':iou(b,predictions[f]) if f in predictions else 0} for f,b in sorted(truth.items()) if f!=seed]
        r={'split':'reference','class':label,'seed_frame':seed,'seconds':time.monotonic()-t,'tested_nonseed_frames':len(measured),
           'iou50_passed':sum(x['iou']>=.5 for x in measured),'per_frame':measured,
           'predictions':[{'frame':f,'bbox_source_xyxy':b} for f,b in sorted(predictions.items())]}
        results.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('per_frame','predictions')}),flush=True)
    t=time.monotonic();predictions=track(ROOT/'data/drone/reconstructed-validation',66,[2591,1455,2698,1544],45,100)
    r={'split':'validation','class_hypothesis':'helicopter','seed_frame':66,'seconds':time.monotonic()-t,'unverified':True,
       'predictions':[{'frame':f,'bbox_source_xyxy':b} for f,b in sorted(predictions.items())]}
    results.append(r);print(json.dumps({'split':'validation','seconds':r['seconds'],'tracked_frames':len(predictions)}),flush=True)
    (OUT/'appearance-tracking.json').write_text(json.dumps({'results':results,'live_queries':0,'limitations':'Reference tests use exact GT seeds; validation boxes have no score evidence. CSRT success is not proof of correct tracking.'},indent=2)+'\n')

if __name__=='__main__':main()
