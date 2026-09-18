"""Test camera-motion prediction plus local appearance matching on reference."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from pilot_feature_search import iou

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'data/drone/reference/helsinki'
OUT=ROOT/'artifacts/drone-api-tests/planning-pilot'

def main():
    cv2.setNumThreads(4);cv2.setRNGSeed(0);start=time.monotonic()
    docs={int(p.stem.split('_')[-1]):json.loads(p.read_text()) for p in (REF/'annotations').glob('*.json')}
    images={f:cv2.imread(str(REF/'images'/f'frame_{f:06d}.png'),0) for f in sorted(docs)}
    thumbs={f:cv2.resize(im,(480,270)).astype(np.float32) for f,im in images.items()}
    edges={f:cv2.phaseCorrelate(thumbs[f],thumbs[f+1]) for f in range(24)}
    results=[]
    for label in ['helicopter','small_launcher','large_launcher','tank','ta-ta','jammer']:
        t=time.monotonic();truth={f:a['bbox'] for f,d in docs.items() for a in d['annotations'] if a['object_id']==label}
        seed=sorted(truth)[len(truth)//2];b=truth[seed];predictions={seed:b};scores={}
        for direction,limit in [(-1,0),(1,24)]:
            prev=seed;box=list(b)
            for f in range(seed+direction,limit+direction,direction):
                edge=prev if direction==1 else f
                (dx,dy),response=edges[edge]
                if response<.08:break
                dx*=8*direction;dy*=8*direction
                x1,y1,x2,y2=map(round,box);patch=images[prev][y1:y2,x1:x2]
                if min(patch.shape)<5:break
                px=x1+dx;py=y1+dy;w=x2-x1;h=y2-y1;pad=28
                sx=max(0,round(px-pad));sy=max(0,round(py-pad));ex=min(3840,round(px+w+pad));ey=min(2160,round(py+h+pad))
                region=images[f][sy:ey,sx:ex]
                if region.shape[0]<h or region.shape[1]<w:break
                corr=cv2.matchTemplate(region,patch,cv2.TM_CCOEFF_NORMED)
                _,score,_,loc=cv2.minMaxLoc(corr)
                if score<.35:break
                box=[sx+loc[0],sy+loc[1],sx+loc[0]+w,sy+loc[1]+h]
                predictions[f]=box;scores[f]=score;prev=f
        measured=[{'frame':f,'iou':iou(box,predictions[f]) if f in predictions else 0,'correlation':scores.get(f)} for f,box in sorted(truth.items()) if f!=seed]
        row={'class':label,'seconds':time.monotonic()-t,'seed_frame':seed,'tested_nonseed_frames':len(measured),'iou50_passed':sum(x['iou']>=.5 for x in measured),'per_frame':measured}
        results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='per_frame'}),flush=True)
    result={'method':'global phase correlation at 1/8 resolution; local normalized template match within 28 source pixels of predicted position','results':results,'total_seconds':time.monotonic()-start,'live_queries':0,'limitation':'Exact ground-truth seeds, fixed dimensions, public reference scene only; no validation completeness claim.'}
    (OUT/'motion-tracking.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
