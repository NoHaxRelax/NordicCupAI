"""Bounded local feature-search pilot for planning; no API calls.

Reference holdouts share physical objects and backgrounds with the templates.
Their recall is a development measurement, not cross-scene validation recall.
"""
import json
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'data/drone/reference/helsinki'
OUT=ROOT/'artifacts/drone-api-tests/planning-pilot'

def iou(a,b):
    z=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return z/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z)

def main():
    cv2.setNumThreads(4);cv2.setRNGSeed(0)
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    sift=cv2.SIFT_create(nfeatures=16000,contrastThreshold=.015,edgeThreshold=15)
    documents={int(p.stem.split('_')[-1]):json.loads(p.read_text()) for p in (REF/'annotations').glob('*.json')}
    holdouts={5,12,20}; byclass=defaultdict(list)
    for f,d in sorted(documents.items()):
        if f in holdouts:continue
        for a in d['annotations']:
            b=a['bbox']
            if min(b[0],b[1])<=0 or b[2]>=3840 or b[3]>=2160:continue
            byclass[a['object_id']].append((f,b))
    bank=[];cache={}
    for label, examples in sorted(byclass.items()):
        selected=[examples[i] for i in sorted(set([0,len(examples)//2,len(examples)-1]))]
        for f,b in selected:
            if f not in cache:cache[f]=cv2.imread(str(REF/'images'/f'frame_{f:06d}.png'),0)
            x1,y1,x2,y2=map(int,b);patch=cache[f][y1:y2,x1:x2]
            kp,ds=sift.detectAndCompute(patch,None)
            if ds is not None and len(kp)>=3:
                bank.append((label,f,(x2-x1,y2-y1),np.float32([k.pt for k in kp]),ds))
    bankinfo={'templates':len(bank),'classes':{k:sum(b[0]==k for b in bank) for k in sorted(byclass)},'seconds':time.monotonic()-start}
    print(json.dumps({'bank':bankinfo}),flush=True)
    records=[]
    for split,frames in [('reference',sorted(holdouts)),('validation',[45,66,100,127,188,249])]:
        for f in frames:
            t=time.monotonic()
            path=REF/'images'/f'frame_{f:06d}.png' if split=='reference' else ROOT/'data/drone/reconstructed-validation'/f'frame_{f:06d}.png'
            image=cv2.imread(str(path),0);keypoints,descs=sift.detectAndCompute(image,None)
            pts=np.float32([k.pt for k in keypoints]);matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=48))
            matcher.add([descs]);matcher.train();proposals=[]
            for label,ref,wh,src,ds in bank:
                pairs=matcher.knnMatch(ds,k=2)
                good=[a for a,b in pairs if a.distance < .72*b.distance]
                if len(good)<3:continue
                a=np.float32([src[m.queryIdx] for m in good]);b=np.float32([pts[m.trainIdx] for m in good])
                matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=3000)
                if matrix is None:continue
                n=int(inliers.sum());fraction=float(inliers.mean());scale=float(np.linalg.norm(matrix[:,0]))
                if n<3 or fraction<.5 or not .4<scale<2.5:continue
                w,h=wh; corners=np.float32([[[0,0],[w,0],[w,h],[0,h]]]); projected=cv2.transform(corners,matrix)[0]
                lo=projected.min(0);hi=projected.max(0);box=[max(0,float(lo[0])),max(0,float(lo[1])),min(3840,float(hi[0])),min(2160,float(hi[1]))]
                if not box[0]<box[2] or not box[1]<box[3]:continue
                proposals.append({'frame':f,'class':label,'bbox_source_xyxy':box,'inliers':n,'fraction':fraction,'template_frame':ref,'unverified':True})
            proposals.sort(key=lambda a:a['inliers']*a['fraction'],reverse=True)
            unique=[]
            for p in proposals:
                if not any(p['class']==q['class'] and iou(p['bbox_source_xyxy'],q['bbox_source_xyxy'])>.5 for q in unique):unique.append(p)
            row={'split':split,'frame':f,'seconds':time.monotonic()-t,'keypoints':len(keypoints),'proposals':unique}
            if split=='reference':
                truth=documents[f]['annotations']
                row['truth_count']=len(truth)
                row['matched_classes']=[a['object_id'] for a in truth if any(p['class']==a['object_id'] and iou(p['bbox_source_xyxy'],a['bbox'])>=.5 for p in unique)]
                row['missed_classes']=[a['object_id'] for a in truth if a['object_id'] not in row['matched_classes']]
            records.append(row)
            print(json.dumps({k:v if k!='proposals' else len(v) for k,v in row.items()}),flush=True)
            (OUT/'feature-search.json').write_text(json.dumps({'bank':bankinfo,'results':records,'total_seconds':time.monotonic()-start,'live_queries':0},indent=2)+'\n')

if __name__=='__main__':main()
