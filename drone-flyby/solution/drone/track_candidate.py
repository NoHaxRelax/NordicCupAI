#!/usr/bin/env python3
"""Propagate a score-confirmed identity geometrically; outputs are unverified proposals."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--anchor',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    anchor=json.loads(args.anchor.read_text())
    records={}
    for path in args.captures.rglob('*.json'):
        r=json.loads(path.read_text())
        if r['view']['resolution_level']==2:records[r['frame']]=(path,r)
    start=anchor['frame'];path,record=records[start]
    origin=np.array(record['view']['source_region_xyxy'][:2],dtype=float)
    x1,y1,x2,y2=anchor['bbox_source_xyxy']
    corners=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float32)-origin
    sift=cv2.SIFT_create(nfeatures=1800)
    cache={}
    def features(frame):
        if frame not in cache:
            p,r=records[frame]
            image=cv2.imread(str(p.parent/r['image_file']),cv2.IMREAD_GRAYSCALE)
            cache[frame]=sift.detectAndCompute(image,None)
        return cache[frame]
    proposals=[]
    for direction in (-1,1):
        frame=start;poly=corners.copy()
        while frame+direction in records:
            nxt=frame+direction
            if records[nxt][1]['view']['source_region_xyxy']!=record['view']['source_region_xyxy']:break
            ka,da=features(frame);kb,db=features(nxt)
            if da is None or db is None:break
            pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
            good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance]
            if len(good)<40:break
            a=np.float32([ka[m.queryIdx].pt for m in good]);b=np.float32([kb[m.trainIdx].pt for m in good])
            transform,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.0,maxIters=3000)
            if transform is None or int(inliers.sum())<30 or inliers.mean()<.5:break
            poly=cv2.transform(poly[None,:,:].astype(np.float32),transform)[0]
            low=poly.min(axis=0);high=poly.max(axis=0)
            if not (0<=low[0]<high[0]<=960 and 0<=low[1]<high[1]<=540):break
            box=[float(low[0]+origin[0]),float(low[1]+origin[1]),float(high[0]+origin[0]),float(high[1]+origin[1])]
            proposals.append({'frame':nxt,'object_id':anchor['object_id'],'bbox_source_xyxy':box,
                              'status':'geometric_track_proposal_not_score_verified','anchor_frame':start,
                              'alignment_inliers':int(inliers.sum()),'alignment_inlier_fraction':float(inliers.mean())})
            frame=nxt
    result={'anchor':str(args.anchor),'method':'SIFT background alignment and robust similarity transforms',
            'limitation':'Background motion need not exactly match the object; boxes and propagated identity require checking.',
            'annotations':sorted(proposals,key=lambda r:r['frame'])}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'proposed_frames':len(proposals),'frame_range':[min(r['frame'] for r in proposals),max(r['frame'] for r in proposals)] if proposals else []}))


if __name__=='__main__':main()
