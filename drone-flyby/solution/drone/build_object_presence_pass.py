#!/usr/bin/env python3
"""Build an offline, explicitly-unverified object-location pass for validation frames.

It uses ORB/RANSAC affine registration only to carry human-seeded boxes across
short visible spans.  It deliberately stops a track when the projection leaves
the canvas or a registration edge fails its quality gate; it never invents a
box for an unobserved frame.
"""
import argparse, csv, json
from pathlib import Path

import cv2
import numpy as np

W, H = 3840, 2160

def box_norm(b): return [b[0]/W,b[1]/H,b[2]/W,b[3]/H]
def valid(b):
    return 0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H and 20 <= (b[2]-b[0]) <= 700 and 15 <= (b[3]-b[1]) <= 700
def apply(m,b):
    p=np.array([[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]],np.float32)
    q=cv2.transform(p,m)[0]; lo=q.min(0); hi=q.max(0)
    return [float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--images',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--confirmed',type=Path,required=True);p.add_argument('--tracks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--csv',type=Path,required=True); a=p.parse_args()
 cv2.setRNGSeed(0)
 pilot=json.loads(a.pilot.read_text()); anchor=json.loads(a.confirmed.read_text()); scored=json.loads(a.tracks.read_text())['annotations']
 frames=list(range(1,250)); scale=.20; sift=cv2.ORB_create(nfeatures=900, fastThreshold=12)
 feats={}
 def feat(f):
  if f not in feats:
   im=cv2.imread(str(a.images/f'frame_{f:06d}.png'),cv2.IMREAD_GRAYSCALE)
   im=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
   feats[f]=sift.detectAndCompute(im,None)
  return feats[f]
 edges={}
 for f in range(1,249):
  ka,da=feat(f);kb,db=feat(f+1)
  if da is None or db is None: continue
  pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2); good=[x for z in pairs if len(z)==2 for x,y in [z] if x.distance < .72*y.distance]
  if len(good)<45: continue
  x=np.float32([ka[m.queryIdx].pt for m in good]);y=np.float32([kb[m.trainIdx].pt for m in good])
  m,ins=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=1.25,maxIters=3000)
  n=int(ins.sum()) if ins is not None else 0; frac=float(ins.mean()) if ins is not None else 0
  if m is not None and n>=40 and frac>=.48:
   m[:,2]/=scale; edges[f]=(m,n,frac)
 # One stable ID per human seed; score-confirmed launcher gets a durable ID.
 seeds=[]
 for i,c in enumerate(pilot['candidates'],1):
  seeds.append({'id':f'visual-{c["frame"]:03d}-{i:02d}','frame':c['frame'],'box':c['bbox_source_xyxy'],'confidence':c['confidence'],'reason':c['reason'],'kind':'visual_seed'})
 seeds.append({'id':'score-large-launcher-140','frame':anchor['frame'],'box':anchor['bbox_source_xyxy'],'confidence':'high','reason':'isolated score-confirmed large_launcher anchor','kind':'score_anchor'})
 anns=[]
 # Keep all seeds, then propagate each direction to its visible interval.
 for s in seeds:
  anns.append({'frame':s['frame'],'track_id':s['id'],'label':'large_launcher' if s['kind']=='score_anchor' else 'uncertain','bbox_source_xyxy':s['box'],'bbox_normalized_xyxy':box_norm(s['box']),'confidence':s['confidence'],'evidence':'score_confirmed' if s['kind']=='score_anchor' else 'manual_visual','review_status':'reviewed_seed','unverified':False if s['kind']=='score_anchor' else True,'provenance':s['reason']})
  for d in (-1,1):
   f=s['frame']; b=list(map(float,s['box'])); steps=0
   while 1 <= f+d <= 249 and steps < 80:
    edge=f if d==1 else f-1
    if edge not in edges: break
    m,n,frac=edges[edge]
    if d==-1: m=cv2.invertAffineTransform(m)
    b=apply(m,b); f+=d; steps+=1
    if not valid(b): break
    anns.append({'frame':f,'track_id':s['id'],'label':'large_launcher' if s['kind']=='score_anchor' else 'uncertain','bbox_source_xyxy':[round(x,2) for x in b],'bbox_normalized_xyxy':[round(x,8) for x in box_norm(b)],'confidence':'medium' if steps<=12 else 'low','evidence':'geometric_propagation','review_status':'propagated_not_individually_reviewed','unverified':True,'provenance':f'ORB/RANSAC adjacent-frame registration from seed frame {s["frame"]}; edge inliers={n}, fraction={frac:.3f}'})
 # Replace any propagated score-launcher 141..146 by the independently score-confirmed boxes.
 known={(x['frame'],'score-large-launcher-140'):x for x in scored}
 anns=[x for x in anns if (x['frame'],x['track_id']) not in known]
 for x in scored:
  b=x['bbox_source_xyxy']; anns.append({'frame':x['frame'],'track_id':'score-large-launcher-140','label':'large_launcher','bbox_source_xyxy':b,'bbox_normalized_xyxy':box_norm(b),'confidence':'high','evidence':'score_confirmed','review_status':'score_confirmed','unverified':False,'provenance':x['score_verification']['provenance']})
 anns.sort(key=lambda x:(x['frame'],x['track_id']))
 coverage=[]
 pilot_frames=set(pilot['reviewed_frames'])|{140,141,142,143,144,145,146}
 for f in frames:
  complete=f>=5
  coverage.append({'frame':f,'native_coverage':'complete' if complete else 'partial_transparent','inspection_status':'reviewed_keyframe' if f in pilot_frames else 'unreviewed','annotation_count':sum(x['frame']==f for x in anns),'note':'Only visible pixels may be assessed' if not complete else 'No assertion of complete object recovery.'})
 out={'purpose':'Offline high-recall object-presence candidate pass; not organizer ground truth','source_dimensions':[W,H],'sequence_frames':249,'method':{'seed_review':'five spaced native-resolution frames plus score-confirmed launcher frames','propagation':'ORB feature matching and RANSAC affine registration at one-fifth resolution, with 40-inlier/0.48-fraction gates and canvas/size stop gates','limitation':'Background registration can diverge from a moving foreground object. Propagated boxes are candidates and need visual review before any score probe or submission.'},'annotations':anns,'coverage':coverage,'summary':{'unique_tracks':len(seeds),'box_instances':len(anns),'score_confirmed_instances':sum(not x['unverified'] for x in anns),'unverified_instances':sum(x['unverified'] for x in anns),'reviewed_keyframes':sorted(pilot_frames),'inspected_empty_frames':[],'unreviewed_frames':[f for f in frames if f not in pilot_frames]}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
 fields=['frame','track_id','label','bbox_source_xyxy','bbox_normalized_xyxy','confidence','evidence','review_status','unverified','provenance']
 with a.csv.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
  for x in anns:w.writerow({k:json.dumps(x[k]) if isinstance(x[k],list) else x[k] for k in fields})
 print(json.dumps(out['summary'],indent=2))
if __name__=='__main__': main()
