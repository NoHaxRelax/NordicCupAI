"""Deterministic hypotheses and seed handoff; never claims exhaustive coverage."""
import math
import random
import bisect
from collections import defaultdict
from pathlib import Path
from common import ROOT, DATA, W, H, atomic, read, hypothesis, iou

def existing():
    out=[]
    anchor=read(ROOT/'data/drone/mined/candidate140.json',{})
    if anchor:
        out.append({'seed_id':'prior-launcher-140','frame':140,'class':anchor['object_id'],
            'bbox_source_xyxy':anchor['bbox_source_xyxy'],'evidence':'score_confirmed',
            'source':'data/drone/mined/candidate140.json'})
    for a in read(ROOT/'data/drone/mined/large-launcher-track-proposals.json',{}).get('annotations',[]):
        if a.get('score_verification'):
            out.append({'seed_id':f'prior-launcher-{a["frame"]}','frame':a['frame'],
                        'class':'large_launcher','bbox_source_xyxy':a['bbox_source_xyxy'],
                        'evidence':'score_confirmed','source':'data/drone/mined/large-launcher-track-proposals.json'})
    a=read(ROOT/'data/drone/mined/planning-helicopter66.json',{})
    if a:
        out.append({'seed_id':'prior-helicopter-66','frame':66,'class':a['object_id'],
                    'bbox_source_xyxy':a['bbox_source_xyxy'],'evidence':'score_confirmed',
                    'source':'data/drone/mined/planning-helicopter66.json'})
    for a in out:
        b=a['bbox_source_xyxy'];a['bbox_normalized_xyxy']=[b[0]/W,b[1]/H,b[2]/W,b[3]/H]
        a['box_certainty']='IoU >= 0.50; not exact organizer ground truth'
    return out

def build():
    priors=read(ROOT/'artifacts/drone-api-tests/planning-pilot/reference-size-priors.json')['classes']
    known=existing();roots=[]
    def add(root,items):
        unique={str(x):x for x in items if x is not None}
        items=list(unique.values())
        if items:
            atomic(DATA/'groups'/f'{root["id"]}.json',items)
            roots.append({**root,'file':f'groups/{root["id"]}.json','hypothesis_count':len(items)})
    visual=[]
    for n,c in enumerate(read(ROOT/'data/drone/mined/object-presence-pilot.json',{}).get('candidates',[])):
        f=c['frame'];b=c['bbox_source_xyxy'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
        if any(a['frame']==f and iou(b,a['bbox_source_xyxy'])>.2 for a in known):continue
        items=[]
        for label,p in priors.items():
            for sx in (.55,.75,1,1.35,1.8):
                for sy in (.55,.75,1,1.35,1.8):
                    w=p['width_median']*sx;h=p['height_median']*sy
                    items.append(hypothesis(f,label,[cx-w/2,cy-h/2,cx+w/2,cy+h/2]))
        visual.append(({'id':f'visual-{n:02d}','kind':'visual_seed','label':f'Candidate in frame {f}','stop_after_first':True},items))
    # One bounded phase of a wider grid per class, spread over time. The first
    # complete frame uses the whole canvas; later frames use top + side entries.
    # 100 hypotheses/class/frame is a sample, not a full geometric certificate.
    entries=[]
    labels=sorted(priors, key=lambda c:(c not in ('helicopter','large_launcher','small_launcher'),c))
    for li,label in enumerate(labels):
        p=priors[label];items=[]
        lattices={}
        for first in (True,False):
            rows=[];ends=[];total=0
            for scale in (.8,1,1.25):
                for rotated in (False,True):
                    w=p['width_median']*scale;h=p['height_median']*scale
                    if rotated:w,h=h,w
                    xs=list(range(math.ceil(w/2),math.floor(W-w/2)+1,max(3,round(w/3))))
                    side=[x for x in xs if x<=160 or x>=W-160]
                    for y in range(math.ceil(h/2),math.floor(H-h/2)+1,max(3,round(h/3))):
                        columns=xs if first or y<=400 else side
                        if columns:
                            rows.append((columns,y,w,h));total+=len(columns);ends.append(total)
            lattices[first]=(rows,ends,total)
        for frame in range(5,250):
            rng=random.Random(90210+li*1000+frame)
            rows,ends,total=lattices[frame==5]
            # Deterministic randomized phases reduce repeated sampling of the
            # same lattice along a moving object's path without claiming recall.
            chosen=[]
            for index in rng.sample(range(total),min(100,total)):
                row=bisect.bisect_right(ends,index);xs,y,w,h=rows[row]
                chosen.append((xs[index-(ends[row-1] if row else 0)],y,w,h))
            for x,y,w,h in chosen:
                b=[x-w/2,y-h/2,x+w/2,y+h/2]
                if not any(a['frame']==frame and a['class']==label and iou(b,a['bbox_source_xyxy'])>=.5 for a in known):
                    items.append(hypothesis(frame,label,b))
        entries.append(({'id':f'entry-{label.replace("-","_")}','kind':'entry_scan','label':f'{label}: initial frame and entry bands','stop_after_first':False},items))
    # Interleave initial visual candidates with searches for new entries, so
    # neither expensive initial-frame work nor a single class monopolizes runs.
    for i in range(max(len(visual),len(entries))):
        if i<len(visual):add(*visual[i])
        if i<len(entries):add(*entries[i])
    atomic(DATA/'roots.json',roots)
    return roots,known
