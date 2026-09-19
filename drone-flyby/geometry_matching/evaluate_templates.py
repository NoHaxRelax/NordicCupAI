"""Evaluate only after inference; participant annotations are a partial audit reference."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from experiment import iou,save

def main(data,out):
    truth=[]
    for p in sorted((data/'src/validation/annotations').glob('*.json')):
        j=json.loads(p.read_text())
        for a,prov in zip(j['annotations'],j['provenance']):
            if prov['review_status']=='directly_reviewed_track':
                truth.append(dict(frame=j['frame'],name=a['object_id'],box=a['bbox'],track=prov['track_id']))
    ds=[]
    for prefix in ['fallback','dense']:
        for d in json.loads((out/f'{prefix}_verified.json').read_text()):
            d['id']=prefix+'-'+d['id'];ds.append(d)
    # One class per spatial object hypothesis: prioritize temporal image correlation.
    kept=[]
    for d in sorted(ds,key=lambda d:-d['median_ncc']):
        if any(np.linalg.norm(np.array(d['position_m'])-q['position_m'])<5 for q in kept):continue
        kept.append(d)
    preds=[dict(frame=f['frame'],name=d['name'],box=f['bbox'],score=d['median_ncc'],id=d['id']) for d in kept for f in d['frames']]
    used=set();tp=0;matched_tracks=set();audits=[];centres=set()
    for d in kept:
        matches=Counter();localized=Counter();maxiou=0
        for f in d['frames']:
            box=np.array(f['bbox']);c=(box[:2]+box[2:])/2
            for g in truth:
                if f['frame']!=g['frame']:continue
                overlap=iou(box,g['box']);maxiou=max(maxiou,overlap)
                if overlap>=.25:localized[g['track']]+=1
                if g['name']==d['name'] and g['box'][0]<=c[0]<=g['box'][2] and g['box'][1]<=c[1]<=g['box'][3]:matches[g['track']]+=1;centres.add(g['track'])
        audits.append(dict(id=d['id'],name=d['name'],median_ncc=d['median_ncc'],class_correct_center_tracks=dict(matches),localized_tracks_iou25=dict(localized),max_anyclass_iou=maxiou))
    for p in sorted(preds,key=lambda p:-p['score']):
        opts=[(iou(p['box'],g['box']),i) for i,g in enumerate(truth) if i not in used and g['frame']==p['frame'] and g['name']==p['name']]
        score,idx=max(opts,default=(0,-1))
        if score>=.5:tp+=1;used.add(idx);matched_tracks.add(truth[idx]['track'])
    perclass={}
    for name in sorted(set(g['name'] for g in truth)):
        gt={g['track'] for g in truth if g['name']==name}
        perclass[name]=dict(reviewed_tracks=len(gt),tracks_with_correct_class_center=len(gt&centres),tracks_with_tp50=len(gt&matched_tracks),candidate_instances=sum(d['name']==name for d in kept))
    save(out/'template_detections.json',kept)
    save(out/'template_evaluation.json',dict(reference='Partial participant directly-reviewed tracks, not organizer ground truth. No validation labels were read by model building or candidate inference. Thresholds exploratory and uncalibrated.',reviewed_tracks=len(set(g['track'] for g in truth)),reviewed_boxes=len(truth),candidate_instances=len(kept),candidate_boxes=len(preds),tp50=tp,unmatched_predictions=len(preds)-tp,fn50=len(truth)-tp,precision50=tp/max(1,len(preds)),recall50=tp/max(1,len(truth)),correct_class_center_tracks=len(centres),tracks_with_tp50=len(matched_tracks),perclass=perclass,audits=audits))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.data,a.out)
