"""Independent LK landmark proxy benchmark, not organizer detection accuracy.
Run after root agent has created artifacts/drone-validation-landmarks/tracks.json.
Reference-only epipole/camera prior; odd-numbered validation seed IDs estimate
online background transforms, even IDs are held-out point trajectory targets.
"""
from collections import defaultdict
import json
import numpy as np
import cv2
from benchmark import ROOT,OUT,SCENE,project,epipole,average_h,stats


def predict(model,obs,f,hnom,epi,hbg,clock):
    times=np.array([x[0] for x in obs],float);p=np.array([x[1] for x in obs],float);end=int(times[-1]);horizon=f-end
    if model=='online_background':
        h=np.eye(3)
        for a in range(end,f):
            if a not in hbg:return None
            h=hbg[a]@h
        return project(p[-1:],h)[0]
    if model=='constant_velocity' and len(obs)<2:return p[-1].copy()
    if model=='nominal_projective' or len(obs)<2:return project(p[-1:],np.linalg.matrix_power(hnom,horizon))[0]
    if model=='rational_clock_ols':
        times=np.array([clock[int(t)] for t in times]);future=clock[f]
    else:future=f
    t=times-times[-1];dt=future-times[-1]
    if model=='constant_velocity':
        v=np.sum(t[:,None]*(p-p[-1]),axis=0)/np.sum(t*t);return p[-1]+v*dt
    v=p-epi;r=np.linalg.norm(v,axis=1);inv=1/r
    if model in ['rational_ols','rational_clock_ols']:
        if np.ptp(t)<.01:return project(p[-1:],np.linalg.matrix_power(hnom,horizon))[0]
        coef=np.polyfit(t,inv,1);denom=np.polyval(coef,dt);direction=np.mean(v/r[:,None],axis=0);direction/=np.linalg.norm(direction)
    else:
        slope=np.sum(t*(inv-inv[-1]))/np.sum(t*t);denom=inv[-1]+slope*dt;direction=v[-1]/r[-1]
    if denom<=0:return np.array([1e6,1e6])
    return epi+direction/denom


def summ(rows):
    if not rows:return None
    groups=defaultdict(list)
    for r in rows:groups[r['track_id']].append(r)
    return {'tracks':len(groups),'predictions':len(rows),'error_px':stats([r['error_px'] for r in rows]),
            'final_error_px':stats([rr[-1]['error_px'] for rr in groups.values()]),
            'worst_error_px':stats([max(r['error_px'] for r in rr) for rr in groups.values()]),
            'fraction_within_5px':float(np.mean([r['error_px']<=5 for r in rows])),
            'whole_tracked_remainder_within_5px_fraction':float(np.mean([max(r['error_px'] for r in rr)<=5 for rr in groups.values()])),
            'whole_tracked_remainder_within_10px_fraction':float(np.mean([max(r['error_px'] for r in rr)<=10 for rr in groups.values()])),
            'whole_tracked_remainder_within_20px_fraction':float(np.mean([max(r['error_px'] for r in rr)<=20 for rr in groups.values()])),
            'forecast_only_within_5px_fraction':float(np.mean([r['error_px']<=5 for r in rows if not r['refreshed']])) if any(not r['refreshed'] for r in rows) else None}


def main():
    raw=json.loads((ROOT/'artifacts/drone-validation-landmarks/tracks.json').read_text());tracks=[t for t in raw['tracks'] if len(t['frames'])>=15]
    background=[t for t in raw['tracks'] if int(t['id'].split('_')[-1])%2];targets=[t for t in tracks if not int(t['id'].split('_')[-1])%2]
    src=json.loads((SCENE/'measurements.json').read_text());prs=[r for r in src['pairs'] if r['sequence']=='reference' and r['to']==r['from']+1]
    hnom=average_h([r['matrices']['homography'] for r in prs]);epi=np.median([epipole(r['matrices']['homography']) for r in prs],axis=0)
    matches=defaultdict(list)
    for tr in background:
        for a,b,x,y in zip(tr['frames'][:-1],tr['frames'][1:],tr['xy'][:-1],tr['xy'][1:]):
            if b==a+1:matches[a].append((x,y))
    hbg={};hmeta=[];clock={f:float(f) for f in range(5,250)};offset=0
    probes=np.array([[x,y] for x in [500,1900,3300] for y in [300,1000,1800]],float)
    norm=np.median((project(probes,hnom)-probes)[:,1])
    for f in range(5,249):
        pairs=matches.get(f,[]);tick=1;ratio=None
        if len(pairs)>=8:
            x=np.array([p[0] for p in pairs]);y=np.array([p[1] for p in pairs]);cv2.setRNGSeed(17)
            h,inlier=cv2.findHomography(x,y,cv2.RANSAC,3.)
            if h is not None:
                hbg[f]=h;ratio=float(np.median((y-x)[:,1]/(project(x,hnom)-x)[:,1]))
                tick=0 if abs(ratio)<.25 else 2 if ratio>1.5 else 1
                hmeta.append({'from':f,'matches':len(pairs),'inliers':int(inlier.sum()),'motion_ratio':ratio,'motion_ticks':tick,'homography':h.tolist()})
        offset+=tick-1;clock[f+1]=f+1+offset
    rowsall=[];summaries=[];pertrack=[]
    for k in [1,2,3,4,5,8]:
        for model in ['constant_velocity','nominal_projective','rational_anchored','rational_ols','online_background','rational_clock_ols']:
            for refresh in ([0,3,5,8,10] if model in ['rational_ols','online_background','rational_clock_ols'] else [0]):
                rows=[]
                for tr in targets:
                    obs=list(zip(tr['frames'][:k],map(np.array,tr['xy'][:k])));anchor=obs[-1][0];lastrefresh=anchor;rr=[]
                    for f,truth in zip(tr['frames'][k:],tr['xy'][k:]):
                        refreshed=bool(refresh and f-lastrefresh>=refresh)
                        if refreshed:
                            obs.append((f,np.array(truth)));obs=obs[-k:];lastrefresh=f;pred=np.array(truth)
                        else:pred=predict(model,obs,f,hnom,epi,hbg,clock)
                        if pred is None:continue
                        rr.append({'track_id':tr['id'],'start_frame':tr['start_frame'],'frame':f,'horizon':f-anchor,'error_px':float(np.linalg.norm(pred-truth)),'prediction_xy':pred.tolist(),'truth_xy':truth,'refreshed':refreshed,'k':k,'model':model,'refresh_interval':refresh,'frozen_frame':f in [9,239]})
                    rows+=rr
                    pertrack.append({'track_id':tr['id'],'start_frame':tr['start_frame'],'k':k,'model':model,'refresh_interval':refresh,'summary':summ(rr),'n_future_available':len(tr['frames'])-k,'end_reason':tr['end_reason']})
                summaries.append({'k':k,'model':model,'refresh_interval':refresh,'summary':summ(rows),'by_start':{str(f):summ([r for r in rows if r['start_frame']==f]) for f in [5,60,120,210,225]},'nonfrozen_frames':summ([r for r in rows if not r['frozen_frame']]),'frozen_frames':summ([r for r in rows if r['frozen_frame']])})
                rowsall+=rows
    meta={'input':str(ROOT/'artifacts/drone-validation-landmarks/tracks.json'),'pseudo_reference':'Bidirectional checked local optical flow, independent of tested motion models. Source pixels. No category/box ground truth.','tracks_input':len(tracks),'background_fit_tracks':len(background),'heldout_target_tracks':len(targets),'split':'All available odd seed IDs fit online H and clock, without future lifetime filtering; even seed IDs with>=15 observations test, within each start group. Physical regions may remain correlated.','camera_prior_sequence':'reference','epipole':epi.tolist(),'nominal_homography':hnom.tolist(),'online_homographies':hmeta,'motion_clock':'At each available background pair, compare measured y motion to reference nominal at each matched point; take median ratio. Ratio<.25 is0ticks,>1.5 is2ticks,otherwise1. Current/past frames only; unavailable pairs default1.','limitations':['Tracked duration ends at image margin or tracking confidence failure, not guaranteed physical exit.','Selecting tracks>=15frames favors easy/stable landmarks; static background dominant; not target detector accuracy.','Multiple tracks share scene content and adjacent image pixels.','Online clock/homography consumes future background images causally at each evaluated frame.','k1 always needs reference camera prior except online_background; cannot learn velocity from one observation.','Refresh is oracle point observation and excludes detection noise.']}
    (OUT/'validation-benchmark.json').write_text(json.dumps({'metadata':meta,'summary':summaries,'per_track':pertrack},indent=2)+'\n')
    with (OUT/'validation-predictions.jsonl').open('w') as o:
        for r in rowsall:o.write(json.dumps(r)+'\n')
    for s in summaries:
        if not s['refresh_interval']:
            m=s['summary'];print(s['model'],s['k'],'n',m['tracks'],'median',round(m['error_px']['median'],2),'p90',round(m['error_px']['p90'],2),'whole5',round(m['whole_tracked_remainder_within_5px_fraction'],3),'whole20',round(m['whole_tracked_remainder_within_20px_fraction'],3))


if __name__=='__main__':main()
