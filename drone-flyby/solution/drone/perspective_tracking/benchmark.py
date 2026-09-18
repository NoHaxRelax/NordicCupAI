"""Causal tracking geometry benchmark. Offline; no detector/classification claims.

Run from repository root: python3 drone/perspective_tracking/benchmark.py
Inputs are existing official reference annotations and cached pixel matches.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/drone-perspective-tracking'
SCENE=ROOT/'artifacts/drone-scene-analysis'
REF=ROOT/'data/drone/reference/helsinki'
BOUNDS=np.array([3839.,2159.])
KS=[1,2,3,4,5,8]
REFRESH=[0,3,5,8,10]


def stats(values):
    a=np.asarray(values,float)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'p90':float(np.percentile(a,90)),'max':float(a.max())} if len(a) else None


def clip(b):
    b=np.array(b,float).copy();b[:2]=np.maximum(0,np.minimum(BOUNDS,b[:2]));b[2:]=np.maximum(0,np.minimum(BOUNDS,b[2:]));return b


def visible(b):return bool(np.all(clip(b)[2:]>clip(b)[:2]))
def unclipped(b):return bool(np.all(np.array(b)[:2]>0) and np.all(np.array(b)[2:]<BOUNDS))
def center(b):return (np.asarray(b)[:2]+np.asarray(b)[2:])/2

def iou(a,b):
    a=clip(a);b=clip(b);wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));inter=wh.prod();union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
    return float(inter/union) if union>0 else 0.


def project(p,h):
    p=np.asarray(p,float);h=np.asarray(h,float);q=np.c_[p,np.ones(len(p))]@h.T;return q[:,:2]/q[:,2,None]


def corners(b):
    x1,y1,x2,y2=b;return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])


def warp(b,h):
    p=project(corners(b),h);return np.r_[p.min(0),p.max(0)]


def normalize(h):return np.asarray(h,float)/np.linalg.det(h)**(1/3)
def average_h(hs):
    h=np.median([normalize(v) for v in hs],axis=0);return h/h[2,2]


def epipole(h):
    grid=np.array([[x,y] for x in [320,1920,3520] for y in [270,1080,1890]],float)
    flow=project(grid,h)-grid;n=np.c_[-flow[:,1],flow[:,0]];n/=np.linalg.norm(n,axis=1)[:,None]
    return np.linalg.lstsq(n,(n*grid).sum(1),rcond=None)[0]


def read_data():
    tracks=defaultdict(dict);byframe={}
    for path in sorted((REF/'annotations').glob('*.json')):
        j=json.loads(path.read_text());byframe[j['frame']]=j['annotations']
        for a in j['annotations']:tracks[a['object_id']][j['frame']]=np.array(a['bbox'],float)
    source=json.loads((SCENE/'measurements.json').read_text())
    validation=[r for r in source['pairs'] if r['sequence']=='validation' and r['to']==r['from']+1 and r['from'] not in [8,9]]
    hnom=average_h([r['matrices']['homography'] for r in validation])
    epi=np.median([epipole(r['matrices']['homography']) for r in validation],axis=0)
    # Refit pixel matching motion after excluding every known foreground target.
    # Cached train indices only; held-out pixels never enter these fits.
    hbg={};bgmeta=[]
    for f in range(24):
        npz=np.load(SCENE/f'matches-reference-{f:03d}-{f+1:03d}.npz')
        x=npz['x'][npz['train']];y=npz['y'][npz['train']];keep=np.ones(len(x),bool)
        for p,ann in [(x,byframe[f]),(y,byframe[f+1])]:
            for a in ann:
                b=np.array(a['bbox'])+[-20,-20,20,20]
                keep&=~((p[:,0]>=b[0])&(p[:,0]<=b[2])&(p[:,1]>=b[1])&(p[:,1]<=b[3]))
        cv2.setRNGSeed(17)
        h,inlier=cv2.findHomography(x[keep],y[keep],cv2.RANSAC,6.)
        hbg[f]=h;bgmeta.append({'from':f,'kept_background_matches':int(keep.sum()),'ransac_inliers':int(inlier.sum()),'homography':h.tolist()})
    return dict(tracks),hbg,hnom,epi,{'nominal_calibration_sequence':'validation','nominal_calibration_pairs':[[r['from'],r['to']] for r in validation],'nominal_homography':hnom.tolist(),'epipole':epi.tolist(),'reference_background_fits':bgmeta}


def compose(hbg,start,end):
    h=np.eye(3)
    for f in range(start,end):h=hbg[f]@h
    return h


def forecast(method,obs,frame,hbg,hnom,epi,bootstrap_h=None):
    times=np.array([v[0] for v in obs]);boxes=np.array([v[1] for v in obs]);t=times-times[-1];dt=frame-times[-1];last=boxes[-1]
    if method=='constant_velocity':
        if len(obs)<2:return last.copy()
        slope=np.sum(t[:,None]*(boxes-last),axis=0)/np.sum(t*t)
        return last+slope*dt
    if method in ['nominal_projective','bootstrap_background_projective','online_background']:
        if method=='online_background':h=compose(hbg,int(times[-1]),frame)
        else:
            prior=hnom
            if method=='bootstrap_background_projective':prior=bootstrap_h if bootstrap_h is not None else hnom
            h=np.linalg.matrix_power(prior,int(dt))
        return warp(last,h)
    if method.startswith('rational_center'):
        if len(obs)<2:return warp(last,np.linalg.matrix_power(hnom,int(dt)))
        centres=(boxes[:,:2]+boxes[:,2:])/2;v=centres-epi;r=np.linalg.norm(v,axis=1);inv=1/r
        slope=float(np.sum(t*(inv-inv[-1]))/np.sum(t*t));denom=inv[-1]+slope*dt
        direction=v[-1]/r[-1]
        if '_ols_' in method:
            coeff=np.polyfit(t,inv,1);denom=np.polyval(coeff,dt)
            direction=np.mean(v/r[:,None],axis=0);direction/=np.linalg.norm(direction)
        if denom<=0:return np.array([1e6,1e6,1e6+1,1e6+1])
        c=epi+direction/denom
        wh=last[2:]-last[:2]
        if method.endswith('linear_size'):
            sizes=boxes[:,2:]-boxes[:,:2];size_slope=np.sum(t[:,None]*(sizes-wh),axis=0)/np.sum(t*t);wh=np.maximum(1,wh+size_slope*dt)
        return np.r_[c-wh/2,c+wh/2]
    raise ValueError(method)


METHODS=['constant_velocity','nominal_projective','bootstrap_background_projective','rational_center_fixed_size','rational_center_linear_size','rational_center_ols_fixed_size','online_background']


def summarize(rows):
    if not rows:return None
    groups=defaultdict(list)
    for r in rows:groups[r['class']].append(r)
    all_forecast=[r for r in rows if not r.get('refreshed',False)]
    return {'tracks':len(groups),'boxes':len(rows),'iou':stats([r['iou'] for r in rows]),'center_error_px':stats([r['center_error_px'] for r in rows]),'fraction_iou_ge_0_5':float(np.mean([r['iou']>=.5 for r in rows])),
            'whole_remaining_track_successes':sum(all(r['iou']>=.5 for r in rr) for rr in groups.values()),
            'whole_remaining_track_success_fraction':float(np.mean([all(r['iou']>=.5 for r in rr) for rr in groups.values()])),
            'macro_track_fraction_iou_ge_0_5':float(np.mean([np.mean([r['iou']>=.5 for r in rr]) for rr in groups.values()])),
            'final_iou':stats([rr[-1]['iou'] for rr in groups.values()]),'final_center_error_px':stats([rr[-1]['center_error_px'] for rr in groups.values()]),
            'worst_track_center_error_px':stats([max(r['center_error_px'] for r in rr) for rr in groups.values()]),
            'forecast_only_boxes':len(all_forecast),'forecast_only_fraction_iou_ge_0_5':float(np.mean([r['iou']>=.5 for r in all_forecast])) if all_forecast else None,
            'clipped_boxes':sum(r['truth_clipped'] for r in rows),'clipped_fraction_iou_ge_0_5':float(np.mean([r['iou']>=.5 for r in rows if r['truth_clipped']])) if any(r['truth_clipped'] for r in rows) else None}


def run_track(track,k,method,hbg,hnom,epi,refresh=0,fixed_anchor=None,noise=0,rng=None,observation_indices=None):
    fs=sorted(track);full=[f for f in fs if unclipped(track[f])]
    if len(full)<k:return [],None
    init=full[:k]
    if fixed_anchor is not None:
        if len(full)<=fixed_anchor:return [],None
        init=full[fixed_anchor-k+1:fixed_anchor+1]
    if observation_indices is not None:
        if max(observation_indices)>=len(full):return [],None
        init=[full[i] for i in observation_indices]
    # Main experiment uses consecutive observations. Explicit spaced study below.
    if len(init)!=k or (observation_indices is None and init[-1]-init[0]!=k-1):return [],None
    anchor=init[-1];futures=[f for f in fs if f>anchor]
    if not futures:return [],None
    noise_by_frame={f:rng.uniform(-noise,noise,4) for f in fs} if noise else {}
    def measured(f):
        b=track[f].copy()
        if noise:b+=noise_by_frame[f]
        return b
    obs=[(f,measured(f)) for f in init];rows=[];lastrefresh=anchor;lastcheck=anchor
    bootstrap_h=average_h([hbg[f] for f in range(init[0],init[-1])]) if len(init)>1 else hnom
    for f in futures:
        requested=bool(refresh and f-lastcheck>=refresh)
        if requested:lastcheck=f
        # Censored rectangles do not reveal an object's full centre or dimensions.
        # Keep the existing forecast when a requested view touches an image edge.
        refreshed=bool(requested and unclipped(track[f]))
        if refreshed:
            obs.append((f,measured(f)));obs=obs[-k:];lastrefresh=f
            pred=obs[-1][1].copy()
        else:pred=forecast(method,obs,f,hbg,hnom,epi,bootstrap_h)
        truth=track[f];rows.append({'frame':f,'horizon':f-anchor,'since_refresh':f-lastrefresh,'iou':iou(pred,truth),'center_error_px':float(np.linalg.norm(center(clip(pred))-center(truth))),'truth_clipped':not unclipped(truth),'refresh_requested':requested,'refreshed':refreshed,'prediction_xyxy_unclipped':pred.tolist(),'truth_xyxy':truth.tolist()})
    departure=None
    if fs[-1]<24:
        # Official absence after the final label is all that is observable. Some
        # objects disappear with a few pixels still projected inside the image.
        # Compare first wholly geometrically-outside frame, not annotation threshold.
        expected=fs[-1]+1;predicted=None
        departure_obs=[(f,track[f]) for f in init]
        for f in range(anchor+1,85):
            if method=='online_background' and f>24:break
            p=forecast(method,departure_obs,f,hbg,hnom,epi,bootstrap_h)
            if not visible(p):predicted=f;break
        departure={'official_first_absent_frame':expected,'predicted_fully_outside_frame':predicted,'signed_frame_error':predicted-expected if predicted is not None else None}
    meta={'initial_frames':init,'anchor':anchor,'first_label':fs[0],'wait_to_full_box':init[0]-fs[0],'future_frames':len(futures),'last_label':fs[-1],'right_censored_at_sequence_end':fs[-1]==24,'departure':departure}
    return rows,meta


def main():
    OUT.mkdir(parents=True,exist_ok=True);tracks,hbg,hnom,epi,cal=read_data();allrows=[];summary=[];pertrack=[]
    common={name for name,t in tracks.items() if len([f for f in sorted(t) if unclipped(t[f])])>=5 and max(t)>[f for f in sorted(t) if unclipped(t[f])][4]}
    for fixed in [None,4]:
        for k in KS:
            if fixed is not None and k>fixed+1:continue
            for method in METHODS:
                for refresh in (REFRESH if fixed is None and method in ['nominal_projective','bootstrap_background_projective','rational_center_fixed_size','rational_center_linear_size','rational_center_ols_fixed_size','online_background'] else [0]):
                    rows=[]
                    for name,track in tracks.items():
                        rr,meta=run_track(track,k,method,hbg,hnom,epi,refresh,fixed)
                        if not rr:continue
                        for r in rr:r.update({'class':name,'k':k,'method':method,'refresh_interval':refresh,'anchor_mode':'earliest_full_box' if fixed is None else 'common_fifth_full_box'})
                        rows+=rr
                        pertrack.append({'class':name,'k':k,'method':method,'refresh_interval':refresh,'anchor_mode':rr[0]['anchor_mode'],'summary':summarize(rr),'track':meta})
                    if rows:
                        summary.append({'k':k,'method':method,'refresh_interval':refresh,'anchor_mode':rows[0]['anchor_mode'],'summary':summarize(rows),'common_eligible_track_summary':summarize([r for r in rows if r['class'] in common])})
                        allrows+=rows
    # Initialization uncertainty only; no artificial drift or later corrections.
    noise=[]
    for amp in [1,2,4]:
        for k in [1,2,3,4,5,8]:
            for method in ['nominal_projective','bootstrap_background_projective','rational_center_fixed_size','rational_center_linear_size','rational_center_ols_fixed_size','online_background']:
                trials=[]
                for seed in range(20):
                    rr=[];rng=np.random.default_rng(17000+seed)
                    for name,track in tracks.items():
                        if name not in common:continue
                        rows,_=run_track(track,k,method,hbg,hnom,epi,noise=amp,rng=rng)
                        for r in rows:r['class']=name
                        rr+=rows
                    trials.append(summarize(rr))
                noise.append({'max_abs_initial_corner_noise_px':amp,'k':k,'method':method,'trials':20,'fraction_iou_ge_0_5_mean':float(np.mean([s['fraction_iou_ge_0_5'] for s in trials])),'whole_remaining_track_success_fraction_mean':float(np.mean([s['whole_remaining_track_success_fraction'] for s in trials])),'median_center_error_px_mean':float(np.mean([s['center_error_px']['median'] for s in trials]))})
    metadata={'official_label_count':sum(map(len,tracks.values())),'official_tracks':len(tracks),'common_tracks_k_up_to_5':sorted(common),'image_bounds':BOUNDS.tolist(),'calibration':cal,'information_budgets':{'constant_velocity':'Only first k unclipped official boxes; k1 has zero velocity.','nominal_projective':'One latest box plus camera prior fitted to separate validation images, no later reference imagery.','bootstrap_background_projective':'First k reference frames and boxes. Median initial interframe background H repeated into unseen future; k1 falls back to disjoint validation prior.','rational_center_fixed_size':'First k boxes plus epipole calibrated on disjoint validation images. Reciprocal radius linear in time; size held fixed; k1 nominal prior.','rational_center_linear_size':'Same rational centre with linear width/height trend from initial k boxes; k1 nominal prior.','online_background':'Latest observed box transported by every subsequent actual background image pair. Known foreground boxes expanded20px used only as exclusion masks.'},'caveats':['16 tracks are one physical instance per class; frame rows are strongly correlated.','Starting/refreshed boxes are organizer oracle boxes. No detection/classification accuracy claimed.','Main bootstrap waits for unclipped boxes; entry partial boxes do not initialize a complete footprint.','Final-frame tracks are right-censored: full remaining observed clip is shorter than true visibility.','Box corners are transformed once by composed H; envelopes are never iteratively rewarped.','Predictions clipped for scoring; unbounded predicted boxes also saved.','Cached pixel matches had a loose40px gate and train/test split from prior study.','Full4K images were used for background feature matching; live legal crop accuracy unmeasured.','Refresh frames included in all-frame metrics and separately removed in forecast-only metrics; edge-clipped refresh observations are requested but never fitted or counted as refreshed.','Departure diagnostic ignores refresh and uses bootstrap forecast; geometric exit differs from minimum-visible-pixel annotation threshold.']}
    noise_refresh=[]
    for method in ['rational_center_fixed_size','rational_center_ols_fixed_size','online_background']:
        for refresh in [0,3,5,10]:
            trials=[]
            for seed in range(20):
                rr=[];rng=np.random.default_rng(18000+seed)
                for name,track in tracks.items():
                    if name not in common:continue
                    rows,_=run_track(track,5,method,hbg,hnom,epi,refresh=refresh,noise=2,rng=rng)
                    for r in rows:r['class']=name
                    rr+=rows
                trials.append(summarize(rr))
            noise_refresh.append({'max_abs_observed_corner_noise_px':2,'k':5,'method':method,'refresh_interval':refresh,'trials':20,'fraction_iou_ge_0_5_mean':float(np.mean([s['fraction_iou_ge_0_5'] for s in trials])),'forecast_only_fraction_iou_ge_0_5_mean':float(np.mean([s['forecast_only_fraction_iou_ge_0_5'] for s in trials])),'whole_remaining_track_success_fraction_mean':float(np.mean([s['whole_remaining_track_success_fraction'] for s in trials])),'median_center_error_px_mean':float(np.mean([s['center_error_px']['median'] for s in trials]))})
    spaced=[]
    for indices in [[2,3,4],[0,2,4],[0,1,2,3,4]]:
        trials=[]
        for seed in range(20):
            rr=[];rng=np.random.default_rng(19000+seed)
            for name,track in tracks.items():
                if name not in common:continue
                rows,_=run_track(track,len(indices),'rational_center_ols_fixed_size',hbg,hnom,epi,noise=2,rng=rng,observation_indices=indices)
                for r in rows:r['class']=name
                rr+=rows
            trials.append(summarize(rr))
        spaced.append({'initial_full_box_indices':indices,'max_abs_initial_corner_noise_px':2,'trials':20,'tracks':trials[0]['tracks'],'future_boxes':trials[0]['boxes'],'fraction_iou_ge_0_5_mean':float(np.mean([s['fraction_iou_ge_0_5'] for s in trials])),'whole_remaining_track_success_fraction_mean':float(np.mean([s['whole_remaining_track_success_fraction'] for s in trials])),'median_center_error_px_mean':float(np.mean([s['center_error_px']['median'] for s in trials]))})
    metadata['information_budgets']['rational_center_ols_fixed_size']='Same disjoint epipole prior; least-squares reciprocal-radius intercept and slope, averaged unit radial bearing, fixed size of latest box.'
    result={'metadata':metadata,'summary':summary,'per_track':pertrack,'noise_sensitivity':noise,'noise_and_refresh_sensitivity':noise_refresh,'spaced_observation_sensitivity':spaced}
    (OUT/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'predictions.jsonl').open('w') as out:
        for r in allrows:out.write(json.dumps(r)+'\n')
    for mode in ['earliest_full_box','common_fifth_full_box']:
        print('\n',mode)
        for s in summary:
            if s['anchor_mode']==mode and s['refresh_interval']==0:
                v=s['summary'];print(s['method'],s['k'],'n',v['tracks'],v['boxes'],'iou50',round(v['fraction_iou_ge_0_5'],3),'full',v['whole_remaining_track_successes'],'median err',round(v['center_error_px']['median'],2),'final med',round(v['final_center_error_px']['median'],2))


if __name__=='__main__':main()
