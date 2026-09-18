"""Legal camera-policy visibility and geometry proxy, without detector claims.

Run: python3 drone/perspective_tracking/camera_policy_benchmark.py
Only creates camera-policy-* artifacts; prior studies remain unchanged.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from benchmark import ROOT,OUT,read_data,unclipped,center,clip,iou,forecast,stats

SOURCE=ROOT/'artifacts/drone-source-2026-09-17'
sys.path.insert(0,str(SOURCE))
from local_evaluator import Camera,CameraRejection

POLICIES=['L0_full','L1_left_centre_right_centre','L1_right_centre_left_centre','hybrid_left_full_right_full','hybrid_right_full_left_full']
NAMES={'L0_full':'L0 full view','L1_left_centre_right_centre':'L1 L-C-R-C','L1_right_centre_left_centre':'L1 R-C-L-C','hybrid_left_full_right_full':'L1 L / L0 / L1 R / L0','hybrid_right_full_left_full':'L1 R / L0 / L1 L / L0'}
LEFT=(1,960,540);RIGHT=(1,2880,540);CENTRE=(1,1920,540);FULL=(0,1920,1080)


def schedule(policy):
    camera=Camera();rows=[]
    cycle={'L0_full':[FULL],'L1_left_centre_right_centre':[LEFT,CENTRE,RIGHT,CENTRE],'L1_right_centre_left_centre':[RIGHT,CENTRE,LEFT,CENTRE], 'hybrid_left_full_right_full':[LEFT,FULL,RIGHT,FULL],'hybrid_right_full_left_full':[RIGHT,FULL,LEFT,FULL]}[policy]
    for f in range(25):
        if f>0:camera.apply(*cycle[(f-1)%len(cycle)])
        rows.append({'frame':f,'level':camera.resolution_level,'center_xy':[camera.center_x,camera.center_y],'region_xyxy':list(camera.source_region),'source_pixels_per_input_pixel':4 if camera.resolution_level==0 else 2})
    return rows


def intersection(b,region):
    b=np.array(b);r=np.array(region);return np.maximum(0,np.minimum(b[2:],r[2:])-np.maximum(b[:2],r[:2])).prod()


def fully_observed(b,view):
    r=view['region_xyxy'];return unclipped(b) and b[0]>=r[0] and b[1]>=r[1] and b[2]<=r[2] and b[3]<=r[3]


def summary(rows):
    if not rows:return None
    by=defaultdict(list)
    for r in rows:by[r['class']].append(r)
    return {'tracks':len(by),'boxes':len(rows),'fraction_iou_ge_0_5':float(np.mean([r['iou']>=.5 for r in rows])),
            'whole_track_success_fraction':float(np.mean([min(a['iou'] for a in rr)>=.5 for rr in by.values()])),
            'center_error_px':stats([r['center_error_px'] for r in rows]),'iou':stats([r['iou'] for r in rows]),
            'final_error_px':stats([rr[-1]['center_error_px'] for rr in by.values()])}


def evaluate(tracks,views,opportunities,model,k,noise,seeds,hnom,epi,hbg,common_horizons=None,save_rows=False):
    trials=[];rows0=[]
    for seed in range(seeds):
        rng=np.random.default_rng(27000+seed)
        noise_bank={name:{f:rng.uniform(-noise,noise,4) for f in sorted(t)} for name,t in sorted(tracks.items())}
        rows=[];all_label_passes=0;emitted=0;acquired=[]
        for name,tr in tracks.items():
            available=opportunities[name]['full_observation_frames'];init=available[:k]
            if common_horizons is not None and name not in common_horizons:continue
            def measured(f):return tr[f]+noise_bank[name][f]*views[f]['source_pixels_per_input_pixel']
            # Directly observed boxes can be output while collecting the bootstrap.
            # Classes are oracle identities; no detector/classification claim.
            before=(init[-1] if len(init)==k else max(tr))
            for f in available:
                if f<=before:
                    emitted+=1;all_label_passes+=int(iou(measured(f),tr[f])>=.5)
            if len(init)<k:continue
            acquired.append(name);obs=[(f,measured(f)) for f in init]
            anchor=init[-1]
            for f,true in tr.items():
                if f<=anchor:continue
                if common_horizons is not None and f<=common_horizons[name]:continue
                method='rational_center_ols_fixed_size' if model=='object_rational' else 'nominal_projective' if model=='one_box_disjoint_nominal' else 'online_background'
                pred=forecast(method,obs,f,hbg,hnom,epi)
                row={'class':name,'frame':f,'horizon':f-anchor,'anchor_frame':anchor,'init_frames':init,'iou':iou(pred,true),'center_error_px':float(np.linalg.norm(center(clip(pred))-center(true))),'prediction_xyxy':pred.tolist(),'truth_xyxy':true.tolist(),'current_camera_fully_observes_object':f in available}
                rows.append(row);emitted+=1;all_label_passes+=int(row['iou']>=.5)
        s=summary(rows)
        if s:
            s.update({'bootstrapped_tracks':len(acquired),'bootstrapped_names':acquired,'all_259_label_proxy_iou50_fraction':all_label_passes/259,'all_259_label_emission_fraction':emitted/259})
            trials.append(s)
        if seed==0:rows0=rows
    if not trials:return None
    result={'trials':seeds,'tracks':trials[0]['tracks'],'future_boxes':trials[0]['boxes'],'bootstrapped_tracks':trials[0]['bootstrapped_tracks'],'fraction_iou_ge_0_5_mean':float(np.mean([a['fraction_iou_ge_0_5'] for a in trials])),'whole_track_success_fraction_mean':float(np.mean([a['whole_track_success_fraction'] for a in trials])),'median_center_error_px_mean':float(np.mean([a['center_error_px']['median'] for a in trials])),'p90_center_error_px_mean':float(np.mean([a['center_error_px']['p90'] for a in trials])),'all_259_label_proxy_iou50_fraction_mean':float(np.mean([a['all_259_label_proxy_iou50_fraction'] for a in trials])),'all_259_label_emission_fraction':trials[0]['all_259_label_emission_fraction']}
    if save_rows:result['seed0_rows']=rows0
    return result


def make_coverage(tracks,views):
    result={}
    for name,tr in tracks.items():
        full=[f for f,b in tr.items() if fully_observed(b,views[f])]
        intersect=[f for f,b in tr.items() if intersection(b,views[f]['region_xyxy'])>0]
        partial=[f for f in intersect if f not in full]
        overlap_fraction={f:float(intersection(b,views[f]['region_xyxy'])/((b[2]-b[0])*(b[3]-b[1]))) for f,b in tr.items()}
        half=[f for f in intersect if overlap_fraction[f]>=.5]
        crop_truncated=[f for f in intersect if intersection(tr[f],views[f]['region_xyxy'])<(tr[f][2]-tr[f][0])*(tr[f][3]-tr[f][1])-.001]
        fsfull=[f for f,b in tr.items() if unclipped(b)]
        top=[f for f,b in tr.items() if unclipped(b) and b[3]<=1080]
        dims=[((tr[f][2:]-tr[f][:2])/views[f]['source_pixels_per_input_pixel']).tolist() for f in full]
        result[name]={'first_label':min(tr),'any_visible_frames':intersect,'at_least_half_annotated_visible_box_frames':half,'source_edge_clipped_frames':[f for f,b in tr.items() if not unclipped(b)],'annotated_visible_box_crop_fraction':overlap_fraction,'first_source_complete_box':min(fsfull),'full_observation_frames':full,'source_top_half_complete_frames':top,'full_observations_in_L1_top_half':[f for f in full if views[f]['level']==1],'first_acquisition_frame':min(full) if full else None,'acquisition_delay_from_first_complete_frame':min(full)-min(fsfull) if full else None,'partial_observation_frames':partial,'crop_boundary_truncation_frames':crop_truncated,'observation_input_wh':dims,'new_entry_after_frame0':min(tr)>0}
    return result


def main():
    tracks,hbg,hnom,epi,cal=read_data()
    c=Camera();c.apply(*LEFT)
    try:c.apply(*RIGHT);raise AssertionError('Unexpectedly legal direct L1 side switch')
    except CameraRejection as e:illegal_reason=str(e)
    schedules={p:schedule(p) for p in POLICIES};coverage={p:make_coverage(tracks,schedules[p]) for p in POLICIES};summaries=[];matched=[]
    for p in POLICIES:
        for noise in [0.,1.]:
            for model,ks in [('one_box_disjoint_nominal',[1]),('one_box_oracle_fullframe_background',[1]),('object_rational',[2,3,5])]:
                for k in ks:
                    result=evaluate(tracks,schedules[p],coverage[p],model,k,noise,1 if noise==0 else 20,hnom,epi,hbg,save_rows=noise==0)
                    summaries.append({'policy':p,'model':model,'k':k,'max_abs_input_pixel_corner_noise':noise,'result':result})
    # Same cohort and same future source frames across policies, for each k.
    # Only the selected initial observation times/resolution differ.
    for k in [1,2,3,5]:
        shared={}
        for name,tr in tracks.items():
            all_obs=[coverage[p][name]['full_observation_frames'] for p in POLICIES]
            if not all(len(x)>=k for x in all_obs):continue
            latest=max(x[k-1] for x in all_obs)
            if max(tr)>latest:shared[name]=latest
        for p in POLICIES:
            model='one_box_disjoint_nominal' if k==1 else 'object_rational'
            result=evaluate(tracks,schedules[p],coverage[p],model,k,1.,20,hnom,epi,hbg,common_horizons=shared)
            if result:
                # All259 metrics have no meaning after common-cohort restriction.
                result.pop('all_259_label_proxy_iou50_fraction_mean',None);result.pop('all_259_label_emission_fraction',None)
            matched.append({'policy':p,'model':model,'k':k,'common_horizons':shared,'result':result})
    opportunities={}
    for p in POLICIES:
        cov=coverage[p];dims=np.array([wh for a in cov.values() for wh in a['observation_input_wh']]);nfull=sum(len(a['full_observation_frames']) for a in cov.values());delays=[a['acquisition_delay_from_first_complete_frame'] for a in cov.values() if a['first_acquisition_frame'] is not None]
        opportunities[p]={'objects_ever_fully_observed':sum(bool(a['full_observation_frames']) for a in cov.values()),'objects_with_any_pixels_observed':sum(bool(a['any_visible_frames']) for a in cov.values()),'any_visible_appearances':sum(len(a['any_visible_frames']) for a in cov.values()),'at_least_half_visible_annotation_appearances':sum(len(a['at_least_half_annotated_visible_box_frames']) for a in cov.values()),'partial_source_or_crop_appearances':sum(len(a['partial_observation_frames']) for a in cov.values()),'source_edge_clipped_appearances':sum(len(a['source_edge_clipped_frames']) for a in cov.values()),'cohorts':{cohort:{'objects':sum((not a['new_entry_after_frame0']) if cohort=='present_at_start' else a['new_entry_after_frame0'] for a in cov.values()),'objects_any_visible':sum(bool(a['any_visible_frames']) for a in cov.values() if (not a['new_entry_after_frame0'])== (cohort=='present_at_start')),'objects_complete_geometry_view':sum(bool(a['full_observation_frames']) for a in cov.values() if (not a['new_entry_after_frame0'])==(cohort=='present_at_start')),'complete_observation_appearances':sum(len(a['full_observation_frames']) for a in cov.values() if (not a['new_entry_after_frame0'])==(cohort=='present_at_start'))} for cohort in ['present_at_start','new_entries']},'objects_never_fully_observed':[n for n,a in cov.items() if not a['full_observation_frames']], 'fully_observed_label_appearances':nfull,'all_label_appearances':259,'crop_boundary_truncated_appearances':sum(len(a['crop_boundary_truncation_frames']) for a in cov.values()),'median_input_width_px':float(np.median(dims[:,0])),'median_input_height_px':float(np.median(dims[:,1])),'acquisition_delay_frames':stats(delays),'eligible_bootstrap_tracks':{str(k):sum(len(a['full_observation_frames'])>=k for a in cov.values()) for k in [1,2,3,5]},'eligible_bootstrap_with_future':{str(k):sum(len(a['full_observation_frames'])>=k and a['full_observation_frames'][k-1]<max(tracks[n]) for n,a in cov.items()) for k in [1,2,3,5]},'new_entry_acquisition_delays':{n:a['acquisition_delay_from_first_complete_frame'] for n,a in cov.items() if a['new_entry_after_frame0']}}
    metadata={'source_contract':str(SOURCE/'local_evaluator.py'),'direct_L1_left_to_right_rejection':illegal_reason,'noise':'Independent uniform±1input-pixel corner noise for each class/frame; paired across policies and k, scaled4sourcepx inL0 and2inL1.20seeds. Frame0sharedL0noise has same scale.','bootstrap':'First k source-unclipped boxes fully contained in actual legal view. After bootstrap, no further target boxes enter predictions, even if visible.','calibration':'Other-sequence validation epipole/nominalH from prior study. No future object labels.','one_box_oracle_background':'Diagnostic only: continued FULL-FRAME actual reference background homographies; violates L1 crop information budget and is not a realistic policy score.','all_label_proxy':'Out of259organizer labels, direct oracle boxes before bootstrap plus blind future forecasts. Missing outputs countasfailure. Oracle identities and absence offalsepositives mean this is not mAP.','limitations':['One physical object perclass; correlated frames; sixtracks remainvisible atclipend for k5cohort.','Fullboxvisibility is an oracle eligibility rule, not measured detectability. Partialcrop examples conservatively cannot initialize a fullobject footprint.','No trained detector, actual classification, resampling aliasing or request latency is simulated.','StrictL1alwaysafterframe0 sees onlysource top1080rows; objectsbelowtophalf cannot be reacquired bythis fixedpolicy.','Noise affects sourcecoordinates according tothe current actual zoom; nominal camera calibration uncertainty is not separately perturbed.']}
    data={'metadata':metadata,'schedules':schedules,'coverage_summary':opportunities,'coverage_per_track':coverage,'experiments':summaries,'matched_horizon_experiments':matched}
    (OUT/'camera-policy-measurements.json').write_text(json.dumps(data,indent=2)+'\n')
    report(data)
    print('Illegal direct switch:',illegal_reason)
    for p in POLICIES:
        a=opportunities[p];print(p,a['eligible_bootstrap_with_future'],'observed',a['fully_observed_label_appearances'],'truncated',a['crop_boundary_truncated_appearances'])
    for r in summaries:
        if r['max_abs_input_pixel_corner_noise']==1 and r['model']=='object_rational':print(r['policy'],r['k'],r['result'])


def report(data):
    def pct(x):return f'{100*x:.1f}%'
    def get(p,k,model='object_rational',noise=1):return next(r['result'] for r in data['experiments'] if r['policy']==p and r['k']==k and r['model']==model and r['max_abs_input_pixel_corner_noise']==noise)
    lines=['# Legal zoom/camera policy benchmark','', '**One object observation can be enough if the shared camera motion is already known.** Multiple object observations are useful for estimating local speed/height effects and reducing box noise; they are not a geometric requirement for every object. This study separates that issue from whether the requested crop schedule is legal and what it actually observes.','',
'## Legal schedules','', 'All policies receive the mandatory initial L0 full frame at frame0. Commands affect the next frame. L1 source crops are1920×1080, transmitted960×540, with centres `(960,540)`, `(1920,540)` and `(2880,540)` for the upper left, centre and right views.','',
'**Direct L1 left-to-right alternation is illegal:** the1920source-pixel jump exceeds the current L1 limit1102. The executable scan is `L→C→R→C→L`, giving an outer-side revisit every4frames. The two start phases were evaluated. A hybrid `L1-left→L0-full→L1-right→L0-full` is also legal because full-view resets ignore the pan limit and L0→either topcorner fits the L0limit2203.','',
'Exact legality is checked by instantiating the organizer\'s saved `Camera` class, applying every movement, and requiring the direct side switch to raise `CameraRejection`.','',
'## What is visible','', '| Policy | Complete object views /259labels | Crop-truncated appearances | Tracks with≥2 /≥3 /≥5observations and later labels | Median observed input box |', '|---|---:|---:|---:|---:|']
    for p in POLICIES:
        a=data['coverage_summary'][p];n=a['eligible_bootstrap_with_future'];lines.append(f'| {NAMES[p]} | {a["fully_observed_label_appearances"]} | {a["crop_boundary_truncated_appearances"]} | {n["2"]} / {n["3"]} / {n["5"]} | {a["median_input_width_px"]:.1f}×{a["median_input_height_px"]:.1f}px |')
    lines+=['','A complete view requires the official box to be uncut by both the source-image boundary and the crop boundary. The strict left-first scan acquires 15 of 16 objects completely within this clip; it misses a complete hangar view before frame 24. The other schedules acquire all 16, helped by the shared initial full view. The hangar is still present when the clip ends, so this is delayed acquisition beyond the observed clip rather than proof it would never be acquired. Initial objects already below the top half are a particularly important exception: strictL1cannot get a second look at them. Objects entering afterframe0 are listed separately in the measurements. Median dimensions are over each policy\'s different observed sample; for the **same** box, L1 gives exactly2×the input width and height ofL0.','',
'Partial views can remain useful for classification. All 16 objects have some observed pixels under every schedule; the missing complete hangar box is not a claim that the hangar cannot be detected. The reference includes 11 objects already present at frame 0 and five later entrants. Complete-box acquisition is a geometry-bootstrap condition, not detector recall.', '', 'Acquisition delay from the first complete source box, for new entries after frame 0. Missing means no complete view before the clip ends:','', '| Object | L0 | L1left-first | L1right-first | Hybridleft-first | Hybridright-first |','|---|---:|---:|---:|---:|']
    entrants=[n for n,a in data['coverage_per_track']['L0_full'].items() if a['new_entry_after_frame0']]
    for n in entrants:lines.append('| '+n+' | '+' | '.join(str(data['coverage_per_track'][p][n]['acquisition_delay_from_first_complete_frame']) if data['coverage_per_track'][p][n]['acquisition_delay_from_first_complete_frame'] is not None else 'Missing' for p in POLICIES)+' |')
    lines+=['','## Geometry and localization-noise proxy','',
'Every observation receives identical ±1input-pixel coordinate uncertainty: ±4source pixels inL0, ±2inL1. Noise is paired by object/frame across policies and k;20seeded trials are averaged. This is a controlled proxy, **not measured detector precision**. Initialframe0isL0for everypolicy, so its measurements have the same uncertainty. No later target observations are used afterbootstrap.','',
'The rational trajectory uses the epipole calibrated on the separate validation sequence and fits reciprocal centre radius from the firstkavailable observations. Irregular observation times are respected. Conditional future accuracy is evaluated on **all** subsequent official labels, including frames when the object is outside the current crop. Eligibility and future horizons differ between the first table\'s rows.','',
'| Policy | Initial observations | Forecast tracks /boxes | Conditional future IoU50 | Whole-track pass | Wait-for-k proxy (diagnostic only) |','|---|---:|---:|---:|---:|---:|']
    for p in POLICIES:
        for k in [2,3,5]:
            a=get(p,k)
            if a:lines.append(f'| {NAMES[p]} | {k} | {a["tracks"]}/{a["future_boxes"]} | {pct(a["fraction_iou_ge_0_5_mean"])} | {pct(a["whole_track_success_fraction_mean"])} | {pct(a["all_259_label_proxy_iou50_fraction_mean"])} |')
    lines+=['','**The wait-for-k diagnostic deliberately withholds forecasts between visible observations until k views have been collected. This is not the recommended policy and should not decide the L0-versus-L1 choice. A working system can predict from its first observation using shared motion, then refine later.** This diagnostic counts missing outputs as failures and includes oracle directly observed boxes during bootstrap. It also assumes perfect class identity and no false positives, so it is **not mAP**. Its purpose is to prevent conditional accuracy from hiding objects with insufficient observations.','',
'For a matched comparison, the following uses the common cohort and identical futureframes after the latest policy\'s kthobservation:','', '| Initial observations | Shared tracks /futureboxes | L0 futureIoU50 | StrictL1 left /right | Hybrid left /right |','|---:|---:|---:|---:|---:|']
    for k in [2,3,5]:
        rr={p:next(x['result'] for x in data['matched_horizon_experiments'] if x['policy']==p and x['k']==k) for p in POLICIES}
        if rr['L0_full']:
            a=rr['L0_full'];lines.append(f'| {k} | {a["tracks"]}/{a["future_boxes"]} | {pct(a["fraction_iou_ge_0_5_mean"])} | {pct(rr[POLICIES[1]]["fraction_iou_ge_0_5_mean"])} / {pct(rr[POLICIES[2]]["fraction_iou_ge_0_5_mean"])} | {pct(rr[POLICIES[3]]["fraction_iou_ge_0_5_mean"])} / {pct(rr[POLICIES[4]]["fraction_iou_ge_0_5_mean"])} |')
    lines+=['','The strictscan trades acquisition count and lower-half coverage for more pixels per object and a longer timebaseline between repeated outer views. A longer baseline can improve velocity fitting; this comparison does not isolate resolution alone.','',
'## One observation and shared motion','', 'Cells are conditional future-box IoU50 under the same ±1 input-pixel noise proxy, not classification accuracy.','', '| Policy | Onebox + transferred nominalH | Onebox + actual fullframe backgroundH (oracle diagnostic) |','|---|---:|---:|']
    for p in POLICIES:
        a=get(p,1,'one_box_disjoint_nominal');b=get(p,1,'one_box_oracle_fullframe_background');lines.append(f'| {NAMES[p]} | {pct(a["fraction_iou_ge_0_5_mean"])} | {pct(b["fraction_iou_ge_0_5_mean"])} |')
    lines+=['','These two columns do **not** establish a minimum required number of object views. The transferred calibration was insufficient here; camera parameters, object height/depth and model approximation can contribute, and their separate contributions were not isolated. The second column shows what one object box plus continuing shared motion can do, but consumes whole-reference-frame background imagery and therefore exceeds strictL1\'s information budget. It is deliberately an oracle diagnostic, not an executable zoom-policy result.','',
'A deployable one-observation system should use camera motion estimated from the legal observed pixels, transform source-coordinate tracks, and keep uncertainty for object height/shape and initial localization. The parent investigation separately measures real pixels atL0andL1.','',
'## Implication for the top scan','',
'- A first confident classification can immediately create a track using already-calibrated shared camera motion. There is no need to wait forfiveobjectlooks merely to start predicting.','- Subsequent visible observations can refine the object-specific trajectory; retain them when the scan naturally revisits the object.','- A fixed top-halfscan cannot implement periodic objectrefresh after the object leaves that half. The earlier five-framerefresh result therefore applies only if the camera is allowed to leave the scan to revisit it.','- Compare the strictscan with the legalhybrid, which keeps the same outer-sidehigh-resolution cadence and restores lower-halfcoverage everyotherframe. This study uses ideal visibility and artificial localizationnoise; actual detector/classificationaccuracy remains unmeasured.','',
'## Files and reproduction','', '```sh','python3 drone/perspective_tracking/camera_policy_benchmark.py','```','',
'`camera-policy-measurements.json` stores exact schedules, perclass observationframes/dimensions, seeds\' aggregate results, and noiseless perframe predictions. No training, GPU service, API submission or external change was used. Existing tracking benchmarks were not modified.','']
    # Keep human prose spaced; code labels/constants remain compact identifiers.
    replacements={'frame0':'frame 0','afterbootstrap':'after bootstrap','futureframes':'future frames','source-pixel':'source-pixel','1920source':'1920 source','1102.':'1102.','L0limit2203':'L0 limit 2203','every4frames':'every 4 frames','All16objects':'All 16 objects','strictL1':'strict L1','The strictscan':'The strict scan','legalhybrid':'legal hybrid','objectrefresh':'object refresh','top-halfscan':'top-half scan','five-framerefresh':'five-frame refresh','everyotherframe':'every other frame','localizationnoise':'localization noise','detector/classificationaccuracy':'detector/classification accuracy','outer-sidehigh-resolution':'outer-side high-resolution','fiveobjectlooks':'five object looks','onebox':'one box','Onebox':'One box','nominalH':'nominal H','backgroundH':'background H','fullframe':'full frame','20seeded':'20 seeded','firstkavailable':'first k available','object/frame':'object/frame','futureIoU50':'future IoU50','futureboxes':'future boxes','kthobservation':'kth observation','timebaseline':'time baseline','separate validation sequence':'separate validation sequence','perclass':'per-class','observationframes':'observation frames','perframe':'per-frame','everypolicy':'every policy','later labels':'later labels','crop schedule':'crop schedule'}
    text='\n'.join(lines)
    for a,b in replacements.items():text=text.replace(a,b)
    for a,b in {
        'are1920':'are 1920','transmitted960':'transmitted 960','the1920':'the 1920',
        'limit1102':'limit 1102','topcorner':'top corner','/259labels':'/ 259 labels',
        'with≥2 /≥3 /≥5observations':'with ≥2 / ≥3 / ≥5 observations',
        'px |':' px |','strict L1cannot':'strict L1 cannot','afterframe 0':'after frame 0',
        'exactly2×the':'exactly 2× the','ofL0':'of L0','L1left-first':'L1 left-first',
        'L1right-first':'L1 right-first','Hybridleft-first':'Hybrid left-first',
        'Hybridright-first':'Hybrid right-first','±1input-pixel':'±1 input-pixel',
        '±4source':'±4 source','inL0':'in L0','±2inL1':'±2 in L1',';20':'; 20',
        'Initialframe 0isL0for':'Initial frame 0 is L0 for','/boxes':'/ boxes',
        'All259-label':'All 259-label','StrictL1':'Strict L1','atL0andL1':'at L0 and L1',
        'forfive':'for five','strictscan':'strict scan','lower-halfcoverage':'lower-half coverage',
        'The parent investigation':'The companion pixel investigation',
    }.items():text=text.replace(a,b)
    partial_table=['## Partial visibility and start/end cohorts','','Overlap is measured against the official annotated **visible** box, which may itself be clipped by the source image. At least half of that box does not imply half of the physical object is visible. All policies share the same 23 source-edge-clipped label appearances; crop boundaries can add truncation.','', '| Policy | Any pixels / ≥50% visible-label box / complete box | Complete geometry views: initial 11 / new 5 |', '|---|---:|---:|']
    for p in POLICIES:
        a=data['coverage_summary'][p];co=a['cohorts'];partial_table.append(f'| {NAMES[p]} | {a["any_visible_appearances"]} / {a["at_least_half_visible_annotation_appearances"]} / {a["fully_observed_label_appearances"]} | {co["present_at_start"]["objects_complete_geometry_view"]} / {co["new_entries"]["objects_complete_geometry_view"]} |')
    text=text.replace('## Geometry and localization-noise proxy','\n'.join(partial_table)+'\n\n## Geometry and localization-noise proxy')
    text=text.replace('## Geometry and localization-noise proxy',
        'This is a short 25-frame reference clip, with many objects already visible at the start and some new objects appearing just before the recording ends. Its acquisition counts are not a steady-state claim for the longer validation or evaluation flights. Partial object crops may still be classifiable, but this conservative study requires a complete box to initialize its full footprint.\n\n## Geometry and localization-noise proxy')
    (OUT/'camera-policy-report.md').write_text(text)


if __name__=='__main__':main()
