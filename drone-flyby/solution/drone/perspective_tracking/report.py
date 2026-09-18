"""Render concise findings and standard scientific plots from saved benchmarks."""
import json
from pathlib import Path
import numpy as np
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nordic-drone-matplotlib")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark import OUT

r=json.loads((OUT/'benchmark.json').read_text());v=json.loads((OUT/'validation-benchmark.json').read_text())

def ref(method,k,refresh=0,mode='earliest_full_box'):
    return next(x['summary'] for x in r['summary'] if x['method']==method and x['k']==k and x['refresh_interval']==refresh and x['anchor_mode']==mode)
def val(method,k,refresh=0):return next(x['summary'] for x in v['summary'] if x['model']==method and x['k']==k and x['refresh_interval']==refresh)
def pct(x):return f'{100*x:.1f}%'

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(16,4.6))
for model,label in [('constant_velocity','Constant image velocity'),('nominal_projective','Transferred nominal homography'),('rational_center_fixed_size','Rational object trajectory'),('online_background','Online background homography')]:
    ks=[1,2,3,4,5,8];axs[0].plot(ks,[100*ref(model,k)['fraction_iou_ge_0_5'] for k in ks],'-o',label=label,markersize=4)
axs[0].set(xlabel='Initial consecutive full-box observations',ylabel='Future boxes with IoU ≥ 0.50 (%)',ylim=(0,103),title='Official reference boxes\nPerfect initial measurements; varying horizons');axs[0].legend(fontsize=7)
for model,label in [('rational_center_fixed_size','Anchored rational'),('rational_center_ols_fixed_size','OLS rational'),('online_background','Background homography')]:
    rows=[x for x in r['noise_and_refresh_sensitivity'] if x['method']==model]
    xs=np.arange(len(rows));axs[1].plot(xs,[100*x['forecast_only_fraction_iou_ge_0_5_mean'] for x in rows],'-o',label=label,markersize=4)
axs[1].set(xticks=range(4),xticklabels=['None','3','5','10'],xlabel='Object refresh interval (frames)',ylabel='Forecast-only boxes with IoU ≥ 0.50 (%)',title='Synthetic ±2 px box-corner jitter\n5 bootstrap observations; 20 trials',ylim=(70,101));axs[1].legend(fontsize=8)
for model,label in [('rational_ols','Blind trajectory'),('rational_clock_ols','Trajectory + observed motion clock')]:
    gaps=[0,3,5,8,10];axs[2].plot(np.arange(5),[100*val(model,5,g)['whole_tracked_remainder_within_5px_fraction'] for g in gaps],'-o',label=label,markersize=4)
axs[2].set(xticks=range(5),xticklabels=['None','3','5','8','10'],xlabel='Point refresh interval (frames)',ylabel='Tracks always within 5 source pixels (%)',title='Independent validation landmark proxy\n5 bootstrap observations; ideal point refresh',ylim=(0,103));axs[2].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'tracking-results.png',dpi=160);plt.close(fig)

# One track illustrates the freeze/double-step rather than only aggregate medians.
rows=[]
with (OUT/'validation-predictions.jsonl').open() as inp:
    for line in inp:
        a=json.loads(line)
        if a['track_id']=='landmark_005_000' and a['k']==4 and not a['refresh_interval'] and a['model'] in ['rational_ols','rational_clock_ols']:rows.append(a)
fig,ax=plt.subplots(figsize=(10,3.5))
for model,label in [('rational_ols','Blind forecast'),('rational_clock_ols','Forecast with observed background clock')]:
    rr=[x for x in rows if x['model']==model];ax.plot([x['frame'] for x in rr],[x['error_px'] for x in rr],'-o',markersize=3,label=label)
ax.axvline(9,color='grey',ls=':',label='Frozen frame 9');ax.set(xlabel='Validation frame',ylabel='Source-pixel point error',title='A single frozen frame breaks blind constant-motion timekeeping');ax.legend();fig.tight_layout();fig.savefig(OUT/'freeze-example.png',dpi=150);plt.close(fig)

lines=['# Perspective tracking experiment','', 'Completed 17 September 2026. Offline data only; no training, API submission or external service.','',
'**Four to five accurate full-object observations are a promising bootstrap, given an independently calibrated camera direction. They are not a guarantee that the object can be ignored for the whole flight.** The reference labels support excellent causal geometric prediction, but realistic box uncertainty and the validation capture freezes make continuing motion checks and occasional object refresh useful.','',
'## Information budgets and model','',
'1. **Blind future prediction:** initial object boxes plus a camera prior from the other sequence. No later images or labels enter predictions. The nominal-homography baseline transfers a full interframe mapping; the better rational model transfers only the epipole and estimates each object\'s speed from its initial observations.','2. **Online background:** current background pixels update motion, without reobserving the target. Reference fits exclude all known foreground target boxes plus a 20-pixel margin. Validation fits use separate odd-ID landmark tracks; even IDs are held-out targets.','3. **Object refresh:** an ideal target box or point is supplied every specified number of frames. Refresh frames are separated from forecast-only metrics. Synthetic box-noise tests perturb refresh boxes too. A refresh view touching an image edge is treated as a censored box and is not used to refit the full centre or shape; prediction continues from the last complete observation.','',
'With fixed camera orientation and straight constant translation, a static point moves along a ray from the epipole `e`. Its radius follows `r(t) = 1 / (a + b*t)`, so reciprocal radius is linear in time. We fit that line from the initial measurements. The OLS variant fits both intercept and slope and averages the observed radial direction. We compare fixed box size against a linearly changing width/height. This needs neither metric depth nor drone velocity. Object-box centres are only approximate static 3D points, which is one source of residual error.','',
f'Reference uses a validation-derived epipole of {np.round(r["metadata"]["calibration"]["epipole"],1).tolist()}; validation uses a reference-derived epipole of {np.round(v["metadata"]["epipole"],1).tolist()}. Every calibration pair and matrix is saved. A single observation cannot establish an unknown trajectory; k=1 projective results depend on the separate camera prior.','',
'## Official reference: initial observations and full remaining clip','',
'The organizer supplies 259 appearances across 16 physical objects, one per class. Bootstrap starts at the first fully visible box; entry-clipped boxes are deliberately not treated as the complete footprint. Predictions are clipped to the image for IoU scoring, including subsequent exit-clipped labels. This is bounding-box geometry with known identity and oracle observations, not detector accuracy or competition mAP.','',
'| Initial observations | Tracks / future boxes | Constant-velocity IoU50 | Rational fixed-size IoU50 | Rational whole-track passes | Median / p90 centre error |', '|---:|---:|---:|---:|---:|---:|']
for k in [2,3,4,5,8]:
    a=ref('rational_center_fixed_size',k);c=ref('constant_velocity',k)
    lines.append(f'| {k} | {a["tracks"]} / {a["boxes"]} | {pct(c["fraction_iou_ge_0_5"])} | {pct(a["fraction_iou_ge_0_5"])} | {a["whole_remaining_track_successes"]}/{a["tracks"]} | {a["center_error_px"]["median"]:.2f} / {a["center_error_px"]["p90"]:.2f} px |')
lines+=['','IoU50 means predicted overlap with the official box is at least 0.50. Whole-track pass means **every remaining annotated frame** passes, including clipped exits. Rows have different horizons and eligibility; longer bootstrap omits very short tracks.','',
'For the five-observation linear-size variant, all 177 future boxes across 13 tracks pass IoU50; median centre error is '+f'{ref("rational_center_linear_size",5)["center_error_px"]["median"]:.2f} px. Seven of those tracks have an observed disappearance; all seven geometric exit predictions match the first unlabelled frame. The other six remain visible at reference frame 24, so their true remaining lifetime is not observed. The longest tested five-observation forecast is 20 frames. This cannot establish full-lifetime performance for those censored objects.','',
'**Matched endpoint check:** using the same 13 tracks and 177 future boxes, all bootstraps end at the fifth full-box observation. Two latest observations give '+pct(ref('rational_center_fixed_size',2,mode='common_fifth_full_box')['fraction_iou_ge_0_5'])+', while three, four and five give 100%. Therefore improvement is not solely caused by shortening the future horizon.','',
'**Box shape matters:** applying a plane homography to original box corners grows and stretches them, but an elevated object\'s visible sides can disappear. With one initial box, actual online background transforms produce '+pct(ref('online_background',1)['fraction_iou_ge_0_5'])+' IoU50, despite using more future information. With refresh every five frames, the saved metrics report all-frame and forecast-only accuracy separately. The simple fixed-size rational method happens to work well over this small reference sample; it is not a universal shape model.','',
'## Measurement uncertainty and observation spacing','',
'The perfect-box result is sensitive to measurement error. The following perturbation test adds independent uniform noise in ±2 native-source pixels to each initial box coordinate, repeats 20 seeds, and uses a common 13-track cohort for k≤5; k=8 has 12 sufficiently long tracks. This is a sensitivity test, **not a measured detector-error distribution**. The OLS rational model keeps the latest observed box size fixed.','',
'| Initial observations | Mean future IoU50 | Mean whole-track pass rate | Mean median centre error |','|---:|---:|---:|---:|']
for k in [2,3,4,5,8]:
    a=next(x for x in r['noise_sensitivity'] if x['max_abs_initial_corner_noise_px']==2 and x['method']=='rational_center_ols_fixed_size' and x['k']==k)
    lines.append(f'| {k} | {pct(a["fraction_iou_ge_0_5_mean"])} | {pct(a["whole_remaining_track_success_fraction_mean"])} | {a["median_center_error_px_mean"]:.2f} px |')
lines+=['','Noise plus refresh, five-observation OLS rational bootstrap, ±2 px on **all** observed boxes:','', '| Refresh interval | Forecast-only IoU50 | Whole-track pass rate |','|---:|---:|---:|']
for a in r['noise_and_refresh_sensitivity']:
    if a['method']=='rational_center_ols_fixed_size':lines.append(f'| {a["refresh_interval"] or "None"} | {pct(a["forecast_only_fraction_iou_ge_0_5_mean"])} | {pct(a["whole_remaining_track_success_fraction_mean"])} |')
lines+=['','Spacing comparison, same fifth-frame endpoint and 177 future boxes, ±2 px initial noise:','', '| Observed full-box indices | Future IoU50 | Whole-track pass rate |','|---|---:|---:|']
for a in r['spaced_observation_sensitivity']:lines.append(f'| {a["initial_full_box_indices"]} | {pct(a["fraction_iou_ge_0_5_mean"])} | {pct(a["whole_remaining_track_success_fraction_mean"])} |')
lines+=['','The three noise tables use separate deterministic seed banks; compare policies within a table. Indices are zero-based. Spreading measurements over time gives a longer baseline for speed estimation, so observation count and elapsed span should be separate scheduling decisions.','',
'## Independent validation landmark proxy','',
f'The companion landmark investigation independently tracked natural/scene features with bidirectional Lucas–Kanade pixel optical flow. Of 325 tracks retained for at least 15 observations, {v["metadata"]["heldout_target_tracks"]} even-ID tracks are prediction targets. Online background estimation uses all {v["metadata"]["background_fit_tracks"]} raw odd-ID seed tracks, without selecting them by future lifetime. Five start groups cover frames 5, 60, 120, 210 and 225. These are image-derived proxy positions, not organizer labels, classification accuracy or target boxes. Fifteen long tracks were visually audited; see [landmark evidence](../drone-validation-landmarks/README.md).','',
'Blind four-observation OLS forecasts have median '+f'{val("rational_ols",4)["error_px"]["median"]:.2f} px and p90 {val("rational_ols",4)["error_px"]["p90"]:.2f} px error, yet only '+pct(val('rational_ols',4)['whole_tracked_remainder_within_5px_fraction'])+' of tracks stay within five pixels for their full tracked remainder. Freezes at frames 9 and 239 create one-step jumps of about 60–90 px; the following double steps catch up. When frame 9 is one of the five bootstrap observations, a naïve fit is poisoned: five-observation blind p90 error is '+f'{val("rational_ols",5)["error_px"]["p90"]:.1f} px.','',
'The corrected model keeps the same per-object geometric trajectory but counts physical motion ticks from current background matches: near-zero motion counts as zero steps, normal motion as one, and doubled motion as two. The comparison uses median measured displacement / reference-nominal displacement at the same matched locations. It consumes **current and past background pixels**, never the future target position.','',
'| Bootstrap / object refresh | Median / p90 point error | Tracks always ≤5 px | Tracks always ≤20 px |','|---|---:|---:|---:|']
for k,g in [(4,0),(5,0),(8,0),(5,3),(5,5),(5,8),(5,10)]:
    a=val('rational_clock_ols',k,g);lines.append(f'| {k} / {g or "none"} | {a["error_px"]["median"]:.2f} / {a["error_px"]["p90"]:.2f} px | {pct(a["whole_tracked_remainder_within_5px_fraction"])} | {pct(a["whole_tracked_remainder_within_20px_fraction"])} |')
lines+=['','These refreshes use ideal LK point observations, so this table does not measure a detector\'s reacquisition quality. Long-track selection favors visually stable landmarks. Tracks can end at a safety margin or tracker confidence failure before physical exit, and most are ordinary scene features. Framewise rows and nearby landmarks are correlated.','',
'![Tracking benchmark](tracking-results.png)','', '![Freeze example](freeze-example.png)','',
'## Practical recommendation','',
'- Start with **five accurate full-box observations spanning at least four frame intervals**, or test three observations distributed across that span when crop scheduling makes five expensive. At the stated 3 fps, four intervals are about 1.33 seconds; partial entry can add a further wait.','- Fit a rational per-object centre trajectory with the independently estimated epipole. Retain class identity. Use a conservative box-size model and keep its uncertainty separate from centre uncertainty.','- Keep a cheap **background motion/timing check every frame**. Observed freezes make a fully blind frame-index forecast unreliable even when the trajectory itself is correct.','- Begin with an **object check about every five frames** (about 1.67 seconds at 3 fps), then extend intervals when observed residuals and box uncertainty are low. The noisy reference experiment and independent validation proxy support this as a starting policy, not a guaranteed universal optimum.','- Prefer earlier refresh for tiny objects, changing silhouettes, occlusion, a moving camera angle, low-texture motion estimation, or suspected identity switch. Large boxes can tolerate larger absolute errors.','- Next measure the real detector\'s box jitter and repeat using only legal 960×540 camera crops. Full-frame alignment here is an optimistic geometry study; crop-local performance and real-time latency are not yet measured.','',
'## Reproduce and inspect','',
'Run from the project root:','', '```sh','python3 drone/perspective_tracking/benchmark.py','python3 drone/perspective_tracking/validation_benchmark.py','python3 drone/perspective_tracking/report.py','```','',
'- [benchmark.json](benchmark.json): reference per-track results, denominators, calibrated matrices, noisy-box and observation-spacing experiments.','- [predictions.jsonl](predictions.jsonl): every reference forecast, unbounded predicted box, clipped IoU and centre error.','- [validation-benchmark.json](validation-benchmark.json): point-proxy results, start-group strata, clock steps and separate-background fits.','- [validation-predictions.jsonl](validation-predictions.jsonl): every validation prediction and proxy position.','',
'No future object labels are used by a blind prediction. Refresh variants consume only observations at the requested refresh times. The bootstrap-background homography is frozen after the original bootstrap even when the object is refreshed. Reference and validation calibrate each other only as an explicit disjoint-scene camera prior. The hidden evaluation set was not accessed.','']
(OUT/'README.md').write_text('\n'.join(lines))
print(OUT/'README.md')
