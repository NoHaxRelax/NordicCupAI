"""Offline scene study using original reference labels and captured pixels only.

Run: python3 drone/analyze_scene_geometry.py
Writes reproducible measurements under artifacts/drone-scene-analysis.
No API calls, training, or competition submissions.
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/drone-scene-analysis'
REF = ROOT / 'data/drone/reference/helsinki'
VAL = ROOT / 'data/drone/reconstructed-validation'


def stats(x):
    x = np.asarray(x)
    return {'n': len(x), 'median': float(np.median(x)),
            'p90': float(np.percentile(x, 90)), 'max': float(np.max(x))} if len(x) else None


def project(x, h):
    return cv2.perspectiveTransform(np.float32(x)[None], h)[0]


def iou(a, b):
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = np.prod(wh)
    return float(intersection / (np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection))


def annotations():
    tracks = defaultdict(list)
    poses = []
    for path in sorted((REF / 'annotations').glob('*.json')):
        j = json.loads(path.read_text())
        poses.append(j['pose'])
        for a in j['annotations']:
            b = a['bbox']
            # Exclude image-clipped boxes; these do not measure object scale.
            if b[0] > 2 and b[1] > 2 and b[2] < 3837 and b[3] < 2157:
                tracks[a['object_id']].append((j['frame'], b))
    results = {}
    forecasts = defaultdict(list)
    for name, items in sorted(tracks.items()):
        t = np.array([x[0] for x in items])
        b = np.array([x[1] for x in items], dtype=float)
        c = (b[:, :2] + b[:, 2:]) / 2
        wh = b[:, 2:] - b[:, :2]
        fit = np.polyfit(t, c, 1) if len(t) >= 2 else np.zeros((2, 2))
        residual = np.linalg.norm(c - np.polyval(fit, t[:, None]), axis=1)
        # Causal three-observation fit, evaluated on genuinely later labels.
        for i in range(2, len(t)):
            if t[i] - t[i-2] != 2:
                continue
            coeff = np.polyfit(t[i-2:i+1]-t[i], b[i-2:i+1], 1)
            for horizon in [1, 3, 5, 10]:
                idx = np.where(t == t[i] + horizon)[0]
                if len(idx):
                    pred = coeff[0]*horizon + coeff[1]
                    true = b[idx[0]]
                    forecasts[horizon].append({'class': name, 'anchor': int(t[i]),
                        'center_error_px': float(np.linalg.norm((pred[:2]+pred[2:]-true[:2]-true[2:])/2)),
                        'iou': iou(pred, true)})
        results[name] = {'frames': t.tolist(), 'boxes': b.tolist(),
            'velocity_px_per_frame': fit[0].tolist(), 'linear_center_residual_px': stats(residual),
            'first_last_wh': [wh[0].tolist(), wh[-1].tolist()],
            'width_ratio_last_first': float(wh[-1, 0]/wh[0, 0]),
            'height_ratio_last_first': float(wh[-1, 1]/wh[0, 1])}
    steps = np.diff([[p['x'], p['y'], p['z']] for p in poses], axis=0)
    return {'unclipped_tracks': results, 'pose_step_m': stats(np.linalg.norm(steps, axis=1)),
        'pose_step_vector_m': np.median(steps, axis=0).tolist(),
        'forecast_method': 'Fit each box coordinate to last 3 unclipped ground-truth observations; predict without refresh. This isolates geometry, not detector quality.',
        'forecast_summary': {h: {'center_error_px': stats([r['center_error_px'] for r in rows]),
            'iou': stats([r['iou'] for r in rows]), 'fraction_iou_ge_0_5': float(np.mean([r['iou'] >= .5 for r in rows]))}
            for h, rows in forecasts.items()}, 'forecasts': forecasts}


def pair_study(a_path, b_path, name, a, b):
    images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in [a_path, b_path]]
    images = [cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_AREA) for im in images]
    sift = cv2.SIFT_create(nfeatures=6500)
    ka, da = sift.detectAndCompute(images[0], None)
    kb, db = sift.detectAndCompute(images[1], None)
    matcher = cv2.BFMatcher()
    forward = matcher.knnMatch(da, db, k=2)
    reverse = {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(db, da, k=2) if m.distance < .7*n.distance}
    good = [m for m, n in forward if m.distance < .7*n.distance and reverse.get(m.trainIdx) == m.queryIdx]
    x = np.float32([ka[m.queryIdx].pt for m in good])*2
    y = np.float32([kb[m.trainIdx].pt for m in good])*2
    # Loose geometric sanity gate. Keep deviations up to 40 original pixels,
    # allowing substantial parallax while rejecting unrelated repeated textures.
    h0, _ = cv2.findHomography(x, y, cv2.RANSAC, 6)
    gate = np.linalg.norm(y-project(x, h0), axis=1) < 40
    x, y = x[gate], y[gate]
    rng = np.random.default_rng(20260917)
    order = rng.permutation(len(x)); ntrain = int(len(x)*.7)
    tr, te = order[:ntrain], order[ntrain:]
    translation = np.eye(3); translation[:2, 2] = np.median(y[tr]-x[tr], axis=0)
    sim, _ = cv2.estimateAffinePartial2D(x[tr], y[tr], method=cv2.RANSAC, ransacReprojThreshold=6)
    aff, _ = cv2.estimateAffine2D(x[tr], y[tr], method=cv2.RANSAC, ransacReprojThreshold=6)
    hom, _ = cv2.findHomography(x[tr], y[tr], cv2.RANSAC, 6)
    transforms = {'translation': translation, 'similarity': np.vstack([sim, [0,0,1]]),
                  'affine': np.vstack([aff, [0,0,1]]), 'homography': hom}
    errs = {k: stats(np.linalg.norm(project(x[te], h)-y[te], axis=1)) for k,h in transforms.items()}
    # Probe the model flow field at fixed image positions.
    grid = np.float32([[320,270],[1920,270],[3520,270],[320,1080],[1920,1080],[3520,1080],[320,1890],[1920,1890],[3520,1890]])
    result = {'sequence': name, 'from': a, 'to': b, 'mutual_ratio_matches': len(good),
        'loose_gate_matches': len(x), 'test_matches': len(te), 'heldout_residual_px': errs,
        'matrices': {k:h.tolist() for k,h in transforms.items()},
        'flow_probe_xy': grid.tolist(), 'homography_flow_px': (project(grid, hom)-grid).tolist(),
        'similarity_scale': float(np.sqrt(np.linalg.det(sim[:,:2])))}
    # Preserve actual feature coordinates for independent inspection/reanalysis.
    np.savez_compressed(OUT/f'matches-{name}-{a:03d}-{b:03d}.npz', x=x, y=y, train=tr, test=te)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4); cv2.setRNGSeed(17)
    result = {'reference_annotations': annotations(), 'image_motion_method':
        'Half-size SIFT, mutual Lowe-ratio 0.7 matches, loose 40-source-pixel homography gate, deterministic 70/30 fit/test split. RANSAC 6 source pixels. All reported distances in native source pixels.', 'pairs': []}
    work = [('reference', REF/'images', i, i+1) for i in range(24)]
    work += [('validation', VAL, i, i+1) for i in [5,8,9,20,40,60,80,100,120,140,160,180,200,220,240]]
    work += [('reference', REF/'images', 0, 12), ('reference', REF/'images', 8, 20),
             ('validation', VAL, 130, 140), ('validation', VAL, 140, 150)]
    for name, folder, a, b in work:
        row = pair_study(folder/f'frame_{a:06d}.png', folder/f'frame_{b:06d}.png', name, a,b)
        result['pairs'].append(row)
        print(name, a,b, 'test=',row['test_matches'], 'median px=',
              {k:round(v['median'],3) for k,v in row['heldout_residual_px'].items()},flush=True)
        (OUT/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
