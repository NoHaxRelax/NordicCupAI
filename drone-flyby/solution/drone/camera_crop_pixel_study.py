"""Actual-pixel crop comparisons; local tracking diagnostic, not detection.

Official reference boxes supply independent scoring positions. Validation
motion clocks receive only the two legal cropped views available at each step.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/drone-camera-pixels"
sys.path.insert(0, str(ROOT / "artifacts/drone-source-2026-09-17"))
from local_evaluator import Camera


def stats(v):
    return {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "max": float(np.max(v))} if v else None


def proj(p, h):
    return cv2.perspectiveTransform(np.float32(p)[None], h)[0]


def prior(sequence):
    j = json.loads((ROOT / "artifacts/drone-scene-analysis/measurements.json").read_text())
    hs = [np.array(r["matrices"]["homography"]) for r in j["pairs"]
          if r["sequence"] == sequence and r["to"] == r["from"]+1
          and not (sequence == "validation" and r["from"] in [8, 9])]
    h = np.median([h / np.linalg.det(h)**(1/3) for h in hs], axis=0)
    return h/h[2, 2]


def schedule(count, phase=0, overview=False):
    camera = Camera()
    rows = []
    centres = [960, 1920, 2880, 1920] if not phase else [2880, 1920, 960, 1920]
    for f in range(count):
        if f and not overview:
            camera.apply(1, centres[(f-1) % 4], 540)
        rows.append({"level": camera.resolution_level, "region": list(camera.source_region),
                     "scale": 4 if camera.resolution_level == 0 else 2})
    return rows


def render(gray, view):
    x1, y1, x2, y2 = view["region"]
    return cv2.resize(gray[y1:y2, x1:x2], (960, 540), interpolation=cv2.INTER_AREA)


def flow(a, b, xy, guess=None, win=31):
    lk = dict(winSize=(win, win), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .005))
    p = np.float32(xy).reshape(-1, 1, 2)
    if guess is None:
        nxt, ok, err = cv2.calcOpticalFlowPyrLK(a, b, p, None, **lk)
    else:
        nxt, ok, err = cv2.calcOpticalFlowPyrLK(a, b, p, np.float32(guess).reshape(-1, 1, 2),
                                              flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
    back, okback, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, p.copy(),
                                           flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
    return nxt.reshape(-1, 2), ok[:, 0] & okback[:, 0], err[:, 0], np.linalg.norm(back-p, axis=2)[:, 0]


def reference_localization():
    base = ROOT / "data/drone/reference/helsinki"
    labels, images = {}, {}
    for f in range(25):
        j = json.loads((base / "annotations" / f"frame_{f:06d}.json").read_text())
        labels[f] = {r["object_id"]: np.array(r["bbox"], float) for r in j["annotations"]}
        images[f] = cv2.imread(str(base / "images" / f"frame_{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
    h = prior("validation")
    overview = {"region": [0, 0, 3840, 2160], "scale": 4}
    plan = schedule(25)
    rows = []
    cache = {}
    def pixels(f, view):
        key = (f, tuple(view["region"]))
        if key not in cache:
            cache[key] = render(images[f], view)
        return cache[key]
    for a in range(1, 24):
        view = plan[a]
        candidates = [f for f in range(a+1, 25) if plan[f]["region"] == view["region"]]
        if not candidates:
            continue
        b = candidates[0]
        region = np.array(view["region"])
        names = []
        for name in sorted(set(labels[a]) & set(labels[b])):
            if all(np.all(labels[f][name][:2] > region[:2]) and np.all(labels[f][name][2:] < region[2:]) for f in [a, b]):
                names.append(name)
        if not names:
            continue
        truth0 = np.array([(labels[a][n][:2]+labels[a][n][2:])/2 for n in names])
        truth1 = np.array([(labels[b][n][:2]+labels[b][n][2:])/2 for n in names])
        guess = proj(truth0, np.linalg.matrix_power(h, b-a))
        for level, v in [(0, overview), (1, view)]:
            scale, origin = v["scale"], np.array(v["region"][:2])
            # OpenCV downsampling pixel centres: source=(view+.5)*scale-.5+origin.
            to_view = lambda x: (x-origin+.5)/scale-.5
            observed, ok, err, fb = flow(pixels(a, v), pixels(b, v), to_view(truth0), to_view(guess))
            source = (observed+.5)*scale-.5+origin
            for i, name in enumerate(names):
                accepted = bool(ok[i] and fb[i]*scale <= 2 and err[i] <= 18
                                and 0 <= observed[i, 0] < 960 and 0 <= observed[i, 1] < 540)
                rows.append({"class": name, "from": a, "to": b, "gap": b-a, "level": level,
                             "region": v["region"], "accepted": accepted,
                             "centre_error_source_px": float(np.linalg.norm(source[i]-truth1[i])),
                             "forward_backward_source_px": float(fb[i]*scale),
                             "input_box_wh": ((labels[a][name][2:]-labels[a][name][:2])/scale).tolist()})
    keys = {(r["class"], r["from"], r["to"]) for r in rows if r["level"] == 0 and r["accepted"]}
    keys &= {(r["class"], r["from"], r["to"]) for r in rows if r["level"] == 1 and r["accepted"]}
    summary = []
    for level in [0, 1]:
        rr = [r for r in rows if r["level"] == level]
        common = [r for r in rr if (r["class"], r["from"], r["to"]) in keys]
        summary.append({"level": level, "eligible_pairs": len(rr), "distinct_objects": len({r["class"] for r in rr}),
                        "accepted_pairs": sum(r["accepted"] for r in rr),
                        "accepted_centre_error_source_px": stats([r["centre_error_source_px"] for r in rr if r["accepted"]]),
                        "same_accepted_pairs_error_source_px": stats([r["centre_error_source_px"] for r in common]),
                        "input_box_width_px": stats([r["input_box_wh"][0] for r in rr]),
                        "input_box_height_px": stats([r["input_box_wh"][1] for r in rr])})
    result = {"summary": summary, "rows": rows,
              "method": "Official initial box centre seeds LK with transferred validation motion prior; actual future legal crop pixels refine position. Scored against official future box centre. Identical frame/object pairs at both resolutions, using L1 same-view revisit gaps2/4. Box labels only select visibility and score future positions, never initialize future flow.",
              "limitations": ["Local registration with known initial object and no identity search, not trained detector accuracy or a full causal tracking pipeline.",
                              "Object box centre may move relative to its surface as visible shape changes. Errors combine this effect with pixel tracking error.",
                              "Same31pixel LK window has different source footprint at each resolution; this tests a fixed pixel algorithm, not a pure information-theoretic resolution limit.",
                              "L0 is deliberately compared on the same sparse times and upper regions; actual L0 could observe every frame and has broader coverage."]}
    (OUT / "reference-localization.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Reference localization", json.dumps(summary), flush=True)


def intersection_pixels(a, va, b, vb):
    ra, rb = np.array(va["region"]), np.array(vb["region"])
    r = np.r_[np.maximum(ra[:2], rb[:2]), np.minimum(ra[2:], rb[2:])]
    scale = max(va["scale"], vb["scale"])
    out = []
    for im, view in [(a, va), (b, vb)]:
        local = (r-np.tile(view["region"][:2], 2)) // view["scale"]
        patch = im[local[1]:local[3], local[0]:local[2]]
        target = tuple(((r[2:]-r[:2])//scale).astype(int))
        if (patch.shape[1], patch.shape[0]) != target:
            patch = cv2.resize(patch, target, interpolation=cv2.INTER_AREA)
        out.append(patch.copy())
    return out, r, scale


def validation_clocks():
    h = prior("reference")
    plans = {"L0_full": schedule(245, overview=True), "L1_left_first": schedule(245), "L1_right_first": schedule(245, phase=1)}
    rows = {k: [] for k in plans}
    previous = {}
    for index, f in enumerate(range(5, 250)):
        gray = cv2.imread(str(ROOT / "data/drone/reconstructed-validation" / f"frame_{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
        for name, plan in plans.items():
            current = render(gray, plan[index])
            if index:
                started = time.perf_counter()
                (old, new), region, scale = intersection_pixels(previous[name], plan[index-1], current, plan[index])
                p = cv2.goodFeaturesToTrack(old, maxCorners=400, qualityLevel=.025, minDistance=10, blockSize=7)
                ratio, count = None, 0
                if p is not None:
                    xy = p.reshape(-1, 2)
                    nxt, ok, err, fb = flow(old, new, xy)
                    keep = (ok == 1) & (fb <= 1) & (err <= 18)
                    keep &= (nxt[:, 0] >= 12) & (nxt[:, 0] < new.shape[1]-12) & (nxt[:, 1] >= 12) & (nxt[:, 1] < new.shape[0]-12)
                    count = int(keep.sum())
                    if count >= 8:
                        source = (xy[keep]+.5)*scale-.5+region[:2]
                        expected = proj(source, h)-source
                        ratio = float(np.median((nxt[keep]-xy[keep])[:, 1]*scale/expected[:, 1]))
                tick = None if ratio is None else 0 if abs(ratio) < .25 else 2 if ratio > 1.5 else 1
                rows[name].append({"from": f-1, "to": f, "shared_source_region": region.tolist(), "comparison_scale": scale,
                                   "accepted_correspondences": count, "ratio": ratio, "motion_ticks": tick,
                                   "pixel_processing_ms": (time.perf_counter()-started)*1000})
            previous[name] = current
        if index % 60 == 0:
            print("Validation clock through", f, flush=True)
    summary = {}
    for name, rr in rows.items():
        summary[name] = {"transitions": len(rr), "motion_estimates": sum(r["motion_ticks"] is not None for r in rr),
                         "nonunit_or_unknown": [r for r in rr if r["motion_ticks"] != 1],
                         "pixel_processing_ms": stats([r["pixel_processing_ms"] for r in rr])}
    (OUT / "validation-clocks.json").write_text(json.dumps({"summary": summary, "rows": rows,
        "method": "One legal960x540 view per policy per frame. Fixed L0 overview vs L1left-centre-right-centre and reverse. Only source-region overlap in past and current delivered views is matched; known camera offsets remove pan. No target labels. Reference-only nominal motion prior. First saved frame5 treated as L0warm-up; subsequent ideal command delivery, no latency/skips.",
        "limitations": ["Zero/normal/double thresholds informed by previously observed anomalies, not a blind anomaly detector evaluation.",
                        "Fixed upper sweep does not establish accuracy for arbitrary camera policies, invisible targets or real network delays.",
                        "Clock estimates are coarse motion-step classifications, not precise metric speed or independent object localization.",
                        "Pixel processing times exclude image delivery/decoding, object inference and service overhead."]}, indent=2)+"\n")
    print("Validation clock summary", json.dumps(summary), flush=True)


if __name__ == "__main__":
    cv2.setNumThreads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    reference_localization()
    validation_clocks()
