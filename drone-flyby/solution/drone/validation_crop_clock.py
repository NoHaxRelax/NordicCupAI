"""Check motion timing from one native 960x540 top-centre crop per frame.

Uses fresh pixel features on every pair, without object labels or lookahead.
The crop is a legal level-2-sized input; this is not a complete camera policy.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/drone-validation-landmarks"
REGION = [1440, 0, 2400, 540]


def read(frame):
    im = cv2.imread(str(ROOT / "data/drone/reconstructed-validation" / f"frame_{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return im[REGION[1]:REGION[3], REGION[0]:REGION[2]].copy()


def main():
    cv2.setNumThreads(4)
    data = json.loads((ROOT / "artifacts/drone-scene-analysis/measurements.json").read_text())
    matrices = [np.array(r["matrices"]["homography"]) for r in data["pairs"] if r["sequence"] == "reference" and r["to"] == r["from"]+1]
    prior = np.median([h / np.linalg.det(h)**(1/3) for h in matrices], axis=0)
    prior /= prior[2, 2]
    old = read(5)
    rows = []
    lk = dict(winSize=(31, 31), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .005))
    for frame in range(6, 250):
        new = read(frame)
        start = time.perf_counter()
        points = cv2.goodFeaturesToTrack(old, maxCorners=400, qualityLevel=.025, minDistance=12, blockSize=7)
        if points is None:
            rows.append({"from": frame-1, "to": frame, "accepted_correspondences": 0,
                         "median_measured_to_nominal_dy_ratio": None, "motion_ticks": None,
                         "pixel_processing_ms": (time.perf_counter()-start)*1000})
            old = new
            continue
        nxt, ok, err = cv2.calcOpticalFlowPyrLK(old, new, points, None, **lk)
        back, backok, _ = cv2.calcOpticalFlowPyrLK(new, old, nxt, None, **lk)
        x, y = points.reshape(-1, 2), nxt.reshape(-1, 2)
        fb = np.linalg.norm(back.reshape(-1, 2)-x, axis=1)
        good = (ok[:, 0] == 1) & (backok[:, 0] == 1) & (fb <= 1) & (err[:, 0] <= 18)
        good &= (y[:, 0] >= 16) & (y[:, 0] < 944) & (y[:, 1] >= 16) & (y[:, 1] < 524)
        x, y = x[good], y[good]
        source = x+np.float32(REGION[:2])
        if len(x) >= 8:
            expected = cv2.perspectiveTransform(source[None], prior)[0]-source
            ratio = float(np.median((y[:, 1]-x[:, 1])/expected[:, 1]))
        else:
            ratio = None
        tick = None if ratio is None else 0 if abs(ratio) < .25 else 2 if ratio > 1.5 else 1
        rows.append({"from": frame-1, "to": frame, "accepted_correspondences": len(x),
                     "median_measured_to_nominal_dy_ratio": ratio, "motion_ticks": tick,
                     "pixel_processing_ms": (time.perf_counter()-start)*1000})
        old = new
    report = {"source_region_xyxy": REGION, "input_shape_wh": [960, 540],
              "prior_sequence": "reference only", "transitions": len(rows),
              "minimum_accepted_correspondences": min(r["accepted_correspondences"] for r in rows),
              "nonunit_ticks": [r for r in rows if r["motion_ticks"] != 1],
              "pixel_processing_ms_median": float(np.median([r["pixel_processing_ms"] for r in rows])),
              "pixel_processing_ms_p90": float(np.percentile([r["pixel_processing_ms"] for r in rows], 90)),
              "limitations": ["Exploratory thresholds target observed zero/double movement; not independent evidence for arbitrary timing faults.",
                              "Camera remains fixed at this source region and level. Changing crops, missed input frames and initial zoom transitions were not tested. Water-dominated views cause two insufficient-match transitions in this run.",
                              "A crop clock does not validate object localization or acquisition across the scene.",
                              "Timing covers feature processing on decoded crops on this machine, not image delivery, neural inference, full-frame decoding or API latency."],
              "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crop-clock.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
