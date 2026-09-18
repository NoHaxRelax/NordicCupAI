"""One-position forecasts using shared early background calibration, no jitter.

Odd-ID pixel tracks calibrate the model; even-ID tracks are held-out targets.
Only the target coordinate at calibration end initializes each forecast.
"""
import json
from pathlib import Path

import numpy as np

from benchmark import epipole, project, stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/drone-shared-motion"
CENTRE = np.array([1920., 1080.])
SCALE = 1000.
STARTS = [5, 60, 120, 210, 225]


def summarize(rows):
    if not rows:
        return None
    groups = {}
    for r in rows:
        groups.setdefault(r["track_id"], []).append(r)
    return {"tracks": len(groups), "future_positions": len(rows),
            "error_source_px": stats([r["error_source_px"] for r in rows]),
            "positions_within_5px_fraction": float(np.mean([r["error_source_px"] <= 5 for r in rows])),
            "whole_remainder_within_5px_fraction": float(np.mean([max(r["error_source_px"] for r in rr) <= 5 for rr in groups.values()])),
            "whole_remainder_within_20px_fraction": float(np.mean([max(r["error_source_px"] for r in rr) <= 20 for rr in groups.values()]))}


def fit_motion(background, start, end, clock, epi, parallel):
    e = np.r_[(epi-CENTRE)/SCALE, 1.]
    pairs = []
    local = []
    for tr in background:
        known = [(f, np.array(p)) for f, p in zip(tr["frames"], tr["xy"]) if start <= f <= end]
        for (f, p), (g, q) in zip(known, known[1:]):
            if g == f+1:
                pairs.append(((p-CENTRE)/SCALE, (q-CENTRE)/SCALE, clock[f]-clock[start], clock[g]-clock[start]))
        if len(known) >= 2 and known[-1][0] == end:
            t = np.array([clock[f]-clock[start] for f, p in known])
            if np.ptp(t) > 0:
                p = np.array([p for f, p in known])
                r = np.linalg.norm(p-epi, axis=1)
                slope = float(np.polyfit(t, 1/r, 1)[0])
                local.append((p[-1], slope))
    x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    t0 = np.array([p[2] for p in pairs]); t1 = np.array([p[3] for p in pairs])
    def matrix(parameters):
        q = np.r_[parameters, -np.dot(parameters, e[:2])] if parallel else parameters
        return np.outer(e, q)
    def residual(parameters):
        a = matrix(parameters)
        q = a[2]
        alpha = np.trace(a)
        beta = (t1-t0)/(1+t0*alpha)
        delta = beta*(np.c_[x, np.ones(len(x))]@q)
        pred = (x+delta[:, None]*e[:2])/(1+delta[:, None])
        return ((pred-y)*SCALE).ravel()
    # Cross-multiplication makes the constant-translation fit linear in q.
    # Robust reweighting uses actual reprojection errors; no SciPy dependency.
    delta = y-x
    design = t0[:, None, None]*delta[:, :, None]*e[None, None, :] + (t1-t0)[:, None, None]*(y-e[:2])[:, :, None]*np.c_[x, np.ones(len(x))][:, None, :]
    design = design.reshape(-1, 3)
    if parallel:
        design = design@np.array([[1., 0.], [0., 1.], [-e[0], -e[1]]])
    response = (-delta).ravel()
    weights = np.ones(len(response))
    for _ in range(20):
        parameters = np.linalg.lstsq(design*weights[:, None], response*weights, rcond=None)[0]
        errors = np.linalg.norm(residual(parameters).reshape(-1, 2), axis=1)
        weights = np.repeat((1+errors**2)**(-.25), 2)
    a = matrix(parameters)
    return a, local, {"background_pairs": len(pairs), "parameters": parameters.tolist(),
                       "matrix_A": a.tolist(), "trace_A": float(np.trace(a)),
                       "calibration_residual_source_px": stats(np.linalg.norm(residual(parameters).reshape(-1, 2), axis=1).tolist())}


def predict(model, point, anchor_tick, future_tick, a, local, epi):
    if model.startswith("plane_"):
        h = (np.eye(3)+future_tick*a)@np.linalg.inv(np.eye(3)+anchor_tick*a)
        return project(((point-CENTRE)/SCALE)[None], h)[0]*SCALE+CENTRE
    positions = np.array([p for p, slope in local])
    slopes = np.array([slope for p, slope in local])
    distance = np.linalg.norm(positions-point, axis=1)
    ids = np.argsort(distance)[:8]
    # Nearby background rates provide an empirical local-depth approximation.
    # All values come from past background frames, never target motion.
    weights = 1/np.maximum(distance[ids], 40.)**2
    slope = float(np.average(slopes[ids], weights=weights))
    v = point-epi; r = np.linalg.norm(v)
    den = 1/r+slope*(future_tick-anchor_tick)
    return epi+(v/r)/den


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads((ROOT / "artifacts/drone-validation-landmarks/tracks.json").read_text())["tracks"]
    background = [t for t in raw if int(t["id"].split("_")[-1]) % 2]
    targets = [t for t in raw if not int(t["id"].split("_")[-1]) % 2 and len(t["frames"]) >= 15]
    m = json.loads((ROOT / "artifacts/drone-scene-analysis/measurements.json").read_text())
    epi = np.median([epipole(r["matrices"]["homography"]) for r in m["pairs"] if r["sequence"] == "reference" and r["to"] == r["from"]+1], axis=0)
    cr = json.loads((ROOT / "artifacts/drone-camera-pixels/validation-clocks.json").read_text())["rows"]["L1_left_first"]
    observed = {5: 0.}
    for r in cr:
        observed[r["to"]] = observed[r["from"]]+(r["motion_ticks"] if r["motion_ticks"] is not None else 1)
    clocks = {"blind_frame_ticks": {f: float(f-5) for f in range(5, 250)}, "observed_legal_L1_clock": observed}
    rows, summaries, calibrations = [], [], []
    for intervals in [1, 3, 5]:
        for clock_name, clock in clocks.items():
            model_rows = {k: [] for k in ["plane_parallel", "plane_general", "local_background_rate"]}
            for start in STARTS:
                end = start+intervals
                bg = [t for t in background if t["start_frame"] == start]
                fits = {}
                for model in ["plane_parallel", "plane_general"]:
                    a, local, meta = fit_motion(bg, start, end, clock, epi, model == "plane_parallel")
                    fits[model] = (a, local)
                    calibrations.append({"start": start, "end": end, "clock": clock_name, "model": model, **meta})
                fits["local_background_rate"] = fits["plane_general"]
                for tr in targets:
                    if tr["start_frame"] != start or end not in tr["frames"]:
                        continue
                    i = tr["frames"].index(end)
                    point = np.array(tr["xy"][i])
                    for model, (a, local) in fits.items():
                        for f, truth in zip(tr["frames"][i+1:], tr["xy"][i+1:]):
                            pred = predict(model, point, clock[end]-clock[start], clock[f]-clock[start], a, local, epi)
                            row = {"track_id": tr["id"], "start_frame": start, "anchor_frame": end,
                                   "frame": f, "initial_object_observations": 1, "background_calibration_intervals": intervals,
                                   "clock": clock_name, "model": model, "prediction_xy": pred.tolist(), "truth_xy": truth,
                                   "error_source_px": float(np.linalg.norm(pred-truth))}
                            model_rows[model].append(row)
            for model, rr in model_rows.items():
                summaries.append({"calibration_intervals": intervals, "clock": clock_name, "model": model,
                                  "summary": summarize(rr), "by_start": {str(s): summarize([r for r in rr if r["start_frame"] == s]) for s in STARTS}})
                rows.extend(rr)
                s = summaries[-1]["summary"]
                print(intervals, clock_name, model, "median", round(s["error_source_px"]["median"], 2), "p90", round(s["error_source_px"]["p90"], 2), "whole5", round(s["whole_remainder_within_5px_fraction"], 3), flush=True)
    (OUT / "validation-shared-motion.json").write_text(json.dumps({"metadata": {
        "target_tracks": len(targets), "raw_background_tracks": len(background), "epipole": epi.tolist(),
        "epipole_source": "reference sequence only", "artificial_noise_added": False,
        "target_observations_per_prediction": 1,
        "calibration": "First1/3/5 motion intervals per group, odd-ID background points only. Models frozen after calibration. Targets supply only one position at calibration end.",
        "clock": "Blind variants use frame number and no later images. Observed variants use later images only for motion timing from legal L1 crop overlap; no subsequent target position updates enter either model.",
        "limitations": ["Pixel-derived pseudo-reference, not organizer labels; long stable target tracks selected for evaluation.",
                        "Early background calibration uses source-wide landmark positions; this does not restrict calibration to a single camera crop.",
                        "A target's single initial position is supplied exactly from the independent pixel tracker.",
                        "Background and target tracks are disjoint IDs, but neighboring points may share physical objects.",
                        "Future pixel tracks are used only for scoring. Endpoints may be tracker censoring, not physical object exit."]},
        "summary": summaries, "calibrations": calibrations}, indent=2)+"\n")
    with (OUT / "validation-shared-predictions.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row)+"\n")


if __name__ == "__main__":
    main()
