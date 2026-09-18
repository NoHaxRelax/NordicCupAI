"""Independent pixel-tracked validation landmarks, not organizer ground truth.

Forward/backward pyramidal Lucas-Kanade tracking supplies a noisy reference for
trajectory experiments. No homography or trajectory prediction gates these tracks.
Run: /opt/anaconda3/bin/python3 drone/validation_landmark_study.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/drone/reconstructed-validation"
OUT = ROOT / "artifacts/drone-validation-landmarks"
STARTS = [5, 60, 120, 210, 225]


def read(frame):
    im = cv2.imread(str(SOURCE / f"frame_{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_AREA)


def summary(values):
    return {"n": len(values), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "max": float(np.max(values))} if values else None


def run_start(start):
    old = read(start)
    mask = np.zeros(old.shape, dtype=np.uint8)
    mask[50:170, 90:1830] = 255
    points = cv2.goodFeaturesToTrack(old, maxCorners=160, qualityLevel=.035,
                                    minDistance=25, mask=mask, blockSize=7).reshape(-1, 2)
    # Limit concentration in a single textured tree or roof.
    chosen, bins = [], {}
    for point in points:
        cell = int(point[0] // 160)
        if bins.get(cell, 0) < 6:
            chosen.append(point)
            bins[cell] = bins.get(cell, 0) + 1
    points = np.float32(chosen).reshape(-1, 1, 2)
    tracks = [{"id": f"landmark_{start:03d}_{i:03d}", "start_frame": start,
               "frames": [start], "xy": [(p[0]*2).tolist()],
               "forward_backward_error_px": [], "lk_patch_error": [],
               "end_reason": None} for i, p in enumerate(points)]
    ids = np.arange(len(points))
    lk = dict(winSize=(31, 31), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .005))
    for frame in range(start + 1, min(start + 40, 249) + 1):
        new = read(frame)
        nxt, ok, err = cv2.calcOpticalFlowPyrLK(old, new, points, None, **lk)
        back, backok, _ = cv2.calcOpticalFlowPyrLK(new, old, nxt, None, **lk)
        fb = np.linalg.norm(back.reshape(-1, 2) - points.reshape(-1, 2), axis=1)*2
        xy = nxt.reshape(-1, 2)
        margins = (xy[:, 0] >= 16) & (xy[:, 0] < 1904) & (xy[:, 1] >= 16) & (xy[:, 1] < 1064)
        keep = (ok[:, 0] == 1) & (backok[:, 0] == 1) & (fb <= .8) & (err[:, 0] <= 18) & margins
        for i, trackid in enumerate(ids):
            track = tracks[trackid]
            if keep[i]:
                track["frames"].append(frame)
                track["xy"].append((xy[i]*2).tolist())
                track["forward_backward_error_px"].append(float(fb[i]))
                track["lk_patch_error"].append(float(err[i, 0]))
            else:
                # A flow estimate outside the margin is suggestive only, never
                # treated as independently confirmed disappearance from the frame.
                track["end_reason"] = "margin_or_possible_exit" if not margins[i] else "quality_check_failed"
                track["first_failed_frame"] = frame
        points = nxt[keep].reshape(-1, 1, 2)
        ids = ids[keep]
        old = new
        if not len(points):
            break
    for track in tracks:
        if track["end_reason"] is None:
            track["end_reason"] = "sequence_end" if track["frames"][-1] == 249 else "study_window_end"
    print(f"start {start}: {len(tracks)} seeded; {sum(len(t['frames']) >= 15 for t in tracks)} tracks >=15 frames", flush=True)
    return tracks


def gallery(tracks):
    # Longest track in different horizontal sectors per seed frame, audited by
    # looking at the same local landmark at the beginning, middle and end.
    selected = []
    for start in STARTS:
        rows = [t for t in tracks if t["start_frame"] == start and len(t["frames"]) >= 15]
        for sector in range(3):
            candidates = [t for t in rows if int(t["xy"][0][0] // 1280) == sector]
            if candidates:
                selected.append(max(candidates, key=lambda t: len(t["frames"])))
    thumb, header, pad = 240, 34, 12
    canvas = Image.new("RGB", (3*(thumb+pad)+pad, len(selected)*(thumb+header+pad)+pad), "#f4f4f4")
    draw = ImageDraw.Draw(canvas)
    for r, track in enumerate(selected):
        n = len(track["frames"])
        for col, index in enumerate([0, n//2, n-1]):
            frame, (x, y) = track["frames"][index], track["xy"][index]
            with Image.open(SOURCE / f"frame_{frame:06d}.png") as im:
                crop = im.convert("RGB").crop((round(x)-80, round(y)-80, round(x)+80, round(y)+80))
                crop = crop.resize((thumb, thumb), Image.Resampling.BICUBIC)
            d = ImageDraw.Draw(crop)
            c = thumb//2
            d.ellipse((c-5, c-5, c+5, c+5), outline="#ff2475", width=2)
            px, py = pad+col*(thumb+pad), pad+r*(thumb+header+pad)
            canvas.paste(crop, (px, py+header))
            draw.text((px, py), f"{track['id']} | f{frame}\n{n} observations | source 160px crop", fill="black")
    canvas.save(OUT / "landmark-audit.jpg", quality=92)
    (OUT / "audit-selection.json").write_text(json.dumps([t["id"] for t in selected], indent=2)+"\n")


def main():
    cv2.setNumThreads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = []
    for start in STARTS:
        tracks.extend(run_start(start))
        # Checkpoint makes tracks available to independent predictor experiments.
        (OUT / "tracks.json").write_text(json.dumps({"method": "Pixel-only bidirectional pyramidal Lucas-Kanade on half-resolution images. Source-pixel xy. No trajectory or homography gate. These are pseudo-reference landmarks, not official object labels.",
            "source_resolution": [3840, 2160], "tracking_resolution": [1920, 1080],
            "max_forward_backward_error_source_px": .8, "max_lk_patch_error": 18,
            "seed_region_source_xyxy": [180, 100, 3660, 340],
            "tracks": tracks}, indent=2)+"\n")
    long = [t for t in tracks if len(t["frames"]) >= 15]
    report = {"seeded_tracks": len(tracks), "tracks_at_least_15_frames": len(long),
              "retained_lengths": summary([len(t["frames"]) for t in long]),
              "retained_forward_backward_error_px": summary([v for t in long for v in t["forward_backward_error_px"]]),
              "limitations": ["Bidirectional flow consistency does not prove identity; repeated textures and gradual drift remain possible.",
                              "Selected long stable tracks overrepresent trackable, textured landmarks.",
                              "Forward/backward checking uses future pixels only to build the offline reference, never the causal predictor.",
                              "Track end is censored on failure, image margin, sequence end or study window; not confirmed full object lifetime.",
                              "These points cannot establish class accuracy, object box IoU, detection accuracy or independent sample count."]}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    gallery(tracks)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
