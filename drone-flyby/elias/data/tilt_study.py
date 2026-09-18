"""Tilt study: how far may a sprite be rotated in the image plane before it stops looking real?

Four measurements, each a subcommand. Run from drone-flyby/ with the nordic-drone interpreter.

  python elias/data/tilt_study.py camera
      Fits a pinhole camera (focal length, forward pitch, in-plane yaw) to the helsinki box-centre
      tracks, using the known altitude (600 m) and step (13.89 m per frame). Writes
      elias/out/tilt_camera.json and the plot elias/out/tilt_camera.png (dy against y, dx against x,
      off-nadir angle map).

  python elias/data/tilt_study.py rotation
      For pairs of sprites from elias/sprites/bank.json, rotates the first over 0..359 degrees (with a
      small scale and shift search) and reports the best masked NCC on gray, the angle and the mask IoU.
      Pair kinds: same track in neighbouring frames (ceiling), same track top against bottom of the
      frame (view angle only), different tracks (heading change), helsinki against validation, and
      different classes (the floor that the 360 degree search reaches by chance).
      For two tracks the pair is the one sprite from each with the SMALLEST difference in off-nadir
      angle (bank field off_nadir_deg), so that a heading change is not mixed up with a change of
      view angle. Every row also carries the turn that the lean of a vertical structure predicts:
      view_azimuth_deg of B minus that of A (counter-clockwise, the sense of the rotation search).
      Writes elias/out/tilt_rotation.json and contact sheets elias/out/tilt_pairs_*.png.

  python elias/data/tilt_study.py lean
      Does the best rotation between two tracks follow the lean prediction? For every class with two
      or more tracks in one scene, takes ALL sprite pairs across two tracks whose off-nadir angles
      differ by at most 3 degrees and compares the best NCC angle with view_azimuth_deg(B) minus
      view_azimuth_deg(A). A class whose appearance is ruled by its lean (a tower) shows a small
      error on every pair; a class ruled by its own heading (a tank) shows the same offset on every
      pair of the same two tracks, whatever the lean says. Writes elias/out/tilt_lean.json.

  python elias/data/tilt_study.py sun
      Fits a brightness plane inside each sprite mask and compares the direction of the bright side
      between instances whose heading differs by the angle found above, lists the mean sprite gray per
      class and scene, and looks for cast shadows in a band around every sprite. Needs the rotation
      step first. Writes elias/out/tilt_sun.json.

Masked NCC used here: a = (gray_a - mean_a) inside mask_a and 0 outside, b likewise;
ncc = sum(a * b) / sqrt(sum(a^2) * sum(b^2)). Pixels that only one mask covers add energy to the
denominator and nothing to the numerator, so a silhouette mismatch lowers the score.
"""

import argparse
import glob
import json
import math
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ELIAS = os.path.dirname(HERE)
ROOT = os.path.dirname(ELIAS)
OUT = os.path.join(ELIAS, "out")
BANK = os.path.join(ELIAS, "sprites", "bank.json")

W, H = 3840, 2160
ALTITUDE_M = 600.0
STEP_M = 13.8888889


# ----------------------------------------------------------------------------- camera

def load_tracks(scene, reviewed_only=True):
    """Box-centre tracks {track_id: [(frame, cx, cy, w, h), ...]} from boxes that do not touch the border."""
    tracks = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "src", scene, "annotations", "*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        prov = data.get("provenance")
        for i, ann in enumerate(data["annotations"]):
            if prov is not None:
                if reviewed_only and prov[i]["review_status"] != "directly_reviewed_track":
                    continue
                track = prov[i]["track_id"]
            else:
                track = ann["object_id"]
            x1, y1, x2, y2 = ann["bbox"]
            if x1 <= 1 or y1 <= 1 or x2 >= W - 2 or y2 >= H - 2:
                continue
            tracks.setdefault(track, []).append(
                (data["frame"], 0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1))
    return {k: v for k, v in tracks.items() if len(v) >= 3}


def project(f, pitch, yaw, ground_x, ground_y, height=ALTITUDE_M):
    """Pixel position of a ground point (metres; x right, y forward of the point below the camera).

    The optical axis is pitched forward by `pitch` from straight down, so the top of the image looks
    ahead. `yaw` rotates the image about the principal point (travel direction not exactly image-up).
    """
    depth = ground_y * math.sin(pitch) + height * math.cos(pitch)
    u = f * ground_x / depth
    v = f * (-ground_y * math.cos(pitch) + height * math.sin(pitch)) / depth
    c, s = math.cos(yaw), math.sin(yaw)
    return W / 2 + c * u - s * v, H / 2 + s * u + c * v


def off_nadir_deg(f, pitch, yaw, px, py):
    """Angle between the viewing ray through pixel (px, py) and straight down, in degrees."""
    c, s = math.cos(yaw), math.sin(yaw)
    du, dv = px - W / 2, py - H / 2
    u = c * du + s * dv
    v = -s * du + c * dv
    down = v * np.sin(pitch) + f * np.cos(pitch)
    norm = np.sqrt(u * u + v * v + f * f)
    return np.degrees(np.arccos(np.clip(down / norm, -1, 1)))


def fit_camera(tracks, step_over_height, fixed_f=None, robust=False):
    """Least squares over f, pitch, yaw (and step/height when f is fixed) plus one ground point per track.

    robust=True uses a soft L1 loss (scale 3 px) so that a few misplaced pseudo-label boxes do not
    drag the fit; the reported rms is then taken over the residuals below 10 px.
    """
    from scipy.optimize import least_squares

    names = sorted(tracks)
    obs = []
    for ti, name in enumerate(names):
        for frame, cx, cy, _, _ in tracks[name]:
            obs.append((ti, frame, cx, cy))
    obs = np.array(obs, dtype=float)

    def unpack(p):
        if fixed_f is None:
            return p[0], p[1], p[2], step_over_height, p[3:]
        return fixed_f, p[0], p[1], p[2], p[3:]

    def residuals(p):
        f, pitch, yaw, r, ground = unpack(p)
        res = []
        for ti, frame, cx, cy in obs:
            gx = ground[2 * int(ti)]
            gy = ground[2 * int(ti) + 1] - frame * r * ALTITUDE_M
            px, py = project(f, pitch, yaw, gx, gy)
            res.extend((px - cx, py - cy))
        return np.array(res)

    ground0 = []
    for name in names:
        frame, cx, cy, _, _ = tracks[name][0]
        scale = ALTITUDE_M / 3000.0
        ground0.extend(((cx - W / 2) * scale, -(cy - H / 2) * scale + frame * step_over_height * ALTITUDE_M))
    if fixed_f is None:
        p0 = np.array([3000.0, 0.2, 0.0] + ground0)
    else:
        p0 = np.array([0.2, 0.0, step_over_height] + ground0)
    if robust:
        sol = least_squares(residuals, p0, method="trf", loss="soft_l1", f_scale=3.0)
        res = residuals(sol.x)
        inliers = np.abs(res) < 10
        rms = float(np.sqrt(np.mean(res[inliers] ** 2)))
    else:
        sol = least_squares(residuals, p0, method="lm")
        inliers = np.ones(len(sol.fun), bool)
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    f, pitch, yaw, r, _ = unpack(sol.x)
    return {"inlier_fraction": float(inliers.mean()),"f_px": float(f), "pitch_deg": float(np.degrees(pitch)), "yaw_deg": float(np.degrees(yaw)),
            "step_over_height": float(r), "rms_px": rms, "n_tracks": len(names), "n_points": int(len(obs))}


def displacement_table(tracks):
    """Per-frame displacement (x, y, dx, dy) between consecutive frames of each track."""
    rows = []
    for name, pts in tracks.items():
        pts = sorted(pts)
        for a, b in zip(pts[:-1], pts[1:]):
            if b[0] - a[0] == 1:
                rows.append((0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2]), b[1] - a[1], b[2] - a[2]))
    return np.array(rows)


def cmd_camera(args):
    os.makedirs(OUT, exist_ok=True)
    report = {}
    tracks = load_tracks("helsinki")
    disp = displacement_table(tracks)
    # Straight-line summaries that need no camera model.
    ky, cy0 = np.polyfit(disp[:, 1], disp[:, 3], 1)
    kx, cx0 = np.polyfit(disp[:, 0], disp[:, 2], 1)
    sq_slope, sq_icpt = np.polyfit(disp[:, 1], np.sqrt(disp[:, 3]), 1)
    report["helsinki_linear"] = {
        "dy_at_y0": float(cy0), "dy_at_y2159": float(cy0 + ky * 2159), "dy_slope_per_px": float(ky),
        "dx_at_x0": float(cx0), "dx_at_x3839": float(cx0 + kx * 3839), "dx_slope_per_px": float(kx),
        "dx_zero_at_x": float(-cx0 / kx),
        "sqrt_dy_zero_at_y (horizon row)": float(-sq_icpt / sq_slope),
        "n_displacements": int(len(disp)),
    }
    fit = fit_camera(tracks, STEP_M / ALTITUDE_M)
    f, pitch, yaw = fit["f_px"], math.radians(fit["pitch_deg"]), math.radians(fit["yaw_deg"])
    nadir = project(f, pitch, yaw, 0.0, 0.0)
    fit["nadir_pixel"] = [float(nadir[0]), float(nadir[1])]
    fit["hfov_deg"] = float(2 * np.degrees(np.arctan(W / 2 / f)))
    fit["vfov_deg"] = float(2 * np.degrees(np.arctan(H / 2 / f)))
    fit["ground_m_per_px_at_nadir"] = float(ALTITUDE_M / f)
    probes = {"top_centre": (W / 2, 0), "centre": (W / 2, H / 2), "bottom_centre": (W / 2, H - 1),
              "top_left": (0, 0), "top_right": (W - 1, 0), "bottom_left": (0, H - 1),
              "bottom_right": (W - 1, H - 1), "left_middle": (0, H / 2), "right_middle": (W - 1, H / 2)}
    fit["off_nadir_deg"] = {k: float(off_nadir_deg(f, pitch, yaw, *v)) for k, v in probes.items()}
    rows = np.arange(0, H, 270)
    fit["off_nadir_deg_by_row_at_x1920"] = {int(y): float(off_nadir_deg(f, pitch, yaw, W / 2, y)) for y in rows}
    report["helsinki_fit"] = fit

    # Validation: altitude and step are unknown there, so keep f from helsinki and fit step/height.
    vtracks = load_tracks("validation")
    vdisp = displacement_table(vtracks)
    vky, vcy0 = np.polyfit(vdisp[:, 1], vdisp[:, 3], 1)
    vkx, vcx0 = np.polyfit(vdisp[:, 0], vdisp[:, 2], 1)
    report["validation_linear"] = {
        "dy_at_y0": float(vcy0), "dy_at_y2159": float(vcy0 + vky * 2159),
        "dx_at_x0": float(vcx0), "dx_at_x3839": float(vcx0 + vkx * 3839),
        "dx_zero_at_x": float(-vcx0 / vkx), "n_displacements": int(len(vdisp)),
        "note": "participant pseudo-label boxes, directly_reviewed_track only; noisier than helsinki"}
    # The joint fit is not usable on validation (hand-placed boxes jitter by tens of pixels and a robust
    # fit keeps too few inliers), so use the closed form of the same camera model on a trimmed line:
    # dy(v) = f r (cos p + sin p v / f)^2  gives  tan p = slope * f / (2 dy_centre)  at the image centre.
    keep = np.ones(len(vdisp), bool)
    for _ in range(5):
        slope, icpt = np.polyfit(vdisp[keep, 1], vdisp[keep, 3], 1)
        keep = np.abs(vdisp[:, 3] - (icpt + slope * vdisp[:, 1])) < 6
    dy_centre = icpt + slope * H / 2
    vpitch = math.atan(slope * f / (2 * dy_centre))
    hk, hc = np.polyfit(disp[:, 1], disp[:, 3], 1)
    vfit = {"method": "trimmed line through dy(y), closed form, f fixed to the helsinki value",
            "kept_displacements": int(keep.sum()), "of": int(len(vdisp)),
            "dy_at_y0": float(icpt), "dy_at_y2159": float(icpt + slope * 2159),
            "pitch_deg": float(np.degrees(vpitch)),
            "step_over_height": float(dy_centre / (f * math.cos(vpitch) ** 2)),
            "nadir_row": float(H / 2 + f * math.tan(vpitch)),
            "same_closed_form_on_helsinki_pitch_deg": float(np.degrees(math.atan(hk * f / (2 * (hc + hk * H / 2)))))}
    report["validation_fit"] = vfit

    with open(os.path.join(OUT, "tilt_camera.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))
    plot_camera(disp, vdisp, fit, os.path.join(OUT, "tilt_camera.png"))


def plot_camera(disp, vdisp, fit, path):
    """Three panels drawn with cv2 only: dy(y), dx(x), and the off-nadir angle over the frame."""
    pw, ph, pad = 560, 400, 50
    canvas = np.full((ph + 2 * pad, 3 * (pw + pad) + pad, 3), 255, np.uint8)

    def panel(ox, xs, ys, xs2, ys2, xr, yr, title, model=None):
        cv2.rectangle(canvas, (ox, pad), (ox + pw, pad + ph), (0, 0, 0), 1)
        def to_px(x, y):
            return (int(ox + (x - xr[0]) / (xr[1] - xr[0]) * pw), int(pad + ph - (y - yr[0]) / (yr[1] - yr[0]) * ph))
        for x, y in zip(xs2, ys2):
            cv2.circle(canvas, to_px(x, y), 2, (200, 160, 60), -1)
        for x, y in zip(xs, ys):
            cv2.circle(canvas, to_px(x, y), 2, (40, 40, 200), -1)
        if model is not None:
            pts = [to_px(x, model(x)) for x in np.linspace(xr[0], xr[1], 50)]
            for p, q in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, p, q, (0, 0, 0), 1)
        cv2.putText(canvas, title, (ox, pad - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for frac in (0, 0.5, 1):
            xv = xr[0] + frac * (xr[1] - xr[0]); yv = yr[0] + frac * (yr[1] - yr[0])
            cv2.putText(canvas, f"{xv:.0f}", (int(ox + frac * pw) - 12, pad + ph + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(canvas, f"{yv:.0f}", (ox - 34, int(pad + ph - frac * ph) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    f, pitch, yaw = fit["f_px"], math.radians(fit["pitch_deg"]), math.radians(fit["yaw_deg"])
    r = fit["step_over_height"]
    def dy_model(y):
        v = y - H / 2
        return f * r * (math.cos(pitch) + math.sin(pitch) * v / f) ** 2
    panel(pad, disp[:, 1], disp[:, 3], vdisp[:, 1], vdisp[:, 3], (0, H), (40, 100),
          "dy per frame against y (red helsinki, blue validation, line = fitted camera at x=1920)", dy_model)
    panel(2 * pad + pw, disp[:, 0], disp[:, 2], vdisp[:, 0], vdisp[:, 2], (0, W), (-20, 20),
          "dx per frame against x")
    ox = 3 * pad + 2 * pw
    ys, xs = np.mgrid[0:ph, 0:pw]
    ang = off_nadir_deg(f, pitch, yaw, xs / pw * W, ys / ph * H)
    shade = np.clip(ang / 45.0 * 255, 0, 255).astype(np.uint8)
    canvas[pad:pad + ph, ox:ox + pw] = cv2.applyColorMap(shade, cv2.COLORMAP_VIRIDIS)
    for level in (5, 10, 15, 20, 25, 30, 35, 40):
        contour = (np.abs(ang - level) < 0.15)
        canvas[pad:pad + ph, ox:ox + pw][contour] = (255, 255, 255)
        yy, xx = np.nonzero(contour)
        if len(xx):
            k = np.argmin(np.abs(xx - pw // 2) + 0.01 * yy)
            cv2.putText(canvas, str(level), (ox + int(xx[k]), pad + int(yy[k]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    nx, ny = fit["nadir_pixel"]
    cv2.drawMarker(canvas, (int(ox + nx / W * pw), int(pad + ny / H * ph)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(canvas, "off-nadir angle in degrees over the 4K frame (cross = nadir)", (ox, pad - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(path, canvas)
    print("wrote", path)


# ----------------------------------------------------------------------------- shared sprite helpers

def load_bank():
    with open(BANK) as fh:
        bank = json.load(fh)
    sprites = []
    for entry in bank["sprites"]:
        img = cv2.imread(os.path.join(ELIAS, "sprites", entry["file"]), cv2.IMREAD_UNCHANGED)
        entry = dict(entry)
        entry["bgra"] = img
        sprites.append(entry)
    return sprites


def zero_mean(bgra):
    gray = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = bgra[:, :, 3] > 0
    out = np.zeros_like(gray)
    out[mask] = gray[mask] - gray[mask].mean()
    return out, mask.astype(np.float32)


def rotate_scale(img, angle_deg, scale, nearest=False):
    """Rotate counter-clockwise by angle_deg and scale about the centre on a canvas that holds the result."""
    h, w = img.shape[:2]
    side = int(math.ceil(math.hypot(h, w) * scale)) + 4
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
    mat[0, 2] += side / 2 - w / 2
    mat[1, 2] += side / 2 - h / 2
    return cv2.warpAffine(img, mat, (side, side), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def best_rotation(a_bgra, b_bgra, angles=range(0, 360, 2), scales=(0.85, 0.92, 1.0, 1.08, 1.17), max_shift=6):
    """Best masked NCC of rotated/scaled A against B. Returns a dict with ncc, angle, scale, iou and the curve."""
    a0, ma0 = zero_mean(a_bgra)
    b0, mb0 = zero_mean(b_bgra)
    eb = float((b0 ** 2).sum())
    best = {"ncc": -2.0}
    curve = {}
    for angle in angles:
        top = -2.0
        for scale in scales:
            a = rotate_scale(a0 * 1.0, angle, scale)
            ma = rotate_scale(ma0, angle, scale, nearest=True)
            a = a * ma
            ea = float((a ** 2).sum())
            if ea <= 0 or eb <= 0:
                continue
            side = a.shape[0]
            pad_y = max(0, (side - b0.shape[0] + 1) // 2) + max_shift
            pad_x = max(0, (side - b0.shape[1] + 1) // 2) + max_shift
            bp = cv2.copyMakeBorder(b0, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
            corr = cv2.matchTemplate(bp, a, cv2.TM_CCORR)
            # Only shifts that keep the two mask centres within max_shift of each other.
            cy, cx = (corr.shape[0] - 1) / 2.0, (corr.shape[1] - 1) / 2.0
            y0, y1 = int(max(0, cy - max_shift)), int(min(corr.shape[0], cy + max_shift + 1))
            x0, x1 = int(max(0, cx - max_shift)), int(min(corr.shape[1], cx + max_shift + 1))
            window = corr[y0:y1, x0:x1]
            k = np.unravel_index(np.argmax(window), window.shape)
            ncc = float(window[k] / math.sqrt(ea * eb))
            if ncc > top:
                top = ncc
            if ncc > best["ncc"]:
                mbp = cv2.copyMakeBorder(mb0, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
                yy, xx = y0 + k[0], x0 + k[1]
                sub = mbp[yy:yy + side, xx:xx + side]
                inter = float((sub * ma).sum())
                union = float(mbp.sum() + ma.sum() - inter)
                best = {"ncc": ncc, "angle": int(angle), "scale": float(scale), "iou": inter / union,
                        "shift": [int(xx - pad_x), int(yy - pad_y)]}
        curve[int(angle)] = top
    best["curve"] = curve
    # How peaked is the curve: best NCC at angles at least 30 degrees away from the winner.
    far = [v for ang, v in curve.items() if min((ang - best["angle"]) % 360, (best["angle"] - ang) % 360) >= 30]
    best["ncc_best_elsewhere"] = float(max(far)) if far else None
    best["ncc_at_0"] = float(curve[0])
    return best


# ----------------------------------------------------------------------------- rotation study

def checker(h, w, cell=8):
    ys, xs = np.mgrid[0:h, 0:w]
    board = (((ys // cell) + (xs // cell)) % 2).astype(np.uint8)
    return np.dstack([np.where(board, 200, 150).astype(np.uint8)] * 3)


def composite(bgra, size, zoom):
    """Sprite centred on a checkerboard tile of size x size, upscaled by zoom with nearest neighbour."""
    h, w = bgra.shape[:2]
    tile = checker(size, size)
    s = min(1.0, (size - 2) / max(h, w))
    if s < 1.0:
        bgra = cv2.resize(bgra, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        bgra[:, :, 3] = np.where(bgra[:, :, 3] > 127, 255, 0)
        h, w = bgra.shape[:2]
    y0, x0 = (size - h) // 2, (size - w) // 2
    alpha = bgra[:, :, 3:4] > 0
    tile[y0:y0 + h, x0:x0 + w] = np.where(alpha, bgra[:, :, :3], tile[y0:y0 + h, x0:x0 + w])
    return cv2.resize(tile, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)


def rotated_sprite(bgra, angle, scale):
    color = rotate_scale(bgra[:, :, :3], angle, scale)
    alpha = rotate_scale(bgra[:, :, 3], angle, scale, nearest=True)
    return np.dstack([color, alpha])


def pick_pairs(sprites):
    """Choose the pairs to test. Each pair is (kind, a, b)."""
    by_track = {}
    for s in sprites:
        by_track.setdefault((s["class_name"], s["source"], s["track"]), []).append(s)
    for lst in by_track.values():
        lst.sort(key=lambda s: s["frame"])
    pairs = []
    # Same track: neighbouring frames (ceiling) and first against last (view angle only).
    for key, lst in sorted(by_track.items()):
        if len(lst) >= 2:
            gaps = [(lst[i + 1]["frame"] - lst[i]["frame"], i) for i in range(len(lst) - 1)]
            _, i = min(gaps)
            pairs.append(("same_track_near", lst[i], lst[i + 1]))
            if len(lst) >= 3 or gaps[0][0] > 3:
                pairs.append(("same_track_top_bottom", lst[0], lst[-1]))
    # Different tracks of the same class: the two sprites whose off-nadir angles are closest, so the
    # comparison measures the heading change and not a change of view angle.
    classes = sorted({k[0] for k in by_track})
    mids = {k: lst[len(lst) // 2] for k, lst in by_track.items()}
    for cls in classes:
        keys = [k for k in sorted(by_track) if k[0] == cls]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                kind = "cross_scene" if keys[i][1] != keys[j][1] else "cross_track"
                a, b = min(((a, b) for a in by_track[keys[i]] for b in by_track[keys[j]]),
                           key=lambda ab: (abs(ab[0]["off_nadir_deg"] - ab[1]["off_nadir_deg"]), ab[0]["frame"], ab[1]["frame"]))
                pairs.append((kind, a, b))
    # Floor: one sprite per class against the most similar sized sprite of a different class.
    reps = [mids[[k for k in sorted(mids) if k[0] == cls][0]] for cls in classes]
    for a in reps:
        others = [b for b in reps if b["class_name"] != a["class_name"]]
        area = a["size"][0] * a["size"][1]
        b = min(others, key=lambda o: abs(math.log(o["size"][0] * o["size"][1] / area)))
        pairs.append(("different_class", a, b))
    return pairs


def cmd_rotation(args):
    os.makedirs(OUT, exist_ok=True)
    sprites = load_bank()
    pairs = pick_pairs(sprites)
    print(len(pairs), "pairs")
    results = []
    tiles = {}
    for kind, a, b in pairs:
        res = best_rotation(a["bgra"], b["bgra"], angles=range(0, 360, args.angle_step))
        curve = res.pop("curve")
        predicted = (b["view_azimuth_deg"] - a["view_azimuth_deg"]) % 360
        error = min((res["angle"] - predicted) % 360, (predicted - res["angle"]) % 360)
        row = {"kind": kind, "class_a": a["class_name"], "class_b": b["class_name"],
               "a": a["id"], "b": b["id"], "a_center_src": a["center_src"], "b_center_src": b["center_src"],
               "off_nadir_a": a["off_nadir_deg"], "off_nadir_b": b["off_nadir_deg"],
               "predicted_lean_turn_deg": round(predicted, 1), "angle_minus_predicted_deg": round(error, 1),
               "ncc_at_predicted": round(curve[int(round(predicted / args.angle_step)) * args.angle_step % 360], 3),
               **res, "curve_every_30": {k: round(v, 3) for k, v in curve.items() if k % 30 == 0}}
        results.append(row)
        print(f"{kind:22s} {a['class_name']:16s} {a['id']:44s} -> {b['id']:44s} ncc={res['ncc']:.3f} "
              f"angle={res['angle']:3d} scale={res['scale']:.2f} iou={res['iou']:.2f} "
              f"ncc0={res['ncc_at_0']:.3f} elsewhere={res['ncc_best_elsewhere']:.3f} "
              f"offnadir {a['off_nadir_deg']:.1f}/{b['off_nadir_deg']:.1f} lean_turn={predicted:.0f} "
              f"ncc_at_lean_turn={row['ncc_at_predicted']:.3f}")
        size = 96
        ta = composite(a["bgra"], size, 2)
        tr = composite(rotated_sprite(a["bgra"], res["angle"], res["scale"]), size, 2)
        tb = composite(b["bgra"], size, 2)
        label = np.zeros((size * 2, 300, 3), np.uint8)
        lines = [f"{kind}", f"{a['class_name']}" + ("" if a["class_name"] == b["class_name"] else f" vs {b['class_name']}"),
                 f"A {a['source'][:3]} {a['track'][:22]} f{a['frame']}", f"B {b['source'][:3]} {b['track'][:22]} f{b['frame']}",
                 f"NCC {res['ncc']:.2f} at {res['angle']} deg", f"IoU {res['iou']:.2f}  NCC(0) {res['ncc_at_0']:.2f}",
                 f"off-nadir {a['off_nadir_deg']:.0f} / {b['off_nadir_deg']:.0f}  lean turn {predicted:.0f}"]
        for i, text in enumerate(lines):
            cv2.putText(label, text, (4, 20 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1)
        tiles.setdefault(kind, []).append(np.hstack([label, ta, tr, tb]))
    with open(os.path.join(OUT, "tilt_rotation.json"), "w") as fh:
        json.dump({"angle_step": args.angle_step, "pairs": results}, fh, indent=1)
    header_h = 26
    for kind, rows in tiles.items():
        per_col = 7
        cols = []
        for i in range(0, len(rows), per_col):
            chunk = rows[i:i + per_col]
            while len(chunk) < per_col and len(rows) > per_col:
                chunk.append(np.zeros_like(rows[0]))
            head = np.zeros((header_h, rows[0].shape[1], 3), np.uint8)
            cv2.putText(head, "pair                                                   A            A rotated to best        B",
                        (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cols.append(np.vstack([head] + chunk))
        # Sheets are split so that each PNG stays readable: at most two columns per file.
        for j in range(0, len(cols), 2):
            sheet = np.hstack(cols[j:j + 2])
            path = os.path.join(OUT, f"tilt_pairs_{kind}_{j // 2}.png")
            cv2.imwrite(path, sheet)
            print("wrote", path, sheet.shape)


# ----------------------------------------------------------------------------- lean check

def circular_error(a, b):
    """Smallest absolute difference between two angles in degrees."""
    d = (a - b) % 360
    return min(d, 360 - d)


def cmd_lean(args):
    sprites = load_bank()
    by_track = {}
    for s in sprites:
        by_track.setdefault((s["class_name"], s["source"], s["track"]), []).append(s)
    rows = []
    keys = sorted(by_track)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i][:2] != keys[j][:2]:
                continue  # same class and same scene only
            for a in by_track[keys[i]]:
                for b in by_track[keys[j]]:
                    if abs(a["off_nadir_deg"] - b["off_nadir_deg"]) > args.max_off_nadir_difference:
                        continue
                    res = best_rotation(a["bgra"], b["bgra"], angles=range(0, 360, args.angle_step))
                    predicted = (b["view_azimuth_deg"] - a["view_azimuth_deg"]) % 360
                    rows.append({"class": a["class_name"], "a": a["id"], "b": b["id"],
                                 "off_nadir_a": a["off_nadir_deg"], "off_nadir_b": b["off_nadir_deg"],
                                 "best_angle": res["angle"], "ncc": round(res["ncc"], 3), "iou": round(res["iou"], 3),
                                 "ncc_best_elsewhere": round(res["ncc_best_elsewhere"], 3),
                                 "predicted_lean_turn_deg": round(predicted, 1),
                                 "error_deg": round(circular_error(res["angle"], predicted), 1)})
                    r = rows[-1]
                    print(f"{r['class']:15s} {a['track'][-12:]:>12s} f{a['frame']:<4d}-> {b['track'][-12:]:>12s} f{b['frame']:<4d} "
                          f"offnadir {r['off_nadir_a']:4.1f}/{r['off_nadir_b']:4.1f}  best {r['best_angle']:3d}  lean turn "
                          f"{r['predicted_lean_turn_deg']:5.1f}  error {r['error_deg']:5.1f}  ncc {r['ncc']:.2f} (elsewhere {r['ncc_best_elsewhere']:.2f})")
    summary = {}
    for cls in sorted({r["class"] for r in rows}):
        errs = [r["error_deg"] for r in rows if r["class"] == cls]
        summary[cls] = {"pairs": len(errs), "median_error_deg": float(np.median(errs)), "max_error_deg": float(max(errs)),
                        "within_20_deg": int(sum(e <= 20 for e in errs)),
                        "median_ncc": float(np.median([r["ncc"] for r in rows if r["class"] == cls]))}
    print("per class (an angle drawn at random has a median error of 90 and lands within 20 degrees in 11 % of draws):")
    for cls, v in summary.items():
        print(f"  {cls:15s} {v}")
    with open(os.path.join(OUT, "tilt_lean.json"), "w") as fh:
        json.dump({"max_off_nadir_difference": args.max_off_nadir_difference, "angle_step": args.angle_step,
                   "summary": summary, "pairs": rows}, fh, indent=1)


# ----------------------------------------------------------------------------- sun

def brightness_plane(bgra):
    """Least-squares plane gray = c + gx*x + gy*y inside the mask. Returns direction of the bright side
    (degrees, image convention: 0 = right, 90 = up) and the relative strength (gray range over the sprite / mean)."""
    gray = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float64)
    ys, xs = np.nonzero(bgra[:, :, 3] > 0)
    vals = gray[ys, xs]
    A = np.column_stack([np.ones_like(xs), xs - xs.mean(), ys - ys.mean()])
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    gx, gy = coef[1], coef[2]
    extent = math.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    strength = math.hypot(gx, gy) * extent / max(vals.mean(), 1.0)
    explained = 1.0 - ((vals - A @ coef) ** 2).sum() / max(((vals - vals.mean()) ** 2).sum(), 1e-9)
    return math.degrees(math.atan2(-gy, gx)) % 360, float(strength), float(explained), float(vals.mean())


def cmd_sun(args):
    os.makedirs(OUT, exist_ok=True)
    sprites = {s["id"]: s for s in load_bank()}
    with open(os.path.join(OUT, "tilt_rotation.json")) as fh:
        pairs = json.load(fh)["pairs"]
    per_sprite = {}
    for sid, s in sprites.items():
        direction, strength, explained, mean = brightness_plane(s["bgra"])
        per_sprite[sid] = {"class_name": s["class_name"], "source": s["source"], "track": s["track"],
                           "frame": s["frame"], "bright_side_deg": round(direction, 1),
                           "relative_strength": round(strength, 3), "plane_r2": round(explained, 3),
                           "mean_gray": round(mean, 1)}
    rows = []
    for p in pairs:
        if p["kind"] not in ("cross_track", "cross_scene", "same_track_top_bottom"):
            continue
        a, b = per_sprite[p["a"]], per_sprite[p["b"]]
        delta = (b["bright_side_deg"] - a["bright_side_deg"]) % 360
        def circ(x):
            x = x % 360
            return min(x, 360 - x)
        rows.append({"kind": p["kind"], "class": p["class_a"], "a": p["a"], "b": p["b"],
                     "heading_change_deg": p["angle"], "ncc": round(p["ncc"], 3),
                     "bright_side_change_deg": round(delta, 1),
                     "error_if_shading_turns_with_object": round(circ(delta - p["angle"]), 1),
                     "error_if_shading_fixed_in_image": round(circ(delta), 1),
                     "strength_a": a["relative_strength"], "strength_b": b["relative_strength"],
                     "r2_a": a["plane_r2"], "r2_b": b["plane_r2"]})
    for r in rows:
        print(f"{r['kind']:22s} {r['class']:16s} heading {r['heading_change_deg']:4d}  ncc {r['ncc']:.2f}  bright side "
              f"{r['bright_side_change_deg']:6.1f}  err(turns with object) {r['error_if_shading_turns_with_object']:6.1f}  "
              f"err(fixed) {r['error_if_shading_fixed_in_image']:6.1f}  strength {r['strength_a']:.2f}/{r['strength_b']:.2f} "
              f"r2 {r['r2_a']:.2f}/{r['r2_b']:.2f}")
    # Mean gray of the sprites per class and scene: the simplest lighting difference between the scenes.
    table = {}
    for v in per_sprite.values():
        table.setdefault(v["class_name"], {}).setdefault(v["source"], []).append(v["mean_gray"])
    gray = {cls: {scene: round(float(np.mean(vals)), 1) for scene, vals in scenes.items()} for cls, scenes in table.items()}
    print("mean sprite gray per class and scene:")
    for cls, scenes in sorted(gray.items()):
        print(f"  {cls:16s} {scenes}")

    # Cast shadows: mean gray of a 6 px band just outside the mask in 8 direction sectors, relative to
    # the median of a wider band further out. A cast shadow would make one sector clearly darker, and
    # under a fixed sun the dark sector would point the same way for every object.
    shadows = []
    frames = {}
    for sid, sprite in sprites.items():
        key = (sprite["source"], sprite["frame"])
        if key not in frames:
            frames[key] = cv2.cvtColor(cv2.imread(os.path.join(ROOT, "src", key[0], "images", f"frame_{key[1]:06d}.png")), cv2.COLOR_BGR2GRAY)
        frame = frames[key]
        ox, oy = sprite["origin_src"]
        h, w = sprite["bgra"].shape[:2]
        pad = 26
        x1, y1, x2, y2 = ox - pad, oy - pad, ox + w + pad, oy + h + pad
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            continue
        mask = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
        mask[pad:pad + h, pad:pad + w] = sprite["bgra"][:, :, 3] > 0
        near = cv2.dilate(mask, np.ones((17, 17), np.uint8)) - cv2.dilate(mask, np.ones((5, 5), np.uint8))
        far = cv2.dilate(mask, np.ones((49, 49), np.uint8)) - cv2.dilate(mask, np.ones((29, 29), np.uint8))
        patch = frame[y1:y2, x1:x2].astype(np.float32)
        reference = float(np.median(patch[far > 0]))
        spread = float(np.std(patch[far > 0]))
        ys, xs = np.nonzero(near)
        my, mx = np.nonzero(mask)
        angles = np.degrees(np.arctan2(-(ys - my.mean()), xs - mx.mean())) % 360
        sector = (angles // 45).astype(int)
        ratios = [float(patch[ys[sector == k], xs[sector == k]].mean() / max(reference, 1.0)) if (sector == k).any() else 1.0 for k in range(8)]
        k = int(np.argmin(ratios))
        shadows.append({"id": sid, "class_name": sprite["class_name"], "source": sprite["source"],
                        "darkest_sector_centre_deg": k * 45 + 22.5, "darkest_ratio": round(ratios[k], 3),
                        "terrain_std_over_median": round(spread / max(reference, 1.0), 3)})
    print("cast shadow check: darkest 45 degree sector of the 6 px band outside the mask, relative to the terrain further out")
    for scene in ("helsinki", "validation"):
        rows_scene = [r for r in shadows if r["source"] == scene]
        ratios = np.array([r["darkest_ratio"] for r in rows_scene])
        quiet = [r for r in rows_scene if r["terrain_std_over_median"] < 0.08]
        print(f"  {scene}: n {len(rows_scene)}, median darkest ratio {np.median(ratios):.3f}, below 0.80: {(ratios < 0.8).sum()}, "
              f"on quiet terrain (std/median < 0.08): n {len(quiet)}, median darkest ratio "
              f"{np.median([r['darkest_ratio'] for r in quiet]) if quiet else float('nan'):.3f}")
        dark = [r for r in rows_scene if r["darkest_ratio"] < 0.8]
        hist = {}
        for r in dark:
            hist[r["darkest_sector_centre_deg"]] = hist.get(r["darkest_sector_centre_deg"], 0) + 1
        print(f"    direction histogram of sectors below 0.80: {dict(sorted(hist.items()))}")
    with open(os.path.join(OUT, "tilt_sun.json"), "w") as fh:
        json.dump({"per_sprite": per_sprite, "pairs": rows, "mean_gray_per_class_scene": gray, "shadow_sectors": shadows}, fh, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("camera", help="fit the camera tilt from the helsinki tracks")
    rot = sub.add_parser("rotation", help="best in-plane rotation NCC between sprite pairs")
    rot.add_argument("--angle-step", type=int, default=2, help="angle step in degrees (default 2)")
    lean = sub.add_parser("lean", help="does the best rotation between two tracks follow the lean prediction?")
    lean.add_argument("--angle-step", type=int, default=2)
    lean.add_argument("--max-off-nadir-difference", type=float, default=3.0)
    sub.add_parser("sun", help="compare the bright side of sprites across headings")
    args = parser.parse_args()
    {"camera": cmd_camera, "rotation": cmd_rotation, "lean": cmd_lean, "sun": cmd_sun}[args.cmd](args)


if __name__ == "__main__":
    main()
