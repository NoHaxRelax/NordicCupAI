"""Cut extra sprites from the 4K frames with GrabCut and write elias/sprites/bank.json.

Oscar's bank has no helicopter, one large_tower and only 3 or 4 sprites for several classes. This
module cuts more, from the organiser boxes on helsinki and from the directly reviewed tracks on
validation, and keeps only what passed a visual review on a contact sheet.

Run from drone-flyby/ with the nordic-drone interpreter. Two steps:

  python elias/data/sprites.py cut
      For every class picks frames spread over each track (helsinki: up to 8 per class; validation:
      3 per directly_reviewed_track, 4 when the class has two tracks, 8 when it has one, plus
      EXTRA_FRAMES) and runs GrabCut per candidate: variant "z" on a 3x enlarged crop, variant "n" at
      native size and, for the classes in RECIPES, variant "r" (refined, see below). With
      --classes a,b only those classes are redone. Writes BGRA PNGs to
      elias/sprites/_candidates/<class>/, elias/sprites/_candidates/candidates.json and one review
      sheet per class, elias/out/sprites_review_<class>.png, with one row per candidate: source
      crop with the annotation box, every variant over magenta, and a mask outline on the crop
      (variant r when it exists, else z). LOOK at these sheets and write elias/sprites/review.json.
      The whole cut took 345 s on CPU (208 candidates); the large classes (hangar, large_launcher) are
      the slow ones, so redo single classes with --classes.

  python elias/data/sprites.py zoom --cls tank --out some.png [--ids 0,3,5] [--variants z,n,r]
      Large side-by-side view (crop, mask outline, checkerboard, magenta) of chosen candidates.

  python elias/data/sprites.py bank
      Reads elias/sprites/review.json:
        {"discard": {"<candidate id>": "reason", ...}, "use_variant": {"<candidate id>": "n" or "r", ...}}
      (the older key "use_native": [ids] is still read). Every candidate that is not discarded is
      kept, in variant z unless use_variant says otherwise.
      Copies the kept PNGs to elias/sprites/<class>/ and writes elias/sprites/bank.json in Oscar's
      schema, the contact sheets elias/out/sprites_bank_checker.png and sprites_bank_magenta.png
      (one row per class, every kept sprite) and elias/out/sprites_bank_counts.json (per class:
      candidates, discards with reasons, kept per scene, per variant and per split train/dev).

GrabCut set-up (cv2.grabCut, GC_INIT_WITH_MASK): box interior = probable foreground, a 2 px gap =
probable background, a ring outside of width max(10, 0.3 x longer side) = sure background.
Variant z gives GrabCut a 3x bicubic enlargement of the crop (2x when the box is longer than 130 px)
and reduces the mask back by area averaging and a 0.5 threshold. At native size GrabCut dropped thin
or low-contrast parts (jet tail, front of the large launcher) or returned nothing at all on the grainy
helsinki terrain; enlarged it keeps them, but it leaks into the terrain more often on validation.
Neither variant wins everywhere, so both are cut and the review picks. Sprite pixels are always the
original source pixels. Afterwards: largest connected component plus any component of at least 10 %
of its area, 3x3 closing, holes smaller than max(3 px, 0.5 % of the mask) filled. Masks stay binary
(alpha 0 or 255), nothing is feathered. On validation the pseudo-label boxes are often offset, so
when the foreground runs along a box side the box is grown on that side by 20 % and GrabCut is run
again (at most 3 times); box_in_sprite always refers to the ORIGINAL annotation box.

Variant r, per class (RECIPES), added after a reviewer found parts missing or terrain stuck:
  bg_pad        large_launcher, medium_launcher, medium_plane. The object stands on a uniform pad that
                is larger than the box, so plain GrabCut returns the pad. Pixels within a Lab distance
                of the BOX median colour become sure background (only when at least half of the box
                is that uniform).
  thin_dark     tank, medium_launcher. Black-hat line detector for the gun barrel and the side arms:
                line-shaped groups that start within 3 px of the mask are added, bridged to the mask
                and followed outwards at half the threshold.
  rim           hangar. GrabCut follows the black roof only. The grey end wall stubs at the two ends
                of the open mouth and the light band of the arch front are added from the convex
                hull minus the mask, with a colour test against the ground seen through the door.
  mirror        medium_plane. GrabCut loses the light olive left wing tip. The mask is united with its
                reflection about the fuselage axis (found by search); reflected-in pixels must
                differ in colour from the ground around the plane.
  no_close, colour_holes   medium_launcher. No 3x3 closing (it bridges the 1 px gaps between the
                launch tubes) and a hole is filled only when its colour is closer to the object
                than to the ring.
Tried for the helsinki helicopter and removed again because no version gave the whole rotor: a
dark-blob foreground seed with the box as probable background, and a mask built without GrabCut
from the dark blob plus black-hat and Lab-b top-hat line groups.

Tracks that are left out (EXCLUDED_TRACKS), frames that are swapped (REPLACED_FRAMES), a track
whose boxes are shifted (TRACK_BOX_MARGIN) and extra frames outside Oscar's development split
(EXTRA_FRAMES) are listed with their reasons below and copied into bank.json.

Bank entry fields beyond Oscar's: track, label_status, split (validation frames 100..180 are Oscar's
development split and are marked "dev"), box_src, origin_src, center_src, grabcut_variant,
grabcut_refinement (the notes of variant r), off_nadir_deg and view_azimuth_deg (from the camera
fitted in tilt_study.py; the azimuth is the direction in which tall objects lean in the image,
counter-clockwise from image right). class_id follows dtos.OBJECT_CLASSES as SPEC.md asks (Oscar's
bank numbers the classes alphabetically), so key on class_name when the two banks are merged.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ELIAS = os.path.dirname(HERE)
ROOT = os.path.dirname(ELIAS)
sys.path.insert(0, ROOT)
from dtos import OBJECT_CLASSES  # noqa: E402

CLASSES = [c for c in OBJECT_CLASSES if c != "background"][:16]
SPRITES = os.path.join(ELIAS, "sprites")
CANDIDATES = os.path.join(SPRITES, "_candidates")
OUT = os.path.join(ELIAS, "out")
W, H = 3840, 2160
MAGENTA = (255, 0, 255)

# Camera fitted on helsinki by tilt_study.py camera (focal length px, forward pitch degrees).
CAMERA_F = 3140.0
CAMERA_PITCH_DEG = 18.25

# Tracks that are not used, with the reason.
EXCLUDED_TRACKS = {
    "mine-roller-a-005-009": "shows a gun barrel and a boxy hull: looks like the tank, not the organiser's mine roller "
                             "(Oscar flagged the same track)",
    "large-launcher-c2-080-105": "pseudo-label error: a 50 px dark vehicle with two forked roller arms (the build of "
                                 "validation mine-roller-b), 0.37 x the size of the launcher in track d-194-227 and in "
                                 "helsinki (sqrt box area 51 against 136 and 129); altitude is fixed, so not a launcher",
    "large-launcher-c-047-078": "the label box lies on a sand wedge next to the same small dark vehicle as track c2; "
                                "no launcher in or near the box",
}

# A chosen frame that is swapped for a neighbour of the same track, with the reason.
REPLACED_FRAMES = {
    ("small-tower-c-155-164", 156): (157, "at f156 the box top is 16 px from the frame edge and the roof pokes out above "
                                          "it, so the sure-background ring above is clipped and GrabCut loses the roof"),
}

# Pseudo-label boxes that are offset in the same way along a whole track: pixels added on the
# (left, top, right, bottom) side of the box before GrabCut. box_in_sprite keeps the original box.
TRACK_BOX_MARGIN = {
    "small-tower-c-155-164": ((0, 8, 0, 0), "the box sits about 6 px too low in every frame: the dark green roof pokes out "
                                            "above it, lands in the sure-background ring and GrabCut then drops the roof"),
}

# Extra frames per track, added to the evenly spread ones. Validation frames 100..180 are Oscar's
# development split; these tracks had almost all their chosen frames in it.
EXTRA_FRAMES = {
    "medium-launcher-a-092-123": [93, 94, 95, 97, 98, 99],
    "helicopter-b-early-077-104": [82, 85, 91, 93, 99],
}

# Third GrabCut variant "r" (refined), cut only for the classes listed here. Each key switches on one
# step; see refined_steps() and the module docstring.
RECIPES = {
    "large_launcher": {"bg_pad": 16.0},
    "hangar": {"rim": True, "enlarge": False},
    "tank": {"thin_dark": True},
    "medium_launcher": {"bg_pad": 30.0, "thin_dark": True, "no_close": True, "colour_holes": True},
    "medium_plane": {"bg_pad": 14.0, "mirror": True},
}


# ----------------------------------------------------------------------------- annotations

def load_instances(scene):
    """List of dicts (class_name, track, frame, box) for boxes that lie fully inside the frame."""
    items = []
    for path in sorted(glob.glob(os.path.join(ROOT, "src", scene, "annotations", "*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        prov = data.get("provenance")
        for i, ann in enumerate(data["annotations"]):
            if scene == "validation":
                if prov[i]["review_status"] != "directly_reviewed_track":
                    continue
                track = prov[i]["track_id"]
                label_status = "directly_reviewed_track"
            else:
                track = ann["object_id"]
                label_status = "organiser"
            if track in EXCLUDED_TRACKS:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in ann["bbox"]]
            if x1 < 3 or y1 < 3 or x2 > W - 4 or y2 > H - 4:
                continue  # the object is cut by the frame border
            items.append({"class_name": ann["object_id"], "track": track, "frame": int(data["frame"]),
                          "box": [x1, y1, x2, y2], "scene": scene, "label_status": label_status})
    return items


def spread(items, count):
    """Pick `count` items evenly spread over a list that is sorted by frame."""
    if len(items) <= count:
        return list(items)
    idx = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def choose_candidates(per_class_helsinki, per_track_validation):
    chosen = []
    hel = load_instances("helsinki")
    for cls in CLASSES:
        track = sorted([it for it in hel if it["class_name"] == cls], key=lambda it: it["frame"])
        chosen.extend(spread(track, per_class_helsinki))
    val = load_instances("validation")
    for cls in CLASSES:
        tracks = sorted({it["track"] for it in val if it["class_name"] == cls})
        for tr in tracks:
            seq = sorted([it for it in val if it["track"] == tr], key=lambda it: it["frame"])
            # One track only: take twice as many frames, otherwise the class cannot reach six sprites.
            if len(tracks) == 1:
                count = 2 * per_track_validation
            elif len(tracks) == 2:
                count = per_track_validation
            else:
                count = max(3, per_track_validation - 1)
            picked = spread(seq, count)
            by_frame = {it["frame"]: it for it in seq}
            for i, it in enumerate(picked):
                swap = REPLACED_FRAMES.get((tr, it["frame"]))
                if swap and swap[0] in by_frame:
                    picked[i] = by_frame[swap[0]]
            for frame in EXTRA_FRAMES.get(tr, []):
                if frame in by_frame and by_frame[frame] not in picked:
                    picked.append(by_frame[frame])
            chosen.extend(sorted(picked, key=lambda it: it["frame"]))
    return chosen


# ----------------------------------------------------------------------------- GrabCut

def ring_statistics(crop, init_mask):
    """Lab image of the crop, median Lab colour of the sure-background ring, and the Lab distance to it."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = init_mask == cv2.GC_BGD
    median = np.median(lab[ring], axis=0)
    return lab, median, np.linalg.norm(lab - median, axis=2)


def seed_mask(crop, mask, recipe):
    """Recipe seeds written into the GrabCut initialisation mask (in place). Returns notes for the info dict."""
    notes = {}
    if "bg_pad" in recipe:
        # The object stands on a uniform pad that is larger than the box, so the pad is the dominant
        # colour INSIDE the box (the ring further out holds buildings and trees and says nothing about
        # it). Every pixel within a small Lab distance of the box median is sure background. Skipped
        # when less than half of the box is that uniform, because then the median is not a pad.
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        inside = mask == cv2.GC_PR_FGD
        median = np.median(lab[inside], axis=0)
        dist = np.linalg.norm(lab - median, axis=2)
        uniform = float((dist[inside] <= recipe["bg_pad"]).mean())
        notes["pad_fraction_of_box"] = round(uniform, 2)
        if uniform >= 0.5:
            pad = cv2.morphologyEx((dist <= recipe["bg_pad"]).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask[pad > 0] = cv2.GC_BGD
    return notes


def grabcut_once(image, box, enlarge, recipe=None):
    """One GrabCut run.

    Returns (binary mask uint8 over the crop, crop origin x, crop origin y, crop, initialisation mask
    before the recipe seeds, pad mask = pixels a seed made sure background, notes).
    """
    x1, y1, x2, y2 = box
    ring = max(10, int(round(0.3 * max(x2 - x1, y2 - y1))))
    gap = 2
    cx1, cy1 = max(0, x1 - gap - ring), max(0, y1 - gap - ring)
    cx2, cy2 = min(W, x2 + gap + ring), min(H, y2 + gap + ring)
    crop = image[cy1:cy2, cx1:cx2].copy()
    mask = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    mask[max(0, y1 - gap - cy1):y2 + gap - cy1, max(0, x1 - gap - cx1):x2 + gap - cx1] = cv2.GC_PR_BGD
    mask[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = cv2.GC_PR_FGD
    init = mask.copy()
    notes = seed_mask(crop, mask, recipe) if recipe else {}
    # Pixels that a seed turned into sure background (the pad). Later steps never add them back.
    init_pad = ((mask == cv2.GC_BGD) & (init != cv2.GC_BGD)).astype(np.uint8)
    zoom = 1
    if enlarge:
        zoom = 3 if max(x2 - x1, y2 - y1) <= 130 else 2
    work = crop if zoom == 1 else cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    work_mask = mask if zoom == 1 else cv2.resize(mask, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    # GrabCut starts its colour models with k-means on OpenCV's global random generator, so without
    # a fixed seed the mask of a candidate depends on how many candidates were cut before it in the
    # same process (measured: 95 of 169 kept sprites changed pixels between two runs). The review is
    # tied to the exact pixels, so every GrabCut call starts from the same seed.
    cv2.setRNGSeed(20260919)
    try:
        cv2.grabCut(work, work_mask, None, bgd, fgd, 8 if zoom == 1 else 6, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros(crop.shape[:2], np.uint8), cx1, cy1, crop, init, init_pad, notes
    fg = ((work_mask == cv2.GC_FGD) | (work_mask == cv2.GC_PR_FGD)).astype(np.float32)
    if zoom != 1:
        fg = cv2.resize(fg, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
    return (fg > 0.5).astype(np.uint8), cx1, cy1, crop, init, init_pad, notes


def clean_mask(fg, crop=None, init=None, close=True):
    """Largest component(s), 3x3 closing, small holes filled. Binary in, binary out.

    With crop and init given (variant r) a hole is filled only when its mean colour is closer to the
    mask's mean colour than to the ring median, so terrain seen between launch tubes stays out.
    close=False skips the 3x3 closing, which bridges one pixel gaps on objects under 30 px.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if count <= 1:
        return fg * 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= 0.10 * areas.max()]
    out = np.isin(labels, keep).astype(np.uint8)
    before_close = out.copy()
    if close:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    lab = median = inside_mean = None
    if crop is not None and init is not None and out.sum() > 0:
        lab, median, _ = ring_statistics(crop, init)
        inside_mean = lab[before_close > 0].mean(axis=0)
        # Undo the closing where it added terrain-coloured pixels.
        added = (out > 0) & (before_close == 0)
        terrain_like = np.linalg.norm(lab - median, axis=2) < np.linalg.norm(lab - inside_mean, axis=2)
        out[added & terrain_like] = 0
    # Holes: background components that do not touch the crop border.
    inv = (1 - out).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    limit = max(3, 0.005 * out.sum())
    h, w = out.shape
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches or area > limit:
            continue
        if lab is not None:
            colour = lab[labels == i].mean(axis=0)
            if np.linalg.norm(colour - median) < np.linalg.norm(colour - inside_mean):
                continue
        out[labels == i] = 1
    return out


# ----------------------------------------------------------------------------- refinement steps of variant r

def extend_line(response, out, part, centre, low):
    """Follow an accepted line group outwards (away from the mask centre) while the response stays above `low`.

    The far half of a gun barrel is fainter than the near half, so the first threshold finds only a
    stub. Walks along the fitted line axis in 1 px steps, looks at the 3x3 neighbourhood of each step,
    stops after 2 steps without a response, and never goes further than twice the group's length.
    """
    ys, xs = np.nonzero(part)
    vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(np.column_stack([xs, ys]).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()]
    t = (xs - x0) * vx + (ys - y0) * vy
    # The end to extend is the one further away from the mask centre.
    ends = [(t.max(), 1.0), (t.min(), -1.0)]
    t_end, sign = max(ends, key=lambda e: math.hypot(x0 + e[0] * vx - centre[0], y0 + e[0] * vy - centre[1]))
    misses, added = 0, 0
    for step in range(1, int(2 * (t.max() - t.min() + 1)) + 1):
        x, y = int(round(x0 + (t_end + sign * step) * vx)), int(round(y0 + (t_end + sign * step) * vy))
        if not (1 <= x < out.shape[1] - 1 and 1 <= y < out.shape[0] - 1):
            break
        if response[y - 1:y + 2, x - 1:x + 2].max() > low:
            out[y, x] = 1
            added += 1
            misses = 0
        else:
            misses += 1
            if misses >= 2:
                break
    return added


def thin_parts(crop, fg, init, box_in_crop, kernel=5, grow=0.25, max_width=4, reach=3, passes=3, percentile=98.0):
    """Thin dark lines (gun barrel, side arm) that start at the mask.

    Black-hat on gray with a 5x5 ellipse answers to dark lines narrower than 5 px. Threshold: the
    larger of 14 gray levels and the 98th percentile of the response over the sure-background ring.
    Line pixels outside the mask are grouped (after a 3x3 closing that joins a dashed barrel) and a
    group is added when it comes within `reach` px of the mask, is at least 5 px long, at most
    max_width px wide and at least 2.5 times longer than wide. Blobs of dark terrain next to the hull
    fail the shape test. Only the box grown by `grow` x its longer side is searched.
    reach: the black-hat answer fades right next to a dark hull, so a barrel starts 1 to 2 px away from
    the mask; the gap is bridged with a 1 px line so the sprite stays in one piece. passes: how often
    the touch test is repeated with the grown mask, so that the dashes of a broken barrel chain
    outwards. Each accepted group is then followed outwards at half the threshold (extend_line).
    """
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    response = cv2.morphologyEx(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.MORPH_BLACKHAT, shape)
    level = max(14.0, float(np.percentile(response[init == cv2.GC_BGD], percentile)))
    x1, y1, x2, y2 = box_in_crop
    margin = int(grow * max(x2 - x1, y2 - y1))
    allowed = np.zeros(fg.shape, np.uint8)
    allowed[max(0, y1 - margin):y2 + margin, max(0, x1 - margin):x2 + margin] = 1
    outside = (fg == 0).astype(np.uint8)
    lines = ((response > level) & (allowed > 0)).astype(np.uint8) & outside
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) & outside
    count, labels = cv2.connectedComponents(lines, connectivity=8)
    line_like = []
    for i in range(1, count):
        ys, xs = np.nonzero(labels == i)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        long_side, short_side = max(side_a, side_b) + 1, min(side_a, side_b) + 1
        if long_side >= 5 and short_side <= max_width and long_side >= 2.5 * short_side:
            line_like.append(i)
    mys, mxs = np.nonzero(fg)
    centre = (float(mxs.mean()), float(mys.mean()))
    out = fg.copy()
    for _ in range(passes):
        near_mask = cv2.dilate(out, np.ones((2 * reach + 1, 2 * reach + 1), np.uint8))
        for i in list(line_like):
            part = labels == i
            if (part & (near_mask > 0)).any():
                # Bridge from the group's pixel nearest to the mask to the mask pixel nearest to it.
                pys, pxs = np.nonzero(part)
                oys, oxs = np.nonzero(out)
                d2 = (pxs[:, None] - oxs[None, :]) ** 2 + (pys[:, None] - oys[None, :]) ** 2
                k = np.unravel_index(int(np.argmin(d2)), d2.shape)
                cv2.line(out, (int(pxs[k[0]]), int(pys[k[0]])), (int(oxs[k[1]]), int(oys[k[1]])), 1, 1)
                out[part] = 1
                extend_line(response, out, part, centre, 0.5 * level)
                line_like.remove(i)
    return out, int(out.sum() - fg.sum()), round(level, 1)


def hangar_rim(crop, fg):
    """Add the grey end wall stubs at both ends of the open hangar mouth.

    GrabCut follows the black roof and leaves a concave bite at the open end. The roof footprint is
    convex, so the bite is hull minus mask. Its middle is ground seen through the door (or a parked
    jet) and must stay out; the two end wall stubs sit where the bite meets the chord between the two
    hull corners of the mouth. Pixels of the bite within 0.22 x chord length of a chord end are added
    when they are clearly brighter than the roof and connected to the mask. The front face of the arch
    is a light band of 2 to 3 px along the bite, so bite pixels within 3 px of the roof are candidates
    too. Every candidate must also differ by more than 18 in Lab from the median colour of the deep
    bite (more than 6 px from the roof), which is the ground seen through the door.
    """
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return fg, 0
    contour = max(contours, key=cv2.contourArea)
    hull_points = cv2.convexHull(contour)
    hull = np.zeros_like(fg)
    cv2.fillConvexPoly(hull, hull_points, 1)
    bite = ((hull > 0) & (fg == 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bite, connectivity=8)
    if count <= 1:
        return fg, 0
    mouth = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[mouth, cv2.CC_STAT_AREA] < 0.01 * fg.sum():
        return fg, 0
    # The chord is the longest hull edge that borders the mouth (a point just inside it lies in the mouth).
    pts = hull_points[:, 0, :].astype(np.float32)
    ys, xs = np.nonzero(fg)
    centre = np.array([xs.mean(), ys.mean()], np.float32)
    chord = None
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        mid = (a + b) / 2.0
        length = float(np.hypot(*(b - a)))
        inward = centre - mid
        probe = mid + 2.0 * inward / max(float(np.hypot(*inward)), 1e-6)
        px, py = int(round(float(probe[0]))), int(round(float(probe[1])))
        if 0 <= py < fg.shape[0] and 0 <= px < fg.shape[1] and labels[py, px] == mouth:
            if chord is None or length > chord[2]:
                chord = (a, b, length)
    if chord is None:
        return fg, 0
    ys, xs = np.nonzero(labels == mouth)
    to_a = np.hypot(xs - chord[0][0], ys - chord[0][1])
    to_b = np.hypot(xs - chord[1][0], ys - chord[1][1])
    near = np.minimum(to_a, to_b) <= 0.22 * chord[2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    roof = float(np.percentile(gray[fg > 0], 95))
    bright = gray[ys, xs] > roof + 25
    to_roof = cv2.distanceTransform((fg == 0).astype(np.uint8), cv2.DIST_L2, 3)[ys, xs]
    deep = to_roof > 6
    off_ground = np.ones(len(xs), bool)
    band = np.zeros(len(xs), bool)
    if deep.sum() >= 30:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        ground = np.median(lab[ys[deep], xs[deep]], axis=0)
        off_ground = np.linalg.norm(lab[ys, xs] - ground, axis=1) > 18
        band = to_roof <= 3
    keep = (near | band) & bright & off_ground
    add = np.zeros_like(fg)
    add[ys[keep], xs[keep]] = 1
    merged = ((fg > 0) | (add > 0)).astype(np.uint8)
    _, labels2 = cv2.connectedComponents(merged, connectivity=8)
    out = np.isin(labels2, np.unique(labels2[fg > 0])).astype(np.uint8)
    return out, int(out.sum() - fg.sum())


def mirror_union(fg):
    """Union of the mask with its mirror image about the symmetry axis (planes: the fuselage).

    The axis is found by search: angle 0..179 degrees and offset -6..6 px from the mask centroid,
    maximising the IoU of the mask with its own reflection. Returns (mask, angle, offset, iou).
    """
    ys, xs = np.nonzero(fg)
    if len(xs) == 0:
        return fg, None, None, 0.0
    cx, cy = float(xs.mean()), float(ys.mean())
    h, w = fg.shape
    best = (0.0, None, None, None)
    src = fg.astype(np.float32)
    for angle in range(0, 180):
        t = math.radians(angle)
        dx, dy = math.cos(t), math.sin(t)          # axis direction
        nx, ny = -dy, dx                           # axis normal
        a, b, d = 2 * dx * dx - 1, 2 * dx * dy, 2 * dy * dy - 1
        for offset in range(-6, 7):
            px, py = cx + offset * nx, cy + offset * ny
            # Reflection about the line through (px, py) with direction (dx, dy).
            mat = np.array([[a, b, px - a * px - b * py], [b, d, py - b * px - d * py]], np.float32)
            refl = cv2.warpAffine(src, mat, (w, h), flags=cv2.INTER_NEAREST)
            inter = float((refl * src).sum())
            union = float(((refl + src) > 0).sum())
            if union > 0 and inter / union > best[0]:
                best = (inter / union, angle, offset, refl)
    iou, angle, offset, refl = best
    return ((fg > 0) | (refl > 0)).astype(np.uint8), angle, offset, round(iou, 3)


def refined_steps(fg, crop, init, pad, box_in_crop, recipe):
    """Post-processing of variant r in a fixed order. Returns (mask, notes)."""
    notes = {}
    if fg.sum() == 0:
        return fg, notes
    if recipe.get("thin_dark"):
        fg, added, level = thin_parts(crop, fg, init, box_in_crop)
        notes["thin_dark_added_px"] = added
        notes["thin_dark_level"] = level
    if recipe.get("rim"):
        fg, added = hangar_rim(crop, fg)
        notes["rim_added_px"] = added
    if recipe.get("mirror"):
        before = int(fg.sum())
        union, angle, offset, iou = mirror_union(fg)
        # The view is oblique, so the reflection never fits exactly and also lands on the apron next to
        # the tail. A reflected-in pixel is kept only when its Lab colour is more than 30 away from the
        # median colour of a 3 px band around the union (the ground the plane stands on).
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        band = (cv2.dilate(union, np.ones((7, 7), np.uint8)) > 0) & (union == 0)
        ground = np.median(lab[band], axis=0)
        brought_in = (union > 0) & (fg == 0) & (pad == 0) & (np.linalg.norm(lab - ground, axis=2) > 30)
        fg = clean_mask(((fg > 0) | brought_in).astype(np.uint8))
        notes.update(mirror_axis_deg=angle, mirror_offset_px=offset, mirror_self_iou=iou,
                     mirror_added_px=int(fg.sum()) - before)
    return fg, notes


def side_contact(fg, box, ox, oy):
    """Fraction of each box side (left, top, right, bottom) along which the foreground touches it."""
    x1, y1, x2, y2 = box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy
    inner = fg[y1:y2, x1:x2]
    if inner.size == 0:
        return [0, 0, 0, 0]
    return [float(inner[:, 0].mean()), float(inner[0, :].mean()), float(inner[:, -1].mean()), float(inner[-1, :].mean())]


def cut_sprite(image, box, allow_growth, enlarge, recipe=None):
    """GrabCut from the box, optionally growing the box where the object runs into a side.

    recipe (a RECIPES entry) switches on the seeds and the post-processing of variant r.
    Returns (bgra sprite or None, origin (x, y) in source pixels or None, info dict).
    """
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    work = list(box)
    grown = 0
    while True:
        fg, ox, oy, crop, init, pad, notes = grabcut_once(image, work, enlarge, recipe)
        if recipe and recipe.get("colour_holes"):
            fg = clean_mask(fg, crop, init, close=not recipe.get("no_close"))
        else:
            fg = clean_mask(fg)
        if recipe:
            fg, more = refined_steps(fg, crop, init, pad, (work[0] - ox, work[1] - oy, work[2] - ox, work[3] - oy), recipe)
            notes.update(more)
        contact = side_contact(fg, work, ox, oy)
        if not allow_growth or grown >= 3 or max(contact) < 0.12:
            break
        bw, bh = work[2] - work[0], work[3] - work[1]
        if contact[0] >= 0.12:
            work[0] = max(3, work[0] - int(0.2 * bw))
        if contact[1] >= 0.12:
            work[1] = max(3, work[1] - int(0.2 * bh))
        if contact[2] >= 0.12:
            work[2] = min(W - 4, work[2] + int(0.2 * bw))
        if contact[3] >= 0.12:
            work[3] = min(H - 4, work[3] + int(0.2 * bh))
        grown += 1
    info = {"grown": grown, "side_contact": [round(c, 2) for c in contact], **notes}
    if fg.sum() == 0:
        info["auto_reject"] = "empty mask"
        return None, None, info
    ys, xs = np.nonzero(fg)
    mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    sprite = np.dstack([crop[my1:my2, mx1:mx2], fg[my1:my2, mx1:mx2] * 255]).astype(np.uint8)
    info["fill_of_box"] = round(float(fg.sum()) / box_area, 3)
    if info["fill_of_box"] < 0.04:
        info["auto_reject"] = "mask below 4 % of the box"
    elif info["fill_of_box"] > 0.92 and max(contact) > 0.5:
        info["auto_reject"] = "mask fills the box: no separation from the terrain"
    return sprite, (int(ox + mx1), int(oy + my1)), info


# ----------------------------------------------------------------------------- view geometry

def view_geometry(cx, cy):
    """Off-nadir angle (degrees) at a source pixel and the image direction pointing away from nadir."""
    pitch = math.radians(CAMERA_PITCH_DEG)
    u, v = cx - W / 2, cy - H / 2
    down = v * math.sin(pitch) + CAMERA_F * math.cos(pitch)
    off = math.degrees(math.acos(down / math.sqrt(u * u + v * v + CAMERA_F ** 2)))
    nadir_x, nadir_y = W / 2, H / 2 + CAMERA_F * math.tan(pitch)
    azimuth = math.degrees(math.atan2(-(cy - nadir_y), cx - nadir_x)) % 360
    return round(off, 1), round(azimuth, 1)


# ----------------------------------------------------------------------------- drawing

def checker(h, w, cell=8):
    ys, xs = np.mgrid[0:h, 0:w]
    board = ((ys // cell + xs // cell) % 2).astype(bool)
    return np.dstack([np.where(board, 205, 150).astype(np.uint8)] * 3)


def fit_tile(img, size):
    """Scale to fit size x size: whole-number nearest enlargement, or area reduction."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    if s >= 1:
        s = max(1, int(s))
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=interp)


def centred(img, size, fill):
    tile = np.full((size, size, 3), fill, np.uint8)
    y0, x0 = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    tile[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return tile


def sprite_tile(bgra, size, background, pad=3):
    """The sprite over a checkerboard or over flat magenta, as a size x size tile."""
    padded = cv2.copyMakeBorder(bgra, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    ph, pw = padded.shape[:2]
    bg = checker(ph, pw, 4) if background == "checker" else np.full((ph, pw, 3), MAGENTA, np.uint8)
    img = np.where(padded[:, :, 3:4] > 0, padded[:, :, :3], bg)
    return centred(fit_tile(img, size), size, 60 if background == "checker" else MAGENTA)


def crop_views(image, cand, variant, margin):
    """Source crop around the annotation box, the variant's mask placed in it, and the crop origin."""
    x1, y1, x2, y2 = cand["box"]
    cx1, cy1, cx2, cy2 = max(0, x1 - margin), max(0, y1 - margin), min(W, x2 + margin), min(H, y2 + margin)
    var = cand["variants"].get(variant)
    full = np.zeros((cy2 - cy1, cx2 - cx1), np.uint8)
    if var and var.get("file"):
        bgra = cv2.imread(os.path.join(CANDIDATES, var["file"]), cv2.IMREAD_UNCHANGED)
        ox, oy = var["origin_src"]
        h, w = bgra.shape[:2]
        sx1, sy1 = max(ox, cx1), max(oy, cy1)
        sx2, sy2 = min(ox + w, cx2), min(oy + h, cy2)
        if sx2 > sx1 and sy2 > sy1:
            full[sy1 - cy1:sy2 - cy1, sx1 - cx1:sx2 - cx1] = bgra[sy1 - oy:sy2 - oy, sx1 - ox:sx2 - ox, 3] > 0
    return image[cy1:cy2, cx1:cx2].copy(), full, (cx1, cy1)


def outlined(crop, full, size):
    scaled = fit_tile(crop, size)
    mask = cv2.resize(full, (scaled.shape[1], scaled.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(scaled, contours, -1, (0, 255, 255), 1)
    return scaled


def review_sheet(cls, cands, images, path, size=144, per_row=2):
    cells = []
    for number, cand in enumerate(cands):
        image = images[(cand["scene"], cand["frame"])]
        x1, y1, x2, y2 = cand["box"]
        margin = max(12, int(0.3 * max(x2 - x1, y2 - y1)))
        crop, full, (cx1, cy1) = crop_views(image, cand, "z", margin)
        boxed = fit_tile(crop, size)
        k = boxed.shape[0] / crop.shape[0]
        cv2.rectangle(boxed, (int((x1 - cx1) * k), int((y1 - cy1) * k)), (int((x2 - cx1) * k), int((y2 - cy1) * k)), (0, 255, 255), 1)
        tiles = [centred(boxed, size, 0)]
        for variant in ("z", "n", "r"):
            if variant not in cand["variants"]:
                continue
            var = cand["variants"][variant]
            if var.get("file"):
                bgra = cv2.imread(os.path.join(CANDIDATES, var["file"]), cv2.IMREAD_UNCHANGED)
                tile = sprite_tile(bgra, size, "magenta")
                if var["info"].get("auto_reject"):
                    cv2.putText(tile, "auto reject", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            else:
                tile = np.zeros((size, size, 3), np.uint8)
                cv2.putText(tile, "empty", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles.append(tile)
        last = "r" if "r" in cand["variants"] else "z"
        if last == "r":
            crop, full, _ = crop_views(image, cand, "r", margin)
        tiles.append(centred(outlined(crop, full, size), size, 0))
        body = np.hstack(tiles)
        label = np.zeros((34, body.shape[1], 3), np.uint8)
        cv2.putText(label, f"#{number} {cand['scene'][:3]} f{cand['frame']} {cand['track'][:30]}", (3, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        zi, ni = cand["variants"]["z"]["info"], cand["variants"]["n"]["info"]
        extra = (f"y{y1} offnadir {cand['off_nadir_deg']}  z: fill {zi.get('fill_of_box', 0)} grown {zi['grown']}  "
                 f"n: fill {ni.get('fill_of_box', 0)} grown {ni['grown']}")
        cv2.putText(label, extra, (3, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 220, 255), 1)
        cells.append(np.vstack([label, body]))
    rows = []
    for i in range(0, len(cells), per_row):
        chunk = cells[i:i + per_row]
        while len(chunk) < per_row:
            chunk.append(np.zeros_like(cells[0]))
        rows.append(np.hstack([np.hstack([c, np.zeros((c.shape[0], 8, 3), np.uint8)]) for c in chunk]))
    sheet = np.vstack(rows)
    title = np.zeros((24, sheet.shape[1], 3), np.uint8)
    third = " | variant r (refined) | r outline" if cls in RECIPES else " | z outline"
    cv2.putText(title, f"{cls}: crop with box | variant z (3x GrabCut) | variant n (native GrabCut)" + third, (3, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(path, np.vstack([title, sheet]))


def bank_sheet(entries, background, path, size=96, max_cols=14):
    rows = []
    for cls in CLASSES:
        items = [e for e in entries if e["class_name"] == cls]
        for start in range(0, max(1, len(items)), max_cols):
            label = np.zeros((size, 190, 3), np.uint8)
            if start == 0:
                hel = sum(e["source"] == "helsinki" for e in items)
                cv2.putText(label, cls, (3, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(label, f"{len(items)} = {hel} hel + {len(items) - hel} val", (3, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            tiles = [label]
            for e in items[start:start + max_cols]:
                bgra = cv2.imread(os.path.join(SPRITES, e["file"]), cv2.IMREAD_UNCHANGED)
                tile = sprite_tile(bgra, size, background)
                tag = ("h" if e["source"] == "helsinki" else "v") + str(e["frame"])
                cv2.putText(tile, tag, (2, size - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
                tiles.append(tile)
            while len(tiles) < max_cols + 1:
                tiles.append(np.zeros((size, size, 3), np.uint8))
            rows.append(np.hstack(tiles))
    cv2.imwrite(path, np.vstack(rows))


# ----------------------------------------------------------------------------- commands

def load_frames(cands):
    images = {}
    for cand in cands:
        key = (cand["scene"], cand["frame"])
        if key not in images:
            images[key] = cv2.imread(os.path.join(ROOT, "src", cand["scene"], "images", f"frame_{cand['frame']:06d}.png"))
    return images


def cmd_cut(args):
    os.makedirs(OUT, exist_ok=True)
    chosen = choose_candidates(args.per_class_helsinki, args.per_track_validation)
    kept_from_before = []
    if args.classes:
        # Redo only some classes and keep the other candidates as they are on disk.
        redo = set(args.classes.split(","))
        with open(os.path.join(CANDIDATES, "candidates.json")) as fh:
            kept_from_before = [c for c in json.load(fh) if c["class_name"] not in redo]
        chosen = [c for c in chosen if c["class_name"] in redo]
        for cls in redo:
            if os.path.isdir(os.path.join(CANDIDATES, cls)):
                shutil.rmtree(os.path.join(CANDIDATES, cls))
    else:
        if os.path.isdir(CANDIDATES):
            shutil.rmtree(CANDIDATES)
        os.makedirs(CANDIDATES)
    images = load_frames(chosen)
    print(len(chosen), "candidates from", len(images), "frames")
    for cand in chosen:
        image = images[(cand["scene"], cand["frame"])]
        cand["id"] = f"cut:{cand['scene']}-f{cand['frame']:06d}-{cand['track']}"
        bx1, by1, bx2, by2 = cand["box"]
        cand["off_nadir_deg"], cand["view_azimuth_deg"] = view_geometry(0.5 * (bx1 + bx2), 0.5 * (by1 + by2))
        cand["variants"] = {}
        runs = [("z", True, None), ("n", False, None)]
        if cand["class_name"] in RECIPES:
            runs.append(("r", RECIPES[cand["class_name"]].get("enlarge", True), RECIPES[cand["class_name"]]))
        for variant, enlarge, recipe in runs:
            left, top, right, bottom = TRACK_BOX_MARGIN.get(cand["track"], ((0, 0, 0, 0), ""))[0]
            start_box = [max(3, bx1 - left), max(3, by1 - top), min(W - 4, bx2 + right), min(H - 4, by2 + bottom)]
            sprite, origin, info = cut_sprite(image, start_box, cand["scene"] == "validation", enlarge, recipe)
            entry = {"info": info, "file": None}
            if sprite is not None:
                rel = f"{cand['class_name']}/cut__{cand['scene']}-f{cand['frame']:06d}-{cand['track']}.{variant}.png"
                os.makedirs(os.path.join(CANDIDATES, cand["class_name"]), exist_ok=True)
                cv2.imwrite(os.path.join(CANDIDATES, rel), sprite)
                entry.update(file=rel, origin_src=list(origin), size=[int(sprite.shape[1]), int(sprite.shape[0])])
            cand["variants"][variant] = entry
    redone = {c["class_name"] for c in chosen}
    everything = sorted(kept_from_before + chosen, key=lambda c: (CLASSES.index(c["class_name"]), c["scene"], c["track"], c["frame"]))
    with open(os.path.join(CANDIDATES, "candidates.json"), "w") as fh:
        json.dump(everything, fh, indent=1)
    for cls in CLASSES:
        cands = [c for c in everything if c["class_name"] == cls and cls in redone]
        if cands:
            path = os.path.join(OUT, f"sprites_review_{cls}.png")
            review_sheet(cls, cands, images, path)
            print(f"{cls:16s} {len(cands):3d} candidates -> {path}")


def cmd_zoom(args):
    with open(os.path.join(CANDIDATES, "candidates.json")) as fh:
        cands = [c for c in json.load(fh) if c["class_name"] == args.cls]
    ids = [int(i) for i in args.ids.split(",")] if args.ids else list(range(len(cands)))
    picked = [(i, cands[i]) for i in ids]
    images = load_frames([c for _, c in picked])
    cells = []
    for number, cand in picked:
        for variant in args.variants.split(","):
            var = cand["variants"][variant]
            if not var.get("file"):
                continue
            crop, full, _ = crop_views(images[(cand["scene"], cand["frame"])], cand, variant, args.margin)
            mag = np.where(full[:, :, None] > 0, crop, np.array(MAGENTA, np.uint8)).astype(np.uint8)
            chk = np.where(full[:, :, None] > 0, crop, checker(crop.shape[0], crop.shape[1], 4)).astype(np.uint8)
            body = np.hstack([fit_tile(crop, args.size), outlined(crop, full, args.size), fit_tile(chk, args.size), fit_tile(mag, args.size)])
            label = np.zeros((18, body.shape[1], 3), np.uint8)
            cv2.putText(label, f"#{number}{variant} {cand['scene'][:3]} f{cand['frame']} {cand['track'][:28]}", (2, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cells.append(np.vstack([label, body]))
    width = max(c.shape[1] for c in cells)
    cells = [np.hstack([c, np.zeros((c.shape[0], width - c.shape[1], 3), np.uint8)]) for c in cells]
    columns = []
    per_col = (len(cells) + args.columns - 1) // args.columns
    for i in range(0, len(cells), per_col):
        columns.append(np.vstack(cells[i:i + per_col]))
    height = max(c.shape[0] for c in columns)
    columns = [np.vstack([c, np.zeros((height - c.shape[0], c.shape[1], 3), np.uint8)]) for c in columns]
    cv2.imwrite(args.out, np.hstack(columns))
    print("wrote", args.out)


def cmd_bank(args):
    with open(os.path.join(CANDIDATES, "candidates.json")) as fh:
        cands = json.load(fh)
    review_path = os.path.join(SPRITES, "review.json")
    discards, chosen_variant = {}, {}
    if os.path.exists(review_path):
        with open(review_path) as fh:
            review = json.load(fh)
        discards = review.get("discard", {})
        chosen_variant = {cid: "n" for cid in review.get("use_native", [])}
        chosen_variant.update(review.get("use_variant", {}))
    unknown = (set(discards) | set(chosen_variant)) - {c["id"] for c in cands}
    if unknown:
        raise SystemExit(f"review.json names ids that are not candidates: {sorted(unknown)}")
    for cls in CLASSES:
        if os.path.isdir(os.path.join(SPRITES, cls)):
            shutil.rmtree(os.path.join(SPRITES, cls))
    entries = []
    counts = {cls: {"candidates": 0, "discarded": 0, "kept": 0, "kept_helsinki": 0, "kept_validation": 0,
                    "kept_split_train": 0, "kept_split_dev": 0,
                    "kept_variant_z": 0, "kept_variant_n": 0, "kept_variant_r": 0, "kept_tracks": [],
                    "discard_reasons": {}} for cls in CLASSES}
    for cand in cands:
        cls = cand["class_name"]
        counts[cls]["candidates"] += 1
        variant = chosen_variant.get(cand["id"], "z")
        var = cand["variants"][variant]
        reason = discards.get(cand["id"])
        if reason is None and (not var.get("file") or var["info"].get("auto_reject")):
            reason = "GrabCut failed: " + var["info"].get("auto_reject", "empty mask")
        if reason is not None:
            counts[cls]["discarded"] += 1
            counts[cls]["discard_reasons"][cand["id"]] = reason
            continue
        rel = var["file"].replace(f".{variant}.png", ".png")
        dst = os.path.join(SPRITES, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(CANDIDATES, var["file"]), dst)
        with open(dst, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        ox, oy = var["origin_src"]
        bx1, by1, bx2, by2 = cand["box"]
        split = "dev" if cand["scene"] == "validation" and 100 <= cand["frame"] <= 180 else "train"
        entries.append({
            "file": rel.replace(os.sep, "/"), "id": cand["id"], "class_name": cls, "class_id": CLASSES.index(cls),
            "zoom": 2, "source": cand["scene"], "frame": cand["frame"], "track": cand["track"],
            "label_status": cand["label_status"], "split": split, "review_status": "agent_approved_on_contact_sheet",
            "size": var["size"], "box_in_sprite": [bx1 - ox, by1 - oy, bx2 - ox, by2 - oy],
            "box_src": cand["box"], "origin_src": [ox, oy],
            "center_src": [round(0.5 * (bx1 + bx2), 1), round(0.5 * (by1 + by2), 1)],
            "off_nadir_deg": cand["off_nadir_deg"], "view_azimuth_deg": cand["view_azimuth_deg"],
            "grabcut_variant": {"z": "enlarged_3x", "n": "native", "r": "refined_3x"}[variant],
            "grabcut_refinement": {k: v for k, v in var["info"].items() if k not in ("grown", "side_contact", "fill_of_box")},
            "grabcut_box_grown": var["info"]["grown"], "sha256": digest})
        counts[cls]["kept"] += 1
        counts[cls]["kept_" + cand["scene"]] += 1
        counts[cls]["kept_variant_" + variant] += 1
        counts[cls]["kept_split_" + split] += 1
        if cand["track"] not in counts[cls]["kept_tracks"]:
            counts[cls]["kept_tracks"].append(cand["track"])
    bank = {
        "classes": CLASSES,
        "note": "GrabCut sprites cut by elias/data/sprites.py from the 4K frames. Original pixels, binary alpha, "
                "native source resolution (zoom 2). box_in_sprite is the annotation box relative to the sprite's "
                "top-left and can be negative. source names the scene. split = dev marks validation frames 100..180 "
                "(Oscar's development split): filter on split == 'train' when a number is reported on those frames. "
                "class_id follows dtos.OBJECT_CLASSES. Discards and reasons: review.json and "
                "elias/out/sprites_bank_counts.json. medium_plane has only 4 sprites here (4 un-clipped helsinki "
                "frames exist, no directly reviewed validation track): merge Oscar's 6 medium_plane entries. No "
                "helsinki helicopter and no helsinki medium_launcher sprite passed the review.",
        "camera": {"f_px": CAMERA_F, "pitch_deg": CAMERA_PITCH_DEG, "source": "elias/out/tilt_camera.json"},
        "excluded_tracks": EXCLUDED_TRACKS,
        "replaced_frames": {f"{track} f{frame}": {"used_instead": swap[0], "reason": swap[1]}
                            for (track, frame), swap in REPLACED_FRAMES.items()},
        "track_box_margin": {track: {"left_top_right_bottom_px": list(v[0]), "reason": v[1]} for track, v in TRACK_BOX_MARGIN.items()},
        "kept_per_class_and_split": {cls: {"train": counts[cls]["kept_split_train"], "dev": counts[cls]["kept_split_dev"]}
                                     for cls in CLASSES},
        "sprites": entries,
    }
    with open(os.path.join(SPRITES, "bank.json"), "w") as fh:
        json.dump(bank, fh, indent=1)
    os.makedirs(OUT, exist_ok=True)
    bank_sheet(entries, "checker", os.path.join(OUT, "sprites_bank_checker.png"))
    bank_sheet(entries, "magenta", os.path.join(OUT, "sprites_bank_magenta.png"))
    with open(os.path.join(OUT, "sprites_bank_counts.json"), "w") as fh:
        json.dump(counts, fh, indent=1)
    print(f"{'class':16s} cand disc kept (hel+val) tracks  z/n/r  train/dev split")
    for cls in CLASSES:
        c = counts[cls]
        print(f"{cls:16s} {c['candidates']:4d} {c['discarded']:4d} {c['kept']:4d} ({c['kept_helsinki']}+{c['kept_validation']}) "
              f"{len(c['kept_tracks']):3d}    {c['kept_variant_z']}/{c['kept_variant_n']}/{c['kept_variant_r']}"
              f"   {c['kept_split_train']}/{c['kept_split_dev']}")
    print("total kept", len(entries), "of", len(cands))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cut = sub.add_parser("cut", help="run GrabCut on the chosen frames and write the review sheets")
    cut.add_argument("--per-class-helsinki", type=int, default=8)
    cut.add_argument("--per-track-validation", type=int, default=4)
    cut.add_argument("--classes", default="", help="comma separated classes to redo; the rest of candidates.json is kept")
    zoom = sub.add_parser("zoom", help="large view of chosen candidates of one class")
    zoom.add_argument("--cls", required=True, choices=CLASSES)
    zoom.add_argument("--out", required=True)
    zoom.add_argument("--ids", default="", help="comma separated candidate numbers as printed on the review sheet")
    zoom.add_argument("--variants", default="z", help="z, n or z,n")
    zoom.add_argument("--size", type=int, default=160)
    zoom.add_argument("--margin", type=int, default=8)
    zoom.add_argument("--columns", type=int, default=2)
    sub.add_parser("bank", help="apply review.json and write bank.json and the bank contact sheets")
    args = parser.parse_args()
    {"cut": cmd_cut, "zoom": cmd_zoom, "bank": cmd_bank}[args.cmd](args)


if __name__ == "__main__":
    main()
