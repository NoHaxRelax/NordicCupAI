"""Synthetic generator v2: scale-preserving 64x64 windows built from sprites (see elias/SPEC.md).

What it does
------------
Every sample is composed at NATIVE source resolution (3840x2160 pixel grid) and only then
reduced to the 64x64 window with cv2.INTER_AREA, in the same two stages the evaluator plus a
window cutter would apply: first by the level factor f (4, 2, 1 for level 0, 1, 2), which is what
the evaluator does when it renders a 960x540 view, then by s / f down to the window scale s.
Nothing is added after the reduction (an optional +-1 grey level noise exists, default off).

Per positive sample:
  1. class (balanced over the classes that have a sprite in the allowed sprite scenes), sprite,
     level L, rotation, scale jitter 0.92..1.08;
  2. label box = tight bbox of the rotated alpha mask plus the per-class median margin between
     tight mask bbox and organiser box (measured from box_in_sprite). The four side margins are
     carried along with the rotation (cos^2 blend), so a lopsided footprint stays lopsided on
     the correct side after a 90 or 180 degree turn;
  3. window scale s from the SPEC rule applied to that label box;
  4. background patch of 64 s native pixels from an allowed background scene, overlapping no
     labelled box (helsinki), or at least 96 px from every labelled box and inside the row band
     covered by labels in that frame (validation, whose labels are incomplete). Where in the
     frame: see "Ground" below;
  5. sprite photometric jitter (LUT: brightness +-8 %, contrast +-8 %, gamma 0.9..1.1),
     cv2.warpAffine (colour interpolation drawn per sprite from --warp-interp, default lanczos4
     or nearest; the mask is warped linearly, cut at 0.5 and feathered by a 3x3 Gaussian),
     optional Gaussian blur (--sprite-blur-max, default off), alpha compositing so that the
     LABEL BOX centre lands on the window centre plus the recorded offset `off`;
  6. with p=0.3 one distractor sprite of another class, partly inside the window, never
     overlapping the main object and never close enough to the centre to count as a positive;
  7. milder global photometric LUT on the whole patch (+-4 %, +-4 %, gamma 0.95..1.05);
  8. INTER_AREA by f; with p=0.15 a zero-filled border strip (up to 20 window px) on one side,
     written in delivered pixels as a detector sees at a view edge; INTER_AREA by s / f.

Defaults that go beyond the letter of that recipe, each because of what was measured:
  * Sharpness. At window scale 1 a bilinear warp plus blur 0..0.6 keeps only 0.55 to 0.71 of the
    Laplacian energy of real objects (jammer, tank, ta-ta; mean abs Laplacian in the central
    12x12 px). lanczos4 without blur keeps 0.84 to 0.94, nearest 0.95 to 1.15. The default draws
    one of the two per sprite, which brackets the real sharpness for any heading (measured mix:
    0.91 to 1.06), and --lossless-prob 0.25 pastes a quarter of the sprites with quarter turns
    only (no resampling at all, scale 1). --warp-interp linear --sprite-blur-max 0.6
    --lossless-prob 0 gives the literal recipe back.
  * Ground (--bg-open-prob 0.7, --bg-near-frac 0.5). Labelled objects stand on open ground,
    while a uniformly drawn patch is mostly forest or water, and brightness alone also picks
    roofs. Every background frame is therefore described by six layers at 8 px cells (grey,
    blue-dominance, water, straight-edge structure, edge density, red-dominance; see
    ground_layers) and the rings of 56 px around the labelled boxes of the background scenes
    give the limits: at least as bright as the 10th percentile of the rings, no more blue, red,
    structured or edgy than their 90th percentile, no water. 70 % of all windows need such open ground under their central
    half; half of those are drawn 100 to 500 source px from a labelled box ("near", the very
    ground real objects stand on), the other half anywhere in the frame ("open"). The remaining
    30 % go anywhere, except that a window which gets a sprite pasted never lies on water or on
    ground more blue or red than the rings allow (slate roofs, deep shade, tiled roofs); pure
    background windows may. --bg-open-prob 0 switches the open-ground share off. Known limit:
    large pale flat roofs of the urban validation frames look like paved yards to these layers,
    so some objects still stand on them.

Half of the positives are centred (true offset within +-0.5 window px, the rounding of a window
cut in delivered pixels), half are jittered by up to +-6 window px, as real_crops.py does.

Negatives (label 16) are half pure background and half hard negatives in which an object sits
off-centre by more than max(0.75 x its longer side, 10 window px). They use the same
(level, scale) mix as the positives because each negative first draws a virtual object exactly
like a positive would; box_w, box_h and size_px of a negative are those of that object.

Which bank entries are pasted (read_bank_entries; everything dropped is listed in the manifest)
-----------------------------------------------------------------------------------------------
  * Vetoes: --exclude-sprite-ids, --exclude-tracks, the `excluded_tracks` block of every bank
    (the default banks are read for it even when they are not used for sprites) and
    KNOWN_BAD_TRACKS. The track of an entry is its `track` field or the provenance track of the
    labelled box it was cut from. This removes Oscar's `lib:mine_roller` (validation track
    mine-roller-a-005-009, which looks like a tank).
  * Duplicates: one entry per (class, scene, frame, zoom, labelled box); the first bank wins.
  * Resolution: per (class, scene) only the highest zoom is kept. In Oscar's bank only zoom 2
    sprites hold native pixels and masks made at native resolution. Zoom 0 and 1 masks are 4 or
    2 px coarse: the small_launcher mask missed the top of the object and the tank masks ended
    before the gun barrel. They are used only for a class with nothing better in that scene
    (large_tower when Oscar's bank is used alone): then the colour is re-cut from the source
    frame under the bank alpha (--no-recut-native keeps the upsampled pixels and limits such a
    sprite to levels L <= zoom), and masks below --min-recut-mask-px are dropped.

Rim rule, the same for every sprite (hard_mask_and_colour)
----------------------------------------------------------
Colour is trusted and the hard mask is set only where the bank alpha is at least 0.8. The rim
below that holds the bank's un-mixed colours (Oscar's mattes: cyan, pink or black outlines when
shown opaque) or old ground; it is left out of the mask and repainted from the nearest mask
pixel, and the 3x3 feather after the warp provides the soft edge. Parts too thin to have any
solid pixel (gun barrel, rotor blade) stay in the mask with their colour clipped to the range
of the solid colours. Sprites with a binary mask (elias/sprites) first get a coverage estimate
for their outermost pixels, otherwise old bright ground shows as a halo on dark ground.
Identity test (each zoom 2 helsinki sprite pasted back on its own place): mean error in the
band around the silhouette 10.2 grey levels with the old rule (mask and colour at alpha 0.5),
3.8 with this one.

Usage (run from drone-flyby/)
-----------------------------
  python elias/data/synth.py --sprite-scenes helsinki --background-scenes helsinki \
      --n 4000 --rot-max 180 --seed 0 --out elias/out/synth_demo_helsinki.npz --compare

Writes OUT.npz (SPEC container), OUT.json (manifest: counts per class x level, parameters,
measured throughput) and OUT.png (contact sheet, one row per class, x3 nearest). With --compare
it also writes OUT_vs_real_s{1,2,4}.png: real windows (from --compare-real NPZ, default
elias/out/real_<scene>.npz, else cut here from the first sprite scene) next to synthetic ones.

As an on-the-fly dataset (duck-typed for torch.utils.data.DataLoader, picklable, lazy):

  from elias.data.synth import SynthWindows
  ds = SynthWindows(n=200000, sprite_scenes=['helsinki'], background_scenes=['helsinki'], seed=1)
  ds.set_epoch(3)            # new random samples, still reproducible per (seed, epoch, index)
  sample = ds[0]             # dict: x uint8 (64,64,3) BGR, label, level, scale, size_px,
                             #       box_w, box_h, off (2,), scene, frame, track, source

Decoded frames are cached as .npy under elias/out/frame_cache (memory mapped, shared between
worker processes; about 25 MB per frame). Use --no-frame-cache to keep them in RAM instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
ELIAS_DIR = HERE.parents[1]
TASK_DIR = HERE.parents[2]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from dtos import OBJECT_CLASSES  # noqa: E402  (class order used by the scorer)

CLASSES = list(OBJECT_CLASSES)
BACKGROUND = len(CLASSES)  # label 16
WINDOW = 64
MAX_OBJECT_WINDOW_PX = 48
FRAME_W, FRAME_H = 3840, 2160
LEVEL_FACTOR = {0: 4, 1: 2, 2: 1}

# bank "source" value -> scene folder under src/
SCENE_OF_SOURCE = {"reference": "helsinki", "helsinki": "helsinki", "validation": "validation"}
# distance kept from every labelled box when choosing background, in source px
KEEPOUT_PX = {"helsinki": 0, "validation": 96}
# on these scenes labels are incomplete, so background rows are limited to the labelled band
BAND_LIMITED = {"validation"}

DEFAULT_BANKS = [
    TASK_DIR / "oscar-sprite-synthetic" / "sprite-bank" / "bank.json",
    ELIAS_DIR / "sprites" / "bank.json",
]
DEFAULT_FRAME_CACHE = ELIAS_DIR / "out" / "frame_cache"

GREY_WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)   # BGR -> grey

SPRITE_PAD = 3            # transparent border kept around every cached sprite
# One rule for every sprite (bank matte, binary alpha or re-cut): colour is trusted and the hard
# mask is set only where the bank alpha reaches this level. The rim below it holds either old
# background or the bank's extrapolated ("decontaminated") colour, which shows as a coloured
# outline when pasted opaque; it is repainted from the nearest solid pixel and left to the feather.
SOLID_ALPHA = 0.8

INTERPOLATIONS = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
                  "cubic": cv2.INTER_CUBIC, "lanczos4": cv2.INTER_LANCZOS4}

# ground layers: one cell is GROUND_CELL source px; layer order inside the cached uint8 array
GROUND_CELL = 8
GROUND_LAYERS = ("grey", "blue", "water", "structure", "edges", "red")
GROUND_VERSION = 2        # part of the cache file name: bump when ground_layers() changes
WATER_SHARE_MAX = 0.02    # no sprite is ever pasted on a patch whose centre holds more water
NEAR_RING_PX = (100, 500)  # "near" backgrounds: patch centre this far from a labelled box

# Pseudo-label tracks of `validation` known to carry the wrong class. The banks' own
# `excluded_tracks` blocks are read as well; this copy keeps the veto alive when a bank is absent.
KNOWN_BAD_TRACKS = {
    "mine-roller-a-005-009": "boxy hull with a gun barrel: looks like the tank, not the "
                             "organiser's mine roller (flagged by Oscar and in elias/sprites)",
}


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def pick_scale(factor: int, longer_side_src: float) -> int:
    """SPEC rule: smallest s in {f, 2f, 4f} with longer_side / s <= 48 (4f if none fits)."""
    for s in (factor, 2 * factor, 4 * factor):
        if longer_side_src / s <= MAX_OBJECT_WINDOW_PX:
            return s
    return 4 * factor


def _env_range(name: str, lo: float, hi: float):
    """'lo,hi' from the environment. Helsinki objects are 1.3 to 2.0 times brighter than the same
    objects in validation while its terrain is darker (median grey 54 against 98), so sprite and
    background exposure must vary independently and widely for a third, unseen flight."""
    text = os.environ.get(name, '')
    if not text:
        return lo, hi
    a, b = (float(v) for v in text.split(','))
    return a, b


def make_lut(brightness: float, contrast: float, gamma: float, pivot: float) -> np.ndarray:
    """256-entry lookup table for gamma, then contrast about `pivot` (0..1), then brightness."""
    x = np.arange(256, dtype=np.float64) / 255.0
    x = x ** gamma
    p = pivot ** gamma
    x = ((x - p) * contrast + p) * brightness
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def reduce_like_evaluator(patch: np.ndarray, factor: int, scale: int, strip=None) -> np.ndarray:
    """Native patch (64 s square) -> 64x64 window: INTER_AREA by f, optional strip, then by s/f.

    `strip` is (side, width_in_delivered_px); it is zero-filled in delivered pixels because that
    is where a window that pokes out of a view gets its padding.
    """
    img = patch
    if factor > 1:
        size = patch.shape[0] // factor
        img = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    if strip is not None:
        if img is patch:
            img = patch.copy()
        side, width = strip
        if side == 0:
            img[:, :width] = 0
        elif side == 1:
            img[:width, :] = 0
        elif side == 2:
            img[:, -width:] = 0
        else:
            img[-width:, :] = 0
    if img.shape[0] != WINDOW:
        img = cv2.resize(img, (WINDOW, WINDOW), interpolation=cv2.INTER_AREA)
    return img


def fill_from_nearest_solid(bgr: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Replace the colour of every non-solid pixel by the colour of the nearest solid pixel."""
    if solid.all() or not solid.any():
        return bgr
    _, labels = cv2.distanceTransformWithLabels(
        (~solid).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    flat_index = np.arange(solid.size).reshape(solid.shape)
    index_of_label = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    index_of_label[labels[solid]] = flat_index[solid]
    return bgr.reshape(-1, 3)[index_of_label[labels]].reshape(bgr.shape)


def ground_layers(image):
    """Six uint8 layers [6, H/8, W/8] that describe the ground of one 4K frame (0..255 each).

    grey       mean grey level;
    blue       share of blue-dominant pixels (water, slate roofs, deep shadow);
    water      blue-dominant AND smooth AND at least 60 px across, grown by 20 px;
    structure  within 32 px of a straight edge of 64 px or more (roofs, roads, jetties, hedges);
    edges      Canny edge density in a 60 px neighbourhood (buildings, boats, tree crowns);
    red        share of strongly red-dominant pixels (tiled roofs; also the helsinki clay court).
    All of it is computed on the frame reduced by 4 and then averaged down to 8 px cells.
    """
    quarter = cv2.resize(np.asarray(image), (FRAME_W // 4, FRAME_H // 4),
                         interpolation=cv2.INTER_AREA)
    blue_c, green_c, red_c = [quarter[:, :, i].astype(np.float32) for i in range(3)]
    grey = 0.114 * blue_c + 0.587 * green_c + 0.299 * red_c
    blue = (blue_c >= red_c + 8) & (blue_c >= green_c - 10)
    mean = cv2.blur(grey, (9, 9))
    deviation = np.sqrt(np.maximum(cv2.blur(grey * grey, (9, 9)) - mean * mean, 0.0))
    water = (blue & (deviation < 5.0)).astype(np.uint8)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    water = cv2.dilate(water, np.ones((11, 11), np.uint8))
    edges = cv2.Canny(cv2.GaussianBlur(quarter, (0, 0), 1.0), 60, 140)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=18, minLineLength=16, maxLineGap=2)
    structure = np.zeros(edges.shape, np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(structure, (int(x1), int(y1)), (int(x2), int(y2)), 1, 1)
    structure = cv2.dilate(structure, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    density = cv2.blur((edges > 0).astype(np.float32), (15, 15))
    red = (red_c >= green_c + 70) & (red_c >= blue_c + 70)
    stack = [grey, blue.astype(np.float32) * 255.0, water.astype(np.float32) * 255.0,
             structure.astype(np.float32) * 255.0, density * 255.0,
             red.astype(np.float32) * 255.0]
    cells = (FRAME_W // GROUND_CELL, FRAME_H // GROUND_CELL)
    return np.stack([np.clip(cv2.resize(layer, cells, interpolation=cv2.INTER_AREA) + 0.5, 0, 255)
                     .astype(np.uint8) for layer in stack])


# --------------------------------------------------------------------------------------
# frames and labels
# --------------------------------------------------------------------------------------

class FrameStore:
    """Frames and labelled boxes of the scenes, with decoded frames cached (npy memmap or RAM)."""

    def __init__(self, background_scenes, max_frames_per_scene=48, cache_dir=DEFAULT_FRAME_CACHE):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.frames = {}          # (scene, frame) -> ndarray / memmap
        self.decoded_now = 0      # PNG decodes done by this process (cold cache indicator)
        self.keys = []            # background (scene, frame) keys
        self.keepout_boxes = {}   # key -> float array [k, 4], already grown by the keep-out
        self.boxes = {}           # key -> float array [k, 4], the labelled boxes as they are
        self.band = {}            # key -> (row_min, row_max) usable for background
        for scene in background_scenes:
            frames = self.list_frames(scene)
            if max_frames_per_scene and len(frames) > max_frames_per_scene:
                pick = np.linspace(0, len(frames) - 1, max_frames_per_scene).round().astype(int)
                frames = [frames[i] for i in pick]
            for frame in frames:
                key = (scene, frame)
                boxes = np.array([a["bbox"] for a in self.annotations(scene, frame)],
                                 dtype=np.float64).reshape(-1, 4)
                if scene in BAND_LIMITED:
                    if len(boxes) == 0:
                        continue
                    self.band[key] = (int(boxes[:, 1].min()), int(boxes[:, 3].max()))
                else:
                    self.band[key] = (0, FRAME_H)
                grow = KEEPOUT_PX.get(scene, 96)
                self.keepout_boxes[key] = boxes + np.array([-grow, -grow, grow, grow])
                self.boxes[key] = boxes
                self.keys.append(key)
        if not self.keys:
            raise RuntimeError(f"no usable background frames in scenes {list(background_scenes)}")
        self._eligible = {}
        self.ground = {}            # key -> uint8 [6, H/8, W/8], see ground_layers()
        self.ground_limits = None   # limits measured on the rings around the labelled boxes
        self.open_cells = {}        # key -> flat indices of ground cells that pass the limits
        self._open_keys, self._open_cumulative = [], None
        self.mode_counts = {"near": 0, "open": 0, "any": 0, "relaxed": 0}

    # ---- ground: what the surroundings of real objects look like, and where else it looks so
    def ground_of(self, key):
        layers = self.ground.get(key)
        if layers is None:
            npy = None
            if self.cache_dir is not None:
                npy = self.cache_dir / f"{key[0]}_{key[1]:06d}.ground{GROUND_VERSION}.npy"
                png = self.image_path(*key)
                if npy.exists() and npy.stat().st_mtime >= png.stat().st_mtime:
                    layers = np.load(npy)
            if layers is None:
                layers = ground_layers(self.frame(*key))
                if npy is not None:
                    tmp = npy.with_name(f"{npy.stem}.{os.getpid()}.tmp.npy")
                    np.save(tmp, layers)
                    os.replace(tmp, npy)
            self.ground[key] = layers
        return layers

    def measure_ground(self, ring_px=56, low=10.0, high=90.0):
        """Ground limits from the rings around the labelled boxes of the background scenes.

        Labelled objects stand on open ground. The ring of `ring_px` around every box is measured
        in the six ground layers; a background patch counts as open ground when the central
        half of it is at least as bright as the `low` percentile of those rings and no more
        blue, red, structured or edgy than their `high` percentile, and holds no water. Nothing here
        is tuned by hand to a scene: the limits come from the background scenes themselves.
        """
        ring_cells = max(1, int(round(ring_px / GROUND_CELL)))
        rows = []
        for key in self.keys:
            layers = self.ground_of(key)
            inside = np.zeros(layers.shape[1:], bool)
            cell_boxes = []
            for box in self.boxes[key]:
                x1, y1 = np.floor(box[:2] / GROUND_CELL).astype(int)
                x2, y2 = np.ceil(box[2:] / GROUND_CELL).astype(int)
                cell_boxes.append((x1, y1, x2, y2))
                inside[max(y1 - 1, 0):y2 + 1, max(x1 - 1, 0):x2 + 1] = True
            for x1, y1, x2, y2 in cell_boxes:
                ring = np.zeros_like(inside)
                ring[max(y1 - ring_cells, 0):y2 + ring_cells,
                     max(x1 - ring_cells, 0):x2 + ring_cells] = True
                ring &= ~inside
                if ring.sum() >= 12:
                    rows.append(layers[:, ring].mean(axis=1))
        if not rows:
            self.ground_limits = None
            return None
        stats = np.array(rows, dtype=np.float64)
        self.ground_limits = {
            "grey_min": float(np.percentile(stats[:, 0], low)),
            "blue_share_max": float(max(np.percentile(stats[:, 1], high) / 255.0, 0.2)),
            "water_share_max": WATER_SHARE_MAX,
            "structure_share_max": float(max(np.percentile(stats[:, 3], high) / 255.0, 0.2)),
            "edge_density_max": float(np.percentile(stats[:, 4], high) / 255.0),
            "red_share_max": float(max(np.percentile(stats[:, 5], high) / 255.0, 0.2)),
            "rings_measured": int(len(rows)),
        }
        # cells whose 32 px neighbourhood (the central half of a scale 1 window) passes
        counts = []
        for key in self.keys:
            smooth = np.stack([cv2.blur(layer.astype(np.float32), (4, 4))
                               for layer in self.ground[key]])
            passing = self._passes(smooth, strict=True)
            for x1, y1, x2, y2 in self.keepout_boxes[key] / GROUND_CELL:
                passing[max(int(y1), 0):int(math.ceil(y2)), max(int(x1), 0):int(math.ceil(x2))] = False
            row_min, row_max = self.band[key]
            passing[:row_min // GROUND_CELL] = False
            passing[-(-row_max // GROUND_CELL):] = False
            self.open_cells[key] = np.flatnonzero(passing)
            counts.append(len(self.open_cells[key]))
        self._open_keys = [k for k, c in zip(self.keys, counts) if c > 0]
        self._open_cumulative = np.cumsum([c for c in counts if c > 0], dtype=np.float64)
        self.ground_limits["open_ground_share_of_cells"] = round(
            float(sum(counts)) / (len(self.keys) * self.ground[self.keys[0]][0].size), 4)
        return self.ground_limits

    def measure_open_threshold(self):
        """Old name from the first version: runs measure_ground() and returns its grey limit."""
        limits = self.measure_ground()
        return limits["grey_min"] if limits else 0.0

    def _passes(self, values, strict):
        """Ground test on layer means (shape [6] or [6, ...]).

        strict = open ground (all six limits). Otherwise only what no real object stands on is
        refused: water, and ground more blue-dominant or red-dominant than the bluest and reddest
        rings (slate roofs, deep shade, tiled roofs), so dark forest floor, busy yards and the
        like stay possible.
        """
        limits = self.ground_limits
        ok = (values[2] <= limits["water_share_max"] * 255.0) \
            & (values[1] <= limits["blue_share_max"] * 255.0) \
            & (values[5] <= limits["red_share_max"] * 255.0)
        if strict:
            ok = ok & (values[0] >= limits["grey_min"]) \
                & (values[3] <= limits["structure_share_max"] * 255.0) \
                & (values[4] <= limits["edge_density_max"] * 255.0)
        return ok

    def centre_means(self, key, x0, y0, size):
        """Means of the ground layers over the central half of a patch."""
        quarter = size // 4
        c1, r1 = (x0 + quarter) // GROUND_CELL, (y0 + quarter) // GROUND_CELL
        c2 = max(c1 + 1, -(-(x0 + size - quarter) // GROUND_CELL))
        r2 = max(r1 + 1, -(-(y0 + size - quarter) // GROUND_CELL))
        return self.ground_of(key)[:, r1:r2, c1:c2].reshape(len(GROUND_LAYERS), -1).mean(axis=1)

    @staticmethod
    def image_path(scene, frame):
        return TASK_DIR / "src" / scene / "images" / f"frame_{frame:06d}.png"

    @staticmethod
    def annotation_file(scene, frame):
        """The whole annotation JSON of a frame (annotations, and provenance on validation)."""
        path = TASK_DIR / "src" / scene / "annotations" / f"frame_{frame:06d}.json"
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def annotations(cls, scene, frame):
        return cls.annotation_file(scene, frame)["annotations"]

    @classmethod
    def list_frames(cls, scene):
        """Frame numbers that have an annotation file and a real image (placeholders are tiny)."""
        frames = []
        for path in sorted((TASK_DIR / "src" / scene / "annotations").glob("frame_*.json")):
            frame = int(path.stem.split("_")[1])
            image = cls.image_path(scene, frame)
            if image.exists() and image.stat().st_size > 200_000:
                frames.append(frame)
        return frames

    def frame(self, scene, frame):
        key = (scene, frame)
        if key in self.frames:
            return self.frames[key]
        png = self.image_path(scene, frame)
        image = None
        if self.cache_dir is not None:
            npy = self.cache_dir / f"{scene}_{frame:06d}.npy"
            if not npy.exists() or npy.stat().st_mtime < png.stat().st_mtime:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                ignore = self.cache_dir / ".gitignore"
                if not ignore.exists():
                    ignore.write_text("*\n", encoding="utf-8")
                decoded = self._decode(png)
                tmp = npy.with_name(f"{npy.stem}.{os.getpid()}.tmp.npy")
                np.save(tmp, decoded)
                os.replace(tmp, npy)
            image = np.load(npy, mmap_mode="r")
        if image is None:
            image = self._decode(png)
        self.frames[key] = image
        return image

    def _decode(self, png):
        image = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if image is None or image.shape != (FRAME_H, FRAME_W, 3):
            raise RuntimeError(f"cannot read a {FRAME_W}x{FRAME_H} frame from {png}")
        self.decoded_now += 1
        return image

    def _propose(self, rng, size, mode, keys):
        """One candidate (key, x0, y0) for the given mode, or None when the draw is unusable."""
        if mode == "any" or (mode == "open" and not self._open_keys):
            key = keys[int(rng.integers(len(keys)))]
            row_min, row_max = self.band[key]
            if row_max - row_min >= size:
                y0 = int(rng.integers(row_min, row_max - size + 1))
            else:
                y0 = int(rng.integers(row_min, row_max + 1)) - size // 2
            return key, int(rng.integers(0, FRAME_W - size + 1)), min(max(y0, 0), FRAME_H - size)
        if mode == "open":
            pick = int(np.searchsorted(self._open_cumulative,
                                       rng.random() * self._open_cumulative[-1], side="right"))
            key = self._open_keys[min(pick, len(self._open_keys) - 1)]
            cells = self.open_cells[key]
            row, column = divmod(int(cells[int(rng.integers(len(cells)))]), FRAME_W // GROUND_CELL)
            centre = (np.array([column, row]) + rng.random(2)) * GROUND_CELL
        else:   # near: in the ring of NEAR_RING_PX around one labelled box
            key = keys[int(rng.integers(len(keys)))]
            boxes = self.boxes[key]
            if len(boxes) == 0:
                return None
            box = boxes[int(rng.integers(len(boxes)))]
            inner, outer = NEAR_RING_PX
            centre = rng.uniform(box[:2] - outer, box[2:] + outer)
            if np.all(centre > box[:2] - inner) and np.all(centre < box[2:] + inner):
                return None
        if not (0 <= centre[0] < FRAME_W and 0 <= centre[1] < FRAME_H):
            return None
        row_min, row_max = self.band[key]
        x0 = min(max(int(centre[0] - size / 2.0), 0), FRAME_W - size)
        y0 = min(max(int(centre[1] - size / 2.0), 0), FRAME_H - size)
        if row_max - row_min >= size:
            if y0 < row_min or y0 + size > row_max:
                return None
        elif not row_min <= y0 + size // 2 <= row_max:
            return None
        return key, x0, y0

    def sample_patch(self, rng, size, align=1, mode="any", allow_water=False):
        """Random clean background square: returns (scene, frame, x0, y0, uint8 copy).

        mode "near": patch centre 100 to 500 source px from a labelled box AND open ground;
        mode "open": open ground anywhere in the frame (see measure_ground);
        mode "any":  anywhere, but not on water or on ground more blue or red than the rings
                     allow (slate and tiled roofs, deep shade) unless allow_water, which pure
                     background windows pass.
        Every patch is clear of all labelled boxes and their keep-out and respects the row band.
        After 120 failed tries the mode falls back to "any", after 180 anything clean is taken.
        """
        keys = self._eligible.get(size)
        if keys is None:
            keys = [k for k in self.keys if self.band[k][1] - self.band[k][0] >= size]
            if not keys:  # band too thin everywhere: fall back to patch centre inside the band
                keys = list(self.keys)
            self._eligible[size] = keys
        if self.ground_limits is None:
            mode, allow_water = "any", True
        wanted = mode
        for attempt in range(240):
            if attempt == 120:
                mode = "any"
            if attempt == 180:
                allow_water = True
            proposal = self._propose(rng, size, mode, keys)
            if proposal is None:
                continue
            key, x0, y0 = proposal
            if align > 1:
                x0 -= x0 % align
                y0 -= y0 % align
            boxes = self.keepout_boxes[key]
            if len(boxes) and np.any((x0 < boxes[:, 2]) & (x0 + size > boxes[:, 0])
                                     & (y0 < boxes[:, 3]) & (y0 + size > boxes[:, 1])):
                continue
            if mode != "any" or not allow_water:
                if not self._passes(self.centre_means(key, x0, y0, size), strict=mode != "any"):
                    continue
            self.mode_counts[mode if mode == wanted else "relaxed"] += 1
            image = self.frame(*key)
            return key[0], key[1], x0, y0, np.array(image[y0:y0 + size, x0:x0 + size])
        raise RuntimeError(f"no clean {size} px background patch found in 240 tries")


# --------------------------------------------------------------------------------------
# sprites
# --------------------------------------------------------------------------------------

class Sprite:
    """One cached sprite: padded colour (rim filled), hard 0/1 alpha, hull and box margins."""

    __slots__ = ("class_index", "class_name", "sprite_id", "scene", "frame", "zoom", "native",
                 "bgr", "alpha", "centre", "hull", "margins", "pivot", "box_wh", "mask_px",
                 "small_mask")


def locate_instances(entry):
    """Labelled boxes a bank entry can have been cut from: list of (bbox, track_id).

    The bank does not always say which labelled box a sprite belongs to, so the boxes of the same
    class in the same frame are narrowed down by `box_src` when the entry carries it, otherwise
    by the size of the organiser box stored in `box_in_sprite`. On helsinki (no provenance, one
    instance per class) the class name stands in for the track id, as in the SPEC.
    """
    frame = entry.get("frame")
    if frame is None:
        return []
    try:
        data = FrameStore.annotation_file(entry["_scene"], frame)
    except (OSError, ValueError):
        return []
    provenance = data.get("provenance") or []
    found = []
    for index, annotation in enumerate(data.get("annotations", [])):
        if annotation["object_id"] != entry["class_name"]:
            continue
        track = provenance[index].get("track_id") if index < len(provenance) else None
        found.append((tuple(annotation["bbox"]), track or entry["class_name"]))
    box_src = entry.get("box_src")
    if box_src:
        exact = [f for f in found if max(abs(a - b) for a, b in zip(f[0], box_src)) <= 1]
        if exact:
            return exact
    box = entry.get("box_in_sprite")
    if box and len(found) > 1:
        width, height = box[2] - box[0], box[3] - box[1]
        same = [f for f in found if abs((f[0][2] - f[0][0]) - width) <= 2
                and abs((f[0][3] - f[0][1]) - height) <= 2]
        if same:
            found = same
    return found


def read_bank_entries(bank_paths, sprite_scenes, exclude_sprite_ids=(), exclude_tracks=(),
                      min_recut_mask_px=300, dropped=None):
    """Bank entries of the allowed scenes that are fit to paste. Returns a list of entries.

    Applied in this order; every entry that goes is appended to `dropped` (a list) with a reason:
      1. vetoes: ids in `exclude_sprite_ids`; tracks in `exclude_tracks` or in the
         `excluded_tracks` block of ANY bank, the default banks included even when they are
         not used for sprites, or in KNOWN_BAD_TRACKS (a wrong pseudo-label is wrong everywhere). The track of an entry is its `track` field or, failing that, the track of the
         labelled box it was cut from (frame annotation, provenance[i].track_id). An entry that
         cannot be told apart from an excluded track is dropped as well;
      2. duplicates: one entry per (class, scene, frame, zoom, labelled box); the first bank wins;
      3. resolution: per (class, scene) only the highest zoom is kept. Masks made at zoom 0 or 1
         are 4 or 2 px coarse: small objects come out cut off or as blobs, so they are only
         used for a class that has nothing better in that scene;
      4. such low zoom survivors need a mask of at least `min_recut_mask_px` native px, unless
         that would leave the class without any sprite (then the largest one stays).
    """
    dropped = dropped if dropped is not None else []
    entries, excluded = [], {str(t): "listed in --exclude-tracks" for t in exclude_tracks}
    excluded.update({t: f"KNOWN_BAD_TRACKS of synth.py: {why}" for t, why in KNOWN_BAD_TRACKS.items()
                     if t not in excluded})
    given = [Path(b).resolve() for b in bank_paths]
    veto_only = [b for b in DEFAULT_BANKS if b.exists() and b.resolve() not in given]
    for bank_path in list(bank_paths) + veto_only:
        bank_path = Path(bank_path)
        try:
            with open(bank_path, "r", encoding="utf-8") as handle:
                bank = json.load(handle)
        except (OSError, ValueError) as error:
            print(f"[synth] skipping bank {bank_path}: {error}")
            continue
        if isinstance(bank, dict):
            block = bank.get("excluded_tracks") or {}
            for track in block:
                reason = block[track] if isinstance(block, dict) else "no reason given"
                excluded.setdefault(str(track), f"excluded_tracks of {bank_path.parent.name}/"
                                                f"{bank_path.name}: {reason}")
        if bank_path in veto_only:
            continue        # a default bank that was not asked for: only its vetoes count
        for entry in (bank["sprites"] if isinstance(bank, dict) else bank):
            scene = SCENE_OF_SOURCE.get(entry.get("source"))
            if scene is None or scene not in sprite_scenes:
                continue
            if entry.get("class_name") not in CLASSES:
                continue
            if not (bank_path.parent / entry["file"]).exists():
                continue
            item = dict(entry)
            item["_scene"] = scene
            item["_path"] = str(bank_path.parent / entry["file"])
            item["_zoom"] = int(entry.get("zoom", 2))
            item["_id"] = str(entry.get("id", Path(entry["file"]).stem))
            entries.append(item)

    def drop(item, reason):
        dropped.append({"id": item["_id"], "class_name": item["class_name"],
                        "scene": item["_scene"], "frame": item.get("frame"),
                        "zoom": item["_zoom"], "reason": reason})

    vetoed_ids = {str(i) for i in exclude_sprite_ids}
    kept, seen = [], set()
    for item in entries:
        instances = locate_instances(item)
        tracks = {item["track"]} if item.get("track") else {track for _, track in instances}
        bad = sorted(tracks & set(excluded))
        if item["_id"] in vetoed_ids:
            drop(item, "listed in --exclude-sprite-ids")
            continue
        if bad:
            certain = "" if tracks <= set(excluded) else "cannot be told apart from "
            drop(item, f"{certain}track {bad[0]}: {excluded[bad[0]]}")
            continue
        instance = instances[0][0] if len(instances) == 1 else ("size",) + tuple(item.get("size", ()))
        key = (item["class_name"], item["_scene"], item.get("frame"), item["_zoom"], instance)
        if key in seen:
            drop(item, "duplicate of an earlier entry (same class, scene, frame, zoom and box)")
            continue
        seen.add(key)
        kept.append(item)

    best = {}
    for item in kept:
        key = (item["class_name"], item["_scene"])
        best[key] = max(best.get(key, -1), item["_zoom"])
    survivors = []
    for item in kept:
        top = best[(item["class_name"], item["_scene"])]
        if item["_zoom"] == top:
            survivors.append(item)
        else:
            drop(item, f"zoom {item['_zoom']} mask, the class has zoom {top} sprites in this scene")

    # low zoom survivors: the mask has to be large enough to be more than a blob
    areas = {}
    for item in survivors:
        if item["_zoom"] < 2 and min_recut_mask_px > 0:
            bgra = cv2.imread(item["_path"], cv2.IMREAD_UNCHANGED)
            if bgra is not None and bgra.ndim == 3 and bgra.shape[2] == 4:
                areas[item["_id"]] = int(np.sum(bgra[:, :, 3] >= SOLID_ALPHA * 255.0))
    final = []
    for item in survivors:
        area = areas.get(item["_id"])
        if area is None or area >= min_recut_mask_px:
            final.append(item)
            continue
        group = [areas.get(other["_id"], 10 ** 9) for other in survivors
                 if (other["class_name"], other["_scene"]) == (item["class_name"], item["_scene"])]
        if max(group) < min_recut_mask_px and area == max(group):
            item["_small_mask_px"] = area      # nothing better exists: kept, and said so
            final.append(item)
        else:
            drop(item, f"zoom {item['_zoom']} mask of {area} px, below {min_recut_mask_px} px")
    return final


def coverage_of_binary_mask(colour, mask):
    """Soft coverage for a sprite that only has a binary mask (GrabCut cuts, alpha 0 or 255).

    The outermost mask pixels of such a sprite are frame pixels in which object and old ground
    are mixed; pasted opaque on darker or brighter ground they show as a halo. Their coverage
    is estimated by projecting the pixel colour on the line from the nearest outside colour (old
    ground, which both banks keep under alpha 0) to the nearest core colour (mask eroded by one
    pixel). Where the two differ by less than 12 grey levels the pixel counts as fully covered,
    and so does a pixel whose colour lies more than 30 grey levels off that line: it is no
    mixture but a part of its own, such as the white trim of the spacecraft.
    """
    alpha = mask.astype(np.float32)
    core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    rim = mask & ~core
    if not core.any() or not rim.any() or mask.all():
        return alpha
    soft = cv2.GaussianBlur(colour, (3, 3), 0)
    inside = fill_from_nearest_solid(colour, core).astype(np.float32)
    outside = fill_from_nearest_solid(soft, ~mask).astype(np.float32)
    span = inside - outside
    length2 = np.sum(span * span, axis=2)
    along = np.sum((colour.astype(np.float32) - outside) * span, axis=2) / np.maximum(length2, 1.0)
    estimate = np.clip(along, 0.0, 1.0)
    off_line = colour.astype(np.float32) - (outside + estimate[:, :, None] * span)
    own_part = np.sum(off_line * off_line, axis=2) > 30.0 ** 2
    estimate = np.where((length2 < 3 * 12.0 ** 2) | own_part, 1.0, estimate)
    alpha[rim] = estimate[rim]
    return alpha


def hard_mask_and_colour(colour, bank_alpha, keep_thin_parts=True):
    """The one rim rule for every sprite. Returns (hard mask, colour with the rim repainted).

    solid  = alpha >= SOLID_ALPHA (0.8): colour trusted, part of the mask;
    rim    = the rest next to a solid pixel: NOT in the mask and repainted from the nearest mask
             pixel, so neither old ground nor the bank's extrapolated rim colour is ever shown;
             the 3x3 feather applied after the warp provides the soft edge;
    thin   = alpha >= 0.5 with no solid pixel among its 8 neighbours (a gun barrel, a rotor
             blade, an antenna): kept in the mask, otherwise it would vanish; its colour is kept
             but clipped per channel to the 1st..99th percentile of the solid colours, because
             un-mixed colours of thin parts overshoot. Off for re-cut low zoom sprites, whose
             upsampled alpha has a rim several pixels wide.
    Binary masks first get a coverage estimate for their outermost pixels.
    """
    intermediate = (bank_alpha > 0.02) & (bank_alpha < 0.98)
    if intermediate.sum() < 0.01 * max(int((bank_alpha >= 0.5).sum()), 1):
        bank_alpha = coverage_of_binary_mask(colour, bank_alpha >= 0.5)
    solid = bank_alpha >= SOLID_ALPHA
    if not solid.any():
        solid = bank_alpha >= 0.5
    mask = solid
    if keep_thin_parts:
        near_solid = cv2.dilate(solid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        thin = (bank_alpha >= 0.5) & ~near_solid
        count, labels, stats, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), connectivity=8)
        for index in range(1, count):          # specks of one or two pixels are matte noise
            if stats[index, cv2.CC_STAT_AREA] < 3:
                thin[labels == index] = False
        if thin.any():
            low = np.percentile(colour[solid], 1, axis=0)
            high = np.percentile(colour[solid], 99, axis=0)
            colour = colour.copy()
            colour[thin] = np.clip(colour[thin], low, high).astype(np.uint8)
            mask = solid | thin
    return mask, fill_from_nearest_solid(colour, mask)


def recut_from_frame(entry, sprite_bgra):
    """Native pixels for an upsampled sprite, or None when the instance cannot be located."""
    scene, frame = entry["_scene"], entry.get("frame")
    png = FrameStore.image_path(scene, frame)
    if frame is None or not png.exists():
        return None
    try:
        boxes = [a["bbox"] for a in FrameStore.annotations(scene, frame)
                 if a["object_id"] == entry["class_name"]]
    except OSError:
        return None
    image = recut_from_frame.cache.get(str(png))
    if image is None:
        image = cv2.imread(str(png), cv2.IMREAD_COLOR)
        recut_from_frame.cache = {str(png): image}  # keep one frame only
    if image is None:
        return None
    height, width = sprite_bgra.shape[:2]
    bx = entry["box_in_sprite"]
    solid = sprite_bgra[:, :, 3] > 200
    blur_sigma = 0.6 * (4 >> entry["_zoom"])
    best = None
    for box in boxes:
        x0, y0 = int(box[0] - bx[0]), int(box[1] - bx[1])
        if x0 < 0 or y0 < 0 or x0 + width > FRAME_W or y0 + height > FRAME_H:
            continue
        real = image[y0:y0 + height, x0:x0 + width]
        soft = cv2.GaussianBlur(real, (0, 0), blur_sigma)
        error = float(np.abs(soft.astype(np.int16) - sprite_bgra[:, :, :3].astype(np.int16))[solid].mean())
        if best is None or error < best[0]:
            best = (error, real.copy())
    if best is None or best[0] > 25.0:
        return None
    return best[1]


recut_from_frame.cache = {}


def load_sprites(entries, recut_native=True):
    sprites = []
    for entry in entries:
        bgra = cv2.imread(entry["_path"], cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
            print(f"[synth] skipping unreadable sprite {entry['_path']}")
            continue
        colour = bgra[:, :, :3]
        native = entry["_zoom"] >= 2
        if not native and recut_native:
            real = recut_from_frame(entry, bgra)
            if real is not None:
                colour, native = real, True
        bank_alpha = bgra[:, :, 3].astype(np.float32) / 255.0
        if not (bank_alpha >= 0.5).any():
            continue
        colour = cv2.copyMakeBorder(colour, SPRITE_PAD, SPRITE_PAD, SPRITE_PAD, SPRITE_PAD,
                                    cv2.BORDER_REPLICATE)
        bank_alpha = cv2.copyMakeBorder(bank_alpha, SPRITE_PAD, SPRITE_PAD, SPRITE_PAD, SPRITE_PAD,
                                        cv2.BORDER_CONSTANT, value=0)
        mask, colour = hard_mask_and_colour(np.ascontiguousarray(colour), bank_alpha,
                                            keep_thin_parts=entry["_zoom"] >= 2)
        alpha = mask.astype(np.float32)

        ys, xs = np.nonzero(mask)
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()   # inclusive pixel indices
        centre = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        points = np.stack([xs, ys], axis=1).astype(np.int32)
        hull = cv2.convexHull(points).reshape(-1, 2).astype(np.float64) - centre

        # organiser box relative to the mask: box_in_sprite is in unpadded sprite coordinates and
        # uses edge coordinates, the mask bbox in edge coordinates is [x1, x2 + 1).
        bx = entry["box_in_sprite"]
        margins = np.array([
            (x1 - SPRITE_PAD) - bx[0],
            (y1 - SPRITE_PAD) - bx[1],
            bx[2] - (x2 + 1 - SPRITE_PAD),
            bx[3] - (y2 + 1 - SPRITE_PAD),
        ], dtype=np.float64)

        sprite = Sprite()
        sprite.class_name = entry["class_name"]
        sprite.class_index = CLASSES.index(entry["class_name"])
        sprite.sprite_id = str(entry.get("id", Path(entry["_path"]).stem))
        sprite.scene = entry["_scene"]
        sprite.frame = entry.get("frame", -1)
        sprite.zoom = entry["_zoom"]
        sprite.mask_px = int(mask.sum())
        sprite.small_mask = "_small_mask_px" in entry
        sprite.native = native
        sprite.bgr = np.ascontiguousarray(colour)
        sprite.alpha = np.ascontiguousarray(alpha)
        sprite.centre = centre
        sprite.hull = hull
        sprite.margins = margins
        sprite.pivot = float(colour[mask].mean()) / 255.0
        sprite.box_wh = (bx[2] - bx[0], bx[3] - bx[1])
        sprites.append(sprite)
    recut_from_frame.cache = {}
    return sprites


class RenderedSprite:
    """A sprite after photometric jitter and warp, positioned relative to its reference point."""

    __slots__ = ("colour", "alpha", "anchor", "tight", "label")
    # anchor: position of the reference point inside the canvas, before the sub-pixel shift
    # tight, label: [x1, y1, x2, y2] edge coordinates relative to the reference point


# --------------------------------------------------------------------------------------
# the dataset
# --------------------------------------------------------------------------------------

class SynthWindows:
    """Map-style dataset of synthetic windows. Sample i depends only on (seed, epoch, i)."""

    def __init__(self, n, sprite_banks=None, sprite_scenes=("helsinki",),
                 background_scenes=("helsinki",), rot_max=180.0, rot_max_per_class=None,
                 seed=0, neg_frac=0.25, level_probs=(1.0, 1.0, 1.0), center_jitter=6.0,
                 jitter_prob=0.5, neg_guard=10.0, lossless_prob=0.25, bg_open_prob=0.7,
                 bg_near_frac=0.5, warp_interp=("lanczos4", "nearest"), sprite_blur_max=0.0,
                 distractor_prob=0.3, strip_prob=0.15, post_noise=0, max_bg_frames=48,
                 frame_cache=DEFAULT_FRAME_CACHE, recut_native=True, exclude_sprite_ids=(),
                 exclude_tracks=(), min_recut_mask_px=300):
        if sprite_banks is None:
            sprite_banks = [p for p in DEFAULT_BANKS if Path(p).exists()]
        self.params = dict(
            n=int(n), sprite_banks=[str(p) for p in sprite_banks],
            sprite_scenes=list(sprite_scenes), background_scenes=list(background_scenes),
            rot_max=float(rot_max), rot_max_per_class=dict(rot_max_per_class or {}),
            seed=int(seed), neg_frac=float(neg_frac), level_probs=[float(p) for p in level_probs],
            center_jitter=float(center_jitter), jitter_prob=float(jitter_prob),
            neg_guard=float(neg_guard), lossless_prob=float(lossless_prob),
            bg_open_prob=float(bg_open_prob), bg_near_frac=float(bg_near_frac),
            warp_interp=[str(name) for name in warp_interp],
            sprite_blur_max=float(sprite_blur_max),
            distractor_prob=float(distractor_prob), strip_prob=float(strip_prob),
            post_noise=int(post_noise), max_bg_frames=int(max_bg_frames),
            frame_cache=str(frame_cache) if frame_cache else None, recut_native=bool(recut_native),
            exclude_sprite_ids=[str(i) for i in exclude_sprite_ids],
            exclude_tracks=[str(i) for i in exclude_tracks],
            min_recut_mask_px=int(min_recut_mask_px))
        unknown = set(self.params["warp_interp"]) - set(INTERPOLATIONS)
        if unknown or not self.params["warp_interp"]:
            raise ValueError(f"warp_interp must name some of {sorted(INTERPOLATIONS)}")
        unknown = set(self.params["rot_max_per_class"]) - set(CLASSES)
        if unknown:
            raise ValueError(f"unknown classes in the rotation override: {sorted(unknown)}")
        self.epoch = 0
        self.dropped_sprites = []
        self.entries = read_bank_entries(
            sprite_banks, set(sprite_scenes), exclude_sprite_ids=self.params["exclude_sprite_ids"],
            exclude_tracks=self.params["exclude_tracks"],
            min_recut_mask_px=self.params["min_recut_mask_px"], dropped=self.dropped_sprites)
        self.available = sorted({CLASSES.index(e["class_name"]) for e in self.entries})
        if not self.available:
            raise RuntimeError(f"no sprites for scenes {list(sprite_scenes)} in {sprite_banks}")
        self.missing_classes = [c for i, c in enumerate(CLASSES) if i not in self.available]

        # label plan: exact negative share, positives cycle over the available classes
        n = int(n)
        n_neg = int(round(n * float(neg_frac)))
        plan = np.full(n, BACKGROUND, dtype=np.int16)
        plan[n_neg:] = np.resize(np.array(self.available, dtype=np.int16), n - n_neg)
        np.random.default_rng([int(seed), 7]).shuffle(plan)
        self.plan = plan
        self._ready = False

    # lazy heavy state, dropped when pickled so DataLoader workers start light -------------
    def __getstate__(self):
        state = dict(self.__dict__)
        for name in ("store", "sprites", "by_class", "class_margins"):
            state.pop(name, None)
        state["_ready"] = False
        return state

    def _setup(self):
        p = self.params
        cv2.setNumThreads(1)
        self.sprites = load_sprites(self.entries, recut_native=p["recut_native"])
        self.by_class = {}
        for sprite in self.sprites:
            self.by_class.setdefault(sprite.class_index, []).append(sprite)
        self.class_margins = {c: np.median(np.stack([s.margins for s in group]), axis=0)
                              for c, group in self.by_class.items()}
        self.store = FrameStore(p["background_scenes"], p["max_bg_frames"], p["frame_cache"])
        self.store.measure_ground()
        self._ready = True

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.plan)

    # ---------------------------------------------------------------------------------
    def _rot_max(self, class_index):
        return float(self.params["rot_max_per_class"].get(CLASSES[class_index],
                                                          self.params["rot_max"]))

    def _render(self, rng, sprite):
        """Jitter, rotate and scale one sprite. Returns a RenderedSprite."""
        p = self.params
        rot_max = self._rot_max(sprite.class_index)
        lossless = rng.random() < p["lossless_prob"]
        if lossless:
            turns = [k for k in (-1, 0, 1, 2) if abs(k) * 90 <= rot_max + 1e-6]
            angle, scale = 90.0 * turns[int(rng.integers(len(turns)))], 1.0
        else:
            angle, scale = rng.uniform(-rot_max, rot_max), rng.uniform(0.92, 1.08)
        lo, hi = _env_range('SYNTH_SPRITE_GAIN', 0.92, 1.08)
        wide = hi/lo > 1.5  # a wide gain range also widens contrast and gamma
        lut = make_lut(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
                       rng.uniform(0.8, 1.25) if wide else rng.uniform(0.92, 1.08),
                       rng.uniform(0.8, 1.25) if wide else rng.uniform(0.9, 1.1), sprite.pivot)
        sigma = 0.0 if lossless else rng.uniform(0.0, p["sprite_blur_max"])
        shift = (0.0, 0.0) if lossless else (rng.random(), rng.random())
        names = p["warp_interp"]
        interpolation = INTERPOLATIONS[names[int(rng.integers(len(names)))]]

        height, width = sprite.alpha.shape
        matrix = cv2.getRotationMatrix2D((float(sprite.centre[0]), float(sprite.centre[1])),
                                         angle, scale)
        linear = matrix[:, :2].copy()
        corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
                           dtype=np.float64) - sprite.centre
        reach = np.abs(corners @ linear.T).max(axis=0)
        anchor = np.ceil(reach) + 2.0          # reference point inside the canvas
        canvas_w, canvas_h = int(2 * anchor[0] + 2), int(2 * anchor[1] + 2)
        matrix[:, 2] = anchor + np.array(shift) - linear @ sprite.centre
        if lossless:
            # whole-pixel translation and a quarter turn: warpAffine then copies pixels exactly
            matrix[:, 2] = np.rint(matrix[:, 2])
        placed_anchor = linear @ sprite.centre + matrix[:, 2]

        colour = cv2.LUT(sprite.bgr, lut).astype(np.float32)
        # colour: the drawn interpolation (a quarter turn on the lossless path copies pixels
        # whatever the kernel). The mask is always warped linearly, cut at 0.5 and feathered by
        # a 3x3 Gaussian, so no mixed rim pixel of the bank is ever shown opaque.
        colour = cv2.warpAffine(colour, matrix, (canvas_w, canvas_h),
                                flags=cv2.INTER_NEAREST if lossless else interpolation,
                                borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(sprite.alpha, matrix, (canvas_w, canvas_h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
        alpha = cv2.GaussianBlur((alpha >= 0.5).astype(np.float32), (3, 3), 0)
        if sigma >= 0.2:
            colour = cv2.GaussianBlur(colour, (5, 5), sigma)
            alpha = cv2.GaussianBlur(alpha, (5, 5), sigma)

        # tight bbox of the rotated mask, from the rotated convex hull of the mask pixels
        hull = sprite.hull @ linear.T
        half_pixel = 0.5 * scale
        tight = np.array([hull[:, 0].min() - half_pixel, hull[:, 1].min() - half_pixel,
                          hull[:, 0].max() + half_pixel, hull[:, 1].max() + half_pixel])
        # side margins follow the rotation: window side normal -> direction in the sprite frame
        margins = self.class_margins[sprite.class_index]
        unit = linear / scale
        label = tight.copy()
        for side, normal in enumerate(((-1.0, 0.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0))):
            u = unit.T @ np.array(normal)
            weights = np.array([max(0.0, -u[0]), max(0.0, -u[1]), max(0.0, u[0]), max(0.0, u[1])]) ** 2
            grow = float(weights @ margins) * scale
            label[side] += -grow if side < 2 else grow
        if label[2] - label[0] < 2 or label[3] - label[1] < 2:
            label = tight.copy()

        rendered = RenderedSprite()
        rendered.colour, rendered.alpha = colour, alpha
        rendered.anchor = placed_anchor
        rendered.tight, rendered.label = tight, label
        return rendered

    @staticmethod
    def _paste(patch, rendered, reference_xy):
        """Composite so that the sprite reference point lands on reference_xy (patch coords).

        The canvas already carries a sub-pixel shift in [0, 1); the remaining integer part of the
        placement is absorbed here, so the final position is exact to the rounding of one pixel.
        """
        origin = np.rint(np.asarray(reference_xy) - rendered.anchor).astype(int)
        canvas_h, canvas_w = rendered.alpha.shape
        size_y, size_x = patch.shape[:2]
        x1, y1 = max(origin[0], 0), max(origin[1], 0)
        x2, y2 = min(origin[0] + canvas_w, size_x), min(origin[1] + canvas_h, size_y)
        if x2 <= x1 or y2 <= y1:
            return origin + rendered.anchor
        cx1, cy1 = x1 - origin[0], y1 - origin[1]
        cx2, cy2 = cx1 + (x2 - x1), cy1 + (y2 - y1)
        alpha = rendered.alpha[cy1:cy2, cx1:cx2, None]
        below = patch[y1:y2, x1:x2].astype(np.float32)
        mixed = below + alpha * (rendered.colour[cy1:cy2, cx1:cx2] - below)
        patch[y1:y2, x1:x2] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)
        return origin + rendered.anchor   # where the reference point really ended up

    def _choose_sprite(self, rng, class_index, level=None):
        group = self.by_class[class_index]
        if level is not None:
            usable = [s for s in group if s.native or s.zoom >= level]
            group = usable or group
        return group[int(rng.integers(len(group)))]

    def _choose_level(self, rng, class_index):
        probs = np.array(self.params["level_probs"], dtype=np.float64)
        group = self.by_class[class_index]
        if not any(s.native for s in group):      # only upsampled sprites: stay at L <= zoom
            best = max(s.zoom for s in group)
            probs = np.where(np.arange(3) <= best, probs, 0.0)
        probs = probs / probs.sum()
        return int(rng.choice(3, p=probs))

    # ---------------------------------------------------------------------------------
    def __getitem__(self, index):
        if not self._ready:
            self._setup()
        p = self.params
        index = int(index)
        rng = np.random.default_rng([p["seed"], self.epoch, index])
        label = int(self.plan[index])

        # the anchor object decides level and scale, also for negatives (same mix as positives)
        anchor_class = label if label != BACKGROUND else int(
            self.available[int(rng.integers(len(self.available)))])
        level = self._choose_level(rng, anchor_class)
        factor = LEVEL_FACTOR[level]
        sprite = self._choose_sprite(rng, anchor_class, level)
        obj = self._render(rng, sprite)
        box_w = float(obj.label[2] - obj.label[0])
        box_h = float(obj.label[3] - obj.label[1])
        scale = pick_scale(factor, max(box_w, box_h))
        size = WINDOW * scale
        longer = max(box_w, box_h)

        # level 0 views always start at source (0, 0): keep the background on that 4 px grid
        kind = "positive"
        if label == BACKGROUND:
            kind = "pure" if rng.random() < 0.5 else "hard"

        # ground under the window centre: like the surroundings of real objects for the
        # bg_open_prob share (half of it right next to labelled boxes), anywhere for the rest.
        # A window that gets a sprite pasted never lies on water; pure background may.
        mode = "any"
        if rng.random() < p["bg_open_prob"]:
            mode = "near" if rng.random() < p["bg_near_frac"] else "open"
        scene, frame, _, _, patch = self.store.sample_patch(
            rng, size, align=factor if level == 0 else 1, mode=mode, allow_water=kind == "pure")
        blo, bhi = _env_range('SYNTH_BG_GAIN', 1.0, 1.0)
        if bhi > blo:  # scene exposure differs from flight to flight, independently of the objects
            patch = cv2.LUT(patch, make_lut(float(np.exp(rng.uniform(np.log(blo), np.log(bhi)))),
                                            rng.uniform(0.85, 1.18), rng.uniform(0.85, 1.18), 0.45))
        window_centre = np.array([size / 2.0 - 0.5, size / 2.0 - 0.5])  # pixel-centre coords

        offset_src = np.zeros(2)
        main_tight = None
        if kind == "positive":
            # a real window is centred on the box centre rounded to a delivered pixel: +-f/2 px
            offset_src = rng.uniform(-factor / 2.0, factor / 2.0, size=2)
            if p["center_jitter"] > 0 and rng.random() < p["jitter_prob"]:
                offset_src += rng.uniform(-p["center_jitter"], p["center_jitter"], size=2) * scale
        elif kind == "hard":
            nearest = max(0.78 * longer, (p["neg_guard"] + 0.5) * scale)
            offset_src = self._hard_negative_offset(rng, obj, scale, nearest)

        if kind != "pure":
            label_centre = np.array([(obj.label[0] + obj.label[2]) / 2.0,
                                     (obj.label[1] + obj.label[3]) / 2.0])
            wanted = window_centre + offset_src - label_centre
            placed = self._paste(patch, obj, wanted)
            offset_src = offset_src + (placed - wanted)
            main_tight = obj.tight + np.tile(placed, 2)

        if kind == "positive" and rng.random() < p["distractor_prob"] and len(self.available) > 1:
            self._add_distractor(rng, patch, anchor_class, level, scale, main_tight)

        lut = make_lut(rng.uniform(0.96, 1.04), rng.uniform(0.96, 1.04),
                       rng.uniform(0.95, 1.05), 0.45)
        patch = cv2.LUT(patch, lut)

        strip = None
        if rng.random() < p["strip_prob"]:
            per_window_px = scale // factor
            strip = (int(rng.integers(4)), int(rng.integers(1, 20 * per_window_px + 1)))
        window = reduce_like_evaluator(patch, factor, scale, strip)
        if p["post_noise"]:
            noise = rng.integers(-1, 2, size=window.shape)
            window = np.clip(window.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if kind == "pure":
            track = "background"
        elif kind == "hard":
            track = f"hardneg:{sprite.sprite_id}"
        else:
            track = sprite.sprite_id
        return {
            "x": np.ascontiguousarray(window),
            "label": np.int16(label),
            "level": np.int8(level),
            "scale": np.int8(scale),
            "size_px": np.float32(max(box_w, box_h) / factor),
            "box_w": np.float32(box_w),
            "box_h": np.float32(box_h),
            "off": (offset_src / scale).astype(np.float32),
            "scene": scene,
            "frame": np.int32(frame),
            "track": track[:48],
            "source": "synthetic",
        }

    @staticmethod
    def _hard_negative_offset(rng, obj, scale, nearest):
        """Offset (source px) of the object centre: at least `nearest` away, still partly visible."""
        half = WINDOW * scale / 2.0
        half_w = (obj.label[2] - obj.label[0]) / 2.0
        half_h = (obj.label[3] - obj.label[1]) / 2.0
        for _ in range(30):
            distance = rng.uniform(1.0, 2.0) * nearest
            theta = rng.uniform(0.0, 2.0 * math.pi)
            offset = np.array([math.cos(theta), math.sin(theta)]) * distance
            visible_x = min(half, offset[0] + half_w) - max(-half, offset[0] - half_w)
            visible_y = min(half, offset[1] + half_h) - max(-half, offset[1] - half_h)
            if visible_x >= min(2 * half_w, 8 * scale) * 0.75 and \
                    visible_y >= min(2 * half_h, 8 * scale) * 0.75:
                return offset
        sign = rng.choice([-1.0, 1.0], size=2)
        return sign * nearest / math.sqrt(2.0) * 1.02

    def _add_distractor(self, rng, patch, main_class, level, scale, main_tight):
        others = [c for c in self.available if c != main_class]
        other = self._choose_sprite(rng, int(others[int(rng.integers(len(others)))]), level)
        obj = self._render(rng, other)
        size = patch.shape[0]
        centre = np.array([size / 2.0 - 0.5, size / 2.0 - 0.5])
        width, height = obj.label[2] - obj.label[0], obj.label[3] - obj.label[1]
        longer = max(width, height)
        label_centre = np.array([(obj.label[0] + obj.label[2]) / 2.0,
                                 (obj.label[1] + obj.label[3]) / 2.0])
        need = 4.0 * scale   # at least 4 window px of the distractor must be inside
        for _ in range(20):
            position = rng.uniform(-longer / 2.0, size + longer / 2.0, size=2)
            if np.linalg.norm(position - centre) < 0.80 * longer:
                continue
            reference = position - label_centre
            tight = obj.tight + np.tile(reference, 2)
            inside_x = min(size, tight[2]) - max(0.0, tight[0])
            inside_y = min(size, tight[3]) - max(0.0, tight[1])
            if inside_x < min(need, tight[2] - tight[0]) or inside_y < min(need, tight[3] - tight[1]):
                continue
            gap = 2.0 * scale
            if main_tight is not None and not (
                    tight[2] + gap < main_tight[0] or tight[0] - gap > main_tight[2]
                    or tight[3] + gap < main_tight[1] or tight[1] - gap > main_tight[3]):
                continue
            self._paste(patch, obj, reference)
            return True
        return False


# --------------------------------------------------------------------------------------
# dumping, manifest, sheets
# --------------------------------------------------------------------------------------

FIELDS = ("x", "label", "level", "scale", "size_px", "box_w", "box_h", "off", "scene", "frame",
          "track", "source")
DTYPES = {"x": np.uint8, "label": np.int16, "level": np.int8, "scale": np.int8,
          "size_px": np.float32, "box_w": np.float32, "box_h": np.float32, "off": np.float32,
          "scene": "<U16", "frame": np.int32, "track": "<U48", "source": "<U10"}

_WORKER_DATASET = None


def _worker_init(params, epoch):
    global _WORKER_DATASET
    _WORKER_DATASET = SynthWindows(**params)
    _WORKER_DATASET.set_epoch(epoch)


def _worker_chunk(indices):
    return [_WORKER_DATASET[i] for i in indices]


def generate(dataset, workers=1, progress=True):
    """All samples of a dataset as SPEC arrays, plus timing facts."""
    n = len(dataset)
    started = time.perf_counter()
    samples = []
    if workers <= 1:
        dataset[0] if n else None          # builds caches; timed separately below
        setup_seconds = time.perf_counter() - started
        started_steady = time.perf_counter()
        for i in range(n):
            samples.append(dataset[i])
            if progress and (i + 1) % 1000 == 0:
                print(f"[synth] {i + 1}/{n}")
        steady_seconds = time.perf_counter() - started_steady
        decoded = dataset.store.decoded_now
    else:
        import multiprocessing
        chunks = [list(range(i, min(i + 256, n))) for i in range(0, n, 256)]
        with multiprocessing.Pool(workers, initializer=_worker_init,
                                  initargs=(dataset.params, dataset.epoch)) as pool:
            setup_seconds = 0.0
            started_steady = time.perf_counter()
            for chunk in pool.imap(_worker_chunk, chunks):
                samples.extend(chunk)
            steady_seconds = time.perf_counter() - started_steady
        decoded = None
    data = {name: np.array([s[name] for s in samples], dtype=DTYPES[name]) for name in FIELDS}
    if n == 0:
        data["x"] = np.zeros((0, WINDOW, WINDOW, 3), np.uint8)
        data["off"] = np.zeros((0, 2), np.float32)
    timing = {
        "workers": int(workers),
        "samples": int(n),
        "setup_seconds": round(setup_seconds, 3),
        "generation_seconds": round(steady_seconds, 3),
        "samples_per_second": round(n / steady_seconds, 1) if steady_seconds > 0 else None,
        "samples_per_second_including_setup": round(n / (time.perf_counter() - started), 1),
        "png_frames_decoded_this_run": decoded,
        "note": ("generation_seconds for workers > 1 includes worker start-up; for workers = 1 the "
                 "first sample (sprite loading, frame cache opening) is counted as setup and "
                 "generated a second time inside the timed loop"),
    }
    return data, timing


def build_manifest(dataset, data, timing, out_path):
    per_class_level = {}
    for index, name in enumerate(CLASSES + ["background"]):
        rows = data["label"] == index
        per_class_level[name] = [int(np.sum(rows & (data["level"] == lv))) for lv in (0, 1, 2)]
    scales = {str(int(s)): int(np.sum(data["scale"] == s)) for s in np.unique(data["scale"])}
    negatives = data["label"] == BACKGROUND
    positives = data["label"] != BACKGROUND
    ids, uses = np.unique(data["track"][positives], return_counts=True)
    used = {str(i)[:48]: int(u) for i, u in zip(ids, uses)}
    sprites_per_class = {CLASSES[c]: [
        {"id": s.sprite_id, "scene": s.scene, "frame": int(s.frame), "zoom": int(s.zoom),
         "native_pixels": bool(s.native), "mask_px": int(s.mask_px),
         "positives": used.get(s.sprite_id[:48], 0),
         **({"warning": "low zoom mask below min_recut_mask_px, kept because the class has "
                        "nothing better in this scene"} if s.small_mask else {})}
        for s in group] for c, group in dataset.by_class.items()}
    return {
        "file": str(out_path),
        "source": "synthetic",
        "generator": "elias/data/synth.py (v2)",
        "n": int(len(data["label"])),
        "counts_class_x_level": per_class_level,
        "counts_scale": scales,
        "negatives": {
            "total": int(negatives.sum()),
            "pure_background": int(np.sum(data["track"] == "background")),
            "hard_off_centre": int(np.sum(np.char.startswith(data["track"], "hardneg:"))),
        },
        "classes_without_sprite": dataset.missing_classes,
        "sprites_used": sprites_per_class,
        "sprites_dropped": dataset.dropped_sprites,
        "ground_limits": dataset.store.ground_limits,
        "background_modes_drawn": (dict(dataset.store.mode_counts) if timing["workers"] <= 1
                                   else "not counted with --workers > 1"),
        "class_margins_ltrb_px": {CLASSES[c]: [round(float(v), 1) for v in m]
                                  for c, m in dataset.class_margins.items()},
        "parameters": dataset.params,
        "fixed_parameters": {
            "scale_jitter": [0.92, 1.08], "sprite_brightness": 0.08, "sprite_contrast": 0.08,
            "sprite_gamma": [0.9, 1.1],
            "global_brightness": 0.04, "global_contrast": 0.04, "global_gamma": [0.95, 1.05],
            "strip_max_window_px": 20, "solid_alpha": SOLID_ALPHA,
            "mask": "bank alpha >= solid_alpha, warped linearly, cut at 0.5, 3x3 Gaussian feather",
            "near_ring_px": list(NEAR_RING_PX), "ground_cell_px": GROUND_CELL,
            "hard_negative_distance": "U(1, 2) x max(0.78 x longer side, (neg_guard + 0.5) window px)",
            "keepout_px": KEEPOUT_PX, "reduction": "INTER_AREA by f, then INTER_AREA by s/f",
        },
        "throughput": timing,
        "notes": [
            "Negatives follow the real_crops.py convention: box_w, box_h and size_px are those of "
            "the object that fixed level and scale. Pure background: track 'background', off (0, 0).",
            "Hard negatives: track 'hardneg:<sprite id>', `off` is the offset of the off-centre "
            "object (window px).",
            "Positives record the true sub-pixel offset of the label box centre: within +-0.5 window "
            "px for the centred share, up to +-(center_jitter + 0.5) for the jittered share.",
            "Backgrounds cut from `validation` are noisy: its labels are incomplete, so a "
            "background or negative window can contain an unlabelled object.",
            "background_modes_drawn counts every sample_patch call of this process, the setup "
            "sample included; 'relaxed' = the wanted mode found nothing in 120 tries.",
            "Ground limits are measured on the rings around the labelled boxes of the background "
            "scenes. Known limit: large pale flat roofs pass as open ground, so on urban frames "
            "some objects still stand on roofs.",
        ],
    }


def contact_sheet(data, path, columns=16, title=None):
    """One row per class (plus background), tiles x3 nearest with a caption strip underneath."""
    tile, caption, gutter, label_w = WINDOW * 3, 14, 2, 150
    names = CLASSES + ["background"]
    rng = np.random.default_rng(0)
    header = 22 if title else 0
    sheet = np.full((header + len(names) * (tile + caption + gutter),
                     label_w + columns * (tile + gutter), 3), 32, np.uint8)
    if title:
        cv2.putText(sheet, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    for row, name in enumerate(names):
        top = header + row * (tile + caption + gutter)
        rows = np.nonzero(data["label"] == row)[0]
        cv2.putText(sheet, name, (4, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(sheet, f"n={len(rows)}", (4, top + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 180, 180), 1, cv2.LINE_AA)
        if len(rows) == 0:
            continue
        chosen = rng.choice(rows, size=min(columns, len(rows)), replace=False)
        chosen = chosen[np.lexsort((data["level"][chosen], data["scale"][chosen]))]
        for col, i in enumerate(chosen):
            left = label_w + col * (tile + gutter)
            big = cv2.resize(data["x"][i], (tile, tile), interpolation=cv2.INTER_NEAREST)
            sheet[top:top + tile, left:left + tile] = big
            text = f"L{int(data['level'][i])} s{int(data['scale'][i])} {float(data['size_px'][i]):.0f}px"
            cv2.putText(sheet, text, (left + 2, top + tile + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet)


def cut_real_windows(scene, seed=0, max_frames=25):
    """Real centred windows following the SPEC, for looking at next to the synthetic ones."""
    rng = np.random.default_rng(seed)
    frames = FrameStore.list_frames(scene)
    if len(frames) > max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    samples = {name: [] for name in FIELDS}
    for frame in frames:
        image = cv2.imread(str(FrameStore.image_path(scene, frame)), cv2.IMREAD_COLOR)
        for ann in FrameStore.annotations(scene, frame):
            if ann["object_id"] not in CLASSES:
                continue
            x1, y1, x2, y2 = ann["bbox"]
            box_w, box_h = x2 - x1, y2 - y1
            for level, factor in LEVEL_FACTOR.items():
                scale = pick_scale(factor, max(box_w, box_h))
                # the view origin fixes the phase of the delivered grid (always 0 at level 0)
                phase = np.zeros(2, int) if level == 0 else rng.integers(0, factor, size=2)
                centre_delivered = (np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0]) - phase) / factor
                side_delivered = WINDOW * scale // factor
                origin = np.rint(centre_delivered - side_delivered / 2.0).astype(int) * factor + phase
                size = WINDOW * scale
                patch = np.zeros((size, size, 3), np.uint8)
                sx1, sy1 = max(origin[0], 0), max(origin[1], 0)
                sx2, sy2 = min(origin[0] + size, FRAME_W), min(origin[1] + size, FRAME_H)
                if sx2 <= sx1 or sy2 <= sy1:
                    continue
                patch[sy1 - origin[1]:sy2 - origin[1], sx1 - origin[0]:sx2 - origin[0]] = \
                    image[sy1:sy2, sx1:sx2]
                window = reduce_like_evaluator(patch, factor, scale)
                values = dict(x=window, label=CLASSES.index(ann["object_id"]), level=level,
                              scale=scale, size_px=max(box_w, box_h) / factor, box_w=box_w,
                              box_h=box_h, off=(0.0, 0.0), scene=scene, frame=frame,
                              track=ann["object_id"], source="real")
                for name in FIELDS:
                    samples[name].append(values[name])
    return {name: np.array(samples[name], dtype=DTYPES[name]) for name in FIELDS}


def comparison_sheets(synthetic, real, out_prefix, per_side=7):
    """Per window scale: rows = classes, left block real, right block synthetic."""
    tile, caption, gutter, label_w, gap = WINDOW * 3, 14, 2, 150, 24
    written = []
    for scale in (1, 2, 4, 8):
        rows = [c for c in range(len(CLASSES))
                if np.any((synthetic["label"] == c) & (synthetic["scale"] == scale))
                or np.any((real["label"] == c) & (real["scale"] == scale))]
        if not rows:
            continue
        width = label_w + 2 * per_side * (tile + gutter) + gap
        sheet = np.full((24 + len(rows) * (tile + caption + gutter), width, 3), 32, np.uint8)
        cv2.putText(sheet, f"scale {scale}: REAL (left {per_side})   |   SYNTHETIC (right {per_side})",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rng = np.random.default_rng(scale)
        for r, c in enumerate(rows):
            top = 24 + r * (tile + caption + gutter)
            cv2.putText(sheet, CLASSES[c], (4, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            for block, data in enumerate((real, synthetic)):
                idx = np.nonzero((data["label"] == c) & (data["scale"] == scale))[0]
                if len(idx) == 0:
                    continue
                idx = rng.choice(idx, size=min(per_side, len(idx)), replace=False)
                idx = idx[np.argsort(data["level"][idx], kind="stable")]
                for col, i in enumerate(idx):
                    left = label_w + block * (per_side * (tile + gutter) + gap) + col * (tile + gutter)
                    sheet[top:top + tile, left:left + tile] = cv2.resize(
                        data["x"][i], (tile, tile), interpolation=cv2.INTER_NEAREST)
                    text = f"L{int(data['level'][i])} {float(data['size_px'][i]):.0f}px"
                    cv2.putText(sheet, text, (left + 2, top + tile + 11), cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, (200, 200, 200), 1, cv2.LINE_AA)
        path = f"{out_prefix}_vs_real_s{scale}.png"
        cv2.imwrite(path, sheet)
        written.append(path)
    return written


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_rot_json(text):
    if not text:
        return {}
    if os.path.exists(text):
        with open(text, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(text)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Synthetic scale-preserving 64x64 windows from sprites (elias/SPEC.md).")
    parser.add_argument("--sprite-bank", nargs="+", default=None,
                        help="bank.json paths (default: Oscar's bank plus elias/sprites/bank.json if present)")
    parser.add_argument("--sprite-scenes", nargs="+", default=["helsinki"],
                        choices=["helsinki", "validation"])
    parser.add_argument("--background-scenes", nargs="+", default=["helsinki"],
                        choices=["helsinki", "validation"])
    parser.add_argument("--n", type=int, default=4000, help="total samples, negatives included")
    parser.add_argument("--rot-max", type=float, default=180.0,
                        help="rotation uniform in [-DEG, DEG]; 180 = any heading")
    parser.add_argument("--rot-max-json", default=None,
                        help='per-class override, a JSON file or inline JSON such as {"hangar": 20}')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--neg-frac", type=float, default=0.25, help="share of label 16 samples")
    parser.add_argument("--level-probs", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        metavar=("L0", "L1", "L2"))
    parser.add_argument("--center-jitter", type=float, default=6.0,
                        help="uniform +- offset of jittered positives in window px (recorded in off); "
                             "6 matches real_crops.py")
    parser.add_argument("--jitter-prob", type=float, default=0.5,
                        help="share of positives that get the centre jitter, the rest is centred")
    parser.add_argument("--neg-guard", type=float, default=10.0,
                        help="hard negatives keep at least this many window px off-centre")
    parser.add_argument("--lossless-prob", type=float, default=0.25,
                        help="share of sprites pasted with quarter turns only: no resampling, no "
                             "blur, exact native pixels (0 = always the warp pipeline)")
    parser.add_argument("--bg-open-prob", type=float, default=0.7,
                        help="share of backgrounds forced onto open ground: central half as "
                             "bright, as little blue, structured and edgy as the rings around the "
                             "labelled boxes, and free of water (0 = anywhere but water)")
    parser.add_argument("--bg-near-frac", type=float, default=0.5,
                        help="part of the open-ground share whose patch centre lies 100 to 500 "
                             "source px from a labelled box, where real objects stand")
    parser.add_argument("--warp-interp", nargs="+", default=["lanczos4", "nearest"],
                        choices=sorted(INTERPOLATIONS),
                        help="colour interpolation of the warp, drawn per sprite from this list. "
                             "lanczos4 is a little softer than real pixels, nearest a little "
                             "sharper; linear loses 30 to 45 %% of the detail at scale 1")
    parser.add_argument("--sprite-blur-max", type=float, default=0.0,
                        help="sprite Gaussian blur sigma is uniform in [0, this] on the warp path "
                             "(0 = none; the warp alone already softens)")
    parser.add_argument("--exclude-sprite-ids", nargs="*", default=[],
                        help="bank ids never to paste (manual veto), e.g. lib:mine_roller")
    parser.add_argument("--exclude-tracks", nargs="*", default=[],
                        help="validation track ids whose sprites are dropped, on top of the "
                             "excluded_tracks blocks of the banks")
    parser.add_argument("--min-recut-mask-px", type=int, default=300,
                        help="zoom 0 and 1 sprites with a smaller mask are dropped when the class "
                             "has another sprite in that scene")
    parser.add_argument("--distractor-prob", type=float, default=0.3)
    parser.add_argument("--strip-prob", type=float, default=0.15)
    parser.add_argument("--post-noise", type=int, default=0, choices=[0, 1],
                        help="1 adds uniform {-1, 0, +1} grey levels after the reduction")
    parser.add_argument("--max-bg-frames", type=int, default=48,
                        help="evenly spaced background frames kept per scene")
    parser.add_argument("--frame-cache", default=str(DEFAULT_FRAME_CACHE))
    parser.add_argument("--no-frame-cache", action="store_true",
                        help="decode PNG frames into RAM instead of the npy memmap cache")
    parser.add_argument("--no-recut-native", action="store_true",
                        help="keep the upsampled pixels of zoom < 2 sprites")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sheet-cols", type=int, default=16)
    parser.add_argument("--compare", action="store_true",
                        help="also write OUT_vs_real_s*.png against real windows")
    parser.add_argument("--compare-real", default=None,
                        help="real npz for --compare (default elias/out/real_<scene>.npz, else cut here)")
    args = parser.parse_args(argv)

    banks = args.sprite_bank or [p for p in DEFAULT_BANKS if Path(p).exists()]
    dataset = SynthWindows(
        n=args.n, sprite_banks=banks, sprite_scenes=args.sprite_scenes,
        background_scenes=args.background_scenes, rot_max=args.rot_max,
        rot_max_per_class=parse_rot_json(args.rot_max_json), seed=args.seed,
        neg_frac=args.neg_frac, level_probs=args.level_probs, center_jitter=args.center_jitter,
        jitter_prob=args.jitter_prob, neg_guard=args.neg_guard,
        lossless_prob=args.lossless_prob, bg_open_prob=args.bg_open_prob,
        bg_near_frac=args.bg_near_frac, warp_interp=args.warp_interp,
        exclude_sprite_ids=args.exclude_sprite_ids, exclude_tracks=args.exclude_tracks,
        min_recut_mask_px=args.min_recut_mask_px,
        sprite_blur_max=args.sprite_blur_max, distractor_prob=args.distractor_prob,
        strip_prob=args.strip_prob, post_noise=args.post_noise, max_bg_frames=args.max_bg_frames,
        frame_cache=None if args.no_frame_cache else args.frame_cache,
        recut_native=not args.no_recut_native)
    if dataset.missing_classes:
        print(f"[synth] no sprite for: {', '.join(dataset.missing_classes)}")
    reasons = {}
    for item in dataset.dropped_sprites:
        reasons.setdefault(item["reason"].split(":")[0], []).append(item["id"])
    for reason, ids in reasons.items():
        print(f"[synth] dropped {len(ids)} bank entries ({reason}): {', '.join(ids[:6])}"
              f"{' ...' if len(ids) > 6 else ''}")

    data, timing = generate(dataset, workers=args.workers)
    if not dataset._ready:
        dataset._setup()   # workers did the generation; the manifest still wants sprite facts

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    manifest = build_manifest(dataset, data, timing, out)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    contact_sheet(data, out.with_suffix(".png"), columns=args.sheet_cols,
                  title=f"{out.name}: sprites {args.sprite_scenes}, backgrounds {args.background_scenes}")
    print(f"[synth] wrote {out} ({len(data['label'])} samples), manifest and contact sheet")
    print(f"[synth] throughput: {timing['samples_per_second']} samples/s "
          f"({timing['workers']} worker(s), setup {timing['setup_seconds']} s, "
          f"PNG frames decoded this run: {timing['png_frames_decoded_this_run']})")

    if args.compare:
        scene = args.sprite_scenes[0]
        real_path = Path(args.compare_real) if args.compare_real else ELIAS_DIR / "out" / f"real_{scene}.npz"
        real = None
        if real_path.exists():
            try:   # another builder may be rewriting the file at this very moment
                with np.load(real_path) as loaded:
                    real = {name: loaded[name] for name in FIELDS}
                print(f"[synth] comparing with {real_path}")
            except Exception as error:
                print(f"[synth] cannot read {real_path} ({error})")
        if real is None:
            real = cut_real_windows(scene, seed=args.seed)
            print(f"[synth] cut {len(real['label'])} real windows from {scene} for the comparison")
        for path in comparison_sheets(data, real, str(out.with_suffix(""))):
            print(f"[synth] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
