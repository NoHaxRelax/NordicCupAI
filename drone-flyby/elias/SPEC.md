# elias/: recognition at small pixel sizes. Shared spec for every module in this folder.

Everything here lives under `drone-flyby/elias/`. Nothing outside this folder is edited except through
Oscar's `DRONE_DETECTOR=module:factory` hook. Python: `~/venvs/nordic-drone/Scripts/python.exe`
(torch 2.11 cu128, cv2, numpy, ultralytics, faster-coco-eval). Run everything from `drone-flyby/`.

## Geometry (from the organisers' README and local_evaluator.py)

- Source frames are 3840x2160 BGR PNG. A delivered view is always 960x540.
- Level L has factor f = 4, 2, 1 for L = 0, 1, 2. The view is the source region of size (960 f, 540 f)
  centred on the camera centre, resized to 960x540 with **cv2.INTER_AREA**. Nothing else is in-distribution.
- Boxes are `[x1, y1, x2, y2]` in source pixels. Organiser boxes are loose: a few pixels of margin around
  the silhouette (the asset footprint), not a tight mask box.

## Scenes

| name | frames | labels | trust |
|---|---|---|---|
| `helsinki` | `src/helsinki/images/frame_%06d.png`, 25 frames | `src/helsinki/annotations/frame_%06d.json`, 16 classes, one instance per class | organiser ground truth |
| `validation` | `src/validation/images/frame_%06d.png`, 249 frames (1 to 3 are empty placeholders) | `src/validation/annotations/*.json`, 11 classes, 987 boxes, `provenance[i].track_id` and `review_status` aligned with `annotations[i]` | participant pseudo-labels from score probing: noisy and INCOMPLETE (an unlabelled region is not proof of emptiness) |

Class list and order: `dtos.OBJECT_CLASSES`. Class index 16 is `background`.

## The sample: a scale-preserving window

One sample is a **64x64 BGR uint8 window** plus metadata. Altitude is fixed, so physical size is a class cue
and must survive: the object is never resized to fill the window.

- `scale` s in {1, 2, 4, 8}: source pixels per window pixel. A window covers 64 s source pixels.
- For an object seen at level L (factor f): s is the smallest value in {f, 2f, 4f} with
  `max(box_w, box_h) / s <= 48`. The window is cut from the delivered view around the box centre
  (64 s / f delivered pixels wide) and, if s > f, reduced with INTER_AREA by s / f.
- Metadata per sample: `label` (0..16), `level` (0,1,2), `scale`, `box_src` (w, h in source px),
  `size_px` = longer box side in DELIVERED pixels (the x axis of the accuracy curve), `scene`,
  `frame`, `track` (string; class name on helsinki), `center_offset` (dx, dy of the object centre from the
  window centre in window px; (0,0) when centred), `source` in {real, synthetic}.
- Negatives (`label` 16): windows whose centre is at least 0.75 x the nearest object's longer side away
  from that object's centre, at the same (level, scale) mix as the positives. On `validation` draw
  negatives only from the horizontal band of rows that are covered by labelled tracks in that frame AND
  at least 96 source px from any labelled box; note in the manifest that they are noisy.

## Dataset container (the interface between modules)

`np.savez_compressed(path, x=uint8[N,64,64,3], label=int16[N], level=int8[N], scale=int8[N],
size_px=float32[N], box_w=float32[N], box_h=float32[N], off=float32[N,2], scene=<U16[N],
frame=int32[N], track=<U48[N], source=<U10[N])` plus `path.with_suffix('.json')` manifest with counts per
class x level and the generation parameters. Every builder also writes a contact sheet PNG (one row per
class, columns = samples, native 64 px tiles upscaled x3 nearest) next to the npz, because a human or
an agent must be able to LOOK at what was built.

## Sprite bank format (Oscar's, extended)

`oscar-sprite-synthetic/sprite-bank/bank.json`: list of entries with `file` (BGRA PNG at native source
resolution, tight silhouette), `class_name`, `zoom`, `source` (reference = helsinki, validation),
`frame`, `size`, `box_in_sprite` (the organiser box relative to the sprite's top-left, may be negative),
`review_status`. New sprites go to `elias/sprites/<class>/*.png` with `elias/sprites/bank.json` in the
same schema, `source` naming the scene they were cut from.

## Hold-out rule

A number only counts if the model never saw backgrounds or sprites from the scene it is tested on.
Two directions are always reported: (train: helsinki sprites+backgrounds, test: validation real crops)
and (train: validation, test: helsinki real crops). Synthetic generators therefore take
`--sprite-scenes` and `--background-scenes`.
