# Synthetic sprite dataset v1 (18 Sep 2026)

180 images (256 px), 320 objects, 15 classes, YOLO labels in `labels/`, provenance in `manifest.json`.

- Sprites: `data/drone/sprite-bank-reviewed-20260918` (68 sprites Oscar approved or redrew in the review tool),
  all from training-split tiles. No helicopter (all its crops were rejected); large tower only at L1.
- Backgrounds: training-split tiles verified empty only (organiser reference tiles with complete labels, and
  tiles Oscar reviewed as empty). Validation-video frames used are all at or before frame 85.
- Held out: the development split (validation frames 100-180) is never read, so it remains a clean test set.
- Zoom mix: L1 86, L0 57, L2 37. 1-3 objects per image, quarter turns, flips, 0.9-1.1 scale, no overlaps.
- Labels: the organiser's box of the source instance moved with the sprite (loose, like the real labels).
- Blending: none beyond the sprite bank's rim-free edges. A Higgsfield seam pass was tested and dropped:
  Nano Banana 2 Lite re-textured and re-coloured the surroundings (see `pilot-nano-banana-2-lite/`), and
  Higgsfield's MCP ignores the inpainting mask (the mask role is coerced to a plain reference image).

Rebuild: `python3 -m drone.grid_training.synthetic_sprites prepare BANK OUT --count N` then
`python3 -m drone.grid_training.synthetic_sprites finalize OUT --without-model`.
