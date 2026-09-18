# Drone sprite bank and synthetic set (Oscar, 18 Sep 2026)

Clean, background-free cutouts of the drone-flyby objects, and a synthetic training set built from them.
Everything here comes from the **training split** of the grid dataset (`grid-comparison-20260918-v1/256`).
The development split (validation-video frames 100–180) was never read, so it remains a clean test set.

## What is here

| Folder | Contents |
|---|---|
| `sprite-bank/` | 68 transparent BGRA PNGs, one folder per class, reviewed by Oscar. `bank.json` gives class, zoom, source tile and the organiser box in sprite coordinates. |
| `synthetic-set-v1/` | 180 images (256 px), 320 objects, 15 classes. YOLO labels in `labels/`, provenance in `manifest.json`, previews in `contact-sheet*.jpg`. |
| `sprite-library-v4/` | One reference mask per class appearance (19 masks), used to transfer masks to every training instance. `spec/` holds the source masks. |
| `review/` | Oscar's review: `decisions.json` (approved / redrawn / rejected), redrawn masks, the 66-item `shortlist.json`, and `label_exclusions.json`. |
| `code/drone_sprites/` | The code. It expects the local `data/drone/...` layout of the working repo. |

## How the sprites were made

1. One mask per class, from Nano Banana Pro (Higgsfield) asked to paint the terrain magenta, given the
   crop plus views of the same sprite on other terrain. The model's pixels are only used as a mask;
   **sprite pixels always come from the original tile**. The small tower's slab and the helicopter's
   blades were added locally.
2. Masks are transferred to every training instance by masked template matching over heading and scale,
   constrained to sit centred inside the organiser box.
3. Oscar reviewed one example per class × zoom × appearance in `mask_editor` and approved, redrew or rejected it.
4. Export un-mixes the blurred edge pixels from the old background, so pasted sprites carry no light rim.
   Real frames show no cast shadows; none are added.

Gaps: no helicopter sprite (all rejected), large tower only at L1, validation-appearance jet plane rejected.

## Synthetic set

Sprites are pasted onto training-split tiles verified empty (organiser reference tiles with complete labels,
and tiles Oscar reviewed as empty). 1–3 objects per image, quarter turns, flips, 0.9–1.1 scale, no overlaps.
Half the images are L1. Labels use the organiser's box of the source instance, moved with the sprite.

A Higgsfield seam-blending pass was tested and dropped: Nano Banana 2 Lite re-textured and re-coloured the
surroundings, and the Higgsfield MCP coerces the inpainting mask role to a plain reference image.

## Label problems found (not fixed in the dataset)

- The validation "mine roller" track `mine-roller-a-005-009` looks like the tank (boxy hull, gun barrel),
  unlike the organiser's reference mine roller. It supplies 42 of the 57 mine-roller training examples.
- Jet-plane track `jet-plane-d-040-082`, frames 40 and 46: the box lies on empty forest at all zooms.

Both are excluded from the sprites; the training labels themselves still need a decision.

## Commands

```sh
python3 -m drone.grid_training.mask_editor serve                         # review tool, localhost:8765
python3 -m drone.grid_training.synthetic_sprites bank OUT_BANK           # export reviewed sprites
python3 -m drone.grid_training.synthetic_sprites prepare OUT_BANK OUT_SET --count 180 --seed 1918
python3 -m drone.grid_training.synthetic_sprites finalize OUT_SET --without-model
```
