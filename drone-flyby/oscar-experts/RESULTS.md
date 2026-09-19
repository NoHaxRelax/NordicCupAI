# Overnight results, 19 Sep 2026 (draft, updated as stages finish)

All numbers are on TRAINING tiles (grid384 train split: reference frames plus training-split validation
tracks), 384 px tiles at the three zooms, plus 60 verified-empty training tiles per class. "Found" means a
candidate with the right class and IoU >= .5 against the organizer-style box. Candidates per tile are
BEFORE the gate; the gate columns are out-of-fold (5-fold by tile).

## Stage 1: experts

Wave 1 (half-resolution proposer, unrotated organizer boxes) -> wave 2 (full-resolution proposer, boxes
rotated with the pose, blob expert for the launchers, separate colour-proposal cap):

| class | wave 1 found | wave 2 found | note |
|---|---|---|---|
| jammer | 35/72 | 69/72 | full-res proposer + green panel |
| small_plane | 40/63 | 62/63 | red markings carry the gate |
| small_launcher | 2/69 | 54/69 (blob expert, wave 1b) | L0 hopeless at 2-3 px |
| medium_launcher | 1/18 | 17/18 (blob expert, wave 1b) | |
| ta-ta | 0/69 | 67/69 | low-chroma blob proposer, colour cap raised |
| medium_plane | 19/48 | 36/48 | full-res proposer |
| jet_plane | 49/84 | 57/84 | L0 still weak (11/28) |
| helicopter | 41/48 | 43/48 | unreviewed v4 mask |
| large_tower | 54/57 | 48/57 | slight drop: tall object, box rotated with heading (to revisit) |
| mine_roller | 15/57 | 15/57 | the other 42 targets are the excluded mislabelled validation track; on the reference object 15/15 with IoU 1.0 |
| small_tower, spacecraft | 51/51, 57/57 | pending | |

Gated pass (gates v2 applied, 60 empty training tiles per class), 02:55-03:35:

| class | found | cand/tile | false alarms / 60 empties |
|---|---|---|---|
| condor | 42/42 | 23.4 | 1263 (gate weak) |
| hangar | 9/9 | 1.2 | 53 |
| helicopter | 43/48 | 14.6 | 154 |
| jammer | 69/72 | 1.0 | 0 |
| jet_plane | 57/75 | 2.8 | 189 |
| large_launcher | 53/102 | 6.3 | 144 |
| large_tower | 48/57 | 1.0 | 3 |
| medium_launcher | 18/18 | 2.5 | 9 |
| medium_plane | 24/48 | 1.1 | 1 |
| mine_roller | 15/15 | 1.0 | 0 |
| small_launcher | 54/69 | 0.8 | 0 |
| small_plane | 62/63 | 1.0 | 2 |
| small_tower | 48/51 | 2.0 | 0 |
| spacecraft | 57/57 | 1.4 | 6 |
| ta-ta | 61/69 | 1.9 | 22 |
| tank | 95/171 | 0.6 | 0 |

Lesson: gates v2 were fitted on candidates from an earlier expert build; the fine-pose grid changed in
between, so tank, medium plane and large launcher lost recall to a miscalibrated gate. Gates must be
fitted on candidates produced by the deployed expert code; `full_pass.sh` does exactly that (pass 3).

## Gates (per-class logistic over recorded features, training tiles, out-of-fold)

Wave-1 fit (`gates-v0`): recall target 98%.

| class | positives | negatives | oof recall | background kept | heaviest features |
|---|---|---|---|---|---|
| jammer | 35 | 1693 | 1.00 | 0.1% | panel_area, green_pixels, green_fraction |
| small_plane | 40 | 1471 | 1.00 | 0.5% | red_blobs, colour_distance, bright_fraction |
| small_tower | 53 | 1136 | 1.00 | 0.1% | slab_fraction, colour_distance, low_chroma_fraction |
| spacecraft | 60 | 1294 | 0.98 | 0.6% | low_chroma_fraction, proposer_score, dark_fraction |
| large_tower | 54 | 1311 | 0.98 | 0.0% | colour_distance, camo_mean_L, camo_green |
| helicopter | 47 | 1236 | 1.00 | 13.8% | correlation, green_fraction, region_chroma |

Gates v2 (wave-2 candidates, all 16 classes, 5-fold by tile, recall target 98%):

| class | positives | negatives | oof recall | background kept |
|---|---|---|---|---|
| jammer | 69 | 3688 | 1.00 | 0.1% |
| small_plane | 62 | 3323 | 0.98 | 0.2% |
| medium_plane | 36 | 2976 | 1.00 | 0.2% |
| large_tower | 48 | 2756 | 1.00 | 0.3% |
| mine_roller | 15 | 2760 | 1.00 | 0.0% |
| small_launcher | 54 | 1765 | 0.98 | 0.0% |
| medium_launcher | 18 | 976 | 1.00 | 3.8% |
| jet_plane | 71 | 3285 | 0.99 | 10% |
| helicopter | 50 | 3023 | 0.98 | 27% |
| large_launcher | 60 | 3641 | 0.98 | 40% |
| condor | 159 | 1430 | 0.98 | 44% |

The three weak gates (condor, large launcher, helicopter) are big camouflaged or low-contrast objects whose
recorded features do not separate them from vegetation; those classes rely on the verifier.

## Stage 3: box fit (wave 2, matched candidates, IoU with the organizer box)

| class | n | p25 | median | p75 | share >= .7 |
|---|---|---|---|---|---|
| small_plane | 62 | .89 | .93 | .96 | 1.00 |
| spacecraft | 57 | .88 | .91 | .96 | 1.00 |
| helicopter | 48 | .76 | .90 | .93 | .77 |
| large_tower | 49 | .79 | .90 | .94 | .94 |
| jammer | 69 | .80 | .84 | .94 | .96 |
| small_tower | 48 | .75 | .83 | .95 | .88 |
| medium_plane | 48 | .54 | .94 | .99 | .67 |
| jet_plane | 63 | .59 | .87 | .95 | .52 |
| small_launcher | 54 | .74 | .81 | .84 | .85 |
| medium_launcher | 18 | .70 | .84 | .87 | .72 |
| ta-ta | 67 | .56 | .64 | .74 | .43 |
| hangar | 9 | .89 | .92 | .92 | 1.00 |

Boxes are the posed sprite outline plus the organizer offset stored with each sprite, rotated with
the pose. Adequate for most classes; ta-ta, jet plane and medium plane have a loose-box tail worth a
per-class box model later. Tall objects (towers, hangar) grow down the frame; the tracker's placement
layer handles that after first sighting.

## Stage 2: verifier v0

ResNet-18 (ImageNet init) fine-tuned 12 epochs on 21,915 expert-candidate crops from training tiles
(positives by label overlap, background from unmatched candidates), held-out tiles 99.7%.
Evaluate-only synthetic set (training sprites on dev-tile backgrounds, 795 crops): accuracy 95.1%,
background rejection 98.4%; per-class recall condor 1.0, hangar 1.0, jammer 1.0, large_launcher 1.0,
mine_roller 1.0, small_plane 1.0, spacecraft 1.0, small_tower .96, tank .92, medium_launcher .92,
small_launcher .82, ta-ta .78, medium_plane .78, jet_plane .70.

## Pass 3 (current code, full training set, gates v3 fitted on its own candidates) — pending
## Stage 4: camera strategies — pending (chained after pass 3)


## Proposer fix and fair-share capping (19 Sep 06:30-07:00)

Passes 4-6 were invalid: `SharedProposer` prepared the scene with the first class's blur/downscale (condor's:
half resolution, blur 1.0) for every class. Fixed (one scene per setting group); verified single-class ==
all-16 == CPU proposer. Pass 7 = first valid full pass.

Fair-share proposal capping (`GenericExpert._fair_share`): merge per template, interleave templates by rank,
then cap at 24. Random 6 training tiles per zoom (seed 7, exclusions not applied), single-class proposer:

| class | current L0/L1/L2 | fair share L0/L1/L2 |
|---|---|---|
| jet_plane | 3/6 5/6 5/6 | 5/6 5/6 5/6 |
| tank | 3/6 6/6 6/6 | 4/6 6/6 6/6 |
| large_launcher | 2/6 4/6 3/6 | 3/6 4/6 3/6 |
| medium_plane | 4/6 6/6 6/6 | 3/6 6/6 6/6 |
| jammer, small_tower | 6/6 6/6 5-6/6 | same |
| mine_roller | 0/6 2/6 1/6 | same (tiles of the excluded track mine-roller-a-005-009; not a pass target) |

Same candidate counts (22-35 per tile). Template ordering (same-zoom vs cross-zoom, proposer vs fine pose) made
no difference at all on the same sample.

Per-template robust z-score ranking (alternative to the interleave) on the same sample: identical except tank L2
5/6 vs 6/6. Interleave kept (simpler). Giving the large launcher all its same-zoom sprites (fine/proposer templates
6): miss audit L1 5/12 -> 7/12, L0 8/12; remaining misses are validation-scene launchers whose proposals score
only .21-.44 (different articulation/side), a sprite-coverage problem, not a ranking one.

## Box size refinement (07:30)

Miss audit (12-16 random training targets per class and zoom, seed 11, exclusions applied). The dominant L0 loss
was box size: the candidate sits on the object (correlation .6-.7) but the box is 1.3-1.5x too large (IoU .44-.49).

| class / zoom | scale 1 only | full sweep (.8, 1, 1.25) x all poses | refinement at the winning pose only |
|---|---|---|---|
| jet_plane L0 | 11/12 | 12/12 | 12/12 |
| medium_plane L0 | 6/16 | 10/16 | 10/16 |
| medium_plane L1 | - | 12/16 | 12/16 |
| tank L0 | 9/12 | 10/12 | 9/12 |
| large_launcher L1 | 7/12 | - | 7/12 |

The refinement (`ClassSpec.box_scales = (.8, 1.25)`, two extra masked correlations per candidate) captures the
sweep's gain at ~10% of its cost; it goes into pass 8. Remaining medium-plane L0 misses are at IoU .49 with the
.8 pose scoring below the 1.0 pose, and two reference-frame tiles whose proposal (score .57) falls outside the cap.

## Pass 7 (07:40): first valid full training pass with the fixed shared proposer, 1108 s on 12 shards, no expert errors

| class | found | L0 / L1 / L2 | cand/tile | gate v7 oof recall / background kept |
|---|---|---|---|---|
| condor | 42/42 | 14/14 14/14 14/14 | 11.1 | .98 / .196 |
| hangar | 9/9 (+3/24 partial) | 3/3 each | 1.2 | 1.00 / .000 |
| helicopter | 43/48 | 14/16 15/16 14/16 | 26.8 | .98 / .193 |
| jammer | 65/72 | 24/24 24/24 17/24 | 20.2 | .99 / .001 |
| jet_plane | 61/75 | 13/25 24/25 24/25 | 17.0 | .99 / .031 |
| large_launcher | 57/102 | 22/34 17/34 18/34 | 22.3 | .98 / .152 |
| large_tower | 48/57 | 16/19 16/19 16/19 | 18.4 | 1.00 / .000 |
| medium_launcher | 17/18 | 5/6 6/6 6/6 | 42.7 | 1.00 / .040 |
| medium_plane | 25/48 | 6/16 7/16 12/16 | 26.6 | 1.00 / .002 |
| mine_roller | 15/15 | 5/5 each | 18.5 | 1.00 / .000 |
| small_launcher | 54/69 | 8/23 23/23 23/23 | 20.7 | 1.00 / .001 |
| small_plane | 63/63 | 21/21 each | 18.6 | .98 / .008 |
| small_tower | 45/51 | 11/17 17/17 17/17 | 28.5 | 1.00 / .000 |
| spacecraft | 57/57 | 19/19 each | 30.5 | .98 / .003 |
| ta-ta | 67/69 | 22/23 23/23 22/23 | 30.4 | .99 / .019 |
| tank | 135/171 | 37/57 47/57 51/57 | 17.7 | .99 / .000 |

Weak spots: medium_plane (box size, L0/L1), large_launcher (validation-scene articulation), jet_plane L0, tank L0,
small_launcher L0 (2-3 px objects). Pass 8 adds fair-share capping, all large-launcher sprites and the size refinement.
