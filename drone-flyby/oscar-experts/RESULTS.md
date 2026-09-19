> Clock note: the times written as "CEST" in this file are session-relative and run about 9.5 hours ahead of the
> laptop's local time (the session started at midnight local and the labels drifted); the UTC timestamps in
> VALIDATION-SCORES.md and the organizer receipts are authoritative.

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

## Verifier v1 (08:10, pod 2, `/workspace/experts/runs/verifier-0052/model/best.pt`, copied to pod 1)

Trained on pass 7 candidates (55.7k real crops, 98% background) + 2.1k synthetic composites, 12 epochs, ResNet-18
fine-tune on 96-px crops, 17 classes. Held-out tiles: 99.8% accuracy (background 7195/7211, every object class
near-perfect). Synthetic validation set (795 crops, dev backgrounds, evaluate-only): v1 96.4% accuracy, 99.3% background
rejection (v0: 95.1% / 98.4%). Per-class recall v1: jet_plane .73 (4 of 30 called background), medium_plane .81,
ta-ta .85, medium_launcher .83, large_launcher .91, mine_roller .92, small_launcher .94, tank .97, rest 1.0.

### Refinement acceptance margin (08:50, same 7 x 18 sample, speed batch 1 applied, launcher sprites 6)

| variant | jet_plane | medium_plane | small_tower | large_launcher | total of 126 |
|---|---|---|---|---|---|
| no refinement | 5/5/5 | 3/6/6 | 6/6/6 | 3/4/3 | 94 |
| margin 0 (pass 8) | 6/4/2 | 5/6/6 | 6/5/5 | 2/4/4 | 91 |
| margin .03 | 6/5/3 | 5/6/6 | 6/5/5 | 2/4/4 | 93 |
| margin .06 (default) | 6/5/5 | 5/6/6 | 6/5/5 | 2/4/4 | 95 |
| margin .1 | 5/5/5 | 5/6/6 | 6/5/6 | 2/4/4 | 95 |

(L0/L1/L2 found of 6; jammer, tank, mine_roller unchanged.) A scaled pose that wins by a hair gives a worse box for
jet planes and towers; with a .06 margin only clear wins are taken. Speed batch 1 verified identical on this sample.

## Gated + verified rotating sample (verify7, pod 2, 591 s): gates v7 + verifier v1, 8 tiles per class and zoom, 96 empty tiles

| class | found | L0/L1/L2 | cand/tile | false alarms on 96 empty tiles |
|---|---|---|---|---|
| condor | 24/24 | 8/8 8/8 8/8 | 5.9 | 133 |
| hangar | 9/9 | 3/3 each | 0.4 | 0 |
| helicopter | 17/19 | 6/6 6/7 5/6 | 12.9 | 202 |
| jammer | 23/24 | 8/8 8/8 7/8 | 1.0 | 1 |
| jet_plane | 29/34 | 12/12 9/11 8/11 | 1.3 | 72 |
| large_launcher | 14/24 | 5/8 5/8 4/8 | 5.8 | 297 |
| large_tower | 19/24 | 7/8 6/8 6/8 | 0.9 | 0 |
| medium_launcher | 18/18 | 6/6 each | 2.3 | 21 |
| medium_plane | 21/30 | 5/10 8/10 8/10 | 1.0 | 5 |
| mine_roller | 15/15 | 5/5 each | 1.0 | 0 |
| small_launcher | 19/24 | 3/8 8/8 8/8 | 0.8 | 3 |
| small_plane | 23/24 | 7/8 8/8 8/8 | 1.2 | 20 |
| small_tower | 18/27 | 6/9 5/9 7/9 | 1.0 | 1 |
| spacecraft | 23/24 | 8/8 8/8 7/8 | 1.1 | 6 |
| ta-ta | 18/23 | 5/8 7/7 6/8 | 1.6 | 23 |
| tank | 20/26 | 6/9 10/10 4/7 | 0.7 | 0 |

After gate + verifier most classes keep about one candidate per tile with near-zero false alarms on empty tiles;
helicopter, large_launcher and condor still pass 2-6 candidates per empty tile. Caveat: the deployed code had the
size refinement at margin 0 while gates v7 were fitted without it (small_tower L1, tank L2 dips); verify9 repeats
this with matching code and gates v9.

## Pass 8 (09:05, 1207 s): fair-share capping + launcher sprites 6 + size refinement at margin 0 (vs pass 7)

medium_plane 25 -> 33, jet_plane 61 -> 62 (L0 13 -> 25, L2 24 -> 15), large_launcher 57 -> 61, jammer 65 -> 69,
small_tower 45 -> 37, ta-ta 67 -> 59 (L0 22 -> 14), tank 135 -> 132, spacecraft 57 -> 55, helicopter 43 -> 41,
large_tower 48 -> 46; others unchanged. Net +10 but the losses are on L1/L2 where boxes were already right: the
margin-0 refinement swaps a good box for a marginally better-correlating wrong one. Pass 9 = margin .06 (sample: keeps
the gains, drops the losses); ta-ta (21 px) gets no refinement at all. Gates v8: large_launcher background kept .59
(six heterogeneous templates), helicopter .22, condor .20 - those three rely on the verifier.

Miss-by-track (pass 7): large-launcher-c-047-078 41/48 missed, medium-plane-d/e 8/12 each, large-tower-038-067 9/9,
tank-015-036 18/66, tank-d-029-061 15/54, helicopter-043-073 5/12 - none of these tracks has a sprite in the bank.
`add_track_sprites.py` cuts auto-masked sprites (GrabCut seeded by the organizer box, review_status claude-auto,
contact sheet for Oscar) from their training tiles; pass 10 measures the enlarged bank against pass 9.

## Pass 9 (09:45, 1166 s): final code, old bank - refinement margin .06 (vs pass 7 / pass 8)

jet_plane 61/75 -> 73/75 (L0 13 -> 25, L2 back to 24), medium_plane 25 -> 33, jammer 65 -> 69, large_launcher 57 -> 61,
tank 135 -> 134, ta-ta 67 -> 65 (refinement still on for ta-ta here; off from pass 10), small_tower 45 -> 38 (the one
regression left, under investigation), everything else at pass 7 level. Total complete targets found 803 -> 821.
Gates v9 background kept: large_launcher .58, helicopter .24, condor .20, jet_plane .14; the rest at or below .02.

### verify9 (10:05, pod 1, 308 s with speed batch 1): gates v9 + verifier v1, rotating sample seed 3, 8 tiles per class and zoom

jet_plane 32/33 (3.6 cand/tile, 278 alarms on 96 empty tiles), small_plane 24/24, spacecraft 24/24, condor 24/24,
mine_roller 15/15, medium_launcher 18/18, jammer 23/24, helicopter 20/22, large_tower 21/24, ta-ta 20/22,
medium_plane 22/30, small_launcher 19/24, small_tower 18/25, tank 18/26, large_launcher 13/24 (16 cand/tile, 1069
alarms: gate v9 keeps 58% background and verifier v1 never saw the six-template candidates). Most classes stay
at about one candidate per tile with single-digit false alarms over 96 empty tiles.

## Pass 10 (10:20, pod 2, 1519 s): pass 9 code + auto track sprites + template caps 6 (vs pass 9)

tank 134 -> 171/171, medium_plane 33 -> 45/48, ta-ta 65 -> 67 (no refinement), jet_plane 73 (same, but gate v10
background kept .60 vs .14), large_launcher 61 -> 37 (the colour-method launcher sprite, 20% box fill, hurt), condor
0/42 (every tile raised "need at least one array to stack": a competitor part model built from the junk jet-plane auto
sprite; guarded now). Decision: keep the tank (6 rows, two frames per track) and medium-plane (6 rows) auto sprites,
drop launcher and jet-plane autos. Final bank: 86 sprites, on both pods; pod-1 old bank at
/workspace/experts/bank-backup-pod1-old, pod-2 pre-auto bank at /workspace/experts/bank-backup-prepass10.

## Box scaling bug (11:00): `Template.posed_with_box` scaled the organizer box twice

For any pose with scale != 1 the mask was resized first and its extent (w, h) then multiplied by the scale again, so
a .8 pose produced a .64 box and a delivered-resolution (.5) pose a .25 box. Effects: the first delivered-resolution
pass (pass 11d, old box) found nothing at L0/L1; the size refinement's losses at margin 0 (small towers, jet planes
at L2) were partly this bug, not the refinement. Fixed: only the organizer margins are scaled. Pass 11 (pod 1,
native, final bank), pass 11d (pod 2, delivered), the margin re-check and the four recordings all restarted with it.

### Refinement margin re-check with the fixed box (11:15, same 7 x 18 sample, final bank)

| variant | jet_plane | medium_plane | small_tower | tank | large_launcher | total of 126 |
|---|---|---|---|---|---|---|
| no refinement (old bank) | 5/5/5 | 3/6/6 | 6/6/6 | 4/6/6 | 3/4/3 | 94 |
| old box, margin .06 | 6/5/5 | 5/6/6 | 6/5/5 | 4/6/6 | 2/4/4 | 95 |
| fixed box, margin 0 | 6/4/3 | 5/6/6 | 6/6/6 | 6/6/6 | 2/4/4 | 96 |
| fixed box, margin .06 (deployed) | 6/5/5 | 5/6/6 | 6/6/6 | 6/6/6 | 2/4/4 | 99 |

Towers and ta-ta run without refinement (their organizer boxes are right at sprite size). Margin .06 stays.

## Pass 11 (11:50, pod 1, 844 s): final code + final bank, native resolution - the deployed configuration

| class | found | L0 / L1 / L2 | gate v11 oof recall / background kept |
|---|---|---|---|
| condor | 42/42 | 14/14 each | .98 / .196 |
| hangar | 9/9 (+3/24 partial) | 3/3 each | 1.00 / .000 |
| helicopter | 43/48 | 14/16 15/16 14/16 | .98 / .243 |
| jammer | 69/72 | 24/24 24/24 21/24 | .99 / .000 |
| jet_plane | 72/75 | 24/25 24/25 24/25 | .99 / .109 |
| large_launcher | 61/102 | 21/34 20/34 20/34 | .98 / .582 |
| large_tower | 48/57 | 16/19 each | 1.00 / .000 |
| medium_launcher | 17/18 | 5/6 6/6 6/6 | 1.00 / .040 |
| medium_plane | 45/48 | 13/16 16/16 16/16 | 1.00 / .001 |
| mine_roller | 15/15 | 5/5 each | 1.00 / .000 |
| small_launcher | 54/69 | 8/23 23/23 23/23 | 1.00 / .001 |
| small_plane | 63/63 | 21/21 each | .98 / .020 |
| small_tower | 46/51 | 12/17 17/17 17/17 | .98 / .001 |
| spacecraft | 57/57 | 19/19 each | .98 / .005 |
| ta-ta | 67/69 | 22/23 23/23 22/23 | .99 / .019 |
| tank | 171/171 | 57/57 each | .98 / .006 |

Total 879 of 1069 complete training targets (pass 7: 803, 75% -> 82%). Code: speed batches 1-3, fair-share capping,
size refinement (margin .06, off for towers and ta-ta), fixed box scaling, launcher sprites 6, template caps 6,
pose-cache fix; bank: 86 sprites (tank and medium-plane track sprites added). Remaining gaps: large_launcher (the
c-047-078 track has no usable sprite; its gate keeps 58% background), small_launcher at L0 (2-3 px), helicopter and
condor gates (.2 background kept; the verifier carries them).

Box audit on pass 11 (candidates near a target, IoU >= .5 share per zoom): tank 1.00/1.00/1.00, jet_plane .96/.96/.96,
medium_plane .81/1.00/1.00, ta-ta .96/1.00/.96, small_tower .71/1.00/1.00, helicopter .88/.94/.88, large_tower .84 each,
large_launcher .56/.53/.53 (pose/position on the uncovered track, not size). Median size ratios are 1.0 +- .1, so no
per-class box calibration is needed any more.

### verify11 (12:20, pod 1, 346 s): gates v11 + verifier v1, rotating sample seed 4, 8 tiles per class and zoom

tank 28/28, helicopter 21/21, small_plane 24/24, spacecraft 24/24, condor 24/24, mine_roller 15/15, medium_launcher
18/18, hangar 9/9, medium_plane 28/30, jet_plane 31/33, small_tower 23/26, ta-ta 21/22, jammer 23/24, large_tower
21/24, small_launcher 19/24 (L0 3/8), large_launcher 15/24 (17 cand/tile, 1069 alarms on 96 empty tiles). All other
classes about one candidate per tile; false alarms over 96 empty tiles: helicopter 214, jet_plane 231, condor 133,
small_plane 42, the rest <= 21.

Verifier v2 (`/workspace/experts/runs/verifier-0211/model/best.pt`, pass 11 candidates + synthetic, 12 epochs):
99.86% on held-out tiles (background 7236/7245); synthetic validation set 95.7% accuracy / 98.6% background rejection
(v1: 96.4% / 99.3%). verify11v2 = same sample with v2, pending; the better one on real candidates is deployed.

### verify11v2 (12:35): gates v11 + verifier v2, same sample as verify11

Identical recall to verifier v1 on every class; fewer false alarms on the 96 empty tiles: large_launcher 1069 -> 977,
helicopter 214 -> 193, jet_plane 231 -> 218, condor 133 -> 115, small_plane 42 -> 41. Verifier v2 is deployed
(`runs/verifier-0211/model/best.pt` on pod 1, `runs/verifier-0211-model/best.pt` on pod 2). The launcher/helicopter/
jet-plane false alarms survive both gate and verifier; the launcher gate (58% background kept since six templates)
is the next accuracy item: per-template gates.

## Per-template gates (12:50): `fit_gates.py` fits a sub-gate per sprite and keeps it only where it beats the class gate out-of-fold

| class | class gate background kept | with per-template gates | recall |
|---|---|---|---|
| large_launcher | .582 | .284 (2 sprite gates) | .984 |
| condor | .196 | .153 (2) | .982 |
| jet_plane | .109 | .079 (1) | .976 |
| small_plane | .020 | .004 (1) | .984 |
| ta-ta | .019 | .004 (3) | 1.000 |
| tank | .006 | .003 (4) | .983 |
| helicopter (one sprite) | .243 | .243 | .981 |

Without the selection rule the split hurt jet_plane (.11 -> .41) and condor (.20 -> .22): too few positives per sprite.
Deployed gates: `runs/pass11/gates-t2.json` (pod 1) = `runs/pass11-gates-t2.json` (pod 2). verify11t2 pending.
Speed batch 4 (opt-in DRONE_EXPERT_WINDOW_GPU: batched GPU fine-pose windows, L2 live 3.5 -> 1.95 s) applied for the
next recordings.

## Delivered-resolution recall check (pass 11d, 13:00, pod 2, 848 s): experts on delivered pixels (L0 x1/4, L1 x1/2), templates scaled

| class | native L0 / L1 (pass 11) | delivered L0 / L1 | verdict |
|---|---|---|---|
| tank | 57 / 57 | 33 / 57 | L1 fine, L0 needs more than 1/4 |
| jet_plane | 24 / 24 | 8 / 17 | native (or 1/2 at L0) |
| medium_plane | 13 / 16 | 8 / 4 | native |
| small_plane | 21 / 21 | 0 / 12 | native |
| ta-ta | 22 / 23 | 0 / 19 | native at L0 |
| small_launcher | 8 / 23 | 0 / 23 | L1 fine |
| medium_launcher | 5 / 6 | 3 / 5 | roughly fine |
| large_launcher | 21 / 20 | 22 / 20 | fine |
| large_tower | 16 / 16 | 13 / 16 | L1 fine |
| small_tower | 12 / 17 | 14 / 17 | fine (L0 even better) |
| jammer, condor, spacecraft, mine_roller, hangar | full | full | fine |
| helicopter | 14 / 15 | 15 / 15 | fine |

Total 749 vs 879 native. So the resolution can be chosen per class and zoom: at L1 all classes except the three
planes run on delivered pixels; at L0 tank, planes, ta-ta and the small launcher need native (or half) resolution.
pass11d2 (L0 at 1/2, L1 native) is running to fill in the middle setting.

verify11t (unselected per-template gates, verifier v2, same sample): false alarms on 96 empty tiles large_launcher
977 -> 592, small_plane 41 -> 8, ta-ta 20 -> 4, spacecraft 7 -> 2, tank 2 -> 1; but jet_plane 218 -> 617 and condor
115 -> 154 (the undertrained splits), recall unchanged except jet_plane 31 -> 30. verify11t2 (selected sub-gates) pending.

verify11t2 (selected per-template gates + verifier v2): false alarms large_launcher 565, jet_plane 163, condor 94,
small_plane 9, ta-ta 4, spacecraft 2, tank 1 (class gates: 977 / 218 / 115 / 41 / 20 / 7 / 2); recall unchanged
(jet_plane 30/33 vs 31/33). Deployed: `gates-t2.json`.

## First strategy recording (13:20): V-D (l2 top + revisit 3), gates v9 + verifier v1, min conf .3, 249 frames, 0 skipped

Local evaluator proxy: COCO mAP@0.50 = 0.158 (the team's run O scored 0.456 on this proxy and 0.511 on the API).
Submitted to the validation API as attempt V-D-l2-rev3; diagnosis of the proxy score in progress.

## Validation API, attempt V-D-l2-rev3 (13:40 CEST, uuid df97afdc): score **0.2390**

Recording D served by replay_server.py behind a quick tunnel. Frames 1-8 failed (tunnel 530/502 before it routed;
readiness check added), the rest scored. Local proxy 0.158 -> API 0.239 (run O: 0.456 -> 0.511). Recording B (l2 top,
no revisit) proxy 0.198, submitted next.

Where the proxy score goes (recording D vs local ground truth, 987 boxes, 249 frames):
- Coverage: only 12% of ground-truth centres ever lay inside a requested view (l2_top band); the tracker carried the
  rest. Raw detector hits on objects in view: tank 14/19, small_launcher 15/15, medium_plane 6/8, launcher 2/18,
  jet_plane 2/17, helicopter 2/10, large_tower 0/7, small_tower 0/6, mine_roller 2/4.
- Helicopter answers are half the labelled size (w .53, h .61): it flies higher than the reference sprite; scale sweep
  (1, 1.5, 2) added for helicopter.
- large_launcher: 649 answered boxes for 149 labels (6 hit); ta-ta 112 and condor 77 answers with no such object in
  the scene. gates-t2 + verifier v2 cut these on training empties; verified next on the API.
- Towers and mine roller: no answers at all in this recording (in view 7/6/4 times only).

### pass 11d2 (14:05): L0 at half native scale (2x upsample instead of 4x), L1/L2 native

L0 found: tank 56/57, small_tower 17/17 (native 12), large_tower 16/19, jammer 24/24, condor 14/14, spacecraft 19/19,
helicopter 13/16, large_launcher 20/34, medium_plane 10/16 (13), jet_plane 18/25 (24), small_plane 14/21 (21),
ta-ta 10/23 (22), small_launcher 5/23 (8), mine_roller 5/5. L0 total 262 vs 278 native (94%) at a quarter of the
pixels. Live detector knob: DRONE_EXPERT_LEVEL_FACTORS=2,2,1 (recording W-A restarted with it: L0 frames were 44 s).

Validation attempt V-B-l2top (uuid 106ce2d0): score 0.0056 with no reported errors, but the replay server received a
single /predict request during the attempt (D's server received 239), so the tunnel dropped the traffic; the number
is not the recording's. Resubmitted through a fresh quick tunnel with request counting.

Recording W-A (l1 sweep + overview, gates-t2, verifier v2, helicopter scales, GPU windows, LEVEL_FACTORS 2,2,1) on pod 1:
L0 frames 8.2 s (44 s at native scale), L1 frames 6.3 s, with 16 pool processes.

Attempt V-B-l2top via a second quick tunnel (uuid 909c1a04): score 0.0, no errors, again a single request reached the
server. Quick tunnels are abandoned; submissions use the Runpod HTTP proxy (port 19123, `validate_proxy.sh`).

## W-D (improved config) vs V-D on the local proxy (14:50): 0.133 vs 0.158

| | V-D (gates v9, verifier v1) | W-D (gates-t2, verifier v2, helicopter scales) |
|---|---|---|
| hits / answered | 241 / 1593 | 171 / 548 |
| tank hit / answered | 118 / 177 | 60 / 82 |
| small_launcher | 49 / 138 | 43 / 84 |
| hangar | 17 / 56 | 11 / 67 |
| medium_plane, jet_plane | 31 / 63, 20 / 31 | 31 / 31, 20 / 31 |
| large_launcher answered | 649 (6 hit) | 48 (6 hit) |
| ta-ta / condor answered (absent) | 112 / 77 | 35 / 0 |

The false boxes fell by two thirds, but tank recall on the validation scene halved: the per-sprite gates (fitted
at 98% recall on training candidates) or verifier v2 reject validation-scene tanks that the class gate accepted.
Attribution recordings X-D-gv11 (class gates v11 + verifier v2) and X-D-gt2v1 (gates-t2 + verifier v1) running.
Proxy B attempt (uuid b8a2ed08) scored 0.0 because I killed the replay server during the run; resubmitted.
The organizer's client matches the local evaluator when the server is reachable (249 requests, B proxy 0.201).

## Validation API, attempt V-B-l2top via the Runpod proxy (15:05, uuid 943198d5): score **0.2208** (proxy 0.198-0.201)

All frames reached the server. The 16 reported errors are camera rejections, an organizer rule the team's local
evaluator does not enforce: "ignored camera request L2 (3360, 270) from L2 (2400, 270): center movement 960 px
exceeds the L2 limit of 551 px". Our l2 sweeps step 960 px per frame, so on the real service the camera stays put
and the served views diverge from the recording (scores of recorded answers are unaffected; live strategies are).

API scores so far: D (l2 top + revisit 3) 0.239, B (l2 top) 0.221, both gates v9 + verifier v1.

Attempt W-D-l2-rev3 via proxy (uuid 67d3ebf0): score 0.066, no errors, but again only one request reached the server
(a laptop burst through the same proxy arrives completely). Three of five attempts so far stopped after the first
request without any reported error; resubmitted to test whether it is transient.

## Validation API, attempt W-D-l2-rev3 via proxy with a cleanly started server (15:50, uuid 9c4474b9): **0.2146**

219 requests served; the 16 errors are the L2 camera-step rejections. The earlier one-request attempts all used a
replay server started inside an ssh session that never returned; servers started with `( setsid nohup ... & )` in
their own ssh call serve every frame. API scores: V-D 0.239 (gates v9, verifier v1), V-B 0.221 (same), W-D 0.215
(gates-t2, verifier v2, helicopter scales). Attribution recordings X-D-gv11 (class gates v11 + v2), X-D-gt2v1 (gates-t2
+ v1) and X-D-v9v1 (gates v9 + v1, helicopter scales only) decide which change costs the tank recall.

Recording W-A (l1 sweep + overview, gates-t2, verifier v2, helicopter scales, L0 at half scale): proxy **0.198**,
249 frames, 0 skipped: coverage beats the L2-top strategies (W-D 0.133 with the same configuration). Submitted from
pod 1 via proxy.

Attribution recordings X-D-gv11 (0.063) and X-D-gt2v1 (0.059) are INVALID: run concurrently, their 32 pool workers
shared the GPU for the batch-4 windows and experts failed with "cuFFT error: CUFFT_INTERNAL_ERROR" (166 and 113
class failures; planes never answered). Rule: one GPU-window recording per pod at a time. X-D-v9v1 reruns alone.

## Validation API, attempt W-A-l1sweep via pod 1's proxy (16:10, uuid 05726c63): **0.1501** (proxy 0.198)

202 of 249 frames reached the server; the listed errors are L1 camera-step rejections (limit 1102 px, our sweep
steps 1920 px). The missing frames cost the score (the proxy evaluator gives 0.198 with all frames answered).
API so far: V-D 0.239, V-B 0.221, W-D 0.215, W-A 0.150; local proxies 0.158, 0.198, 0.133, 0.198.

Frame loss through the proxy is scattered (48 single or paired frames over the run, no long stall): the organizer
emits frames every 333 ms in real time and skips the frames a slow round trip overlaps, so transport latency alone
costs 12-19% of frames on proxy attempts (W-D 219/249, W-A 201/249; the tunnel attempt D served 239/249). Scores
from different transports are therefore not directly comparable; the local proxy stays the like-for-like measure.

Attribution (valid, one recording per pod): X-D-v9v1 (gates v9, verifier v1, helicopter scales) proxy 0.158 = V-D, so
the helicopter scale sweep is neutral on the proxy; the W-D drop to 0.133 comes from gates-t2 and/or verifier v2.
X-D-gt2v1 (pod 1) and X-D-gv11 (pod 2) running.

X-D-gt2v1 (gates-t2, verifier v1, helicopter scales), alone on pod 1: proxy 0.150. Attribution on the l2-top strategy:
gates v9 + verifier v1 0.158 (V-D, X-D-v9v1); per-sprite gates-t2 + v1 0.150; gates-t2 + verifier v2 0.133 (W-D).
The training-fitted refinements generalise worse to the validation scene than the plainer pair, so gates v9 +
verifier v1 stay deployed. Y-A (l1 sweep, that pair, L0 at half scale, batch 7, MPS) recording on pod 1.

X-D-gv11 (class gates v11, verifier v2): proxy 0.153. Full attribution on l2 top + revisit 3: v9+v1 0.158, v11+v2 0.153,
t2+v1 0.150, t2+v2 0.133. Y-C (l1 sweep + overview + revisit 3, v9+v1, latest code) recording on pod 2.

Recording Y-A (l1 sweep + overview, gates v9 + verifier v1, helicopter scales, L0 at half scale, speed batches 1-9,
MPS): proxy **0.208**, 249 frames, no errors, ~2.7 s per frame at L0 and L1. Best local proxy so far
(W-A 0.198 with gates-t2 + v2; V-D 0.158). Submitted from pod 1.

## Validation API, attempt Y-A-l1sweep via pod 1's proxy (17:20, uuid 15ee68c3): **0.2035** (proxy 0.208, 222/249 frames served, 7 camera errors)

API scores so far: V-D 0.239 (tunnel, 239 frames), V-B 0.221, W-D 0.215, Y-A 0.203, W-A 0.150. Local proxies:
Y-A 0.208, W-A 0.198, V-B 0.198, V-D 0.158, W-D 0.133. The API rewards the l2-top recordings relative to the proxy
(the organizer's labels vs the team's quick labels differ per class); the l1 sweep is the better local proxy but not
yet the better API score. Y-C (l1 sweep + revisit 3) recording on pod 2; Y-A2 (l1 sweep without L0 overviews) next.

Recording Y-C (l1 sweep + overview + revisit 3, gates v9 + verifier v1, latest code, MPS): proxy 0.193 (Y-A without
revisits 0.208). Submitted from pod 2. Y-A2 (l1 sweep, no L0 overviews) recording on pod 1.

Attempt Y-C-l1-rev3 via proxy (uuid b3034191): 0.054 with one request delivered again, so the launch method was not
the cause. Hypothesis: the Runpod proxy answers later requests from a cache or a reused connection. replay_server.py
on both pods now sends Cache-Control: no-store and Connection: close and runs uvicorn with keep-alive off; Y-C
resubmitted through it.

## Recording Y-A2 (18:00): l1 sweep WITHOUT L0 overviews, gates v9 + verifier v1, latest code: proxy **0.284**

Every earlier recording had L0 overview frames between the L1 sides (Y-A 0.208, W-A 0.198). Dropping them gives
0.284: our L0 answers cost more than they add (weak L0 recall plus false tracks) and the L1-only sweep spends every
frame at a resolution the experts handle. Y-A2 is the next submission from pod 1 (after Y-C).

Attempt Y-C from pod 1 (uuid 973d2ed8): one request delivered, 0.0. So the single-request failure is neither the pod
nor the server: of 12 attempts, 5 ran fully and 6 stopped after frame 1 with no reported error and an 83 s duration,
alternating in time. Most likely one of the organizer's queue workers loses our endpoint after the first frame.
Countermeasure: `validate_retry.sh` resubmits until more than 100 requests reach the server (Y-A2 running).

## Validation API, attempt Y-A2-l1only from pod 1 (18:20, uuid 599de753, first try full): **0.2346** (proxy 0.284)

191 of 249 frames delivered; the 16 listed errors are L1 camera-step rejections (our sweep moves 1920 px, the limit
is 1102 px). API table: V-D 0.239 (239 frames), Y-A2 0.235 (191), V-B 0.221, W-D 0.215, Y-A 0.203, W-A 0.150.
Per delivered frame the L1-only sweep is clearly the best strategy; the transport pacing now costs more than any
pipeline change. Next: Y-A3 (birth confidence .3), Y-C2 (L1-only + revisits), Y-A5 (min confidence .15).

Memory note (18:40): with speed batches 7-9 (6000-pose cache, GPU windows per worker) one 16-process recording holds
the whole A100 (workers 4-14 GB each); a second recording on pod 2 failed at startup with CUDA OOM, and on pod 1
the second one runs on CPU fallbacks. Launches now use DRONE_EXPERT_POSED_CACHE=1500 and one recording per pod.

Recording Y-A3 (l1 sweep with L0 overviews, track birth confidence .3 instead of .5): proxy 0.208 = Y-A, so the
tracker's birth threshold is neutral; not submitted. Strategy proxies with the deployed pair: L1-only sweep 0.284,
L1 sweep + L0 overviews 0.208, + revisits 0.193, l2 top + revisits 0.158.

Recording Y-C2 (L1-only sweep + revisit every 3 frames): proxy 0.254 (Y-A2 without revisits 0.284). Revisits at
L2 cost more sweep coverage than they add; not submitted.

Recording Y-A5 (L1-only sweep, detector minimum confidence .15 instead of .3): proxy 0.284 = Y-A2; the confidence
floor is neutral (the gate and verifier already decide). Not submitted. Remaining knobs under test: update
confidence .5 (Y-A7, pod 1), half-height band (Y-A6, pod 2).

Recording Y-A6 (L1-only sweep with the band at half frame height, vertical fraction .5): proxy 0.164 (top band
0.284). Objects are caught entering at the top; the mid band sees far fewer of them. Not submitted.

Recording Y-A7 (L1-only sweep, track update confidence .5): proxy 0.284 = Y-A2. Every tracker knob tested (birth .3,
update .5, detector floor .15) is neutral; 0.284 is the L1-only sweep's local ceiling with the deployed detector.

Y-A2 rerun on the API (19:35, uuid cb481c4d): **0.209** with 180/249 frames delivered (first run 0.235 with 191).
The API score of a fixed recording moves with the frames the organizer's pacing delivers: about 0.0025 per frame,
so two attempts of the same recording differ by 0.02-0.03. Configuration differences below that need repeats.

Recording Y-A8 (L1-only sweep, class gates v11 fitted on the final code + verifier v1): proxy 0.240 (gates v9:
0.284). Gates fitted on the later passes reject more validation-scene objects than the pass-9 gates although their
training out-of-fold numbers are equal or better; gates v9 stay deployed. Not submitted.

Recording Y-A9 (L1-only sweep with L1 at DELIVERED resolution, factor 1): proxy 0.184 (native L1: 0.284) at 1.4 s
per frame instead of 2.7 s. L1 stays native for the score; per-class routing (planes and ta-ta native, the rest
delivered) is the only cheaper option left and is tested as Y-A11.

Recording Y-A11 (L1-only sweep, routing: tank, planes and ta-ta at native L1, other classes at delivered
resolution): proxy 0.255 at 1.8 s per frame (all native: 0.284 at 2.7 s; all delivered: 0.184 at 1.4 s). The
routing recovers most of the native score for a third less time; which classes to keep native is now a
speed-versus-recall knob for the live endpoint. Not submitted (below the all-native recording).

Recording Y-A10 (L1-only sweep with the 68-sprite bank from before the track sprites): proxy 0.233 (86-sprite bank
with the tank and medium-plane track sprites: 0.284). The auto-cut training-split track sprites are worth +0.05 on
the validation scene, the largest single detector gain of the day. Y-A13 tests restoring the launcher and jet-plane
auto sprites that were dropped for hurting the training gate.

Validation API, attempt Y-C-l1-rev3 full run from pod 1 (20:40, uuid ad1604f7): **0.164** (proxy 0.193, 179/249
frames). Confirms the proxy ordering: L1-only sweep 0.235/0.209 > L1 sweep + revisits 0.164.

Recording Y-A12 (L1-only sweep, bank without the large-tower box-mask sprite): proxy 0.284 = Y-A2. The crude
tower sprite neither helps nor hurts on the validation scene; it stays in the bank pending Oscar's review sheet.

## Recording Y-A13 (20:55): L1-only sweep with ALL track sprites (tank, medium plane, launcher c-047-078, jet-plane c): proxy **0.305**

The launcher and jet-plane auto sprites that were dropped after pass 10 (they widened the training gate) add
+0.021 on the validation scene on top of the 0.284 of the 86-sprite bank. Validation recall beats training-gate
tightness here; the 92-sprite bank (`bank-allauto`) becomes the deployed bank on both pods. Submitted from pod 1.

Validation API, attempt Y-A13-allauto from pod 1 (21:05, full run first try): **0.2342** with 179/249 frames (Y-A2:
0.2346 with 191 frames, 0.209 with 180). Per delivered frame the 95-sprite bank is ahead, matching the proxy
(0.305 vs 0.284); on the API the difference sits inside the delivery noise.

Recording Y-A14 (L1-only sweep, gates v9 only, NO verifier): proxy 0.169 (with verifier v1: 0.284). The verifier is
the single most valuable stage on the validation scene (+0.115): the gates alone pass far too many false boxes.
Not submitted.

Recording Y-A15 (L1-only sweep, 95-sprite bank, routing: tank, planes and ta-ta native at L1, rest delivered):
proxy 0.271 at 1.8 s per frame (all native on the same bank: 0.305 at 2.7 s). Routing costs about 0.03 on either
bank; the live endpoint can spend it if the frame budget demands.

Recording Y-A16 (L1-only sweep, 95-sprite bank, verifier-confidence floor .5): proxy 0.300 (floor .3: 0.305). The
low-confidence tail holds slightly more true boxes than false ones; the floor stays at .3. Not submitted.

Recording Y-A17 (95-sprite bank + box-mask sprites of helicopter-043-073): proxy 0.283 (without them 0.305). Box
masks with 2-9% real foreground add false helicopter boxes; discarded (bank-heli not deployed). Training-split
sprite sources are now exhausted: every track with training tiles is either in the bank or measured as harmful.

Proxy determinism: re-scoring the Y-A13 recording with the local evaluator gives 0.303 (first 0.305), so the local
proxy's own noise is about 0.002 and differences of 0.02 between recordings are real.

Budget (22:15): pod 1 about 26 h and pod 2 about 20 h at $1.59/h, roughly $74 of the $100 night budget. Pod 2 is
stopped now (disk kept, never terminated); pod 1 stays up for recordings and submissions.

## Recording Y-A18 (22:20): L1-only sweep, 95-sprite bank, sprite caps 8 instead of 6 for the multi-sprite classes: proxy **0.310**

New best (+0.005 over caps 6, above the 0.002 proxy noise). Deployed: `fine_templates=8, proposer_templates=8` for
tank, medium_plane, large_tower, helicopter, jet_plane, large_launcher (working tree and pod 1). Submitted from pod 1.
Pod 2 stopped at 22:15 (EXITED, disk kept).

Validation API, attempt Y-A18-caps8 from pod 1 (22:30, full run first try): **0.2282** with 165/249 frames
delivered (0.00138 per delivered frame; Y-A13 0.00131, Y-A2 0.00123). Per delivered frame the proxy ordering holds
(caps 8 > 95-sprite bank > 86-sprite bank); the raw API number is dominated by how many frames the organizer's
pacing delivers (165-191 across attempts).

## Recording Y-A19 (22:55): sprite caps 12 (every sprite of the multi-sprite classes takes part): proxy **0.315**

Caps 6 / 8 / 12: 0.305 / 0.310 / 0.315 at 2.7 / 2.9 / 3.3 s per L1 frame. Every additional sprite view adds recall on
the validation scene; caps 12 deployed (working tree and pod 1). Submitted from pod 1.

Validation API, attempt Y-A19-caps12 from pod 1 (23:05, full run first try): **0.2302** with 172/249 frames
(0.00134 per delivered frame; Y-A18 0.00138, Y-A13 0.00131, Y-A2 0.00123). The last three configurations are
indistinguishable on the API within the delivery noise; the proxy separates them by 0.005 each.

Recording Y-A20 (sprite caps 16, i.e. every sprite of every class): proxy 0.314 (caps 12: 0.315) at 3.6 s per frame.
Saturated; caps 12 stay deployed. Training-only levers are exhausted for this session.

## Continuation on the 4-GPU pod (morning of 20 Sep local): baseline Z-A-base proxy 0.315, 249 camera moves applied, 0 refused

The policy already keeps every move legal (limits in dtos.py equal the organizer's), so the API camera errors were
replay divergence only. Per-class losses of the baseline (987 labels, 409 hit = 41%):

| class | GT | hit | answered | ever in view | raw hits in view | note |
|---|---|---|---|---|---|---|
| tank | 164 | 116 | 174 | 70 | 48 | found 69% when seen; the rest never enters a view |
| medium_plane | 66 | 63 | 80 | 28 | 23 | solved |
| jet_plane | 112 | 63 | 67 | 50 | 26 | half found in view |
| small_launcher | 97 | 64 | 167 | 42 | 39 | in-view recall fine, 77 confident false boxes |
| large_launcher | 149 | 42 | 138 | 33 | 3 | detector barely fires on the validation launchers |
| helicopter | 105 | 27 | 45 | 36 | 8 | boxes now right-sized (w/h 1.01/1.10); recall low |
| hangar | 69 | 34 | 61 | 28 | 6 | |
| large_tower, small_tower, mine_roller | 96, 45, 52 | 0 | 0 | 22, 23, 18 | 0 | validation-scene looks absent from the training bank |
| medium_launcher | 32 | 0 | 193 | 13 | 0 | answers 2x wider than Oscar's labels (organizer-style box margins), 172 with conf > .5 |

Reachable without validation-scene sprites: coverage (tank, jet plane) and in-view recall of jet plane/hangar.
Being tested: L0 overviews with the current bank and caps (Z-A-ov), band lowered 15% (Z-A-v15).

Recording Z-A-v15 (L1-only sweep, band lowered by 15% of its range): proxy 0.257 (top band 0.315). The top edge is
where objects enter; every lower band tested loses. Band settled at the top.

Recording Z-A-ov (L1 sweep + L0 overviews between sides, current 95-sprite bank and caps 12): proxy 0.229
(L1-only 0.315). The verdict on L0 overview frames holds with the better detector: they cost coverage at L1 and
add false tracks. Z-A-w6 (six-waypoint L1 sweep) recording.

Recording Z-A-w6 (six-waypoint L1 sweep: left, centre-left, centre, right, centre-right, centre): proxy 0.276
(four-waypoint 0.315). A longer cycle revisits each side later, and objects at the top edge leave before the
camera returns; wider coverage per cycle loses to revisit frequency. Z-A-w2 (sides only) and Z-A-w4q (left,
centre-left, right, centre-right) recording.

## Recording Z-A-w2 (sides-only L1 sweep, DRONE_L1_WAYPOINTS=2): proxy **0.334**, new best (four-waypoint 0.315)

Requesting only the two sides lets the policy's legal-step clamp produce the intermediate views (960 -> 2061 -> 2880
-> 1779 -> 960): each side is revisited every third frame instead of every fourth, and the centre is still covered
by the clamped steps. Six waypoints 0.276, quarter positions (Z-A-w4q) recording. Submitted from the 4-GPU pod.

## Per-object failure analysis on the Higgsfield synthetic VALIDATION set (Oscar's request, 20 Sep): `eval_synthetic.py`

216 composites on dev backgrounds, 363 pasted reviewed sprites (rotated by quarter turns, half of them mirrored,
scale 0.9-1.1, Higgsfield-blended), run through the deployed pipeline (proposer, expert, gate v9, verifier v1).

| class | objects | found | lost at gate | box (IoU .2-.5) | pose | no proposal |
|---|---|---|---|---|---|---|
| large_launcher | 21 | 86% | 0 | 5% | 10% | 0 |
| medium_launcher | 12 | 92% | 0 | 0 | 0 | 0 |
| hangar | 18 | 78% | 22% | 0 | 0 | 0 |
| condor | 15 | 67% | 33% | 0 | 0 | 0 |
| jet_plane | 30 | 63% | 10% | 27% | 0 | 0 |
| small_tower | 24 | 62% | 33% | 4% | 0 | 0 |
| mine_roller | 24 | 54% | 46% | 0 | 0 | 0 |
| tank | 39 | 51% | 44% | 5% | 0 | 0 |
| small_plane | 30 | 43% | 7% | 40% | 10% | 0 |
| small_launcher | 33 | 42% | 9% | 9% | 0 | 39% |
| spacecraft | 36 | 39% | 39% | 22% | 0 | 0 |
| jammer | 27 | 33% | 44% | 7% | 15% | 0 |
| medium_plane | 27 | 26% | 15% | 52% | 7% | 0 |
| ta-ta | 27 | 0% | 22% | 52% | 26% | 0 |

Unflipped sprites are found 72% (204), mirrored ones 19% (159): jet_plane 15/15 vs 4/15, spacecraft 12/12 vs 2/24,
tank 16/21 vs 4/18, small_plane 12/15 vs 1/15, medium_plane 7/12 vs 0/15, jammer 9/15 vs 0/12. Zoom: L0 39%,
L1 51%, L2 56%. The verifier rejected nothing (it was trained on these composites' cousins).
Reading per object:
- Gate v9 is the biggest loss on new backgrounds: it rejects 44-46% of true tanks, jammers and mine rollers and 33-39%
  of towers, condors and spacecraft, mostly mirrored ones. The gate learned the training tiles' background and
  sprite statistics (rim contrast, colour distance, competitor margins), i.e. it is overfitted to the sprites and
  their backgrounds. Gate-free recording on the real scene running (Z-A-nogate).
- Mirrors: the heading sweep covers rotations, never mirror images, so a mirrored sprite lands slightly off in pose
  and then fails the gate or the IoU. A mirrored copy of every sprite in the bank is being tested (bank-mirror).
- Box failures for planes, spacecraft and ta-ta are 1.3-1.6x too large against the synthetic tight boxes: the
  synthetic labels are the sprite extent while our boxes carry the organizer's margins (the real labels do too), so
  this is mostly a label-convention difference, but mirrored planes also land off-centre.
- ta-ta is broken end to end on composites (0/27): tight boxes plus pose failures; small launchers fail at the
  proposal stage on dev backgrounds (colour blob depends on the background).

## Synthetic composites, per object (19 Sep, Oscar's step-back request)

Pipeline: proposer -> expert -> gate v9 -> verifier v1, current bank (95 sprites, caps 12), delivered scale per zoom.
Outcome per pasted object: found / gate (true object reached the gate and was rejected) / box (found but IoU < 0.5
against the tight synthetic box) / pose / proposal (never proposed).

| set | bank | found unflipped | found mirrored | L0 / L1 / L2 |
|---|---|---|---|---|
| validation composites (216, dev backgrounds) | 95 | 72% | 19% | 39 / 51 / 56% |
| validation composites | 190 (mirror of every sprite) | 69% | 50% | 44 / 64 / 74% |
| training composites (531, train backgrounds) | 95 | 55% | 22% | 31 / 42 / 42% |

Per class on the validation composites with the mirrored bank (found / gate-rejected / box):
condor 80/0/0, hangar 78/22/0, jammer 41/59/0, jet_plane 97/0/3, large_launcher 90/0/10, medium_launcher 100/0/0,
medium_plane 22/26/41, mine_roller 54/46/0, small_launcher 42/9/9 (39% never proposed), small_plane 60/23/13,
small_tower 79/21/0, spacecraft 67/28/6, ta-ta 4/33/37 (26% pose), tank 69/26/5.

Training composites with the 95-sprite bank are worse than the validation ones for the same reason: the gate
rejects 75% of condors, 67% of small launchers, 50% of hangars, 48% of jammers, 45% of medium planes, 39% of
mine rollers even on training backgrounds. The gate is fitted on real training tiles; pasted sprites at random
scale and heading have different feature statistics (competitor scores, colour stats), so gate rejection here is
partly a synthetic artefact. Whether the gate hurts on the real validation scene is the Z-A-nogate recording.

Reading per object:
- Mirrored poses are the largest single gap: 19% -> 50% just by adding mirrored sprites. Cost: 2x proposer kernels.
- The gate is the second: jammer, mine roller, tank, spacecraft, small tower lose 21-59% at the gate.
- Box convention: medium_plane and ta-ta boxes miss IoU 0.5 against tight boxes (organizer boxes carry margins,
  synthetic boxes are tight, so the box column is partly a label-convention artefact; ta-ta is real: 4% found).
- small_launcher: 39% never proposed (too small at delivered scale; needs a lower proposer threshold or L2 only).
- Perspective distortion is not testable on 2D pastes; the sets contain none.

## Speed-motivated A/Bs on the sides-only sweep (baseline Z-A-w2 0.334)
- Z-A-h30 (proposer heading step 30 instead of 15, half the kernels): 0.294 (-0.04). Not free.
- Z-A-nogate (no logistic gate, verifier only): 0.287 (-0.047). The gate helps on the real scene, so the synthetic
  gate rejections are mostly a paste artefact (feature statistics of pasted sprites), not a gate bug. Gate stays.
- Z-A-w2 on the validation API (full run, 149 requests): **0.246**, 16 errors. New expert-pipeline API best (V-D 0.239).
  Proxy 0.334 -> API 0.246, the usual delivery/replay gap.
- Z-A-cap12 (max_candidates 24->12, colour candidates 12->6, ta-ta colour 30->15): 0.320 (-0.014). Cheap but not free.
- Z-A-off0 (fine_offsets (0.,) instead of (-8, 0, 8); one alternative heading as before): 0.324 (-0.010).
- Z-A-hyb (speed batch 13: shared proposer on delivered 960x540 pixels, experts at native scale, DRONE_PROPOSER_FACTORS=1,1,1): 0.237 (-0.097).
  Same loss as the delivered-L1 test: the coarse peak search on delivered pixels loses the small objects. Not viable as is.
- Z-A-mirror (190-sprite mirrored bank, caps still 12): 0.326 (-0.008). Per class vs Z-A-w2: hangar 0.54 -> 0.73,
  helicopter 0.26 -> 0.32, medium_plane 0.95 -> 0.78, tank 0.65 -> 0.59, large_launcher 0.22 -> 0.12; the four zero
  classes stay at zero. Mirrors help the few-sprite classes and crowd out originals in the capped classes (the cap of 12
  now holds 6 originals + 6 mirrors). Next config to test: mirrors only for classes with fewer than 6 sprites, or
  caps 24 for medium_plane / tank / large_launcher with the mirrored bank.
- Z-A-scales (scales 1/1.25/1.5 for large_tower, small_tower, mine_roller, large_launcher; 1/1.5/2 for medium_launcher): 0.325 (-0.009).
  The four zero classes stay at zero and large_launcher drops 0.22 -> 0.125; every other class is identical. Size is not
  what blocks the towers, the mine roller or the medium launcher; the failure is earlier (proposal or gate) and needs a
  stage-by-stage check on the real frames.

## Real-frame stage analysis (eval_real.py: L1 view centred on each labelled object, delivered-then-upsampled like the live detector)
Small sample (every 10th frame, up to 20 objects per class), current pipeline:
| class | objects | found | gate | box | pose | proposal |
|---|---|---|---|---|---|---|
| helicopter | 10 | 30% | 20% | 10% | 30% | 10% |
| large_launcher | 15 | 7% | 0 | 13% | 80% | 0 |
| large_tower | 9 | 22% | 0 | 44% | 33% | 0 |
| medium_launcher | 3 | 33% | 0 | 67% | 0 | 0 |
| mine_roller | 5 | 0 | 0 | 0 | 60% | 40% |
| small_tower | 4 | 0 | 50% | 50% | 0 | 0 |
"pose" = proposals lie on the object (score 0.4-0.5) but none survives the 24-candidate cut, so no fine pose is fitted;
"box" = pose right, box IoU 0.2-0.5. Larger sample (every 3rd frame, 40 per class, all 11 classes) and the same with the
scale sweep are running as runs/real-val-big and runs/real-val-scales.
Box details from the small sample (pred vs local GT): large_tower boxes 1.4-2.3x too WIDE (height about right), small_tower
1.5-2x too wide, medium_launcher 1.6x too wide, helicopter 1.6x too wide; large_launcher boxes too small (0.6-0.9x) and
shifted down by up to 60 px (shadow). Towers are vertical objects: rotating the sprite by "heading" emulates the perspective
lean but the axis-aligned box of the rotated organizer box balloons in width. The local GT boxes are tight around the
object. large_launcher and mine_roller: the true proposals (score 0.4-0.5) lose the 24-candidate cut to background
peaks scoring 0.6-0.98 that the expert then ACCEPTS (13-22 accepted per view = the 77 FPs).

## Oracle replay on the validation API (Oscar's box-positioning check)
oracle-gt: the LOCAL ground truth of the validation scene (all 987 labelled instances, normalised full-frame boxes,
confidence 1, same camera path as Z-A-w2) replayed to the validation API: **0.329** (186 requests, 16 camera errors).
Perfect answers against the local labels score 0.33 on the API. Z-A-w2 (0.334 local proxy) scored 0.246 = 75% of that
ceiling. So the local scene's labels cover only about a third of what the organizer scores; the missing two thirds
are objects no local run has ever labelled, and the local proxy cannot see them. Box positioning is not the gap.
The local validation scene (data/drone/reconstructed-validation) is complete at native resolution for 245 of 249
frames (manifest native_coverage 1.0), so the frames contain every object; only the labels are incomplete
(run_metadata: "participant pseudo-labels, incomplete", 37 tracks, 987 boxes, 4 objects per frame). The missing
objects are visible in the local frames and can be labelled locally.
Single-class API replays of Z-A-w2 (same camera path; the per-class scores add up to the full score):
| replay | API score |
|---|---|
| oracle-gt (local labels) | 0.329 |
| only-medium_plane | 0.054 |
| only-tank | 0.052 |
| only-jet_plane | 0.037 |
| only-hangar | 0.045 |
| only-small_launcher | 0.012 (94 frames) |
| only-helicopter | 0.017 |
| only-large_launcher | 0.003 |

## Frame delivery on the validation API (from the replay-server logs)
Every API run lasts 83-84 s = 249 frames x 333 ms: the organizer sends frames on a fixed clock and SKIPS a frame
whenever the previous request has not returned. Frames that reached the replay server (which answers in ~1 ms):
oracle-gt 185/249, only-medium_plane 178/249, Z-A-w2 151/249; gaps of 2 (one frame skipped) 45-60 times per run,
occasional gaps of 10-25 frames. The round trip organizer -> Runpod HTTP proxy (EUR-IS-1) -> pod exceeds 333 ms a
third of the time. Unanswered frames can only score zero, so delivery alone caps the score at 0.6-0.75 of what the
answers are worth, independent of the detector. Levers: a pod/data centre closer to the organizer, a direct TCP port
instead of the HTTP proxy, and answering every request within a few ms. The same cap applies to every team run.
- Z-A-proute (speed batch 15, delivered-pixel proposer for large_launcher/helicopter/tank/large_tower): 0.279 (-0.055);
  tank 0.65 -> 0.41, helicopter 0.26 -> 0.01, large_launcher 0.22 -> 0.10. Proposer stays native for every class.
Reading of the single-class scores: local AP medium_plane 0.95 / tank 0.65 / jet_plane 0.55 vs API single-class 0.054 /
0.052 / 0.037. If the organizer averages over 16 classes these are AP 0.86 / 0.83 / 0.60 on the API, i.e. tank scores
HIGHER on the API than against the local labels (its local "false positives" are real tanks); if over 11 classes,
0.59 / 0.57 / 0.41. oracle-only-<class> replays (local labels of one class only) are queued to pin down the class count.
Drone main (09:20 UTC): the run O replay (0.511) was served from pod 1 (EUR-IS-1) through a cloudflared quick tunnel,
ran 83 s with no errors; how many of the 249 frames reached its server was never captured (the log would be on the
stopped pod 1). So the frame-loss cap may or may not have applied to the team's best score; it needs one measured run.
E2E Drone Agent (09:30 UTC): run O's replay received 249/249 frames through a cloudflared quick tunnel started ON the
pod (`cloudflared tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:PORT`); its live runs lost 0-3 frames,
all attributable to slow answers. So the third of frames lost is the Runpod HTTP proxy path only. Switched: tunnel on
the 4-GPU pod (tunnel-url file), every replay above is being resubmitted through it (log: single-class-runs.log after
the TUNNEL BATCH line). Proxy-path single-class scores so far: medium_plane 0.054, tank 0.052, hangar 0.045,
jet_plane 0.037, small_launcher 0.012 (94 frames delivered).
Proxy-path single-class sum: 0.054+0.052+0.037+0.045+0.012+0.017+0.003 = 0.220 vs the full run's 0.246 (the
rest is ta-ta/condor/medium_launcher answers, measured next through the tunnel). large_launcher gets almost no API
credit (0.003) although it scores 0.22 against the local labels: its local hits are tracker carries of pseudo-label
tracks, not real detections (raw hits 3/33 in view).
Partial results (pod stopped on Oscar's instruction before the runs finished; same objects compared with the baseline):
| variant | objects | baseline found | variant found | reading |
|---|---|---|---|---|
| box = posed mask extent (tight 0) | 96 | 13 | 7 | worse: large_tower box failures unchanged 16/16, large_launcher 5 -> 1 found |
| box = mask extent + 5% | 96 | 13 | 8 | same |
| max_candidates 64 | 59 | 7 | 8 | large_launcher 5 -> 6 found, pose 29 -> 25; mine_roller still 0 |
So the box failures are not the rotated-box inflation alone: the fitted pose itself is off (IoU 0.2-0.5 with the
mask extent too), and a larger candidate cut barely helps. For towers, mine roller, medium launcher, helicopter and
large launcher the training sprites (one instance each) simply do not match the validation instances well enough for
template matching. Fixes left: validation-scene sprites (Oscar's decision) or a learned detector.

## Pod state, 09:20 UTC
4-GPU pod mcybpfvl2uovkh stopped on Oscar's instruction (disk kept: project copies, banks, runs, logs, tunnel binary).
API runs paused on Oscar's instruction until a change can plausibly score above 0.6.
