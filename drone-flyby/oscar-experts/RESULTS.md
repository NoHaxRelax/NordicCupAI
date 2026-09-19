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
