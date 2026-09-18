# Drone flyby: subsystem map (six readers and a cross-checker), 2026-09-18

## read:sprites-synthetic
Oscar's sprite bank holds 68 reviewed BGRA cutouts, but only about 18 independent appearances across 15 classes. Helicopter is missing entirely, and large_tower exists at L1 only.

**What the bank's zoom labels mean.** Every sprite is at native 4K pixel scale whatever its label. The grid dataset takes one native 256 px crop, shrinks it with `cv2.resize` INTER_AREA by 4 or 2, then upsamples back to 256 with INTER_LINEAR. L0 and L1 sprites are therefore blurred copies of the L2 pixels. Only the 17 L2 sprites carry real detail.

**What I did.** I read all four code files and measured `bank.json`, the synthetic-set manifests, the 25 Helsinki annotation files and the 249 validation pseudo-label files. I viewed the sprites on magenta, green and grey backgrounds, the synthetic contact sheet, six composites at 2.5x and six real Helsinki crops at native, /2 and /4. In total: 3 sprite/composite renders, 1 real-crop sheet, 5 annotation measurements and 1 edge-softness metric that failed. No training or scoring run was made.

**Composite quality.** The composites look plausible. The real frames are themselves sprites pasted on aerial imagery with no cast shadows, so copy-paste matches how the data is generated. Cutouts are clean at L2. Small, light objects (jet_plane, ta-ta, small_launcher) show pink or cyan fringe pixels on a grey background; this comes from the un-mix step dividing by a small alpha.

**The realism gap.** `local_evaluator.py` downsamples with `cv2.resize(..., INTER_AREA)` on exact integer factors, which is plain box averaging, then encodes PNG. A generator should compose in 4K native space and render through that same call. Pasting pre-blurred L0 sprites into upsampled tiles, as the composer does today, does not reproduce this.

**Generalisation risk.** Validation shows different skins and sizes for at least large_launcher, tank and medium_plane. The validation large_launcher sprite is 73x32 against 126x84 in Helsinki. Validation headings differ from Helsinki too. The current augmentation (quarter turns, flips, 0.9-1.1 scale) is too narrow for any of this.

**What cannot run here.** The `bank` and `prepare` commands need the private `data/drone/grid-comparison-20260918-v1/256` tiles and manifest, which are not in this repo. The PNGs plus `bank.json` are enough for a new standalone composer.

### Answers
- **How many sprites per class and per zoom, and which classes or zooms are missing?** [measured]
  68 sprites in total: 16 at L0, 35 at L1, 17 at L2.

Per class as [L0, L1, L2]:
- condor [1,5,1] = 7
- hangar [1,2,1] = 4
- helicopter [0,0,0] = 0
- jammer [1,2,1] = 4
- jet_plane [1,1,1] = 3
- large_launcher [2,5,2] = 9
- large_tower [0,1,0] = 1
- medium_launcher [1,2,1] = 4
- medium_plane [1,3,2] = 6
- mine_roller [1,3,1] = 5
- small_launcher [1,2,1] = 4
- small_plane [1,2,1] = 4
- small_tower [1,1,1] = 3
- spacecraft [1,1,1] = 3
- ta-ta [1,2,1] = 4
- tank [2,3,2] = 7

Other breakdowns:
- Source: 57 from reference (Helsinki) frames, 11 from validation frames.
- Review status: 52 approved, 16 redrawn.
- Type: 53 instance sprites, 15 library sprites.

Missing:
- helicopter has no sprite; all 8 review items were rejected.
- large_tower has no L0 or L2 sprite; 7 items were rejected.
- The validation-appearance jet_plane was rejected (6 items), as was one medium_plane L0.
- Only large_launcher, tank, medium_plane and mine_roller have a validation-appearance sprite. The mine_roller validation library sprite comes from a track Oscar flagged as looking like a tank.
  Evidence: Counted from sprite-bank/bank.json and review/decisions.json (90 decisions: 52 approved, 22 rejected, 16 redrawn). Rejected by class: helicopter 8, large_tower 7, jet_plane 6, medium_plane 1. review/label_exclusions.json lists mine-roller-a-005-009 and jet-plane-d-040-082 frames 40 and 46.
- **What are the sprite pixel sizes, and what do L0/L1/L2 mean in the bank?** [measured]
  Sprite sizes are in native 4K pixels at every zoom label. The L0, L1 and L2 sprites of one object are the same size:
- condor: 114x104 at L0, L1 and L2.
- jammer: 27x35, 27x35 and 29x39.

Native sprite size, with the organiser box in brackets:
- condor 114x104 (173x170)
- hangar 178x104 (189x130)
- jammer 29x39 (32x43)
- jet_plane 42x43 (72x75)
- large_launcher, reference 112x72 to 127x85 (139x105 to 156x110)
- large_launcher, validation 73x32 (83x60)
- large_tower 55x59 (60x66)
- medium_launcher 21x22 (48x45)
- medium_plane 50x40 (57x49)
- mine_roller 49x55 (53x60)
- small_launcher 11x14 (20x30)
- small_plane 35x44 (43x50)
- small_tower 48x46 (60x58)
- spacecraft 39x42 (42x49)
- ta-ta 27x17 (32x18)
- tank, reference 42x41 (54x52)
- tank, validation 48x20 (56x27)

The organiser boxes are looser than the sprites. medium_launcher's box is about 2.2x the sprite width.

On the zoom labels: L0 and L1 tiles come from the same native 256 px crop. They are box-downsampled by 4 or 2 and then linearly upsampled back to 256. L0 and L1 sprites are therefore blurred copies, not independent material.
  Evidence: bank.json size and box_in_sprite fields. codex_drone-training-baseline/drone-flyby/solution/drone/grid_training/prepare.py contains: low=cv2.resize(native[y:v,x:u],(size//divisor,...),INTER_AREA); image=cv2.resize(low,(size,size),INTER_LINEAR). The docstring of sprite_library.transfer_mask says 'Object size is the same at L0, L1 and L2 in the grid inputs'. The Helsinki frame 0 jammer box is [1496,1247,1528,1290], which is 32x43 and matches the bank.
- **Q2 data side: what pixel size does each class have in the delivered image?** [measured]
  Median box size in the 25 Helsinki frames, for fully visible boxes, given as native 4K, then L1 (/2), then L0 (/4):
- condor 172x168, 86x84, 43x42
- hangar 188x129, 94x65, 47x32
- helicopter 116x94, 58x47, 29x24
- large_launcher 150x108, 75x54, 38x27
- jet_plane 77x82, 39x41, 19x21
- large_tower 62x64, 31x32, 16x16
- small_tower 57x59, 29x30, 14x15
- mine_roller 53x60, 27x30, 13x15
- medium_plane 56x49, 28x25, 14x12
- tank 50x47, 25x24, 13x12
- medium_launcher 47x45, 24x23, 12x11
- spacecraft 44x49, 22x25, 11x12
- small_plane 43x50, 22x25, 11x13
- jammer 32x44, 16x22, 8x11
- ta-ta 32x17, 16x9, 8x4
- small_launcher 22x30, 11x15, 6x8

The sprite itself is smaller than its box. The small_launcher sprite is 11x14 native, which is about 3x4 px at L0.

In the real crops at /4:
- ta-ta is still a distinct grey blob on an orange court.
- jammer is a recognisable green rectangle.
- small_launcher is a faint greenish smudge.
- The helicopter's rotor blades nearly vanish.

This is a by-eye reading of one frame. No recognition threshold in pixels was measured by any model.
  Evidence: Computed from drone_oscar-live-tracker/drone-flyby/src/helsinki/annotations/*.json: 259 boxes, 25 frames, edge-clipped boxes excluded. Real crops viewed in _work/real_crops.png.
- **What is the visual quality of the cutouts and of the composites?** [inferred]
  Cutouts:
- L2 sprites are crisp and well masked. condor, spacecraft, tank, mine_roller, large_launcher and small_plane look clean on both green and grey backgrounds.
- jet_plane, ta-ta, small_launcher and jammer show pink or cyan fringe pixels on a grey background. This is consistent with `cutout_rgba` dividing by a small alpha when un-mixing, F=(I-(1-a)B)/a. The jammer L2 sprite keeps a semi-transparent rim.
- L0 and L1 sprites are blurred versions of the same objects, by construction.

Composites (contact sheet plus six images viewed at 2.5x):
- Objects sit plausibly on forest, water, sand and road, with no visible halo. The real frames are also sprites on aerial imagery with no cast shadows, so the look matches.
- Blurred L0 sprites look slightly translucent, in the same way the blurred L0 background does.
- Objects land on water and dense canopy as often as on open ground. Whether real objects do that is unknown.

I tried to measure real edge softness, to check the sigma=1.0 alpha, and the metric failed. It returned 11-14 px because it measured object interior texture rather than the edge. The sigma=1.0 alpha therefore remains unvalidated.
  Evidence: Viewed _work/bank_sheet_magenta.png, _work/l2_sprites.png, synthetic-set-v1/contact-sheet-clean.jpg, _work/synth_examples.png and _work/real_crops.png. The failed edge metric was run once and not retuned.
- **Exactly what augmentation does the composer apply today, and what does it not do?** [read-from-code]
  The shipped set was built with `synthetic_sprites.prepare`, seed 1918, count 180. It applies:
- Zoom drawn with weights (.25, .5, .25). The result was 86 L1, 57 L0 and 37 L2 images.
- Background: one training-split tile verified empty at that zoom. 149 distinct tiles were used: 96 with complete_boxes supervision and 84 human_verified_empty; 96 reference, 69 validation and 15 batch2-validation.
- 1 to 3 objects per image. The result was 80, 60 and 40 images with 1, 2 and 3 objects. Class is drawn uniformly from the classes that have a sprite at that zoom, and the sprite must carry the same zoom label.
- Quarter turns with rng.randrange(4), a horizontal flip with p=.5, and scale uniform 0.9-1.1. Resize uses INTER_LINEAR above 1 and INTER_AREA otherwise.
- A no-overlap test on the union of the sprite and its organiser box, up to 40 placement tries, with the object fully inside the tile.
- Plain alpha compositing.
- The label is the organiser box moved with the sprite.
- band/ and alpha/ files are written for a seam-model pass that was dropped; finalize ran with --without-model.
- Totals: 320 objects, 15 classes, 66 distinct sprites.

`compose_tiles.py` is an older, more general composer working from the manifest pool. It offers:
- box, grabcut or sprite masks;
- --scale .85-1.2 by default;
- optional --rot90 and --noise sigma;
- --onto-positive, default .25;
- --distractors, which pastes unlabelled background patches as hard negatives for paste edges;
- `paste_sprite`, which swaps the old background out of edge pixels using an inpainted estimate.

Neither composer does any of the following:
- arbitrary-angle rotation. Only `transfer_mask` searches 10-degree steps, and that is for mask alignment;
- blur or feather jitter;
- brightness, contrast, colour or gamma jitter;
- native-space composition rendered through the evaluator's box average;
- partial occlusion or overlap;
- edge truncation. Objects are always fully inside the tile;
- negative images. Images with no object are skipped;
- lookalike hard negatives;
- a wide scale range;
- full 960x540 frames. Output is 256 px tiles only.
  Evidence: synthetic_sprites.py prepare() and transform(); compose_tiles.py compose(), distractor_patch() and paste(); synthetic-set-v1/prepared.json (zoom_weights [0.25,0.5,0.25], max_objects 3, scale [0.9,1.1], band_px 4); manifest.json field blend='none: local sprite paste with un-mixed edges'.
- **Which parts of the code depend on the private working-repo layout, and what would it take to run it from this repo?** [read-from-code]
  Hard dependencies on the private layout:
- `mask_editor.py` constants:
  - DATA=Path('data/drone/grid-comparison-20260918-v1/256')
  - LIBRARY=Path('data/drone/sprite-library-20260918-v4')
  - DRAW_KIT=Path('artifacts/drone-sprite-masks-to-draw-20260918')
  - REVIEW=Path('data/drone/sprite-mask-review-20260918')
- `synthetic_sprites.DATA = me.DATA`. `prepare()` reads DATA/'manifest.json' for the empty tiles and asserts that `bank.json`'s source_manifest_sha256 (bb9dfb50...) equals that manifest's sha. `finalize()` reads the same manifest for the class names.
- `compose_tiles.compose` makes the same sha assert against `library.json`.
- The commands are written as `python3 -m drone.grid_training.<module>`, but the files live in code/drone_sprites/.
- `sprite_library.py` tries `from .train_v2 import sha, write` and has a local fallback, so that import is not a blocker.

The drone.grid_training package (prepare.py, train.py, train_v2.py) does exist in the codex_drone-training-baseline export under drone-flyby/solution/drone/grid_training. Its prepare.py needs data/drone/grid384-20260918-v1/manifest-combined-approved.json, which is not here.

Not present in this repo:
- the 256 px tile images and their manifest.json (records with split, kind, zoom, annotations, sha256, supervision);
- the draw kit;
- the pilot-nano-banana-2-lite folder named in the synthetic-set README;
- the reconstructed validation frames. run_metadata.json points at /Users/bumblebee/Github_repos/Projects/nordic-ai-cup-2026/data/drone/reconstructed-validation.

What can run here today: nothing in `bank` or `prepare` as written.

What a rerun needs, either:
- the private data/ tree copied in, and the modules run as `python -m drone_sprites.synthetic_sprites` from code/ with cwd at the data root; or
- a new composer that reads only sprite-bank/*.png and bank.json (class_id, size, box_in_sprite) and takes the 25 Helsinki 4K frames as backgrounds. This is the smaller job.

OpenCV 5.0.0, numpy 2.4.2 and PIL 12.1.1 are installed on this machine. A torch import printed only an OpenMP duplicate-runtime warning, so torch was not confirmed usable.
  Evidence: mask_editor.py lines 32-36; synthetic_sprites.py lines 30, 97-99, 184; compose_tiles.py line 123; sprite_library.py lines 23-33; the find results over the exports; drone_oscar-live-tracker/drone-flyby/src/validation/run_metadata.json.
- **Q4: what is a concrete recipe for a 50k-image synthetic generator, which split should be held out, and what are the realism risks?** [inferred]
  This is an untested proposal. No generator was built and no model was trained.

Recipe:
1. Use only native-detail sprites: the 17 L2 PNGs plus the large_tower L1 sprite. Discard the L0 and L1 sprites; they are blurred copies.
2. Cut a helicopter and a native large_tower. Without a helicopter sprite one of the 16 macro-averaged classes scores 0, which costs 0.0625 of the mAP.
3. Build 4K backgrounds from the 25 Helsinki frames, with annotated boxes inpainted or kept as labelled objects. Add the reconstructed validation frames if they can be obtained.
4. For each image, paste 0-6 sprites in 4K space. Use:
   - arbitrary rotation 0-360 degrees. Validation headings differ from Helsinki: tank boxes range from 58x28 to 59x64.
   - scale of roughly 0.6-1.4. Within a track, box size drifts about 14% from the top to the bottom of the frame. Across instances the spread is much larger: large_launcher 60x44 to 179x103.
   - brightness, contrast and hue jitter; alpha-edge sigma jitter.
   - some overlap, and truncation at the frame edge.
5. Sample a legal camera view at L0, L1 or L2. Crop the source region and call `cv2.resize(view,(960,540),interpolation=cv2.INTER_AREA)` exactly as `local_evaluator.render_view` does. Block phase is then exact by construction. Save as PNG.
6. Emit boxes. They should be loose, to match the organiser's convention. The box-to-sprite ratio differs by class: medium_launcher 48x45 against 21x22, jet_plane 72x75 against 42x43. Store the ratio per class.
7. Keep 10-20% of images empty, and add paste-edge distractors as `compose_tiles.distractor_patch` does.

Hold-out:
- Validate on real frames only, never on synthetic.
- Oscar's existing contract holds out validation frames 100-180.
- The backgrounds he used from the validation video are all at or before frame 85.
- Hold out by track, not by frame. Consecutive frames of one track are near-duplicates.
- Keep sprites cut from validation out of whatever is scored on validation. The bank has 11 of them.

Realism risks:
- The downsampling kernel is only known from the organiser's local evaluator (INTER_AREA on integer factors, a plain box average), not from the live server.
- Transport is lossless PNG, so no JPEG augmentation is needed.
- There are no cast shadows in real frames, so none should be added.
- One Helsinki instance per class means one skin and one heading family per class. Validation already shows other skins and a smaller large_launcher, and evaluation may show more. A detector trained on sprites can memorise texture.
- The un-mix fringe pixels may become a paste cue.
- The L2 sprites carry the source frame's lighting.
  Evidence: local_evaluator.py lines 163-178, including the comment 'INTER_AREA is what the evaluator uses... plain box averaging'. README.md line 207 ('PNG is lossless') and line 465 ('Helsinki holds a single instance of each class'). Sprite README ('Real frames show no cast shadows'). The Helsinki size-versus-y fit gives a bottom-to-top ratio of 1.138. Validation pseudo-label track sizes. synthetic-set-v1/README.md for the hold-out statement.
- **Q3 from the data side: is there enough per-class material for per-class experts?** [inferred]
  No, on the counts available.

Sprite material: each class has one native-detail cutout, which is one object instance. large_launcher, tank and medium_plane have a second, validation skin. helicopter has none.

Real Helsinki boxes per class range from 2 (mine_roller) to 25. All of them are one instance per class on one flight line:
- condor 11
- hangar 6
- helicopter 19
- jammer 13
- jet_plane 22
- large_launcher 25
- large_tower 19
- medium_launcher 10
- medium_plane 5
- mine_roller 2
- small_launcher 25
- small_plane 9
- small_tower 20
- spacecraft 23
- ta-ta 25
- tank 25

Validation pseudo-labels cover only 11 classes. Counting only tracks longer than 2 frames, the instances per class are: medium_launcher 1; hangar, medium_plane, mine_roller and small_tower 2 each; helicopter, jet_plane, large_tower and small_launcher 3 each; large_launcher and tank 5 each. condor, jammer, small_plane, spacecraft and ta-ta have no validation labels at all.

Oscar flagged 42 of the 57 mine_roller training examples as one mislabelled tank-like track.

A per-class expert would see 1-5 distinct instances, so it can only learn that instance's texture. It would also need negatives from the other 15 classes, which a shared model gets for free. tank and mine_roller already confuse each other here.

For scale, Lucas's 79k-parameter CNN was binary object-versus-background with no class output. It was trained on 1314 positive and 2190 negative crops for 500 steps, on 8 of 16 identities, and used only to rerank anomaly proposals. At L2 with the top 25 boxes it reached AP50 0.437% on all objects and 0.261% on held-out identities. Recall was 23.1%, precision 1.1%, and L0 was 0.0%. Its ceiling was the proposal set, so it says little about what a classifier trained on synthetic data can do.

Nobody here has trained per-class against shared models, so the comparison itself is untested.
  Evidence: bank.json counts; Helsinki annotation counts (259 boxes); validation/run_metadata.json (37 tracks, 987 boxes, 11 classes) and the per-track tally from validation/annotations; Oscar's README under 'Label problems found'; anomaly/cnn-results.json and anomaly/README.md.
- **What did Lucas's tiny CNN and the anomaly detector achieve, as a reference?** [read-from-code]
  Hand-written anomaly score, top 25 proposals per view, IoU 0.50:
- L0: AP 0.000%, 0 TP out of 259.
- L1: AP 0.005%, 15 TP, 6460 FP.
- L2: AP 0.162%, 56 TP, 6419 FP, recall 17.9%.
- At 300 proposals, L2 recall is 51.3% with 0.21% precision.

The CNN reranker (`ObjectnessCNN`):
- Conv stack 16, 32, 64, 96; stride 2; 64x64 crops; 79k parameters.
- AdamW, lr 2e-3, 500 steps.
- Horizontal flip plus brightness x0.85-1.15 and offset +-10.

CNN results, top 25 per view:
- L0: 0.000%.
- L1: 0.201% (59 TP).
- L2: 0.437% (72 TP, recall 23.1%, precision 1.11%).
- L2, held-out identities: 0.261%.
- L2, held-out identities at top 100: recall 57.5%, precision 0.575%, AP 0.544%.

The L1 and L2 views were centred on the ground-truth target, so these numbers are an upper bound and not a deployable search. One run, one seed (180926), one alternating identity split.
  Evidence: anomaly/README.md, anomaly/results.json, anomaly/cnn-results.json and anomaly/cnn_experiment.py.

### Numbers
- Sprites in bank (total / L0 / L1 / L2): 68 / 16 / 35 / 17 (sprite-bank/bank.json)
- Sprites by source: reference 57, validation 11 (sprite-bank/bank.json)
- Review decisions: 90: approved 52, redrawn 16, rejected 22 (helicopter 8, large_tower 7, jet_plane 6, medium_plane 1) (review/decisions.json)
- Classes with no sprite: helicopter; large_tower exists at L1 only (bank.json, README.md)
- Library reference masks: 19 variants, all cut at L1 (sprite-library-v4/library.json)
- Synthetic set v1: 180 images of 256 px, 320 objects, 15 classes; L1 86, L0 57, L2 37; 1/2/3 objects in 80/60/40 images; 149 distinct backgrounds; 66 distinct sprites used (synthetic-set-v1/manifest.json, prepared.json)
- Composer settings actually used: seed 1918, zoom_weights [0.25,0.5,0.25], max_objects 3, scale [0.9,1.1], band_px 4, blend none (synthetic-set-v1/prepared.json, manifest.json)
- Helsinki annotations: 25 frames, 259 boxes, one instance per class; mine_roller 2 boxes, medium_plane 5, hangar 6 (helsinki/annotations/*.json, README.md line 465)
- Smallest classes at L0 (median box, px): small_launcher 5.5x7.5, ta-ta 8.0x4.2, jammer 8.1x10.9, small_plane 10.8x12.5, spacecraft 11.0x12.2 (measured from helsinki annotations divided by 4)
- Largest classes at L0 (median box, px): hangar 47x32, condor 43x42, large_launcher 38x27, helicopter 29x24 (measured from helsinki annotations divided by 4)
- Box size drift with vertical position: bottom-to-top size ratio 1.138 (pooled log-size slope 5.97e-05 per px) (fit over helsinki fully visible boxes)
- Validation pseudo-labels: 249 frames, 37 tracks in metadata (34 distinct track ids in the annotation files), 987 boxes, 11 of 16 classes; none for condor, jammer, small_plane, spacecraft, ta-ta (validation/run_metadata.json and the annotation tally)
- large_launcher size spread in validation (median box per track): 60x44 to 179x103 native px across 5 tracks; validation sprite 73x32 against reference 126x84 (validation annotations, bank.json)
- Evaluator downsampling: cv2.resize INTER_AREA on exact integer factors (box average), PNG compression level 3 (local_evaluator.py render_view, lines 163-178)
- Tiny CNN, L2 top-25 AP50 (all / held-out identities): 0.437% / 0.261%; recall 23.1%, precision 1.11%; L0 0.0% (anomaly/cnn-results.json)
- Tiny CNN training set: 1314 positive and 2190 negative 64x64 crops, 500 steps, 79k parameters, 8 train and 8 held-out identities, seed 180926 (anomaly/cnn-results.json, cnn_experiment.py, anomaly/README.md)
- Anomaly score alone, L2 top-25 AP50: 0.162% (56 TP, 6419 FP); at 300 proposals recall 51.3%, precision 0.21% (anomaly/README.md, results.json)

### Gaps
- No detector or classifier was trained on the synthetic set in anything I could read, so whether synthetic data helps the validation score is unknown.
- Real sprite edge softness is unmeasured. My one attempt, a 10-90% luminance rise on ta-ta and spacecraft, returned 11-14 px because it measured interior texture. I did not retune it. The sigma=1.0 alpha in `cutout_rgba` and `paste_sprite` is therefore unvalidated against real frames.
- The grid dataset (data/drone/grid-comparison-20260918-v1/256 tiles and manifest.json, sha bb9dfb50...) is not in this repo. The bank and prepare steps cannot be rerun, and I could not check how many training instances per class the pool holds beyond Oscar's statement of 57 mine_roller examples.
- Reconstructed validation frame images are not in any export; only pseudo-label JSONs are. The metadata points at /Users/bumblebee/Github_repos/Projects/nordic-ai-cup-2026/data/drone/reconstructed-validation. The recipe step that uses validation frames as backgrounds depends on getting them.
- The validation labels are participant pseudo-labels marked incomplete, not organiser truth. The instance counts, sizes and skin differences drawn from them carry that uncertainty.
- The INTER_AREA kernel is taken from the organiser's `local_evaluator.py` comment and code. It was not verified against the live server, and I made no network call.
- The pixel size at which an object is recognisable (Q2) was judged by eye on one frame's crops. No model-based threshold exists here.
- The pilot-nano-banana-2-lite folder and the draw kit (artifacts/drone-sprite-masks-to-draw-20260918) named in the READMEs are absent. The seam-blending result rests on Oscar's written account alone.
- A torch import printed an OpenMP duplicate-runtime warning and no version string, so torch usability on this machine is unconfirmed.
- Whether real objects avoid water or dense canopy is unknown. The composer places them uniformly.
- The appearances in the evaluation sequence are unknown. Validation already differs from Helsinki for large_launcher, tank and medium_plane, and the mine_roller validation track may be mislabelled.

## read:history-results
Reconstructed the drone-flyby history from 36 commits on four branches (Oscar = Badecar, 34 commits; Lucas, 2 commits) plus every results file in the exports. Main findings. (1) No file, commit or receipt in any of the four branches records 0.470 or any other portal score. The only portal-scoring pipeline in the repo is a frame-keyed replay of participant pseudo-labels (solution/drone/discovery/endpoint.py, run_score_shards_live.py, build_full_validation_plan.py, analyze_validation_score_shards.py); every live-detector number in the repo is a local proxy. The most plausible origin of 0.470 is therefore a fixed-plan replay of the score-anchored validation labels, which cannot transfer to the evaluation sequence. This is inferred, not proven; Oscar must confirm from artifacts/drone-api-tests. (2) The only controlled ablation in the history says the detector is the bottleneck: the same live pipeline scores 0.978 mAP@0.50 on Helsinki with the oracle detector and 0.378 with the scratch YOLO26m, so about 0.60 is lost in the detector and about 0.02 in camera + tracker + placement + transport. (3) The COCO-pretrained YOLO26x fine-tuned on the 25 Helsinki frames (one physical instance per class) reaches 93.8/93.8/97.4 % recall on its own training views at L0/L1/L2 but 0/71, 1/71, 0/71 on validation targets: severe overfitting. (4) The frozen fixed-asset bundle asset-precision-20260918-v3 (SIFT geometry + masked pixel correlation + tiny-shape rules + scratch CNN + crop verifier) is accurate at native resolution (26/26 reference, 16/19 development, 13/13 consumed, 7/7 reserved with 12 proposals versus 0/7 with 330 for the old hybrid) but costs a median 4.1 to 4.5 s per 960x540 view on the RTX 4080 against a 333 ms frame interval and a 3333 ms timeout, and at L0 its pose-pixel branch gave 692 false alarms and 0 hits. It is not deployable live as is. (5) The team's validation pseudo-labels (v8: 987 boxes, 249 frames, 34 tracks with boxes) cover only 11 of 16 classes; condor, jammer, small_plane, spacecraft and ta-ta have zero validation labels. (6) Good news for whoever picks this up: the three bundle checkpoints and all 249 reconstructed 4K validation frames are already present as Git LFS blobs in the local clone (.git/lfs/objects, 3.8 GB), and the 246-crop template bank is ordinary git content, so the bundle and the validation replay can be rebuilt without Oscar. Everything under artifacts/, data/drone/, docs/ of the private preparation project (path /Users/bumblebee/Github_repos/Projects/nordic-ai-cup-2026 on Oscar's Mac), the YOLO checkpoints on mypc (hostname BumblebeeV3, C:\Users\oscar\nordic-drone), the grid dataset and all RunPod /workspace volumes are absent. I made one read-only RunPod list-pods call (no pod touched) to confirm the oscar-claude-drone-live-eval pods; no competition endpoint was contacted and nothing was modified.

### Answers
- **Timeline: what was built when, by whom, and what did each step score?** [read-from-code]
  All times CEST. 2026-09-17 10:18 Oscar (Badecar) imports the starter kit (d7764e0). 17:24 to 18:28 Oscar, branch codex/drone-training-baseline: COCO-pretrained YOLO26x fine-tuned on 250 camera views rendered from the 25 Helsinki frames (zoom mix 70 % L1, 20 % L2, 10 % L0, 960 input, 50 epochs) plus an ImageNet ResNet50 crop classifier. HPC jobs 29428577 (classifier, gpua100) and 29428578 (detector, gpul40s) were submitted 17:26 but DTU SSH expired and their outcome is unverified. mypc run v1 failed after 30 s (logger AttributeError), v2 ran 17:44:50 to 17:58:21, last.pt SHA 14f34b0b..., inference p50 20.7 ms / p95 21.1 ms on RTX 4080. Evaluation (c4ffb35, a29927e): recall on its own training views 243/259, 563/600, 75/77 at L0/L1/L2; on 71 manually labelled validation appearances 0/71, 1/71, 0/71 correct (conf 0.25, IoU 0.5); verdict 'not ready for use'. Same day (code only visible in the 13:44 snapshot of 09-18): validation capture and 4K reconstruction (249 frames), portal score probes to mine and confirm labels (score-assessment-20260917, class shards a/b/c), manual review ledger 995 boxes on 230 frames. Night 17 to 18 Sept: RunPod 'overnight' queue (tile detectors yolo26m 80 ep and yolo26x 60 ep, foreground-only detectors, ResNet50/ConvNeXt-tiny proposal classifiers, detector and crop ensembles, a 4-way large_launcher/mine_roller expert) and grid_dataset/grid_training crop CNNs; no results of these are in the repo. 2026-09-18 13:12 (selection timestamp) Oscar freezes fixed-asset bundle asset-precision-20260918-v3; 13:44 commits the stable snapshot f2f765d (solution/drone source, release/, validation annotations). Bundle results: 26/26 reference with 26 proposals, 16/19 development with 20, 13/13 consumed with 22, 7/7 reserved frames (187, 201, 224, 247) with 12, baseline hybrid 0/7 with 330; 44 unit tests pass; bundle smoke on CPU 6.37 s for one tank crop. 13:49 and 14:06 Lucas, branch drone-flyby-anomoly-detection: class-agnostic anomaly proposals and a 79k-parameter binary CNN reranker (numbers under 'rejected'). 19:03 Oscar, branch drone/oscar-live-tracker (cfddb4c): live endpoint = pluggable detector + vendored perspective tracker + L1-left / L0 / L1-right / L0 upper-band sweep; oracle detector 0.978 mAP@0.50 on Helsinki through HTTP, 25/25 camera moves applied, median round trip 27 ms; placeholder scratch YOLO26m 0.378. 19:32 OpenCV thread fix: SIFT calibration 1 s to 250 ms, realtime replay skips no frames (max round trip 383 ms, calibration 277 ms). 20:09 fixed_assets detector adapter, local validation scene from score-anchored-validation-v8 (987 boxes, 249 frames), analyze_run_log.py. 20:30 family gating by zoom level: at L0 pose_pixels 692 false alarms / 0 hits, tiny shapes 8 / 0, geometry 69 hits / 3 false alarms; DRONE_EVAL_TIMEOUT_S added because the bundle exceeds the 3333 ms budget. 20:40 DRONE_CAMERA_MODE=l2_top (7 native crops along the top row, 480 px hops inside the 551 px L2 limit, each column about every 6 frames). 20:56 stale-track revisits (DRONE_REVISIT_EVERY, DRONE_REVISIT_MIN_AGE=6). 20:57 Oscar, branch drone/oscar-sprite-synthetic: 68 reviewed sprites (16 at L0, 35 at L1, 17 at L2), 180 synthetic 256 px images with 320 objects in 15 classes; no model trained on them. 21:09 to 21:49 runner flags, DRONE_CV_THREADS for concurrent replays, DRONE_CLASS_EXTENT (size prior helps large_tower, small_tower, small_launcher; hurts medium_launcher on the validation replay). RunPod: A100 pods oscar-claude-drone-live-eval (created 19:48, still RUNNING with GPU at 0 %) and -2 to -5 (created 20:52 to 21:06, all EXITED) line up with the 20:09 to 21:49 replay commits.
  Evidence: git log --format=fuller origin/<branch> -- drone-flyby for all four branches; training/REPORT-2026-09-17.md, training/EVALUATION-2026-09-17.md, training/receipts/*.json; release/asset-precision-20260918-v3/{RESULTS.md,README.md,comparison.json,bundle-smoke.json,final-tests.txt,selection via comparison.json selected_at=1789729956}; anomaly/README.md + results.json; README-live.md; src/validation/run_metadata.json; oscar-sprite-synthetic/README.md and sprite-bank/bank.json; RunPod list-pods (read-only).
- **Which configuration most plausibly produced the team's 0.470 validation score?** [inferred]
  Unknown from this repository: the string 0.470 (or any portal score) appears in no commit message, document, receipt or JSON on any of the four branches (git grep over all four). Evidence-based ranking of the candidates. Most plausible: a fixed-plan replay of participant pseudo-labels, not a detector. The only code in the repo that talks to the portal is solution/drone/portal.py plus run_probe.py, run_score_shards_live.py and discovery/endpoint.py; that endpoint answers each request with plan['predictions_by_frame'][frame] and requested_view None, i.e. it looks boxes up by frame number and never moves the camera. build_full_validation_plan.py merges three class shards into 'one complete validation plan' for frames 1 to 249, analyze_validation_score_shards.py reports 'summed_full_prediction_map50' and itself warns 'It is not evidence that the same labels or boxes transfer to evaluation'. Label sets score-anchored-validation-v2 to v8 were iteratively repaired from such live score anchors. Against a live-detector origin: the YOLO26x checkpoint finds 0 to 1 of 71 validation targets; the fixed-asset bundle needs a median 4.1 to 4.5 s per view (p95 6.6 s) so a live run would exceed the 3333 ms budget, which is exactly why ac1474e added DRONE_EVAL_TIMEOUT_S 'for offline accuracy studies'; commit 289b0e6 states the validation-scene mAP 'is not the official score'; the live-eval RunPod pods expose only 22/tcp, so they look like offline replay machines, not portal-facing servers. Arithmetic consistency check only: the v8 labels cover 11 of 16 classes, and 11/16 x 0.68 = 0.47. If this reading is right, 0.470 is not a property of any deployable configuration (detector, weights, camera mode and thresholds do not apply: no detector, no camera command, confidences fixed by provenance at 1.0 / 0.98 / 0.90 / 0.80 in build_validation_score_shards.py CONFIDENCE_BY_PROVENANCE), and the evaluation sequence would score far lower with it. Ask Oscar for artifacts/drone-api-tests/score-probes/*/result.json and the portal status JSON (status-before-live-run.json lists validations with scores) to settle it.
  Evidence: git grep -E '0\.470|0\.47[^0-9]' on all four branches returned nothing relevant; solution/drone/discovery/endpoint.py lines 38-46; solution/drone/run_score_shards_live.py; build_full_validation_plan.py; analyze_validation_score_shards.py lines 85-93; build_validation_score_shards.py lines 57-62; commit bodies 289b0e6 and ac1474e; comparison.json view_seconds; EVALUATION-2026-09-17.md; RunPod pod port lists.
- **Which external dependencies are NOT in this repository and must be requested from Oscar?** [measured]
  (a) The private preparation project itself: /Users/bumblebee/Github_repos/Projects/nordic-ai-cup-2026 on Oscar's Mac (key oscar@macbook). Only its drone/ source tree was copied to solution/drone; its docs/, artifacts/ and data/ are absent. (b) Docs: docs/drone-box-placement.md, docs/drone-fixed-asset-result.md, docs/drone-revisit-workflow.md. (c) Artifacts directories referenced by code and reports: artifacts/drone-api-tests (portal status, score-probes/*/result.json, score-assessment-20260917: the only place a real portal score can live), artifacts/drone-fixed-assets-20260918-v2 and -v3 (raw predictions, fresh-detection-gallery.jpg, development-threshold-diagnostic.json, precision-v2-stopped.tar.gz), artifacts/drone-overnight-runpod, artifacts/drone-grid-training-20260918-v1, artifacts/drone-detector-eval-20260917 (zoom-comparison.jpg, l1-full-views.jpg), artifacts/drone-validation-coverage (raw/ and overlay/ review sheets), artifacts/drone-scene-analysis, drone-runtime-tracker, drone-shared-motion, drone-conditioned-shape, drone-camera-pixels, drone-edge-profiles, drone-validation-landmarks, drone-perspective-tracking, drone-revisit-workflow, drone-transfer-review. (d) Data: data/drone/training/score-anchored-validation-v2..v8 (per-track JSON with review_status; only the flattened v8 per-frame export is in src/validation/annotations), data/drone/training/manual-validation, algorithmic-full-validation, data/drone/grid-comparison-20260918-v1 (the 'grid dataset', its /256 training split feeds the sprite bank and frames 100 to 180 are its development split), data/drone/grid384-20260918-v1, data/drone/overnight-runpod-20260917-v1 and -20260918-v2, data/drone/capture/*, data/drone/mined/*, data/drone/fixed-asset-fresh-test-v2 and -reviewed-v2, data/drone/mypc-runs, data/drone/sprite-library-20260918-v4 source, sprite-mask-review-20260918. (e) Weights: the YOLO26x last.pt (118,396,958 bytes, SHA256 14f34b0b97f2...7f5897) at C:\Users\oscar\nordic-drone\runs\mypc-yolo26x-20260917-v2\detector\weights\last.pt on mypc; the 'placeholder scratch YOLO26m' used for the 0.378 figure (most likely the bundle's weights.pt, 44,067,410 bytes, trained with scratch_objects/train.py --model yolo26m.yaml, but this identity is inferred); every overnight RunPod checkpoint (tile, foreground, proposal classifiers, expert); HPC checkpoints under hpc:~/nordic-drone/runs/; pretrained yolo26x.pt (SHA 9fdd44a3...) and resnet50-11ad3fa6.pth. DRONE_WEIGHTS therefore points at nothing in this repo (*.pt is gitignored on the live branch). (f) Machines: 'mypc' = Oscar's Windows PC, hostname BumblebeeV3, RTX 4080 16 GB, Python 3.11.11, torch 2.6.0+cu124, ultralytics 8.4.155; RunPod pods oscar-claude-drone-live-eval (id ypuawkayl3px8t, RUNNING, A100 80GB, GPU util 0 %, 1.59 USD/h) and -2 (6h0kppqrggsd9n), -3 (oxf32el1pf5zgy), -4 (h27ua3c8vea5dg), -5 (9kldtw19fup0y0), all EXITED, each with a 60 GB persistent /workspace that most likely holds the validation replay logs (logs/val-v7 and similar) that the 20:09 to 21:49 commits quote; elias@shopvoices is an authorized SSH key on all five. W&B project badecar-danmarks-tekniske-universitet-dtu/nordic-ai-cup-drone (no runs logged for the completed model). (g) NOT missing, contrary to the export: the three bundle checkpoints (weights.pt d5c85a84..., verifier.pt 95efb7d1..., heatmap.pt 3abf14ab...) and all 249 validation/reconstructed-validation/frame_*.png are present as LFS blobs under C:\Users\edlun\Desktop\lucky shots\NordicCupAI\.git\lfs\objects (checked by oid), and the 246-crop template bank with masks is plain git content, so DRONE_DETECTOR=fixed_assets with DRONE_PROJECT=<export>/solution and DRONE_BUNDLE=<release>/model/manifest.json, and build_validation_scene.py, can be reproduced locally (needs ultralytics, which is not installed here, and faster-coco-eval for local_evaluator.py scoring).
  Evidence: grep of docs/, artifacts/, data/drone/ references over all exports; src/validation/run_metadata.json and tracking/size-prior.json (absolute Mac paths); training/receipts/mypc-final-result.json, mypc-final-smoke.json (hostname BumblebeeV3), first-run-jobs.json; .gitignore on live branch (*.pt, src/validation/images/); LFS pointer files and oid presence check in .git/lfs/objects (3 models present, 249/249 frames present); RunPod list-pods output.
- **What has been tried and rejected, with its number, so nobody repeats it?** [read-from-code]
  1. COCO-pretrained YOLO26x fine-tuned only on the Helsinki reference scene (one physical instance per class, 250 views, 50 epochs): 93.8 / 93.8 / 97.4 % recall on its training views, 0/71, 1/71, 0/71 on validation at L0/L1/L2; lowering conf to 0.10 gives 1/71, 3/71, 1/71, raising to 0.50 gives 0; 5 L1 and 9 L2 jet planes were localized but called condor. Verdict in EVALUATION-2026-09-17.md: build a class-by-pixel-size balanced set before training longer or bigger. 2. Scratch YOLO26m on copy-paste data as the live detector: 0.378 on Helsinki versus 0.978 for the oracle through the identical pipeline. 3. Context CNN continuation (scratch_objects/run_context_mypc.py): inspected at epoch 24, 1/19 development targets, rejected. 4. Original deterministic features: 18/26 reference, 2/19 validation. Original hybrid: 26/26 and 5/19 with 39 proposals; 0/7 with 330 proposals on the reserved frames; 4/13 with 799 on the eight consumed frames. 5. First reserved test of the selected fixed-asset candidate (frames 190, 206, 230, 245): 0/6 versus 1/6 for the hybrid, kept as a historical failure. 6. Permissive compact-CNN combination (dense CNN v3 + crop verifier v4): 17/19 but 739 proposals (392 on reference). Frozen v7: 12/13 but 233 proposals, mostly orange roofs and vegetation. 7. asset-precision-20260918-v2 (both broad CNN proposal branches disabled): 23/26 and 14/19, stopped. 8. Dense CNN v2 (3000 steps): no useful combined improvement. Crop verifier v3: synthetic object size mismatched the detector's proposal crops; v4: better recall but excessive proposals. 9. Strict GPU pixel matcher alone at 0.70: 5/13. 10. Lucas, hand-written anomaly score (rare Lab colour + local contrast + edges), top 25 per view: AP50 0.000 % at L0 (0 TP, 625 FP), 0.005 % at L1 (15 TP, 6460 FP), 0.162 % at L2 (56 TP, 6419 FP); at 300 proposals per view L2 reaches 51.3 % recall with 77,540 FP (0.21 % precision). 11. Lucas, 79k-parameter binary objectness CNN reranking the same boxes: AP50 0.201 % L1, 0.437 % L2, 0.261 % on held-out identities; precision stays near 1 %. 12. Fixed-asset bundle at L0/L1 in the live loop: pose_pixels 692 false alarms / 0 hits and tiny shapes 8 / 0 at L0; only the geometry branch is usable there (69 hits / 3 false alarms) and only for large distinctive objects. 13. Higgsfield seam blending for synthetic sprites: dropped, Nano Banana 2 Lite re-textured and re-coloured the surroundings and the MCP coerces the inpainting mask to a plain reference image. 14. Sprite gaps: all helicopter sprites rejected, validation-appearance jet plane rejected, large tower only at L1. 15. Size prior for medium_launcher hurts on the validation replay (helps the two towers and small_launcher), hence DRONE_CLASS_EXTENT. 16. Tracker options left off after study: class-conditioned shape model and adapt_edges (RevisitConfig.adapt_edges=False; workflow docstring 'No class-conditioned shape model is enabled'). Per Elias's rule that results are size-specific: every number above holds only for the model, data and resolution it was measured on.
  Evidence: training/EVALUATION-2026-09-17.md; commit cfddb4c body; release/asset-precision-20260918-v3/RESULTS.md (tables and 'CNN experiments'), README.md, comparison.json; anomaly/README.md and anomaly/results.json; commit bodies ac1474e, a243ed6, ad8c8f2; oscar-sprite-synthetic/README.md; tracking/revisit.py RevisitConfig, tracking/workflow.py docstring.
- **From the evidence only, where does the gap from 0.470 to 0.908 most plausibly lie?** [inferred]
  Ranked by strength of evidence. 1. Detector recall on an unseen scene inside the latency budget (dominant). The one controlled ablation gives 0.978 with oracle detections and 0.378 with the real scratch detector through the same camera, tracker and placement code, so roughly 0.60 mAP sits in the detector and 0.02 in everything else. The fine-tuned YOLO26x scores 0 to 1 of 71 on validation. The only recognizer that does generalize (the fixed-asset bundle, 7/7 and 13/13 at native views) costs a median 4.1 to 4.5 s per view against 333 ms per frame and a 3333 ms hard timeout, and works only near native resolution, so it cannot currently be served live. 2. Missing classes. The validation pseudo-labels contain 11 of 16 classes (no condor, jammer, small_plane, spacecraft, ta-ta; measured). Since AP is macro-averaged over the classes represented in the ground truth, every represented class the system never emits costs 1/K of the score; with a label-replay score this alone caps the result at 11/K. 3. Box placement below IoU 0.50, especially for small classes. RESULTS.md records 7 of 13 original versus reviewed participant boxes with IoU below 0.50, and the same frozen detector scores 4/7 against original boxes versus 7/7 against reviewed ones. Matched small_launcher IoUs in comparison.json are 0.587, 0.589, 0.592, 0.609, 0.633, 0.636, 0.646: every one is borderline. Measured median sizes on the validation labels: small_launcher 23 x 30 px, medium_launcher 30 x 45, large_tower 49 x 69 at native scale, so a 5 px offset already threatens IoU 0.5. 165 of the 987 v8 boxes are algorithmic bottom completions and 195 are projections from a single score anchor. 4. Classification confusion (documented, size unknown): jet_plane called condor by YOLO26x; the validation track mine-roller-a-005-009 looks like a tank yet supplies 42 of 57 mine-roller training examples and three of the bundle's validation mine_roller templates (frames 5, 7, 9); a dedicated large_launcher versus mine_roller expert had to be built. 5. Camera coverage (secondary, interacts with 1). Measured on the v8 labels, objects move 56 px per frame near the top and 80 near the bottom, live a median 33 frames, and spend about 9 frames in the top 540 rows and about 17 in the top 1080. The l1 cycle therefore gives about 4 L1 looks per side and an L0 look every second frame, which is enough for a detector that works at L1 (short side 11 to 30 px for the small classes at L1, 5 to 15 px at L0). l2_top gives native pixels but only 1 to 2 looks per object, often while it is still entering. 6. Dropped frames from latency: not a factor with a YOLO-class detector (21 ms inference, tracker about 10 ms, one-off calibration 250 to 277 ms, 'skips no frames', max round trip 383 ms); fatal with the bundle. 7. Duplicates: no evidence that they matter. The tracker suppresses at duplicate_iou 0.7, the bundle at nms_iou 0.35, and no log in the repo attributes loss to duplicates; unknown on the portal. Caveat: if 0.470 really came from a pseudo-label replay, then items 2 and 3 (label coverage and box convention) explain 0.470 itself, while item 1 is what stands between the team and any score at all on the evaluation sequence.
  Evidence: commit cfddb4c body (0.978 vs 0.378); EVALUATION-2026-09-17.md; comparison.json (view_seconds, match IoUs); RESULTS.md 'Data use and annotation review'; drone-flyby/README.md lines 379-416 (333 ms, 3333 ms, macro over represented classes, duplicates are FPs); measured with _work/hist_measure.py on src/validation/annotations and src/helsinki/annotations; oscar-sprite-synthetic/review/label_exclusions.json; model/bank file list (validation-mine_roller-000005/7/9); tracking/revisit.py and selection settings.
- **What does the recorded history already say about the six open questions (Q1 to Q6)?** [read-from-code]
  Q1 verifier: already built once, as a crop CNN (verifier.pt, 1.84 MB, crop verifier v5, 800 steps, 23 s of training on mypc). Gates in selection-v3.json: CNN proposals need verifier 0.95, tiny shapes 0.80, geometric matches scoring 0.70 to 0.80 need aligned pixels >= 0.25 plus same-class CNN >= 0.30, heatmap towers 0.40. Effect on the eight consumed frames: 233 proposals to 22 while recall went 12/13 to 13/13; on reserved frames 12 proposals versus 330. Lucas's class-agnostic binary verifier only reached about 1 % precision, so a verifier works when the proposer is already class-specific and precise. Q2 pixel size: EVALUATION-2026-09-17.md already tabulates median L1 training versus validation box sizes per class (for example tank 25 x 24 px in training versus 55 x 32.5 px in validation, condor 85 x 83 versus 45 x 38.5) and concludes that coverage must be balanced over actual object pixel size, not zoom label. In-domain the YOLO reaches 93.8 % recall even at L0, so pixel size alone is not the limit for a trained CNN; for template methods it is (bundle works only near native scale; anomaly cues retain 0 recall at L0). Q3 experts versus shared model: the shared YOLO overfit to one instance per class; the bundle is in effect per-class machinery (246 templates, Lab colour rules only for small_launcher, dense heatmap only for small_tower, a partial-body mask for jets) and generalizes to new views of the same assets but not proven on new scenes; the only trained 'expert' is the 4-way mine_roller / large_launcher / other / background classifier, result not in repo. Q4 synthetic data: copy-paste training exists (scratch_objects/prepare.py; native_data.py was added 'to counter synthetic cutout boundaries'; verifier v3 failed on a synthetic-versus-proposal size mismatch); the sprite branch has only 180 images and no model was trained on it; generative seam blending was rejected. Q5 camera cycle: L1-left / L0 / L1-right / L0 on the upper band is the default (DRONE_CAMERA_MODE=l1, DRONE_VERTICAL_FRACTION=0, DRONE_OVERVIEW_BETWEEN_SIDES=1); l2_top and stale-track L2 revisits were added the same evening because the bundle needs native pixels; no comparative score of the modes is recorded in the repo. Q6 one frame or two: the tracker is by design a 'one-observation object tracker using a frozen shared camera-motion model' (tracking/tracker.py); the motion model is calibrated once from the two opening L0 overviews and a single complete detection then fixes the whole trajectory, with box size from the detection blended with a per-class size prior fitted against image row (size-prior.json). The 0.978 oracle result on Helsinki is the evidence that one observation suffices when the detection is correct.
  Evidence: release/.../comparison.json selection.settings, RESULTS.md, README.md; anomaly/README.md; training/EVALUATION-2026-09-17.md size table; solution/drone/overnight/evaluate_expert.py GROUPS; scratch_objects module docstrings; oscar-sprite-synthetic/README.md and synthetic-set-v1/manifest.json; example.py SETTINGS and CONFIG defaults; tracking/workflow.py LevelOneSweep; tracking/tracker.py docstring; tracking/placement.py SizePrior; commit cfddb4c.

### Numbers
- Team validation score / leader: 0.470 / 0.908 (given in the task; recorded nowhere in the four branches) (task text; git grep on all four branches found no occurrence)
- Live pipeline, oracle detector, Helsinki, via HTTP: mAP@0.50 0.978; 25/25 camera moves applied; median round trip 27 ms (commit cfddb4c body (drone/oscar-live-tracker))
- Live pipeline, placeholder scratch YOLO26m, Helsinki: mAP@0.50 0.378 (commit cfddb4c body)
- Realtime replay after OpenCV thread fix: 0 skipped frames; max round trip 383 ms; calibration 277 ms (was about 1 s, cost 2 skipped frames) (commit 2f6f515 body)
- YOLO26x fine-tune recall on its own training views L0/L1/L2: 243/259 = 93.8 %, 563/600 = 93.8 %, 75/77 = 97.4 % (training/EVALUATION-2026-09-17.md)
- YOLO26x correct on validation targets L0/L1/L2 (conf 0.25, IoU 0.5): 0/71, 1/71, 0/71 (localized any class: 0, 6, 9); at conf 0.10: 1, 3, 1; at 0.50: 0, 0, 0 (training/EVALUATION-2026-09-17.md)
- YOLO26x training run: mypc-yolo26x-20260917-v2, 50 epochs, 17:44:50 to 17:58:21, peak 6.94 GiB, last.pt 118,396,958 bytes, SHA256 14f34b0b97f228342f0b71c5e8fb2c603d0100abc725aa74b675458c3b7f5897 (training/REPORT-2026-09-17.md, receipts/mypc-final-result.json)
- YOLO26x inference latency, RTX 4080, batch 1, 960x540: p50 20.73 ms, p95 21.12 ms, p99 21.22 ms (excludes HTTP, camera, tracker); pretrained on shared A100: p50 30.9 ms (receipts/mypc-final-smoke.json, receipts/smoke-pretrained.json)
- Fixed-asset bundle v3 accuracy: reference 26/26 with 26 proposals; development 16/19 with 20; consumed 13/13 with 22; reserved 7/7 with 12 (4/7 against original boxes); baseline hybrid 0/7 with 330 (release/asset-precision-20260918-v3/comparison.json and RESULTS.md)
- Fixed-asset bundle time per 960x540 view (mypc GPU, loaded): median 4.09 to 4.49 s, p95 4.29 to 6.63 s, max 8.24 s; CPU smoke 6.37 s (comparison.json view_seconds; bundle-smoke.json)
- Time budget: frame every 333 ms; request timeout 3333 ms; L2 move limit 551 px (drone-flyby/README.md lines 149-178, 379-391)
- Bundle families at L0 on validation replay: pose_pixels 692 false alarms / 0 hits; tiny shapes 8 / 0; geometry 69 hits / 3 false alarms (commit ac1474e body)
- Bundle thresholds (selection-v3): cnn 0.60, cnn_verifier 0.95, tiny 0.65, tiny_verifier 0.80, tiny_min_contrast 0.80, feature 0.70, verify_features_below 0.80, weak feature verifier 0.30 and pixel 0.25, calibrated pixel 0.70, pixel 0.60, heatmap 0.30 (small_tower only), heatmap verifier 0.40, nms_iou 0.35 (release/.../comparison.json selection.settings)
- Bundle files: weights.pt 44,067,410 B (sha d5c85a84...), verifier.pt 1,840,190 B (95efb7d1...), heatmap.pt 6,469,458 B (3abf14ab...); bank 246 crops + 246 masks on disk (RESULTS.md says 241: 210 reference + 31 validation; 12 calibrated poses) (LFS pointer files; ls of model/bank; RESULTS.md)
- Validation frames used for bundle training or calibration: 41 (35 frames up to 89, plus 110, 150, 190, 206, 230, 245); development 100, 127, 140, 160; consumed tests 183, 198, 204, 212, 220, 228, 241, 249; reserved 187, 201, 224, 247 (comparison.json data_use)
- Anomaly proposals, top 25 per view, AP50: L0 0.000 % (0 TP / 625 FP), L1 0.005 % (15 / 6460), L2 0.162 % (56 / 6419); L2 at 300 per view: 160 TP / 77,540 FP, recall 51.3 % (anomaly/README.md, anomaly/results.json)
- Anomaly + binary CNN rerank, AP50: L1 0.201 %, L2 0.437 %, L2 held-out identities 0.261 % (27 TP / 2798 FP); top 100 held-out L2: recall 57.5 %, precision 0.575 % (anomaly/README.md)
- Validation pseudo-labels v8 (measured): 987 boxes, 249 frames, 34 tracks with boxes (37 track files), 11 of 16 classes; missing condor, jammer, small_plane, spacecraft, ta-ta; median 3 boxes per frame (max 9) versus Helsinki 10 per frame (max 12) (_work/hist_measure.py on drone_oscar-live-tracker/drone-flyby/src/validation/annotations; run_metadata.json)
- v8 label provenance: directly reviewed 606, score-anchored projection 195, algorithmic bottom completion 165, score-confirmed anchor 14, group projected 4, partial entry 3 (src/validation/run_metadata.json (re-counted))
- Object vertical speed on validation (measured): median 56.0 px/frame for centre y in 0-540, 64.0 in 540-1080, 72.0 in 1080-1620, 79.5 in 1620-2160; median track length 33 frames (max 45); 62 % of tracks start with top edge within 5 px of the frame top (_work/hist_measure.py)
- Median short side of validation boxes, native px (divide by 2 for L1, by 4 for L0): hangar 184, helicopter 102, large_launcher 66, jet_plane 59.5, small_tower 59, mine_roller 54, tank 49, large_tower 48, medium_plane 45, medium_launcher 30, small_launcher 23 (_work/hist_measure.py)
- Median short side of Helsinki boxes for the five classes absent from validation labels, native px: condor 166, spacecraft 44, small_plane 43, jammer 32, ta-ta 17 (_work/hist_measure.py on src/helsinki/annotations)
- Coverage ledger (2026-09-17 state): 1009 candidates = 995 manual boxes on 230 frames + 14 score-confirmed seeds; 69 of 69 review sheets done; later frames reviewed only in the top 540 px at stride 4; max measured displacement between samples 285 px (gate 460 px) (validation/annotations/README.md, coverage-ledger.json)
- Label-quality findings: 7 of 13 original versus reviewed participant boxes have IoU < 0.50; track mine-roller-a-005-009 (looks like a tank) supplies 42 of 57 mine-roller training examples; jet-plane-d-040-082 frames 40 and 46 lie on empty forest (RESULTS.md; oscar-sprite-synthetic/README.md and review/label_exclusions.json)
- Sprite bank and synthetic set: 68 sprites (16 at L0, 35 at L1, 17 at L2), 19 reference masks, 66-item shortlist; 180 images at 256 px, 320 objects, 15 classes (no helicopter), seed 1918; no model trained (oscar-sprite-synthetic/README.md, sprite-bank/bank.json, synthetic-set-v1/manifest.json)
- Live endpoint defaults: DRONE_CONF 0.25, DRONE_IMGSZ 960, birth 0.6, update 0.4, extent blend, camera mode l1, vertical fraction 0, overview between sides 1, revisit every 0 (min age 6), duplicate_iou 0.7, association_iou 0.1, 3 visible misses before retirement; tracker about 10 ms per frame after about 200 ms calibration (example.py SETTINGS/CONFIG, tracking/revisit.py RevisitConfig, README-live.md)
- l2_top sweep geometry: 7 L2 crops along the top row, 480 px hops inside the 551 px L2 limit, each column about every 6 frames (commit a243ed6 body; tracking/workflow.py LevelOneSweep.l2_waypoints (limit 550, low 480))
- RunPod live-eval pods: 5 A100-SXM4-80GB pods at 1.59 USD/h with 60 GB /workspace each; oscar-claude-drone-live-eval RUNNING since 2026-09-18 17:48Z with GPU util 0 %; -2 to -5 EXITED (created 18:52Z to 19:06Z); only 22/tcp exposed (RunPod list-pods, read-only call at about 2026-09-18 20:53Z)
- Local LFS content: 3.8 GB, 250 objects: 249 of 249 reconstructed validation PNGs (about 14.8 MB each) and the 3 bundle checkpoints are present (oid presence check in C:\Users\edlun\Desktop\lucky shots\NordicCupAI\.git\lfs\objects)

### Gaps
- No portal validation score of any kind is committed. The config behind 0.470 cannot be established from the repo; Oscar must supply artifacts/drone-api-tests (portal status JSON with the validations list, score-probes/*/result.json, score-assessment-20260917 summary) and say whether 0.470 was a fixed-plan label replay or a live detector run.
- No diagnostics logs from the live endpoint are in the repo (logs/ is gitignored). The family/zoom numbers in commits ac1474e, a243ed6 and ad8c8f2 come from logs such as logs/val-v7 that presumably sit on the oscar-claude-drone-live-eval pods' /workspace volumes or on Oscar's Mac. No recorded proxy mAP exists for l1 versus l2_top versus revisits, nor for any birth/update threshold sweep.
- The identity of the 'placeholder scratch YOLO26m' behind the 0.378 Helsinki score is inferred (most likely the bundle's weights.pt); its path and training data version are not stated.
- Results of the overnight RunPod queue (tile detectors, foreground detectors, proposal classifiers, ensembles, the mine_roller expert), of grid_training, and of perspective_tracking/camera_policy_benchmark.py, resolution_study.py and the tracking benchmarks are absent: only the code is in solution/drone. The docs that summarize them (docs/drone-box-placement.md, docs/drone-fixed-asset-result.md, docs/drone-revisit-workflow.md) are absent.
- HPC jobs 29428577 and 29428578 were never verified after DTU SSH expired; the ResNet50 crop classifier has no recorded result.
- Bundle speed on an A100 is unknown; only RTX 4080 (loaded, 4.1 to 4.5 s per view) and laptop CPU (6.4 s) timings exist, so whether any branch subset fits in 333 ms is unmeasured.
- How many classes the real validation ground truth contains is unknown. The team's labels have 11; search_missing_classes.py and the missing-class YOLO scan looked for the rest, results absent. The macro-average denominator K is therefore unknown.
- faster-coco-eval and ultralytics are not installed in this Python 3.13 environment, so I did not rerun local_evaluator.py or the bundle; the 0.978 and 0.378 figures are read from the commit, not re-measured.
- RESULTS.md says the bank has 241 crops while the committed bank directory holds 246 crop/mask pairs; the 5-crop difference is unexplained.
- One read-only RunPod list-pods call was made to confirm pod names; no pod was started, stopped or entered. The pod oscar-claude-drone-live-eval is running idle at 1.59 USD/h, which Oscar or Elias may want to stop; I took no action.

## read:data-inventory
I measured everything below with Python from the annotation JSONs, the sprite bank and the Helsinki 4K frames. The validation images are not in any export, so validation appearance could not be checked by eye. Scripts and sheets are in the scratchpad folder `drone\_work`.

**Datasets**
- **Helsinki:** organiser ground truth, 25 frames, 259 boxes, 16 classes with exactly one instance and one heading each, 8 to 12 objects per frame.
- **Oscar's validation v8:** participant pseudo-labels, 249 frames, 987 boxes, 34 track ids that are 31 physical objects. Only 11 classes appear, with 1 to 5 instances per class at different headings and scales, and 1 to 9 objects per frame (median 3).
- **Codex validation:** 995 manual boxes plus 14 score-confirmed seeds, 11 classes.
- **Classes missing from validation labels:** condor, jammer, small_plane, spacecraft and ta-ta have no box in v8. They are the small classes, and nothing in the repo says whether they are truly absent or just unlabelled.

**Geometry**
- Both flights show the same perspective motion field, not a single vector.
  - Helsinki: dy = 51.85 + 0.01333*cy and dx = 0.00708*(cx-1914) px per frame.
  - Validation: dy = 52.38 + 0.01467*cy and dx = 0.00739*(cx-1928).
- This fits a pinhole camera tilted about 19 degrees forward of nadir with a focal length near 3200 px.
- A constant 3x3 homography per frame predicts Helsinki box centres to 0.64 px mean error. The single mean vector (1.0, 65.5) misses by 9.2 px.
- Objects enter at the top edge, cross in about 34 frames and leave at the bottom.
- Width grows about 0.8% per frame. Height grows for flat objects and shrinks for tall ones.

**Q6 (one frame or two)**
- One observation plus the flight's own homography keeps mean IoU at 0.96 / 0.92 / 0.89 / 0.81 / 0.68 after 1 / 3 / 5 / 10 / 20 frames. 96 to 100% of boxes stay at IoU 0.5 or better.
- The single scene-wide vector gives 0.65 / 0.33 / 0.19 / 0.06 / 0.
- Two-frame constant velocity is worse than one frame plus the field: 0.94 / 0.76 / 0.53 / 0.11 / 0 without noise, and it collapses under noise.
- With 4 source px of edge noise (one L0 pixel), fusing a second observation through the field lifts IoU from 0.78 to 0.83 at k=1 and from 0.71 to 0.74 at k=10. The share at IoU 0.5 or better at k=20 rises from 64% to 82%.
- So one frame fixes the trajectory. A second frame is worth it only as noise averaging, mainly for small_launcher and ta-ta.
- The motion model must come from the same flight. The homography fitted on validation labels drifts 10 px after 10 frames and 30 px after 20 when applied to Helsinki.

**Q2 (pixel size)**
- Box sizes at each level are tabulated per class in the numbers below.
- Organiser boxes are loose. The visible object covers only 22% of the box area for medium_launcher, 23% for small_launcher, 29% for jet_plane and 41% for condor.
- A tight box around the pixels therefore cannot reach IoU 0.5 for those classes.

**Label problems**
- Both problems the sprite README mentions show up in the numbers.
- I also found stale duplicate frames 9 and 239, backward-moving boxes in helicopter-b-early, and 25% of boxes shared by the two validation label sets disagreeing at IoU below 0.5.

### Answers
- **Q2: what object pixel size is delivered at L0, L1 and L2 for each class, and which sizes are recognisable?** [measured]
  **Organiser box size (Helsinki median, unclipped, source px = L2 px; divide by 2 for L1, by 4 for L0)**

| class | box (src px) | box at L0 (px) |
|---|---|---|
| condor | 172x168 | 43x42 |
| hangar | 188x129 | 47x32 |
| helicopter | 116x94 | 29x24 |
| large_launcher | 149x108 | 37x27 |
| jet_plane | 77x82 | 19x21 |
| large_tower | 62x64 | 16x16 |
| small_tower | 57x59 | 14x15 |
| medium_plane | 56x49 | 14x12 |
| mine_roller | 53x60 | 13x15 |
| tank | 50x47 | 12.5x12 |
| medium_launcher | 47x45 | 12x11 |
| spacecraft | 44x49 | 11x12 |
| small_plane | 43x50 | 11x12.5 |
| jammer | 32x44 | 8x11 |
| ta-ta | 32x17 | 8x4 |
| small_launcher | 22x30 | 5.5x7.5 |

**The boxes are loose.** The tight extent of the visible object, taken from the reviewed sprite masks in `bank.json`, is much smaller for several classes.

| class | object (src px) | share of box area | longest side at L0 / L1 (px) |
|---|---|---|---|
| medium_launcher | 21x22 | 0.22 | 5.5 / 11 |
| small_launcher | 11x14 | 0.23 | 3.5 / 7 |
| jet_plane | 42x43 | 0.29 | 10.8 / 21 |
| condor | 114x104 | 0.41 | - |
| large_launcher | 112x72 | 0.50 | - |
| tank | 42x41 | 0.73 | 10.5 / 21 |
| ta-ta | 27x17 | 0.84 | 6.8 / 13.5 |

The remaining classes fall between 0.66 and 0.85.

**What I could see by eye on the Helsinki contact sheet** (`_work\helsinki_classes_L2_L1_L0.png`):
- At L0, small_launcher, medium_launcher, jammer, large_tower and ta-ta are not separable from terrain.
- Tank, spacecraft, small_plane and mine_roller are blobs of 10 to 14 px at L0.
- Condor, hangar, helicopter, large_launcher and jet_plane keep their shape at L0.
- At L1, everything except small_launcher (7 px object) and medium_launcher (11 px object) has visible structure.
- At L2, all classes are clear.

**Localisation tolerance.** A pure shift keeps IoU at 0.5 or better only while the shift is under a third of the box side.
- ta-ta: 5.7 source px, which is 1.4 L0 px.
- small_launcher: 7.3 source px, 1.8 L0 px.
- jammer: 10.8 source px, 2.7 L0 px.
- tank: 15.7 source px, 3.9 L0 px.
- hangar: 43 source px.

No detector-recall-versus-pixel-size measurement exists in my data sources, so the recognisability threshold itself is my visual reading, not a measured curve.
  Evidence: - Box sizes: `_work` scripts over `drone_oscar-live-tracker\drone-flyby\src\helsinki\annotations\*.json`.
- Object extents: `drone_oscar-sprite-synthetic\drone-flyby\oscar-sprite-synthetic\sprite-bank\bank.json`, fields `size` and `box_in_sprite`.
- Contact sheet rendered from `helsinki\images` at 1x, 1/2 and 1/4 scale.
- **Q6: can trajectory and box size be fixed from ONE recognising frame, or are TWO needed? IoU after 1, 3, 5, 10, 20 frames.** [measured]
  One frame is enough if position is propagated with a flight-specific perspective motion field. A second frame helps only as noise averaging.

**No observation noise (Helsinki ground truth, pooled over classes). Mean IoU, with share at IoU 0.5 or better in brackets.**

| model | k=1 | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| A. static box | 0.065 | 0 | 0 | 0 | 0 |
| B. one obs + single scene vector (1.0, 65.5), fixed size | 0.650 (79%) | 0.331 (26%) | 0.189 (7%) | 0.056 (1%) | 0.000 |
| C. one obs + own-flight homography, fixed size | 0.958 | 0.912 | 0.868 | 0.773 (98%) | 0.627 (87%) |
| D. as C plus size growth (+0.8%/f width, +0.3%/f height) | 0.960 (100%) | 0.922 (100%) | 0.886 (100%) | 0.810 (99%) | 0.684 (96%) |
| E. two obs, constant velocity, fixed size | 0.941 | 0.755 (94%) | 0.527 (64%) | 0.111 (0%) | 0.000 |

E fails at longer horizons because the apparent speed grows from 52 to 80 px per frame on the way down, which constant velocity cannot follow.

**With 4 source px Gaussian noise per box edge (one L0 pixel)**

| model | k=1 | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| D. one obs + own homography + growth | 0.775 (95%) | 0.761 (94%) | 0.744 (93%) | 0.705 (88%) | 0.588 (64%) |
| F. two obs fused through the homography | 0.825 (99%) | 0.806 (98%) | 0.786 (98%) | 0.741 (93%) | 0.623 (82%) |
| E. two obs, constant velocity | 0.658 | 0.447 | 0.299 | 0.107 | 0.005 |

The second observation is therefore worth about +0.03 to +0.05 IoU. At k=20 the share at IoU 0.5 or better goes from 64% to 82%.

**Per class, model D**

| class | k=1, no noise | k=10, no noise | k=10, 4 px noise (share at IoU 0.5+) |
|---|---|---|---|
| condor | 0.99 | 0.80 | - |
| helicopter | 0.98 | 0.88 | - |
| large_launcher | 0.98 | 0.91 | 0.88 (100%) |
| jet_plane | 0.98 | 0.91 | - |
| small_tower | 0.97 | 0.85 | - |
| tank | 0.96 | 0.80 | 0.70 (97%) |
| jammer | 0.96 | 0.85 | - |
| spacecraft | 0.96 | 0.86 | - |
| large_tower | 0.94 | 0.57 (88%) | 0.57 (75%) |
| small_launcher | 0.94 | 0.77 | 0.55 (65%) |
| ta-ta | 0.91 | 0.69 | 0.53 (60%) |

Hangar, medium_launcher, medium_plane and mine_roller have too few frames for k=10. Large_tower is the weakest because its box height shrinks along the track.

Fusing two frames (F) raises small_launcher at k=10 to 0.62 (89%) and ta-ta to 0.60 (77%).

**Caveat: the motion model must be calibrated on the flight itself.** Using the homography fitted on validation pseudo-labels to propagate Helsinki boxes gives 0.901 / 0.747 / 0.622 / 0.388 / 0.146 (share 100 / 88 / 72 / 46 / 4%).
- The two fitted models differ by 0.6 / 2.1 / 3.8 / 9.9 / 29.8 px after 1 / 3 / 5 / 10 / 20 frames.
- Small classes are lost first under the borrowed model. ta-ta is 0% at IoU 0.5 by k=3, and small_launcher is 10% by k=5.
- Codex's image-registration figure (228.0 px per 4 frames at the top centre, sd 0.63 px over the whole flight) shows the field can be measured from the background online at about 0.16 px per frame precision.

**Size from one frame.** Holding the entry-frame size costs about 0.06 IoU at k=20 (model C against D). Width grows x1.13 to x1.22 over a crossing.
  Evidence: - `_work\q6.py` and `_work\motion.py`, simulated over every unclipped Helsinki ground-truth box pair.
- Homographies saved as `_work\H_hel.npy` and `_work\H_val.npy`.
- Validation pseudo-labels give the same ordering of models (D with the Helsinki-fitted homography: 0.891 / 0.782 / 0.692 / 0.501 / 0.253).
- Registration figures come from `coverage-ledger.json`, field `sampling.motion_measurements`.
- **Is apparent motion the same everywhere, or perspective dependent? Direction, magnitude, variance across objects and across y.** [measured]
  It is perspective dependent and very regular.

**Helsinki, 219 consecutive unclipped pairs**
- Mean displacement is (1.03, 65.52) px per frame, with sd (6.60, 7.41), min (-11.5, 52.5) and max (13.5, 80).
- Vertical: dy = 51.85 + 0.01333*cy, r = 0.994.

| y band | 0-540 | 540-1080 | 1080-1620 | 1620-2160 |
|---|---|---|---|---|
| dy (px/frame) | 55.9 | 62.7 | 69.6 | 76.5 |

  Within each band the sd is about 2.
- Horizontal: dx = 0.00708*(cx - 1914), r = 0.998.

| x band | 0-960 | 960-1920 | 1920-2880 | 2880-3840 |
|---|---|---|---|---|
| dx (px/frame) | -8.0 | -3.4 | +3.6 | +11.8 |

- dx does not depend on y (r = -0.016) and dy does not depend on x (r = 0.014).
- Objects fan out from the vertical centre line and accelerate downward.

**Validation v8, 886 pairs**
- dy = 52.38 + 0.01467*cy (r = 0.988, residual sd 1.31).
- dx = 0.00739*(cx - 1928) (r = 0.982).
- Using directly reviewed boxes only: dy = 52.52 + 0.01449*cy.

**Model error for one step on Helsinki (mean centre error)**

| model | mean error (px) |
|---|---|
| constant vector | 9.18 |
| affine field | 0.77 |
| homography | 0.64 |

The Helsinki homography applied to validation pairs gives 2.12 px.

**Pinhole reading.** Using the Helsinki run metadata (altitude 600 m, step 13.889 m), the fit implies a tilt of 18.9 degrees forward of nadir, focal length about 3196 px, horizontal field of view 62 degrees and vertical 37 degrees.

**Crossing time.** A ground point takes 10 frames from y=0 to y=540, 19 frames to y=1080 and 34 frames to y=2160. Validation tracks that span top to bottom last 32 to 39 frames, median 34.
  Evidence: - Fits in `_work\motion.py` (`fit_affine`, `fit_H`) over the Helsinki annotations and `src\validation\annotations`.
- Poses in the Helsinki annotation JSONs step by exactly (2.4118, 13.6779, 0) m per frame, norm 13.8889.
- **Do objects enter at the top? What is the distribution of x and of entry positions?** [measured]
  Yes, objects enter at the top.

**Validation v8**
- 32 tracks start after frame 10. For 31 of them, the first box has its top edge at y1 of 54 or less, and 19 of those have y1 = 0.
- The only exception is jet-plane-d-040-082 (y1 = 208), which is the flagged bad label.
- Two tracks are already in the frame at sequence start: the large_launcher at y=938 and mine-roller-a at y=810.
- All complete tracks exit through the bottom edge.
- Entry x ranges from 618 to 3211. 41% of entries are left of x=1920, and 78% lie inside the central band 960 to 2880.
- Entry frames: 15, 29, 37, 38, 40, 43, 47, 51, 53, 57, 58, 66, 77, 80, 92, 92, 105, 109, 110, 118, 119, 119, 119, 128, 139, 155, 182, 194, 197, 217, 238, 240. One new object arrives every 8.2 frames on average, with gaps from 0 to 27 frames. Three small_launchers enter together at frame 119.

**Helsinki**
- 11 classes are already present at frame 0.
- The other 5 all enter at y1 = 0: jet_plane at frame 3, large_tower at 6, medium_launcher at 14, hangar at 19 and medium_plane at 20.

**Box-centre x distribution**

| | 5th | 25th | 50th | 75th | 95th percentile |
|---|---|---|---|---|---|
| Validation | 539 | 1256 | 2073 | 2668 | 3237 |
| Helsinki | 746 | 1482 | 2054 | 2746 | 3757 |

In both sets 47% of boxes are in the left half.

**y distribution.** It is roughly uniform, slightly heavier at the top. Validation boxes per 270 px band from top to bottom: 171, 135, 124, 122, 117, 110, 104, 104. This happens because objects move slower near the top.
  Evidence: `_work` scripts over `src\validation\annotations\frame_*.json` (using the `provenance.track_id` field) and the Helsinki annotations.
- **Does the box grow or shrink as an object crosses the frame?** [measured]
  Width grows steadily. Height depends on how tall the object is.

**Helsinki ground truth, unclipped boxes, log-linear fit per frame**
- Width grows +0.64 to +1.02% per frame for every class, median +0.83%. That is x1.13 to x1.22 over a crossing. Examples: large_launcher 140 to 160, tank 46.7 to 54.0, ta-ta 29.3 to 35.7.
- Height grows for flat objects:

| class | height growth (% per frame) |
|---|---|
| tank | +0.80 |
| jet_plane | +0.83 |
| helicopter | +0.95 |
| condor | +0.95 |
| small_plane | +0.96 |

- Height is flat or shrinking for tall objects, as the tilted camera sees less of their side when they come closer:

| class | height change (% per frame) | measured range (px) |
|---|---|---|
| large_tower | -0.92 | 68 to 59 |
| ta-ta | -1.32 | 18.7 to 14.0 |
| small_tower | -0.32 | - |
| small_launcher | -0.13 | - |
| spacecraft | +0.11 | - |
| jammer | +0.19 | - |

**Validation v8 pseudo-labels**
- Median slopes are +0.66% per frame for width and +0.72% for height.
- Individual tracks are inconsistent, from -1.47 to +1.90% per frame.
- tank-b-128-160 even shrinks (width from 25 to 62), which is a label artefact.

The tracker's existing `tracking\size-prior.json` already encodes a per-class linear slope against the centre row, fitted from the same Helsinki boxes. For example large_launcher w = 131.19 + 0.0194*cy.
  Evidence: - `_work` size-change script.
- `drone_oscar-live-tracker\drone-flyby\tracking\size-prior.json`.
- **Do Helsinki and validation use the same 16 assets with the same sizes and headings? One appearance per class or several? Which classes are rare or absent?** [measured]
  **Helsinki**
- Exactly one instance per class at one heading. `run_metadata.json` gives `total_objects` 16, one per class.

| class | boxes |
|---|---|
| large_launcher | 25 |
| small_launcher | 25 |
| tank | 25 |
| ta-ta | 25 |
| spacecraft | 23 |
| jet_plane | 22 |
| small_tower | 20 |
| helicopter | 19 |
| large_tower | 19 |
| jammer | 13 |
| condor | 11 |
| medium_launcher | 10 |
| small_plane | 9 |
| hangar | 6 |
| medium_plane | 5 |
| mine_roller | 2 |

- The rare classes in Helsinki are mine_roller (2 boxes), medium_plane (5), hangar (6), small_plane (9) and medium_launcher (10).

**Validation v8**
- Only 11 classes appear.
- condor, jammer, small_plane, spacecraft and ta-ta have zero boxes. Whether they are really absent from the scene is unknown, because the labels are participant-made and stated to be incomplete.
- Boxes and tracks per class:

| class | boxes | tracks |
|---|---|---|
| tank | 164 | 5 |
| large_launcher | 149 | 5 |
| jet_plane | 112 | 3 |
| helicopter | 105 | 3 |
| small_launcher | 97 | 3 objects |
| large_tower | 96 | 3 objects |
| hangar | 69 | 2 |
| medium_plane | 66 | 2 |
| mine_roller | 52 | 2 |
| small_tower | 45 | 2 objects |
| medium_launcher | 32 | 1 |

**Several instances per class, with different headings and scales.** Per-track median box at mid-frame rows, as width x height (aspect):
- tank: 59x29 (2.03), 67x56, 60x65, 52x40 and 70x49, against Helsinki 53x50.
- large_launcher: 83x60, 113x62, 61x45, 152x141 and 179x103, against Helsinki 152x110.
- large_tower: 51x100, 38x65 and 64x58, against Helsinki 66x60.
- hangar: 226x199 and 203x184, against Helsinki 188x129.
- jet_plane: 75x57, 60x86 and 89x76, against Helsinki 81x85.
- small_launcher is stable: 23x28, 22x30 and 23x30, against Helsinki 22x30.

**Ratio of validation median size to Helsinki median size**

| class | width ratio | height ratio |
|---|---|---|
| large_launcher | x0.79 | x0.62 |
| hangar | x1.11 | x1.44 |
| medium_plane | x0.80 | x1.31 |

So a Helsinki size prior does not transfer to a different heading.

**Appearance.** Sprites cut from validation differ from the reference.
- The validation medium_plane is a dark aircraft, while the reference one is pale.
- The validation large_launcher is axis-aligned at 74x33, while the reference is diagonal at 127x85.
- Validation crops are visibly blurrier.

**Camera and motion.** The geometry and motion field are the same in both sets.
  Evidence: - `_work` tables over both annotation sets.
- `_work\sprites_compare.png`, rendered from `sprite-bank` PNGs.
- `src\helsinki\run_metadata.json` and `src\validation\run_metadata.json`.
- **Label problems: can the mine roller that looks like a tank and the jet-plane boxes on empty forest be seen in the numbers? Any others?** [measured]
  Both are visible, and there are more.

**1. mine-roller-a-005-009**
- The box is 63x45 (aspect 1.40, landscape).
- The Helsinki mine_roller is 53x60 (aspect 0.88), and the other validation mine roller, mine-roller-b, is 58x66 (0.88).
- Its sprite is 50x27 px, a camouflaged hull with a gun barrel. That is nearly the same as the validation tank sprite at 48x20. The reference mine roller is a long vehicle with rollers at 49x55.
- It supplies 19 of the 52 mine_roller boxes in v8. 14 of those 19 have status `algorithmic_bottom_completion`. The sprite README says it supplies 42 of 57 training crops.

**2. jet-plane-d-040-082**
- In frames 40 to 52 the labelled centre moves up the image, from y=231 to y=42, while the ground flows down at about 57 px per frame.
- The offset from the motion-consistent position is 824 px at frame 40, 461 px at frame 46 and 63 px at frame 52.
- From frame 53 on it is consistent.
- So all 13 boxes in frames 40 to 52 are wrong, not only frames 40 and 46. All 13 carry status `directly_reviewed_track`.

**3. Other motion outliers in v8**
- helicopter-b-early-077-104 moves backward at frames 79 and 80 (-35 and -57 px).
- jet-plane-e-109-141 stalls at frame 140 (-2 px).
- tank-b-128-160 jumps +104 px at frame 154, and its width ranges from 25 to 62.

**4. Stale duplicate frames**
- Frames 9 and 239 are duplicates of frames 8 and 238. The boxes are identical, the next step is doubled (142 and 146 px), and codex's registration shows 170 then 285 px over the two 4-frame windows that contain them.
- A tracker that assumes one step per frame index will be one step off there.

**5. The two validation label sets disagree**
- Over the 773 boxes they share, mean IoU is 0.749. 51% are identical and 25% have IoU below 0.5.

| track | v8 box | codex box | IoU |
|---|---|---|---|
| tank-015-036 | 59x29 | 114x68 | 0.21 |
| large-tower-038-067 | 51x100 | 158x135 | 0.24 |
| tank-b | - | - | 0.24 |
| large-tower-b | - | - | 0.33 |
| jet-plane-c | - | - | 0.41 |
| helicopter-043-073 | - | - | 0.45 |

- Class disagreements: the same object is medium_launcher in v8 (medium-launcher-a-092-123) and small_launcher in codex. At 30x45 its box sits between the two Helsinki sizes. Codex's condor-103-150 (90x77) was replaced in v8 by jet-plane-e (88x75). Helsinki's condor is 172x168 and its jet_plane 77x82, so jet_plane fits.

**6. Near-stationary codex tracks**
- Codex has three medium_plane tracks (a, b and c, frames 95 to 150) with median dy of -1.5, 6.0 and -1.2 px per frame, wandering around x 2570-2770, y 295-530.
- They are either airborne objects moving with the drone or tracker artefacts. v8 dropped them.
- This cannot be resolved without the validation frames.

**7. Track count mismatch**
- `run_metadata.json` says 37 tracks, but there are 34 distinct track ids in the provenance.
  Evidence: - `_work` scripts.
- `oscar-sprite-synthetic\review\label_exclusions.json`.
- `sprite-bank\mine_roller\lib__mine_roller.png` against `tank\lib__tank.png`, viewed in `_work\sprites_compare.png`.
- **Q4 data side: what do the existing cutouts and the synthetic set actually contain?** [measured]
  **Sprite bank**
- 68 RGBA sprites (52 approved, 16 redrawn), all from the train split.
- 57 come from Helsinki reference tiles. 11 come from validation (4 large_launcher, 4 tank, 2 medium_plane, 1 mine_roller).
- Sprites per class:

| class | sprites |
|---|---|
| large_launcher | 9 |
| condor | 7 |
| tank | 7 |
| medium_plane | 6 |
| mine_roller | 5 |
| hangar | 4 |
| jammer | 4 |
| medium_launcher | 4 |
| small_launcher | 4 |
| small_plane | 4 |
| ta-ta | 4 |
| jet_plane | 3 |
| small_tower | 3 |
| spacecraft | 3 |
| large_tower | 1 (L1 only) |
| helicopter | 0 |

- For most classes every sprite is the same physical instance at the same heading. Only zoom blur and frame differ, so true appearance diversity is one per class.
- Sprites are stored at source-pixel scale for every zoom label. The condor is 114x104 at L0, L1 and L2.
- The level's blur is baked in. Mean absolute Laplacian inside the mask at L0 / L1 / L2:

| sprite | L0 | L1 | L2 |
|---|---|---|---|
| condor | 2.5 | 5.2 | 18.7 |
| mine_roller | 3.3 | 11.3 | 59.8 |
| large_launcher | 3.6 | 11.0 | 46.9 |

- So the grid pipeline works on views upsampled back to source scale.

**synthetic-set-v1**
- 180 images of 256 px, 320 objects, 15 classes and no helicopter.
- Zoom mix is L1 86, L0 57 and L2 37.
- 80 images have one object, 60 have two and 40 have three.
- There are 149 distinct backgrounds.
- Objects per class range from 12 (large_tower) to 30 (small_plane). Distinct sprites per class range from 1 to 9.
- Augmentation is quarter turns, flips and scale 0.9 to 1.1 only. There is no blending. A Higgsfield seam-blending pass was tried and dropped.

**The gap against validation.** Validation objects appear at arbitrary headings (box aspects from 0.51 to 2.03 within a class) and at up to a 3x size range within large_launcher. Quarter turns and 10% scaling do not cover that.
  Evidence: - `bank.json`, `synthetic-set-v1\manifest.json` and `review\decisions.json` (90 decisions: 52 approved, 22 rejected, 16 redrawn).
- Laplacian measured with OpenCV on the sprite PNGs.
- **Q5 data side: what do object positions say about the L0 -> L1 top-left -> L0 -> L1 top-right cycle?** [measured]
  These numbers come from annotations only. They are not a policy simulation.

**Share of all labelled boxes fully inside each L1 window**

| window | validation v8 | Helsinki |
|---|---|---|
| top-left | 0.260 | 0.247 |
| top-right | 0.283 | 0.251 |
| top-centre (x 960 to 2880) | 0.436 | 0.232 |
| bottom-left | 0.197 | 0.208 |
| bottom-right | 0.227 | 0.259 |

- 56% of validation boxes have their centre in the top half.
- No box in either dataset is cut by the x=1920 seam. 3.3 to 3.5% are cut by y=1080.
- Entries cluster in the middle: 78% of validation entries have x between 960 and 2880.
- An object spends about 19 frames in the top half and about 10 in the top quarter. A 4-frame cycle therefore sees each top quadrant about 4 to 5 times per object while it is in the top half.
- A new object enters every 8.2 frames on average, but 3 can arrive in the same frame.

**Dead reckoning makes unobserved frames cheap.** With the own-flight motion field, IoU is still 0.81 after 10 unobserved frames. Large classes stay at 0.88 to 0.91, while small_launcher and ta-ta are at 0.77 and 0.69.

**Where the bottom half gets its boxes.** These numbers suggest the bottom half could be covered by propagation rather than by looking.

**The small classes.** Only the small classes need L1 or L2 pixels. They are under 12 px at L0.
  Evidence: `_work` coverage script over both annotation sets, and the crossing times from the fitted Helsinki homography.
- **Q1 and Q3 data side: what do the data imply for a verifier model and for per-class experts versus one shared model?** [inferred]
  **Real appearances per class are very few.**
- Helsinki gives exactly one instance at one heading per class, seen in 2 to 25 frames.
- The validation pseudo-labels add 1 to 5 more instances for 11 classes only.
- For condor, jammer, small_plane, spacecraft and ta-ta, the single Helsinki instance is the only real example anywhere in the repo. For mine_roller it is visible in only 2 frames.
- Any per-class expert trained on these is fitted to one object at one heading on one background. Consecutive frames of a track are near-duplicates, shifted 52 to 80 px with about 1% scale change, so they are not independent samples.

**Splitting.** A split by frame leaks the same instance across train and test. The only leak-free split is by physical instance, and that leaves 5 classes with nothing to hold out.

**Box regression.** A verifier or detector that outputs tight boxes will fail IoU 0.5 on medium_launcher (object fills 22% of the box), small_launcher (23%), jet_plane (29%) and condor (41%). Box regression has to learn the organiser's loose box or take it from a size prior.

**Heading.** That prior depends on heading. Validation box aspects differ from Helsinki by up to x1.44.
  Evidence: Instance and track counts, and the fill ratios, measured above. The implications for model design are my reasoning.

### Numbers
- Helsinki dataset size: 25 frames, 259 boxes, 16 classes x 1 instance, 8 to 12 boxes per frame (median 10) (src\helsinki\annotations\*.json)
- Helsinki drone step and altitude: 13.8889 m per frame (pose delta 2.4118, 13.6779, 0), altitude 600 m (helsinki run_metadata.json and the pose field)
- Validation v8 dataset size: 249 frames (1..249), 987 boxes, 34 track ids = 31 physical objects (metadata says 37 tracks), 11 classes, 1 to 9 boxes per frame (median 3) (src\validation\annotations and run_metadata.json)
- Validation v8 review status counts: directly_reviewed_track 606, score_anchored_track_projection 195, algorithmic_bottom_completion 165, score_confirmed_anchor 14, score_positive_group_projected_track 4, directly_reviewed_partial_entry 3 (src\validation\run_metadata.json)
- Codex validation label set: 1009 candidates = 995 manual_track + 14 score_confirmed_seed, boxes in 230 frames (5..249), 38 source files, 11 classes (condor 48, medium_plane 208, no medium_launcher) (codex validation\annotations\candidate-union.json)
- Classes with zero boxes in validation v8: condor, jammer, small_plane, spacecraft, ta-ta (measured)
- Helsinki motion field: dy = 51.85 + 0.01333*cy (r 0.994); dx = 0.00708*(cx-1914) (r 0.998); mean (1.03, 65.52), sd (6.60, 7.41), n=219 (_work\motion.py)
- Helsinki dy by y band (0-540, 540-1080, 1080-1620, 1620-2160): 55.9, 62.7, 69.6, 76.5 px per frame (sd about 2 within each band) (measured)
- Helsinki dx by x band (four 960 px bands): -8.0, -3.4, +3.6, +11.8 px per frame (measured)
- Validation v8 motion field: dy = 52.38 + 0.01467*cy (r 0.988, residual sd 1.31); dx = 0.00739*(cx-1928) (r 0.982); n=886 (_work\motion.py)
- Image-registration motion, validation: 228.0 px per 4 frames at the top centre = 57.0 px per frame, sd 0.63 px over 58 windows; outliers 170 and 285 px around stale frame 9 (coverage-ledger.json, sampling.motion_measurements)
- One-step centre prediction error, Helsinki: constant vector 9.18 px, affine field 0.77 px, homography 0.64 px (mean); Helsinki homography applied to validation 2.12 px (_work\motion.py)
- Pinhole reading of the field: tilt 18.9 degrees forward of nadir, focal length about 3196 px, horizontal FOV 62.0 degrees, vertical FOV 37.3 degrees (derived from the Helsinki affine fit and s/h = 13.889/600)
- Frames for a ground point to pass y=540, y=1080, y=2160 from y=0: 10 / 19 / 34 frames; complete validation tracks last 32 to 39 frames (median 34) (measured)
- Width growth along a track, Helsinki: +0.64 to +1.02% per frame, median +0.83%; x1.13 to x1.22 over a crossing (measured)
- Height change along a track, Helsinki: flat objects +0.80 to +0.96% per frame; large_tower -0.92%, ta-ta -1.32%, small_tower -0.32%, small_launcher -0.13% per frame (measured)
- Q6 mean IoU, one observation + own-flight homography + growth, no noise (k=1,3,5,10,20): 0.960 / 0.922 / 0.886 / 0.810 / 0.684; share at IoU>=0.5: 100 / 100 / 100 / 99 / 96% (_work\q6.py, Helsinki ground truth)
- Q6 mean IoU, one observation + single scene vector: 0.650 / 0.331 / 0.189 / 0.056 / 0.000; share at IoU>=0.5: 79 / 26 / 7 / 1 / 0% (_work\q6.py)
- Q6 mean IoU, two observations with constant velocity, no noise: 0.941 / 0.755 / 0.527 / 0.111 / 0.000 (_work\q6.py)
- Q6 with 4 src px edge noise: one obs + homography, two obs fused + homography, two obs constant velocity: 0.775/0.761/0.744/0.705/0.588; 0.825/0.806/0.786/0.741/0.623; 0.658/0.447/0.299/0.107/0.005 (_work\q6.py, 5 noise trials per sample)
- Q6 with a homography borrowed from the other flight: 0.901 / 0.747 / 0.622 / 0.388 / 0.146; drift between the two fitted models 0.6 / 2.1 / 3.8 / 9.9 / 29.8 px at k=1/3/5/10/20 (_work\q6.py)
- Fill of the organiser box by the visible object (area ratio): medium_launcher 0.22, small_launcher 0.23, jet_plane 0.29, condor 0.41, large_launcher 0.50, small_tower 0.66, medium_plane 0.73, tank 0.73, hangar 0.76, spacecraft 0.76, jammer 0.80, large_tower 0.80, ta-ta 0.84, mine_roller 0.85 (bank.json (size against box_in_sprite) and Helsinki ground truth)
- IoU 0.5 shift tolerance (box short side / 3): ta-ta 5.7 src px (1.4 L0 px), small_launcher 7.3 (1.8), jammer 10.8 (2.7), small_plane 14.3, spacecraft 14.7, medium_launcher 15.0, tank 15.7, hangar 43, condor 56 (derived from Helsinki median boxes)
- Entry statistics, validation: 31 of 32 entering tracks have first-box y1 <= 54 (19 at y1 = 0); entry x 618..3211, 78% within 960..2880; one new object every 8.2 frames (measured)
- Share of all validation boxes fully inside each L1 window: top-left 0.260, top-right 0.283, top-centre 0.436, bottom-left 0.197, bottom-right 0.227; boxes cut by the x=1920 seam 0; cut by y=1080 3.3% (measured)
- Agreement between v8 and codex boxes on the same track and frame: n=773, mean IoU 0.749, identical 51%, IoU below 0.5 for 25% (measured)
- jet-plane-d-040-082 label error: frames 40..52: labelled centre y goes 231 -> 42 against +57 px per frame ground flow; offset from the motion-consistent position 824 px (f40), 461 px (f46), 63 px (f52) (measured with the inverse Helsinki homography from frame 53)
- mine-roller-a against a real mine roller: box 63x45 (aspect 1.40) against Helsinki 53x60 (0.88) and mine-roller-b 58x66 (0.88); sprite 50x27 against validation tank sprite 48x20; 19 of 52 v8 mine_roller boxes (measured; bank.json)
- Sprite bank and synthetic set: 68 sprites (57 reference, 11 validation), helicopter 0, large_tower 1; synthetic set 180 images of 256 px, 320 objects, zoom mix L1 86 / L0 57 / L2 37 (bank.json; manifest.json)
- Sprite blur by zoom label (mean absolute Laplacian inside the mask, L0 / L1 / L2): condor 2.5 / 5.2 / 18.7; mine_roller 3.3 / 11.3 / 59.8; large_launcher (validation) 3.6 / 11.0 / 46.9 (OpenCV on sprite PNGs)

### Gaps
- The validation 4K frames (Oscar's 'reconstructed-validation') are not in any export, so the validation labels could not be checked against pixels. The mine-roller and jet-plane problems are confirmed by geometry and sprite images only.
- It is unknown whether condor, jammer, small_plane, spacecraft and ta-ta are really absent from the validation scene or just unlabelled. All validation labels are participant pseudo-labels that state they are incomplete. There is no organiser ground truth for validation and no per-class server score in the repo.
- The grid dataset the sprite work was built from (`grid-comparison-20260918-v1/256`, under `data/drone/...` in the private working repo) is not here. I inferred that its tiles are at source-pixel scale with the level's blur baked in, from the sprite sizes and their sharpness. I did not read the tile-building code.
- Codex's manual track files (`data/drone/training/manual-validation/*.json`), `data/drone/discovery/confirmed-seeds.json` and the review sheets under `raw/` and `overlay/` are referenced but not present. Only `candidate-union.json` and `coverage-ledger.json` exist.
- No measurement of detector or template recall against object pixel size or level exists in my data sources. The Q2 recognisability thresholds come from my reading of one contact sheet of the 16 Helsinki objects, not from a recall curve.
- It is unknown whether stale frames 9 and 239 come from the participant recording or are part of what the server sends.
- Three near-stationary medium_plane tracks in the codex labels (frames 95 to 150) are either objects flying with the drone or tracker artefacts. v8 dropped them, and this cannot be resolved without the frames.
- Nothing is known about the evaluation sequence: speed, altitude, heading, classes or headings. Helsinki and validation share the same geometry, but label-derived motion models differ by about 1.5%. That is enough to lose small classes after 5 to 10 frames of dead reckoning, so the evaluation flight's motion needs to be measured online.
- The Q6 simulation uses Gaussian edge noise of 2, 4 and 8 source px as a stand-in for detector error. Real error distributions per level and class, including class confusion and misses, are not in the data.
- Helsinki covers only 25 frames, so horizon-20 results rest on few samples (1 to 5 per class, and none for 10 classes). Horizon-10 has no samples for hangar, medium_launcher, medium_plane, mine_roller and small_plane.
- No validation-derived sprites exist for 12 classes, and no helicopter sprite at all. The heading-dependence of the organiser box size cannot be calibrated per class from Helsinki, which has one heading per class.

## read:spec-scorer
Starter kit scoring, camera and timing, read from C:\Users\edlun\Desktop\lucky shots\NordicCupAI\drone-flyby and measured with the real scorer (faster-coco-eval 1.8.0 installed only into the scratch folder; the kit pins >=1.7.2,<2 and was verified on 1.7.2). All scratch scripts are in C:\Users\edlun\AppData\Local\Temp\claude\c--Users-edlun-Desktop-lucky-shots-NordicCupAI\bf48d8ae-2b16-4039-be3b-c570527b37ee\scratchpad\drone\_work\starterkit. No competition endpoint was contacted, nothing in the repo or exports was written.

HEADLINES
1. Score = per-class 101-point interpolated AP at IoU 0.50, area range "all", maxDets 100 per frame per class, pooled over ALL frames of a class and ranked by confidence, then macro-averaged over the classes that occur in the sequence's ground truth. One GT box in one frame is one unit, so a box-frame is worth 1/(K*N_c) mAP (K classes present, N_c box-frames of that class). A whole class is worth 1/16 = 0.0625.
2. Low-confidence extras are FREE, confident false positives cost as much as misses. Measured: 400 junk boxes below all TPs -> 1.000; padded to 500/frame -> 1.000; every box also reported under all 15 wrong labels at 0.05 -> 1.000; one FP per class ranked above the TPs -> 0.914 (exactly N/(N+1) per class); one miss per class -> 0.894; a false track in every class for all 25 frames ranked above the TPs -> 0.368, the same track ranked below -> 1.000; duplicates at equal confidence -> 0.602, at low confidence -> 1.000. So a verifier should re-rank, never hard-delete.
3. The scene motion is NOT a constant velocity. The camera looks forward: objects enter at the top at 52 px/frame and leave at the bottom at 81 px/frame, and drift away from x = 1914. One fixed map fits every object: x' = 1.00709x - 13.57, y' = 1.01333y + 51.85 (1-step residual sd 0.3 px in x, 0.9 px in y, held-out classes). A ground point crosses the frame in 33.4 frames (validation pseudo-label tracks: median 33 frames).
4. Q6: ONE recognising frame plus that shared map beats TWO frames of the same object. Held-out, noise-free: one frame + map gives IoU>=0.5 in 100% of forecasts up to 6 frames ahead, 98.6% at 8, 93.9% at 10, 82% at 16. Two-frame per-object constant velocity: 99.5% at 2, 82% at 4, 38% at 6, 0% at 10. A stale box (no motion compensation) fails for all 16 classes after one frame because the shift (52 to 81 px) exceeds h/3 for every class (largest h/3 is condor at 56 px).
5. Q5: with memory plus forecast, looking away from L0 is nearly free: team cycle 0.958 vs always-L0 0.968 if everything were recognisable at L0, and the cycle wins by 0.18 to 0.45 as soon as the recogniser needs 8 to 16 delivered pixels. WITHOUT whole-frame forecasting the same cycle with a PERFECT detector scores only 0.49 to 0.60, which brackets the team's 0.470 (cause of the 0.470 is unknown to me). The cycle is the right one only if the recogniser works at about 10 to 11 delivered px min-side; at 12 px and above an L2/L1 hop sweep of the top row is better (0.918 vs 0.856 at 12 px, 0.905 vs 0.741 at 16 px). Ceiling of any recognise-then-forecast system is about 0.95 to 0.97 because top-edge entry slivers (2.7 to 3.1% of GT box-frames) and the first-look delay cannot be recovered; the leader's 0.908 sits just under that ceiling.
6. Local --realtime never sleeps and its clock includes the harness's own 4K PNG load and render, measured at about 190 ms per frame on this machine, so it starts dropping frames when the endpoint takes more than about 140 ms. Answered fraction = min(1, 333.3/T_cycle): 350 ms -> 96%, 400 -> 84%, 500 -> 67%, 700 -> 48%, 1000 -> 34%. Dropped frames cost mAP almost linearly (every 2nd frame unanswered -> 0.527).

### Answers
- **Exactly how is the score computed in local_evaluator.py (the faster-coco-eval call and its parameters)?** [measured]
  local_evaluator.score() (lines 421-515) builds a COCO dict with one image per frame (3840x2160), categories = all 16 names with ids 1..16 in dtos.OBJECT_CLASSES order, GT boxes as xywh in source pixels with area = w*h and iscrowd = 0. Predictions are the accepted responses scaled by utils.global_bbox_to_source (plain multiply by 3840 and 2160), boxes with w<=0 or h<=0 silently skipped, score = confidence. Then: COCOeval_faster(coco_gt, coco_dt, 'bbox'); params.imgIds = every frame of the scene; params.catIds = only the classes present in the GT; params.iouThrs = np.array([0.50]); evaluate(); accumulate(). Nothing else is overridden, so the library defaults apply: recThrs = 101 points 0.00..1.00, maxDets = [1, 10, 100], areaRng = all/small/medium/large, useCats = 1. The kit reads precision[0, :, category_index, 0, -1], i.e. IoU 0.50, all 101 recall points, area index 0 = 'all' ([0, 1e10], so NO box is ignored for being small), maxDets index -1 = 100 detections per frame per class. Per class AP = mean of the 101 interpolated precisions (entries equal to -1 dropped), clamped to [0,1]; final = unweighted mean over the evaluated classes. If there are no predictions at all the function returns 0.0. Matching is greedy per frame and per class in descending confidence, each detection takes the best still-unmatched GT with IoU >= 0.5 (IoU exactly 0.5 counts). Measured boundary: every box shifted down by 0.33*h still scores 1.000, by 0.34*h scores 0.023; every box scaled by 0.72 or 1.40 about its centre scores 1.000, by 0.70 or 1.42 scores 0.000 / 0.109.
  Evidence: C:\Users\edlun\Desktop\lucky shots\NordicCupAI\drone-flyby\local_evaluator.py lines 421-515; faster_coco_eval Params('bbox') printed from the scratch install (iouThrs 0.5..0.95, recThrs 101, maxDets [1,10,100], areaRng [[0,1e10],[0,1024],[1024,9216],[9216,1e10]]); cocoeval.py lines 266-269 and 315-318 (mergesort by -score, truncate to maxDets[-1]); exp_score.py runs E0, E8, E9
- **How are classes absent from a sequence treated, how do frames never answered count, and can predictions be submitted for a frame that was never received?** [measured]
  Absent classes: evaluated_classes = classes of OBJECT_CLASSES that occur at least once in the scene's GT (lines 431-438) and params.catIds is restricted to them, so an absent class is not part of the macro-average AND every prediction of that class is ignored. Measured: tank removed from the GT plus 50 confident junk tank boxes in every frame -> mAP 1.000 over 15 classes. The README says the service does the same ("AP is calculated for each class represented in the dataset"). Frames never answered (skipped by the clock, timeout, HTTP error, invalid response): score() uses predictions.get(frame, []), the GT of that frame stays in, so every box in it is a miss. Measured: every 2nd frame unanswered -> 0.527, only every 4th frame answered -> 0.304. Predictions for a frame you never received: impossible. replay() stores predictions[frame] only for the frame it just POSTed, and the response must echo request_id and frame exactly or the whole response is discarded (lines 336-349). There is one response per received frame and it can only carry boxes for that frame. Consequence: forecasting only helps for the frame in hand (objects outside the current crop), never for dropped frames. Also note that a discarded response (invalid, timeout, HTTP error) also loses its camera command, because camera.apply sits inside 'if response is not None' (lines 352-399); a refused camera command on the other hand keeps the detections.
  Evidence: local_evaluator.py lines 330-399, 431-438, 481-495, 504; exp_score.py runs E4, E4b, E7
- **How does confidence ordering matter, what do the 500-annotation cap and maxDets do, and how are duplicates handled?** [measured]
  Confidence is the only thing that orders the pooled per-class list (all frames together, stable mergesort by descending score, ties broken by frame order then by order inside the response). AP only looks at precision ABOVE each recall level, so a false positive hurts only the true positives ranked below it. Measured on helsinki GT: one FP per class ranked above all TPs -> mAP 0.914, per class exactly N/(N+1) (mine_roller N=2 -> 0.667, tank N=25 -> 0.962); 400 FPs (one per class per frame) ranked below all TPs -> 1.000; a false track in every class for all 25 frames ranked above the TPs -> 0.368, ranked below -> 1.000, with uninformative (random) confidences for TPs and FPs -> 0.469. Duplicates: there is no NMS, a second box on a matched GT is an FP. Every TP duplicated at the SAME confidence -> 0.602; duplicated at confidence 0.1 -> 1.000. So constant confidences are dangerous and the confidence must rank by probability of being a true positive, consistently across frames and across L0/L1/L2 and forecast boxes. Caps: dtos.DroneFlybyPredictResponseDto.annotations is conlist(max_length=500); 500 is accepted, 501 rejects the whole response (frame lost, camera move lost). Inside the scorer only the top 100 per frame per class survive: 99 junk tank boxes above the real tank -> tank AP 0.010, 120 junk above it -> tank AP 0.000 (the TP is cut off); junk below the TP is harmless (padding every frame to 500 annotations with junk at 0.01 -> 1.000). Class hedging is free at the tail: every box also reported under one wrong class at 0.05 -> 1.000; under ALL 16 labels (right one 0.9, others 0.05, 192 annotations in the fullest frame) -> 1.000. The opposite order is expensive: wrong class at 0.9 plus right class at 0.05 -> 0.491, and perfect boxes with no class knowledge (16 labels at equal confidence) -> 0.101.
  Evidence: exp_score.py runs E1, E1m, E2, E3, E3b, E3c, E5, E5b, E6, E6b, E6c, E6d, E10, E10c; dtos.py line 290; faster_coco_eval/core/cocoeval.py lines 266-269, 315-318
- **Which box validity rules reject a whole response?** [measured]
  Validation is pydantic (dtos.py) and any single failure discards the whole response, all its detections and its camera command. Tested against DroneFlybyPredictResponseDto.model_validate. REJECTED: x1 == x2 or y1 == y2 (strict inequality, so a box clipped to zero width at the frame edge kills the frame), x1 > x2, any coordinate outside [0,1] even by 1e-9 or 1e-7, NaN/inf, numeric strings, confidence outside [0,1], confidence = true (bool), unknown or misspelled object_id ('Tank', 'ta_ta'; the real name is 'ta-ta'), any extra key in an annotation or at top level (extra='forbid'), 501 annotations, frame as 1.0, requested_view.center_x = 2500.0 (StrictInt). ACCEPTED: coordinates exactly 0 and 1 (also as JSON ints), confidence 0 or 1 as ints, 500 annotations, empty list, requested_view omitted or null, a requested_view with level 3 or out-of-bounds centre (schema passes, the camera then refuses the move and the detections still count; but utils.validate_response in api.py raises on level 3, which would turn into an HTTP 500 on your own server). Python-side pitfall measured: np.float32 and np.float64 bbox values are accepted by the DTO, np.int64 in requested_view is REJECTED, so cast camera centres with int(). Also request_id and frame must be echoed exactly (local_evaluator.py lines 337-345). utils.clip_bbox_to_frame returns None when less than 1e-6 of width or height survives clipping, use it on every forecast box that reaches an edge. GT itself clips to 3839 / 2159 (not 3840 / 2160) and keeps slivers down to 6 px high (hangar f19 [161,0,343,6]) and 9 px wide (medium_launcher f23), so exit slivers from a forecast are worth reporting.
  Evidence: validity_tests.py output; dtos.py lines 96-99, 209-294; utils.py lines 216-228, 235-285; trunc.py listing of truncated GT boxes
- **Camera maths: which L1/L2 centres are reachable, what does a full L1 or L2 sweep of the top band cost, and what does an L0->L1->L2->L1->L0 excursion cost?** [measured]
  Bounds (utils.center_bounds_for_level): L0 only (1920,1080); L1 centre x 960..2880, y 540..1620; L2 centre x 480..3360, y 270..1890, all integers, inclusive. Limits by CURRENT level: 2203 / 1102 / 551 px Euclidean, which are the half-diagonals 2202.91 / 1101.45 / 550.73 rounded up. Level steps 0<->1<->2 only; a request for L0 must be (1920,1080) and ignores the distance limit. From L0 every L1 centre is one move (farthest is 1101.45). From L1 at c, any L1 or L2 centre within 1102 px is one move, so every L2 tile inside the current L1 view is reachable (max 550.7) and so are tiles up to 1102 px away, e.g. L1 (960,540) -> L2 (2028,270). From L2, L1/L2 centres within 551: L2 (480,270) -> L1 (960,540) is 550.73 and legal, so every L2 position can step out to L1 and then to L0 in 2 moves; every L2 centre is 2 moves from L0. L1 tiles: TL (960,540) -> TR (2880,540) is 1920 px = 2 moves (via top-centre (1920,540) or via L0); TL -> BL is 1080 px = 1 move; TL -> BR 2 moves. So an L1 sweep of the top band is TL, TC, TR = 3 frames one way, 4-frame cycle TL,TC,TR,TC, and the team's L0,TL,L0,TR visits each top quadrant exactly as often (once per 4 frames), trading the top-centre L1 frame for an L0 frame. L2 tiles: horizontal neighbour 960 px = 2 moves, vertical neighbour 540 px = 1 move, diagonal 3 moves. Visiting the four top-row L2 tiles (x = 480, 1440, 2400, 3360 at y = 270) takes 7 frames one way by either route: pure L2 (480, 1031, 1440, 1991, 2400, 2951, 3360) or alternating L2/L1 hops (L2 480, L1 (960,540), L2 1440, L1 (1920,540), L2 2400, L1 (2880,540), L2 3360), the hop route giving useful L1 frames in between; a there-and-back cycle is 12 frames. Pure traversal without visiting is 4 moves. Starting from L0 add one L1 frame, returning to L0 add one L1 frame. A full 16-tile L2 raster is about 19 frames (columns of 4 with one intermediate frame per column change), during which the scene moves about 1250 px, so it is pointless. Minimal L2 glimpse L0->L1->L2->L1->L0: 4 moves, 3 consecutive frames without an L0 view (2 L1 + 1 L2), during which objects move 3 x (52..81) = 156..243 px. Dwell times from the measured motion: a ground point spends 18.5 frames in the L1 top band (y<1080), 9.8 frames in the L2 top row (y<540), 14.9 frames in the bottom half; an object 50 px high is fully inside the L2 top row for 9.0 frames, a 170 px one for 6.9, so a 12-frame L2 cycle can miss objects at the end tiles (seen in the simulation: pure L2 sweep 0.866 vs hop sweep 0.918).
  Evidence: dtos.py lines 50-72; utils.py lines 193-213, 288-333; local_evaluator.py Camera.apply lines 92-131; camera_math.py output (BFS over the legal-move graph, dwell times from y' = 1.01333y + 51.851)
- **Timing: what does --realtime do and how does latency convert to dropped frames?** [measured]
  Without --realtime replay() sends every frame and waits (timeout still 3333 ms). With --realtime the only change is lines 405-412: elapsed = monotonic() - started; next_index = max(frame_index + 1, int(elapsed / (1/3 s))); frames in between are counted as skipped and never sent. It never sleeps: a server faster than 333 ms gets the next frame immediately, so the local clock runs faster than real time and builds up credit that later slow frames consume (the real service presumably cannot send a frame before it exists; unknown). The clock also includes the harness's own work inside the loop: load_frame of a 16 MB 4K PNG measured 138 ms plus render_view (INTER_AREA resize + PNG level 3) 51 ms on this machine, about 190 ms per frame, so locally frames start to drop once the endpoint needs more than about 140 ms; this overhead is NOT in the printed 'round trip ms' and does not exist in that form on the real service, so local --realtime is pessimistic here. --simulate-latency-ms just sleeps after the response. Steady-state law, simulated with the exact loop logic for 250 frames: answered fraction = min(1, 333.3 / T_cycle): 334 ms -> 100%, 350 -> 96% (239), 400 -> 84% (209), 450 -> 74%, 500 -> 67% (167), 600 -> 56%, 667 -> 50%, 700 -> 48% (120), 1000 -> 34% (84), 2000 -> 17%, 3333 -> 10%. A request over 3333 ms is abandoned: about 10 frames lost and the camera command lost too. Cost in score is close to linear in the dropped share when drops are spread evenly (every 2nd frame unanswered -> 0.527, 3 of 4 unanswered -> 0.304), plus a second-order cost: the camera only moves when a response is accepted, so drops also slow the camera policy. RESPONSE_TIMEOUT = 10 frame intervals = 3333 ms, FRAME_INTERVAL_MS = 333. The oscar-live-tracker branch only adds a DRONE_EVAL_TIMEOUT_S env override to local_evaluator.py and DRONE_HOST/DRONE_PORT to api.py; the scorer is byte-identical in all four exports.
  Evidence: local_evaluator.py lines 64-73, 281-414; sizes_timing.py output (138 ms load, 51 ms render, PNG about 1.0 MB per view; loop simulation table); diff of kit files across the four exports
- **What does the metric reward: long-visible versus briefly visible objects, a false positive versus a miss, and are low-confidence speculative forecasts free or harmful?** [measured]
  Unit of value is the GT box-frame. Class AP is bounded above by the recall over all box-frames of that class, and classes are macro-averaged, so one box-frame is worth 1/(K * N_c) mAP. Helsinki N_c: mine_roller 2, medium_plane 5, hangar 6, small_plane 9, medium_launcher 10, condor 11, jammer 13, helicopter 19, large_tower 19, small_tower 20, jet_plane 22, spacecraft 23, and 25 each for large_launcher, small_launcher, tank, ta-ta. Validation pseudo-labels (incomplete, 11 classes): medium_launcher 32, small_tower 45, mine_roller 52, medium_plane 66, hangar 69, large_tower 96, small_launcher 97, helicopter 105, jet_plane 112, large_launcher 149, tank 164. A full pass is about 33 to 36 box-frames. Hence: a singleton-class pass (medium_launcher in validation) is worth 1/16 = 0.0625 mAP, about 0.0020 per frame; a tank frame (5 passes) only 0.00038. Inside a class a long-visible object weighs in proportion to its frames (a 34-frame pass vs a 10-frame side-exit object = 77% vs 23% of that class). Every frame of delay before first recognition costs about 1/33 = 3% of that object's share, so EARLY recognition near the top edge is what pays, and rare classes pay most. FP versus miss: a miss always costs 1/N_c of the class AP; an FP costs (share of TPs ranked below it) * 1/(N+1), i.e. about the same as a miss when ranked on top and zero when ranked below all TPs of that class. A confident hallucinated track costs about as much as a missed real track. Speculative boxes: FREE when their confidence is below every real TP of the same class across the whole sequence, and they stay inside 500 per response and 100 per frame per class; they become TPs with pure gain if they hit. They are HARMFUL if their confidence overlaps real detections (E10c: 0.469), if they duplicate a box at similar confidence (0.602), or if one of them is invalid (whole response lost). A forecast of a VERIFIED object is not speculative: it should carry the track's high confidence because it is a TP with probability near 1 for the first 6 to 8 frames. Practical rule: confidence = track-level evidence (verified at L1/L2 > seen twice > single L0 hit > hedge labels and top-edge blobs), never a hard filter.
  Evidence: exp_score.py E1, E1m, E2, E3, E5, E6, E10; measure_gt.py and val_tracks.py counts; src/validation/run_metadata.json in the oscar-live-tracker export (boxes_per_class, note: 'Participant pseudo-labels, incomplete')
- **Q5 from the scoring side: is L0 -> L1 top-left -> L0 -> L1 top-right -> L0 the right policy, and what is the price of looking away?** [measured]
  Price of looking away, measured with the real scorer under an explicit recognition model (object recognised once its GT box is fully inside the view and its min side in the DELIVERED image is >= T px; afterwards reported in every frame): (a) WITH memory and forecast the price is almost nothing. Validation pseudo-labels, T=4: always-L0 0.968, team cycle 0.958, L1-only top sweep 0.952. T=8: 0.773 / 0.952 / 0.952. T=10: 0.741 / 0.950 / 0.952. T=12: 0.590 / 0.856 / 0.863. T=16: 0.289 / 0.741 / 0.749. Helsinki (25 frames, start-up dominated): T=8 0.816 / 0.936 / 0.905. (b) WITHOUT forecasting (report only what the current view shows, perfect detector) the same cycle collapses: 0.605 (T=4), 0.508 (T=8) on validation pseudo-labels and 0.545 / 0.488 on helsinki, versus 0.948 / 0.862 for stateless always-L0. The team's 0.470 lies in that band; I did not establish which configuration produced 0.470, so treat this as a lead, not a finding. Verdict: the cycle is sound and equivalent within 0.01 to the L1-only sweeps C (TL,TC,TR,TC) and D (L0,TL,TC,TR) for every T, PROVIDED whole-frame forecasting is on and the recogniser works at about 10 to 11 delivered px. Its L0 frames do not add recognisability above T about 10; their value is re-anchoring, motion registration, big objects and the start-up population. It is the wrong policy if the recogniser needs >= 12 px: at L1 ta-ta is 8.5 px and small_launcher 11 px min-side (every other class >= 16 px), so two classes (up to 0.125 mAP) are out of reach, and the L2/L1 hop sweep of the top row scores 0.918 (T=12), 0.905 (T=16), 0.889 (T=20) versus 0.856 / 0.741 / 0.670 for the cycle; a mixed cycle with an L0->L1->L2->L1 excursion gives 0.912 at T=12. Cheaper fix than a permanent L2 sweep: keep the cycle and spend the 3-frame L2 excursion only on small unclassified candidates, an object stays 18.5 frames in the L1 top band so there is time. Two structural gaps of the cycle: the bottom half is never seen at L1, so objects already below y=1080 at sequence start and only recognisable at L1 are lost for their remaining <= 15 frames (helsinki, 11 objects in frame 0: a one-off L0,BL,BC,BR start is worth +0.04 at T=10 and +0.09 at T=12 but -0.03 at T=8; validation starts with 1 object so no effect), decide from the first L0 frame; and permanent bottom looks are a net loss (L0+4 quadrants 0.917 vs 0.950 at T=10). Residual ceiling 0.95 to 0.97: top-edge entry slivers (2.7% of box-frames in validation pseudo-labels, 3.1% in helsinki; a 50 px object is partial for 1.0 frame at the top, a 170 px object for 3.3) plus first-look delay.
  Evidence: policy_sim.py, policy_sim2.py, policy_sim3.py outputs (real scorer via scorelib.py, camera legality asserted with the kit's rules); sizes_timing.py size table; camera_math.py dwell times
- **Q6 from the scoring side: can trajectory and box size be fixed from ONE recognising frame, or are TWO needed? What is the payoff of a forecast box at IoU 0.5 given object size and motion per frame?** [measured]
  One frame is enough, and better than two, because the motion is a property of the camera and not of the object. Helsinki GT (220 consecutive untruncated pairs): dx = -13.64 + 0.00709x (zero at x = 1924), dy = 51.25 + 0.01334y, i.e. 51 px/frame at the top, 66 at mid-frame, 80 at the bottom; per-object medians range dy 54.0 (hangar, top) to 80.0 (mine_roller, bottom) and dx -11.25 to +12.25. A single global mean velocity leaves 7.4 px sd per step, a linear map leaves 0.4 px (x) and 0.8 px (y). Fitted homography H = [[1.006736,-0.001474,-12.5993],[0.000337,1.011889,51.7059],[0,-0.000001,1]]; 4-number form x' = 1.00709x - 13.566, y' = 1.01333y + 51.851. Held-out (map fitted on 8 classes, tested on the other 8): 1-step residual sd 0.28-0.32 px x, 0.88-0.90 px y, max 3.5 px. Forecast k frames ahead from ONE GT box, held-out map, share with IoU >= 0.5 (mean IoU): k=1 100% (0.96), 2 100%, 4 100% (0.89), 6 100% (0.84), 8 98.6%, 10 93.9%, 12 90.3%, 16 82.1%, 20 65%. With the map fitted on all classes it is 100% up to k=10 and 94 to 97% at 12 to 15, also for the small objects (min side <= 35 px). TWO frames with per-object constant velocity: k=2 99.5%, 3 93.6%, 4 82.4%, 6 38.4%, 8 8.8%, 10 0%, because the speed grows 1.33% of y per frame (about 0.9 px/frame^2). With detector noise of 2 source px per box edge: one frame + map 99.9% at k=2, 99.0% at 6, 92% at 10; two frames 86.5% at k=2, 57% at 4. With 4 px noise (one L0 pixel): one frame + map 95% at k=1..3, 92% at 6; two frames 80% at k=1, 33% at 4. Box size: map the four corners (best), or keep h and grow w by 1.008 per frame (measured growth w 1.0081, h 1.0017 per frame); freezing the size costs a few points beyond k=8. Tolerance at IoU 0.5 for a pure shift is h/3 (or w/3): ta-ta 6 px, small_launcher 10, jammer 14, tank 16, jet_plane 27, condor 56 source px, against a per-frame motion of 52 to 81 px, so an uncompensated box is wrong after ONE frame for all 16 classes while a compensated one holds for 6 to 10 frames. Payoff: each forecast box that lands is one box-frame = 1/(16*N_c) mAP (0.0004 to 0.002 in validation, up to 0.03 for a 2-frame class in helsinki), about 17 of a pass's 33 frames are frames in which the team cycle is NOT looking at the object, and the downside of a forecast that misses is zero if it is ranked below the verified boxes. Second frames are still useful, but to re-anchor position (every <= 6 frames, L0 is enough for that) and to raise track confidence, not to estimate velocity. Same-map check on validation: pseudo-label fit dx = -14.55 + 0.00748x, dy = 45.26 + 0.01813y (residual sd 12.6 px, labels are noisy), 64.8 px/frame at y=1080 vs 65.7 in helsinki, track length median 33 frames vs 33.4 predicted: consistent, not proven identical. The oscar-live-tracker branch already contains 'tracking/tracker.py: One-observation object tracker using a frozen shared camera-motion model' (vendored from the absent preparation project), which matches this conclusion.
  Evidence: measure_gt.py, motion_model.py, q6_forecast.py, q6_variants.py, val_tracks.py outputs; drone_oscar-live-tracker/drone-flyby/tracking/tracker.py line 1 and tracking/VENDOR.md
- **What does the scoring side imply for Q1 to Q4?** [inferred]
  Q1 (verifier on top of the expert system): the metric wants a RANKING, not a gate. A down-ranked false positive costs 0, a deleted true positive costs 1/N_c of a class. Use the verifier's output as the confidence (and as the track confidence for all later forecast frames), keep rejected proposals at a floor confidence below every verified box, hedge the second-best class at the floor. Earlier answers cannot be revised, so verify early (top band). Q2 (pixel size): delivered min-side medians per class at L0 / L1 / L2: ta-ta 4.2 / 8.5 / 17, small_launcher 5.5 / 11 / 22, jammer 8.1 / 16 / 32, small_plane 10.8 / 21.5 / 43, spacecraft 11 / 22 / 44, medium_launcher 11.2 / 22.5 / 45, tank 11.8 / 23.5 / 47, medium_plane 12.2 / 24.5 / 49, mine_roller 13 / 26.5 / 53, small_tower 14 / 28.5 / 57, large_tower 15.6 / 31 / 62, jet_plane 19 / 38.5 / 77, helicopter 23.6 / 47 / 94, large_launcher 27 / 54 / 108, hangar 32 / 64.5 / 129, condor 42 / 84 / 168. The score as a function of the recognisable size T is the policy table in the Q5 answer; the steep part is T between 8 and 16 px. Objects are about 25 to 30% narrower at the top edge than at the bottom (w grows 1.008 per frame). Q3 (experts versus shared model): scoring is per class and macro-averaged, so a weak class costs a flat 1/16 regardless of how many objects it has; hedged labels at low confidence are free, which removes most of the risk of confusing similar classes (small/medium/large launcher, towers, planes); what is not free is a confident wrong label (E6b 0.491). Whatever the architecture, confidences must be comparable across frames WITHIN a class; they need not be comparable ACROSS classes because each class is ranked alone. Q4 (synthetic data): validate with scorelib.score (exact replica of the kit's scorer that accepts any GT dict) on real frames; helsinki is 25 frames with one instance per class (N_c as low as 2), so per-class AP there moves in steps of up to 0.33 and the kit's README says to treat it as a correctness harness first.
  Evidence: exp_score.py, sizes_timing.py, policy_sim.py outputs; README.md lines 465-466

### Numbers
- Scorer parameters actually used: iouThrs [0.50], recThrs 101 points, area 'all' [0,1e10], maxDets 100 per frame per class, useCats 1, catIds = classes present in GT (local_evaluator.py lines 500-514 plus faster_coco_eval 1.8.0 Params defaults printed)
- Annotation cap per response: 500 accepted, 501 rejects the whole response (dtos.py line 290; validity_tests.py)
- One FP per class ranked above all TPs: mAP 0.914 (per class N/(N+1): 0.667 for N=2, 0.962 for N=25) (exp_score.py E1)
- One miss per class: mAP 0.894 (exp_score.py E1m)
- 400 FPs ranked below all TPs / padded to 500 per frame: 1.000 / 1.000 (exp_score.py E2, E3)
- False track in every class for 25 frames: above TPs / below TPs / random confidences: 0.368 / 1.000 / 0.469 (exp_score.py E10, E10c)
- Duplicates: same confidence / low confidence: 0.602 / 1.000 (exp_score.py E5, E5b)
- Class hedging: all 16 labels per box, right one on top: 1.000 (192 annotations in the fullest frame) (exp_score.py E6c)
- Wrong class 0.9 + right class 0.05 / no class knowledge: 0.491 / 0.101 (exp_score.py E6b, E6d)
- maxDets truncation: 99 vs 120 junk boxes of one class above the TP: class AP 0.010 vs 0.000 (exp_score.py E3c, E3b)
- Unanswered frames: every 2nd / 3 of 4: mAP 0.527 / 0.304 (exp_score.py E4, E4b)
- IoU 0.5 tolerance: pure shift up to 0.33*h passes, 0.34*h fails; centre scaling 0.72..1.40 passes, 0.70 and 1.42 fail (exp_score.py E8, E9)
- Helsinki GT: 25 frames, 259 boxes, 8 to 12 boxes per frame, 16 classes with one instance each, N_c from 2 (mine_roller) to 25 (measure_gt.py)
- Validation pseudo-labels (teammate, incomplete): 249 frames, 987 boxes, 34 tracks in files (run_metadata says 37), 11 of 16 classes, median track length 33 frames, mean 3.96 boxes per frame (val_tracks.py on drone_oscar-live-tracker/drone-flyby/src/validation)
- Per-frame motion model (helsinki GT): x' = 1.00709x - 13.566, y' = 1.01333y + 51.851; dy 51.9 px at y=0, 66.2 at y=1080, 80.6 at y=2160; dx zero at x about 1914-1924 (motion_model.py, q6_variants.py)
- Motion model residuals (held-out classes, 1 step): sd 0.28-0.32 px in x, 0.88-0.90 px in y, max 3.5 px; global constant velocity leaves sd 6.6 / 7.4 px (q6_forecast.py, motion_model.py)
- Frames to cross the frame / L1 top band / L2 top row / bottom half: 33.4 / 18.5 / 9.8 / 14.9 (camera_math.py)
- ONE frame + shared map, share of forecasts with IoU>=0.5 (held-out, noise-free): k=1..6 100%, k=8 98.6%, k=10 93.9%, k=12 90.3%, k=16 82.1%, k=20 65.2% (q6_forecast.py)
- TWO frames, per-object constant velocity, IoU>=0.5 share: k=2 99.5%, k=3 93.6%, k=4 82.4%, k=6 38.4%, k=8 8.8%, k=10 0% (q6_forecast.py)
- Same comparison with 4 source px noise per box edge (one L0 pixel): one frame + map: 95.4% (k=1), 92.0% (k=6), 81.4% (k=10); two frames: 80.0% (k=1), 33.3% (k=4) (q6_forecast.py)
- Box growth per frame: w x1.0081, h x1.0017 (motion_model.py)
- Policy simulation, validation pseudo-labels, memory+forecast, T = 8 / 12 / 16 delivered px: always L0 0.773 / 0.590 / 0.289; team cycle 0.952 / 0.856 / 0.741; L1 top sweep 0.952 / 0.863 / 0.749; L2/L1 hop top row 0.918 / 0.918 / 0.905; pure L2 top row 0.866 / 0.866 / 0.855 (policy_sim.py)
- Team cycle with a perfect detector but NO forecasting (stateless): 0.605 (T=4), 0.508 (T=8), 0.401 (T=12) on validation pseudo-labels; 0.545 / 0.488 / 0.351 on helsinki (policy_sim2.py)
- Ceiling of recognise-then-forecast (T=4): 0.968 validation pseudo-labels, 0.941 helsinki (policy_sim.py row A)
- Truncated GT box-frames: helsinki: top 3.1%, bottom 2.3%, side 3.5%; validation pseudo-labels: top 2.7%, bottom 1.8%, side 0% (policy_sim2.py)
- Smallest delivered min-side at L0 / L1 / L2: ta-ta 4.2 / 8.5 / 17 px; small_launcher 5.5 / 11 / 22 px; jammer 8.1 / 16 / 32 px; all other classes >= 10.8 px at L0 and >= 21 px at L1 (sizes_timing.py)
- Camera limits and half-diagonals: 2203 / 1102 / 551 px versus 2202.91 / 1101.45 / 550.73 (dtos.py lines 68-72; camera_math.py)
- Sweep costs: L1 top band TL,TC,TR = 3 frames one way (TL to TR is 1920 px = 2 moves); four L2 top-row tiles = 7 frames one way, 12-frame cycle; L0->L1->L2->L1->L0 = 4 moves with 3 frames away from L0; any L2 centre is 2 moves from L0 (camera_math.py)
- Local harness overhead inside the realtime clock: load_frame 138 ms + render 51 ms per frame on this machine; PNG about 1.0 MB per view (sizes_timing.py)
- Frames answered of 250 versus cycle time: 334 ms 100%, 350 ms 96%, 400 ms 84%, 500 ms 67%, 700 ms 48%, 1000 ms 34%, 3333 ms 10% (sizes_timing.py (loop logic copied from local_evaluator.py lines 405-412))

### Gaps
- The evaluation service's scorer and clock are not in this repo. That the service uses the same faster-coco-eval call, the same class-presence rule and the same maxDets is asserted only by README.md and docstrings ('the same scorer the competition uses'). Unknown: whether the service waits for a frame's emission time before sending (the local harness never sleeps), whether a camera command from a timed-out response is applied, where the evaluator is hosted and what the network round trip is.
- I ran faster-coco-eval 1.8.0 (installed with pip --target into the scratch folder, a PyPI download, no competition endpoint). The kit says verified on 1.7.2 within the range >=1.7.2,<2. It is not installed in the machine's default Python (miniconda, Python 3.13.12), so python local_evaluator.py would currently fail there with ModuleNotFoundError.
- Validation ground truth does not exist locally. The only validation labels are a teammate's pseudo-labels in drone_oscar-live-tracker/drone-flyby/src/validation/annotations (987 boxes, 11 of 16 classes, no condor, jammer, small_plane, spacecraft or ta-ta, flagged by their own run_metadata.json as incomplete; part of them are algorithmic completions from the teammate's own motion model, so the motion check against them is partly circular). The recorded validation frames are not in any export; run_metadata.json points at /Users/bumblebee/Github_repos/Projects/nordic-ai-cup-2026/data/drone/reconstructed-validation, which is the absent preparation project.
- Whether the evaluation sequence has the same altitude, step, gimbal pitch and therefore the same frame-to-frame map as helsinki is unknown. Validation pseudo-labels are consistent (64.8 vs 65.7 px/frame at mid-frame, 33-frame tracks) but too noisy to confirm the coefficients. The map should be re-estimated online (background registration between L0 frames, or from the tracked boxes).
- The policy tables rest on an ASSUMED recognition model (recognised when fully inside the view and min side >= T delivered px, then forecast perfectly). The real T per class is question Q2 and belongs to the detector subsystems; forecast quality was measured separately and only on helsinki GT (25 frames, so horizons beyond about 20 frames have n <= 23).
- Which configuration produced the team's 0.470 is unknown to me. The stateless-cycle result (0.49 to 0.60 with a perfect detector) is only a reference point; oscar-live-tracker does contain a one-observation tracker with a shared motion model (vendored from the absent 'preparation project' drone/perspective_tracking/), I did not check whether it was active in the 0.470 run.
- Number of objects, class mix and starting population of the evaluation sequence are unknown, so the value of a start-up look at the bottom half cannot be fixed in advance (worth +0.04 to +0.09 on helsinki at T >= 10, zero on validation which starts with one object).
- Helsinki has one instance per class and N_c as low as 2, so all per-class AP numbers measured on it are coarse; the kit's README says the same.
- During my run 'git status' began to show modified files under medical-appointment/bench/hpc/ (pod_bootstrap.sh, pod_bringup.sh, pod_endpoint.sh). I did not touch them; every file I wrote is under the scratch _work/starterkit folder and I imported the kit with PYTHONDONTWRITEBYTECODE=1 (no __pycache__ created). Some other session or process changed them.

## read:live-tracker
Oscar's live endpoint (example.py + tracking/) is a one-observation tracker on a shared perspective motion model plus a fixed camera cycle. Unit tests: 44 run, 43 pass, 1 skipped (needs private artifacts). I reproduced the commit's Helsinki oracle score exactly (0.9777, real pipeline with images, in-process, my own COCO scorer because faster_coco_eval is not installed) and built an image-free simulator around the real DroneTrackingWorkflow/RevisitTracker/LevelOneSweep + organizer Camera that reproduces 0.9777 on Helsinki, then ran it on the 249-frame validation pseudo-labels (no validation images exist in this repo). ORACLE CEILING of the deployed default policy (perfect detector, real camera cycle, blend extents) on validation labels: 0.864 raw, 0.903 after removing one duplicated pseudo-label track. So of the gap from 0.470, roughly 0.39 to 0.43 is detector loss and roughly 0.10 to 0.14 is policy/tracking/placement loss. The policy loss decomposes into fixable items: (1) the Helsinki-fitted size prior under 'blend' hurts validation (large_launcher AP 0.66 -> 0.98 with extent=detector; mAP 0.903 -> 0.932); (2) a tracker rule never births the middle object of a same-class cluster (small_launcher AP 0.64; fix in a scratch copy gives 0.97, mAP -> 0.961); (3) looking away costs only about 0.02 with a perfect detector (L0 hold 0.954/0.983). With a size-gated oracle (object recognised only if its smaller side is >= T delivered pixels) the picture flips: L0 frames become useless and, worse, L0 'visible misses' retire true tracks after 3 L0 frames; an L1-only top-band sweep (left, centre, right, centre) beats the deployed L1/L0 cycle at every T >= 8 (0.863 vs 0.759 at T=12, 0.735 vs 0.570 at T=16, 0.678 vs 0.483 at T=20). l2_top is capped near 0.63 to 0.66 even with a perfect detector because one sighting at the top cannot be forecast 30 frames. Q6: one recognising frame is enough for direction and speed because motion is shared and calibrated from frames 0 and 1 (0.7 px error after 1 frame, 5 px after 8, 11 px after 16 on Helsinki; IoU>=0.5 in 94% at k=8, 62 to 82% at k=16, 32% at k=30), per-object two-frame velocity is far worse, but tracks must be refreshed within roughly 8 to 12 frames, and the corner-warp inflates box height (x1.22 at k=16).

### Answers
- **(a) Motion model: calibration from frames 0 and 1, translation or perspective, and the motion clock** [measured]
  Perspective, not pure translation. tracking/motion.py MotionModel stores a rank-1 3x3 matrix A = e q^T and maps source pixels between ticks with H(a,b) = (I+(b-o)A)(I+(a-o)A)^-1 (a camera translating at constant velocity over a plane; e is the epipole, q the plane term). Calibration: DroneTrackingWorkflow.process keeps frame 0 as 'warmup' (answers with frame_rows of that frame's detections, commands overview), and on the next frame calls calibrate_images: SIFT nfeatures=6500 on the two delivered 960x540 views, BFMatcher ratio 0.7 with mutual check, RANSAC homography (6 px) used only as a loose 40 px outlier gate, then MotionModel.from_matches: IRLS (20 iters) intersection of flow lines for the epipole, then a 3-parameter linear 'amount' field q by IRLS (25 iters). Rejects (CalibrationError, stay in warm-up and retry with the next frame) if <12 matches, no motion, or median residual > 6 source px. Geometry is frozen afterwards, never re-calibrated. Measured on Helsinki frames 0,1: 4264 matches, median residual 0.68 px, epipole at (1920, -8033) source px, flow 53 px/frame at y=100, 65 at y=1080, 79 at y=2060, and +-12 px/frame horizontal at the left/right edges. A pure translation model is useless (IoU>=0.5 only 13% after 4 frames). Motion clock: workflow.observed_step takes the overlap of previous and current delivered views (needs >=48 px), 300 Shi-Tomasi corners, pyramidal LK with forward-backward check <1 px, and computes the median ratio of measured flow to the model's one-tick flow; step = round(ratio) is accepted only if MAD spread <= 0.2, |ratio-step| <= 0.25 and 0 <= step <= max(2, 2*frame gap); otherwise it falls back to the frame_index gap. Step 0 = frozen frame (tick not advanced, boxes stay; RevisitTracker._accept replaces the same-tick observation), step 2 = doubled frame. The validation labels contain exactly this: frozen transitions at frames 8->9 and 238->239, each followed by a double step. Without the clock the oracle ceiling drops 0.903 -> 0.894. On Helsinki the clock returned step 1 with ratio 0.997 on all 23 frames, no fallbacks. Risk found: trace(A) = -1.54e-3 on Helsinki makes the model's flow grow by 1/(1+a*trace): x1.038 at tick 24 (true on Helsinki, measured x1.037), but x1.62 at tick 248; validation labels show flat flow over the whole flight (trend x1.011). Whether the validation calibration fits a near-zero trace is unknown here (no images).
  Evidence: tracking/motion.py (MotionModel.mapping, from_matches, calibrate_images), tracking/workflow.py (observed_step, DroneTrackingWorkflow.process); measured with _work/calib_hel.py, trace_spread.py, clock_stats.py, val_trend.py, forecast_error.py, latency.py
- **(b) Track birth and refresh: detections needed, thresholds, identity and duplicates** [measured]
  ONE detection creates a track. RevisitTracker.update: a detection is 'complete' if it is more than crop_margin_pixels=1 px from every delivered-image edge. Complete + confidence >= DRONE_BIRTH_CONFIDENCE (0.6) + no candidate track + IoU <= 0.5 with every existing forecast -> 'birth'. Detections below DRONE_UPDATE_CONFIDENCE (0.4) are dropped. Refresh: same class only, candidate if IoU with the visible part of the forecast >= association_iou 0.1 OR centre distance <= association_distance 1.0 x diagonal of that visible forecast (min 8 px), size ratio within 0.25..4; greedy by score = IoU + 0.5*max(0, 1 - dist/scale); a match is skipped if the runner-up is within ambiguity_margin 0.1. A complete match replaces the anchor (history max 6, but forecasts use only the latest observation unless adapt_edges, which is off); a partial match only resets the miss counter ('partial_seen'). Duplicates: same-class detections with IoU >= duplicate_iou 0.7 are suppressed within a frame (this is the only NMS); an unmatched detection that had any candidate track is logged 'ambiguous_detection' and discarded; an unmatched detection of any class overlapping a forecast by > 0.5 is 'conflicting_detection' and discarded. Two different-class detections of a NEW object in the same frame both get born (no cross-class NMS at birth). With emit_partials (on), unmatched complete boxes between 0.4 and 0.6 and crop-cut boxes are emitted for one frame at confidence x0.8 ('transient') without a track; with entry_tracks (on), a box cut by the TOP source edge with confidence >= 0.6 creates a provisional track whose height comes from the size prior, retired after 1 clear miss. Retirement: forecast fully outside the source frame ('source_exit'), or visible_misses_before_retirement = 3 frames where the detector ran, the forecast lies fully inside the current crop and nothing matched. No age limit and forecast_decay = 0, so one false positive at >= 0.6 is repeated at constant confidence until it exits (about 33 frames) unless it is looked at and missed 3 times. None of association_iou, association_distance, ambiguity_margin, duplicate_iou, visible_misses_before_retirement, partial_confidence_scale, forecast_decay, adapt_edges is exposed as an env var. BUG measured: the 'had_candidate -> continue' rule never births the middle one of three small launchers about 20 px apart (small-launcher-c missed in 32 of 32 frames with a perfect detector, small_launcher AP 0.64); it is triggered when a neighbour's entry track exists first. A scratch patch (birth when all candidate tracks were claimed by other detections) gives AP 0.97 and mAP 0.903 -> 0.932 (blend) / 0.932 -> 0.961 (detector extents); with a real detector that patch needs a low-overlap guard against near-duplicates.
  Evidence: tracking/revisit.py RevisitConfig defaults and RevisitTracker.update lines 240-373; example.py CONFIG; _work/sim_val4.py, sim_patched.py, sim_final.py, patched/tracking/revisit.py
- **(c) Box extents: size-prior.json, blend/prior/detector, per-class override** [measured]
  size-prior.json was fitted by placement.SizePrior.fit on the 25 Helsinki annotation files only (one physical instance per class, unclipped boxes): per class and axis a line size = intercept + slope*cy, clamped to [0.9*min, 1.1*max]. Medians (w x h source px, n): condor 172x168 (9), hangar 188x129 (3), helicopter 116x94.5 (18), jammer 32.5x43.5 (12), jet_plane 77x82 (21), large_launcher 150x108.5 (20), large_tower 62.5x64.5 (18), medium_launcher 47x45 (5), medium_plane 56x49 (4), mine_roller 53x60 (2, slope 0), small_launcher 22x30 (25), small_plane 43x50 (9), small_tower 57x59 (19), spacecraft 44x49 (22), ta-ta 32x17 (25), tank 50x47 (24). Policies (placement.normalize_extent, applied when an observation is stored, centre kept): 'detector' = unchanged, 'prior' = prior size, 'blend' = geometric mean of detector and prior size (prior_weight 0.5), default DRONE_EXTENT_POLICY=blend. DRONE_CLASS_EXTENT is a JSON map such as {"small_tower":"prior"} that overrides the policy per class (RevisitConfig.policy_for). place_partial/complete_partial extend crop-cut sides towards the prior but leave source-edge sides; entry_box gives top-entering objects the prior height; clip_box clips to index 3839/2159 (DRONE_CLIP_LAST_INDEX). The prior does not transfer to validation: validation label medians are large_launcher 117x66, medium_launcher 30x45, hangar 208x186, helicopter 109x114, medium_plane 45x64 (heading differs per instance). With a perfect detector, extent=prior scores 0.758, blend 0.864, detector 0.888 on the raw labels; large_launcher AP is 0.66 under blend and 0.98 under detector. Entry tracks with the Helsinki prior also cost: removing them raised the oracle run from 0.888 to 0.912. The forecast itself ignores the prior and warps the four corners, which inflates height: predicted/true height 1.10 at k=8, 1.22 at k=16, 1.64 at k=24 on Helsinki (1.08/1.15/1.23 on validation labels) while width stays at 1.00; an isotropic width-scale rule gives 1.02 to 1.05.
  Evidence: tracking/size-prior.json provenance block, tracking/placement.py, tracking/revisit.py frame_rows/update; commit ad8c8f2 message; _work/explore_val.py, sim_val.py, sim_val2.py, forecast2.py
- **(d) Camera policies implemented and every env var that selects them** [measured]
  All in workflow.LevelOneSweep.next_view, one absolute move per response, every target clipped to the request's center_bounds and bisected into the maximum_center_delta circle, L0 reset exempt. Warm-up: overview (L0) until calibration succeeds. Mode 'l1' (DRONE_CAMERA_MODE=l1, default): four waypoints at L1 x = min, mid, max, mid of the L1 bounds and y = min + DRONE_VERTICAL_FRACTION*(max-min) (default 0 = top band, i.e. centres (960,540) and (2880,540), covering source rows 0..1080); with DRONE_OVERVIEW_BETWEEN_SIDES=1 (default) the odd waypoints become L0 (1920,1080). Measured sequence: L0, L0, L1(960,540), L0, L1(2880,540), L0, ... = 125 L0 and 124 L1 frames over 249, 0 refused moves. With DRONE_OVERVIEW_BETWEEN_SIDES=0 it is L1 left, centre, right, centre (2 L0 + 247 L1). Mode 'l2_top' (DRONE_CAMERA_MODE=l2_top): native L2 crops bouncing along the top row, waypoints from x=480 to 3360 spaced <= 550 px (7 columns, each revisited about every 6 frames), entered through L1. Revisit: DRONE_REVISIT_EVERY=k (default 0 = off) and DRONE_REVISIT_MIN_AGE (default 6 ticks): on frames with frame_index % k == 0, workflow.stale_track picks the non-provisional track with the oldest anchor whose centre is within one move, and next_view aims L2 at it (only when L2 is allowed from the current level, i.e. from L1 or L2), then the sweep resumes. Full env list: example.py reads DRONE_DETECT_EVERY, DRONE_VERTICAL_FRACTION, DRONE_OVERVIEW_BETWEEN_SIDES, DRONE_OBSERVE_MOTION, DRONE_CAMERA_MODE, DRONE_REVISIT_EVERY, DRONE_REVISIT_MIN_AGE, DRONE_LOG_DIR, DRONE_EXTENT_POLICY, DRONE_CLASS_EXTENT, DRONE_EMIT_PARTIALS, DRONE_ENTRY_TRACKS, DRONE_CLIP_LAST_INDEX, DRONE_BIRTH_CONFIDENCE, DRONE_UPDATE_CONFIDENCE, DRONE_CV_THREADS; detectors.py reads DRONE_DETECTOR, DRONE_WEIGHTS, DRONE_DEVICE, DRONE_IMGSZ, DRONE_CONF, DRONE_HALF, DRONE_BUNDLE, DRONE_PROJECT, DRONE_FAMILY_MIN, DRONE_FAMILY_LEVELS, DRONE_ORACLE_SCENE; api.py reads DRONE_HOST, DRONE_PORT; local_evaluator.py reads DRONE_EVAL_TIMEOUT_S. README-live.md's table omits CAMERA_MODE, REVISIT_*, CLASS_EXTENT, CV_THREADS, HALF, BUNDLE, PROJECT, FAMILY_*, ORACLE_SCENE. Trap: DRONE_DETECT_EVERY=2 tests frame_index % 2, and in the default cycle the L0 frames after warm-up sit on odd indices, so the detector would run only on L1 frames and nothing below row 1080 would ever be refreshed. There is no policy that alternates bands (vertical_fraction is a constant per run) and no level-aware logic.
  Evidence: tracking/workflow.py LevelOneSweep.next_view, l2_waypoints, DroneTrackingWorkflow.stale_track; example.py SETTINGS/CONFIG; detectors.build_detector; views recorded by _work/replay_inproc.py and sim_val2.py
- **(e) Detector backends and what 'gate recognition families by zoom level' means** [read-from-code]
  detectors.build_detector: 'none' (NullDetector, tracks only), 'ultralytics' (UltralyticsDetector: local checkpoint whose names must equal the 16 classes, imgsz 960, conf 0.25, warm-up call, restores OpenCV thread count because importing Ultralytics sets it to 1 and slowed SIFT calibration from 250 ms to 1 s), 'fixed_assets' (FixedAssetDetector: adapter around the PRIVATE preparation project's drone.scratch_objects.bundle.load_bundle, 'geometry, pixel matching and CNNs', documented as seconds per view; needs DRONE_BUNDLE manifest and DRONE_PROJECT repo root, neither is in this repo), 'oracle' (OracleDetector: reads src/<scene>/annotations by request['frame'], clips each box to the view, confidence 0.9; local only), and 'module:factory'. Default is ultralytics if DRONE_WEIGHTS is set, else none. The bundle returns rows tagged with a recognition 'family' (branch). 'Gate by zoom level' = DRONE_FAMILY_LEVELS, a JSON map family -> allowed resolution levels, e.g. {"pose_pixels":[2]}: a row is dropped when the current view's resolution_level is not in its family's list; DRONE_FAMILY_MIN gives per-family confidence floors. Reason in commit ac1474e: on the validation flight the pose-pixel branch produced 692 false alarms and no hit at L0, the tiny-shape branch 8 and none, while the geometry branch hit 69 times against 3 false alarms. Commit a243ed6 adds that on the L1/L0 replay the bundle 'found large, distinctive objects and nothing else', which motivated l2_top. Which detector actually produced the 0.470 validation score is unknown from this repo; commit cfddb4c mentions a 'placeholder scratch YOLO26m' scoring 0.378 on Helsinki.
  Evidence: detectors.py; commit messages ac1474e, a243ed6, 2f6f515, cfddb4c on origin/drone/oscar-live-tracker
- **(f) Latency per stage** [measured]
  Measured on this laptop CPU with the oracle detector through example.predict on Helsinki (25 frames): total median 33 ms, p90 36 ms; base64+PNG decode 11 ms; detector (oracle) 0.7 ms; tracking median 21 ms, of which the motion clock is about 16 ms when previous/current views are L0<->L1 (quarter overlap) and 63 ms for L0->L0; one-off SIFT calibration on frame index 1: 279 ms alone, 317 ms tracking_ms, 331 ms total, i.e. the whole 333 ms budget before any detector time, so frame 2 is likely skipped in real time. README-live.md claims 'about 10 ms per frame after a one-off calibration of about 200 ms'; commit 2f6f515 reports max round trip 383 ms and calibration 277 ms with no skipped frames after the thread fix; commit cfddb4c reports median round trip 27 ms over HTTP with the oracle. Real detector latency (Ultralytics on the deployment GPU, fixed-asset bundle 'seconds per view') could not be measured here: no weights, no ultralytics install, no bundle.
  Evidence: _work/latency.py on _work/logs/hel_default/local.jsonl (fields detector_ms, tracking_ms, total_ms written by example.py Session.record); _work/calib_hel.py
- **Unit tests** [measured]
  python -B -m unittest tracking.test_tracker tracking.test_placement tracking.test_revisit: Ran 44 tests in 0.85 s, OK, 1 skipped. The skip is ReferenceRegressionTests.test_reproduces_saved_one_observation_experiment, which needs artifacts/drone-scene-analysis/matches-reference-000-001.npz and artifacts/drone-shared-motion/reference-measurements.json from the private preparation project. Tests cover legal camera moves (40-frame l2_top drive, stale-track revisit), freeze/double clock, entry tracks, extent policies, state round trips. No test covers a cluster of three same-class objects with staggered births, which is the bug found above.
  Evidence: test run output; tracking/test_tracker.py line 158
- **ORACLE ceiling of tracker + camera policy** [measured]
  Helsinki (real images, real SIFT calibration, real motion clock, example.predict in-process, organizer ground truth): 0.9777, identical to the 0.978 in commit cfddb4c; only large_tower 0.94, medium_launcher 0.90, medium_plane 0.80 are below 1.0. This number is optimistic because the size prior was fitted on the same 25 frames. Validation (249 frames, 987 participant pseudo-labels, 11 classes, image-free simulator validated at 0.9777 on Helsinki, motion model fitted to the labels with median residual 1.22 px, label-derived clock): deployed default = 0.864 on all labels, 0.827 scoring only directly reviewed labels. One label track (medium-plane-d-058-090) duplicates medium-plane-e box for box and the tracker's 0.7 NMS rightly drops it; without it: 0.903 (reviewed-only 0.872). Same run with extent=detector: 0.932. With the cluster-birth fix in a scratch copy: 0.932 (blend) / 0.961 (detector). Camera held at L0 with a perfect detector: 0.923 blend / 0.954 detector / 0.983 with the fix, so the camera cycle itself costs about 0.02 when the detector is perfect at every level. L1-only sweep 0.886, l2_top 0.626, revisit_every=3 identical to default (never triggers with a perfect detector because no track gets 6 ticks old). Frame-index clock instead of the motion clock: 0.894. Reading against the team's 0.470: about 0.39 to 0.43 of the gap to 1.0 is detector loss, about 0.10 to 0.14 is policy/placement/tracking loss of which about 0.04 is the label artefact. Caveats: the labels are incomplete pseudo-labels (5 classes have no label at all: condor, jammer, small_plane, spacecraft, ta-ta), 360 of 987 boxes are algorithmic projections likely produced with this same motion model (hence the reviewed-only figure), and the label-fitted motion model is more accurate than a two-frame SIFT calibration would be.
  Evidence: _work/replay_inproc.py, sim.py, sim_check_hel.py, sim_val.py, sim_val2.py, sim_final.py, sim_patched.py
- **Q5: is L0 -> L1 top-left -> L0 -> L1 top-right -> L0 the right policy?** [measured]
  It is exactly what is deployed (after two L0 calibration frames), and it is right only if the detector recognises objects at L0. Evidence from a size-gated oracle on the de-duplicated validation labels (object recognised only if its smaller side is >= T delivered px; detector extents): T=0: L0 hold 0.954, deployed cycle 0.932, L1-only top sweep 0.923, l2_top 0.663. T=8: 0.774 / 0.863 / 0.950 / 0.662. T=12: 0.587 / 0.759 / 0.863 / 0.662. T=16: 0.290 / 0.570 / 0.735 / 0.652. T=20: 0.213 / 0.483 / 0.678 / 0.652. T=24: 0.181 / 0.374 / 0.518 / 0.600. For scale, smaller sides in delivered px at L0/L1/L2: small_launcher 5.8/11.5/23, medium_launcher 7.5/15/30, medium_plane 11/22/44, tank 12/24.5/49, large_tower 12/24/48, mine_roller 13.5/27/54, jet_plane 15/30/60, large_launcher 16.6/33/66, helicopter 26/52/105, hangar 46/93/186 (Helsinki-only classes: ta-ta 17 px and jammer 32 px source, i.e. 4 and 8 px at L0). Two mechanisms make the L0 frames harmful once the detector cannot see an object at L0: (1) they are half of all frames and add no detections; (2) every L0 frame is a 'full opportunity' for every track, so an object seen at L1 collects a visible miss per L0 frame and is retired after 3 (visible_misses_before_retirement=3, hard-coded) about 6 frames after it leaves the top band; raising it to 99 lifts the deployed cycle from 0.570 to 0.664 at T=16 and 0.483 to 0.617 at T=20 (20 true tracks were retired this way at T=16). The L1-only sweep also covers the x=1920 seam that the two side crops cut (e.g. the medium launcher at x 1890..1920 is never complete in either side crop). Moving the band down does not help: vertical_fraction 0.25/0.5 score 0.726/0.612 at T=12 for the L1-only sweep, because objects must be caught while entering. l2_top cannot work without revisits: a single native sighting followed by a 30-frame forecast scores about 0.63 to 0.66 even with a perfect detector. Recommendation: choose the policy from the detector's measured recall by delivered pixel size; if small/medium classes need more than about 8 to 10 px, switch to DRONE_OVERVIEW_BETWEEN_SIDES=0 and make miss counting level-aware (or disable it), and keep L0 only for the two calibration frames. The suggestive coincidence that the deployed cycle at T=20 simulates to 0.483 while the team scores 0.470 is not proof.
  Evidence: _work/sim_gate.py, sim_gate2.py, sim_gate3.py; tracking/revisit.py lines 360-371 (miss counting), tracking/workflow.py LevelOneSweep
- **Q6: can trajectory and box size be fixed from ONE recognising frame or are TWO needed?** [measured]
  ONE frame fixes the trajectory, because the motion is not per object: it is the shared scene motion calibrated once from frames 0 and 1. The tracker is built that way (tracker.py docstring 'One-observation object tracker', birth from one detection, forecast = model.box(last observation)). Measured on Helsinki organizer boxes with the real SIFT model, forecast from one exact observation: centre error median/p90 0.7/1.2 px after 1 frame, 2.3/4.2 after 4, 5.0/8.4 after 8, 11.2/17.2 after 16, 14.7/19.9 after 20; IoU>=0.5 in 100%, 100%, 94%, 62%, 41%. On validation reviewed labels (label-fitted model, optimistic): 94% at k=8, 82% at k=16, 68% at k=24, 32% at k=30. Two observations used the naive way (per-object constant velocity) are much worse because the image speed grows from about 53 to 79 px/frame down the frame: 9 px after 4 frames, 33 px after 8 (6% IoU>=0.5), 116 px after 16; with 4 px detector jitter 23/49/138 px. Averaging two warped observations does not help (11.6 vs 11.2 px at k=16). So a second frame is not needed for direction or speed, but a REFRESH is needed within about 8 to 12 frames, and earlier for small boxes: ta-ta (32x17) holds IoU>=0.5 in only 59% at k=8 and 0% at k=16, small_launcher 100% at k=8 and 0% at k=16; an object crosses the frame in about 33 frames, so one sighting at the top cannot carry it to the bottom (this is why l2_top caps at 0.63). The drift is systematic, not random: signed dy error +0.47 px/frame, and fitting a per-object speed factor from two exact observations gives a tight 0.992 (p5 0.984, p95 1.006) on Helsinki, after which the k=16 error falls from 11.3 to 2.4 px and IoU>=0.5 from 59% to 100%. With noisy boxes the same per-object fit hurts (validation labels: 83% -> 64 to 76% at k=16), so the second observation should be used to re-anchor position per object and to estimate ONE global speed correction pooled over all refreshes (the 'refresh' events already log prior_iou; adapt_edges exists but is off). Box SIZE from one frame: width is right (ratio 1.00 at k=16), height is over-predicted by the corner warp (x1.10 at k=8, x1.22 at k=16, x1.64 at k=24 on Helsinki) because objects are upright 3-D renders, not ground-plane patches; centre-warp plus isotropic width scaling gives height ratio 1.02 to 1.05 and median IoU 0.77 vs 0.74 at k=16, without changing the share above 0.5. Size also cannot come from the Helsinki class prior, since instance heading changes the box (large_launcher 150x108 vs 117x66).
  Evidence: tracking/tracker.py, tracking/revisit.py _box/_accept; _work/forecast_error.py, forecast2.py, forecast3.py, motion_stats.py

### Numbers
- Unit tests: 44 run, 43 pass, 1 skipped, 0.85 s (python -B -m unittest tracking.test_tracker tracking.test_placement tracking.test_revisit)
- Helsinki oracle ceiling, real pipeline with images, deployed defaults: mAP50 0.9777 (commit cfddb4c states 0.978); 0 refused camera moves (_work/replay_inproc.py)
- Image-free simulator check on Helsinki: 0.9777 (identical) (_work/sim_check_hel.py)
- Validation oracle ceiling, deployed default (L1/L0 cycle, blend, partials, entry tracks): 0.864 all pseudo-labels; 0.827 reviewed-only; 0.903 / 0.872 after removing the duplicated medium_plane label track (_work/sim_val.py, sim_final.py)
- Validation oracle, extent=detector / prior (raw labels): 0.888 / 0.758 (blend 0.864); large_launcher AP 0.98 vs 0.66 under blend (_work/sim_val.py)
- Validation oracle with cluster-birth fix (scratch copy), raw labels: 0.932 blend, 0.961 detector extents; small_launcher AP 0.64 -> 0.97 (_work/sim_patched.py)
- Validation oracle, camera held at L0: 0.923 blend / 0.954 detector (de-duplicated labels); 0.983 with the fix on raw labels (_work/sim_final.py, sim_patched.py)
- Validation oracle, other policies (blend, de-duplicated): L1-only sweep 0.886; l2_top 0.626; frame-index clock 0.894; revisit_every=3 unchanged (_work/sim_final.py, sim_val2.py)
- Size-gated oracle T=16 delivered px: L0 hold / deployed cycle / L1-only / l2_top: 0.290 / 0.570 / 0.735 / 0.652 (_work/sim_gate2.py)
- Size-gated oracle T=12: L0 hold / deployed / L1-only / l2_top: 0.587 / 0.759 / 0.863 / 0.662 (_work/sim_gate2.py)
- Size-gated oracle T=20: L0 hold / deployed / L1-only / l2_top: 0.213 / 0.483 / 0.678 / 0.652 (_work/sim_gate2.py)
- Deployed cycle with visible_misses_before_retirement 3 vs 99: T=16: 0.570 vs 0.664; T=20: 0.483 vs 0.617; 20 tracks retired by L0 misses at T=16 (_work/sim_gate3.py)
- Helsinki calibration from frames 0,1: 4264 matches, median residual 0.68 px, p90 1.45 px, epipole (1920, -8033), trace(A) -1.54e-3, 279 ms (_work/calib_hel.py)
- Model flow per frame (source px) at x=1920: 53.4 at y=100, 65.4 at y=1080, 78.5 at y=2060; +-12.3 px horizontal at x=200/3640 (_work/calib_hel.py)
- Annotation-measured step per frame: Helsinki dy = 51.25 + 0.01334*y, dx = 0.00709*(x-1924); validation is 1.037x faster (p5 1.021, p95 1.054) (_work/motion_stats.py, clock_stats.py)
- Frozen/doubled frames in validation labels: freeze at 8->9 and 238->239, each followed by a double step (_work/clock_stats.py)
- Model flow inflation implied by trace: x1.038 at tick 24 (Helsinki true x1.037), x1.18 at 100, x1.62 at 248; validation label flow trend x1.011 over 248 frames (_work/drift.py, trace_spread.py, val_trend.py)
- One-observation forecast error on Helsinki (median/p90 px, IoU>=0.5): k=1 0.7/1.2 100%; k=4 2.3/4.2 100%; k=8 5.0/8.4 94%; k=16 11.2/17.2 62%; k=20 14.7/19.9 41% (_work/forecast_error.py)
- Two-observation constant velocity on Helsinki (median px, IoU>=0.5): k=4 9.1 81%; k=8 32.6 6%; k=16 116 0% (_work/forecast_error.py)
- One-observation forecast on validation reviewed labels, IoU>=0.5: 94% k=8, 82% k=16, 68% k=24, 32% k=30 (_work/forecast2.py)
- Predicted/true box height with corner warp: Helsinki 1.10 (k=8), 1.22 (k=16), 1.64 (k=24); validation 1.08, 1.15, 1.23, 1.28 (k=30); width 1.00 (_work/forecast2.py, forecast_error.py)
- Per-object speed factor vs shared model on Helsinki: median 0.992 (p5 0.984, p95 1.006); applying it: k=16 error 11.9 -> 2.4 px, IoU>=0.5 53% -> 100% (_work/forecast3.py)
- Latency on this CPU, oracle detector: total median 33 ms; decode 11 ms; tracking 21 ms (motion clock 16 ms L0<->L1, 63 ms L0->L0); calibration frame 331 ms total (_work/latency.py)
- Thresholds: birth 0.6, update 0.4, association IoU 0.1 or distance 1.0 x diagonal, ambiguity margin 0.1, duplicate IoU 0.7, 3 visible misses, partial confidence x0.8, forecast decay 0 (tracking/revisit.py RevisitConfig, example.py CONFIG)
- Validation pseudo-labels: 987 boxes, 249 frames (1..249), 34 tracks, 11 classes, 3.96 boxes/frame; 606 directly reviewed, 195 projected, 165 algorithmic bottom completion, 14 score-confirmed (src/validation/run_metadata.json, _work/explore_val.py)
- Fixed-asset bundle on validation by family (from commit message, not reproduced): pose_pixels 692 false alarms / 0 hits at L0; tiny-shape 8 / 0; geometry 69 hits / 3 false alarms (commit ac1474e on origin/drone/oscar-live-tracker)

### Gaps
- No validation images in this repo (src/validation has annotations only; frames live in the private preparation project at data/drone/reconstructed-validation), so run_local_eval.py --scene validation cannot run here. The validation ceiling comes from an image-free simulator that uses the real tracker and camera code but a motion model fitted to the labels and a label-derived clock; it reproduces the real pipeline exactly on Helsinki (0.9777).
- faster_coco_eval is not installed, so local_evaluator.score cannot run; I used my own COCO AP@0.50 implementation (101-point, greedy matching, max 100 dets), which reproduced the committed 0.978.
- Validation labels are participant pseudo-labels, incomplete and partly produced by projection with this same motion model (360 of 987 boxes); 5 classes (condor, jammer, small_plane, spacecraft, ta-ta) have no label. Whether those classes exist in the real validation flight is unknown, and if they do they cap the real score independently of everything measured here.
- Unknown which detector and settings produced the team's 0.470; no weights, no ultralytics install, no fixed-asset bundle (DRONE_BUNDLE / DRONE_PROJECT 'drone' package) and no run logs (logs/val-v7 referenced in analyze_run_log.py) are in this repo. Real detector latency and real recall versus pixel size could not be measured.
- Unknown what trace(A) the SIFT calibration fits on validation or evaluation frames 0 and 1. If it is as large as on Helsinki (-1.5e-3) while the true flow is flat (as the validation labels suggest), forecasts and the motion clock degrade late in the flight. Cheap check for whoever has the logs: plot timing.ratio against frame in the DRONE_LOG_DIR JSONL; a downward trend confirms it.
- Docs referenced but absent: docs/drone-box-placement.md, drone/perspective_tracking/ (the vendored source), artifacts/drone-scene-analysis and artifacts/drone-shared-motion (needed by the one skipped test), data/drone/training/score-anchored-validation-v8.
- The size-gated oracle is a model of a detector (hard threshold on the smaller side, no false positives, exact boxes). It ranks policies; it does not predict the absolute score. The scratch cluster-birth patch was only tested with a perfect detector; with real near-duplicate detections it needs a low-overlap guard.
- The shared _work folder is used by other agents; my files are common.py, sim*.py, forecast*.py, calib_hel.py, trace_spread.py, drift.py, latency.py, replay_inproc.py, motion_stats.py, clock_stats.py, val_trend.py, explore_val.py, hel_model.json, val_model.json, logs/hel_default, patched/. Nothing was written to the repo or the exports (checked).

## read:asset-recognizer
BASE = C:\Users\edlun\AppData\Local\Temp\claude\c--Users-edlun-Desktop-lucky-shots-NordicCupAI\bf48d8ae-2b16-4039-be3b-c570527b37ee\scratchpad\drone\codex_drone-training-baseline\drone-flyby

The frozen "asset-precision-20260918-v3" bundle is an OFFLINE, native-resolution (L2 only) recognizer. It was never run live and never scored as mAP. Six proposal branches feed one class-wise NMS: SIFT/RootSIFT geometry, GPU FFT masked NCC on 12 calibrated poses, CPU masked NCC for jet/medium-launcher poses, a Lab colour and silhouette rule for small_launcher, a scratch YOLO proposer, and a 1.61M-parameter dense heatmap CNN used only for small_tower. A 457k-parameter 64x64 crop CNN (verifier v5, 17 classes) gates four of the six branches.

Recorded results:
- 26/26 reference targets with 26 proposals.
- 16/19 development targets with 20 proposals.
- 13/13 with 22 proposals on 8 consumed frames.
- 7/7 with 12 proposals on 4 reserved frames (4 of the 5 extras were background mistakes).

Three facts limit how far those numbers can be trusted:
- **Latency.** Median time is 4.1 to 4.5 s per 960x540 view on an RTX 4080 (p95 6.6 s). That exceeds the 3333 ms request timeout, not just the 333 ms frame interval, so the bundle cannot run live as it is.
- **Same-instance retrieval.** Every successful test shares physical tracks with the calibration crops: the bank holds 31 validation-scene crops, of which 12 are calibrated poses, cut from the same objects. The honest unseen-instance numbers in the receipts are:
  - SIFT only: 2/19.
  - Original hybrid: 5/19.
  - First reserved test: 0/6 (selected) and 1/6 (hybrid).
  - Baseline on the fresh frames: 0/7 with 330 proposals.
  - YOLO26x trained on the reference scene: 0/71, 1/71, 0/71 at L0/L1/L2.
  My own rotated masked-NCC test (reference templates against validation-scene crops) classified 7/31 correctly; only tank transferred.
- **Labels.** The labels are participant pseudo-labels, not organizer ground truth, and both ledger reviewers are the Codex agent. 3 of 7 fresh boxes and 7 of 13 consumed boxes had IoU below 0.50 against a tight re-review. The exported coverage-ledger.json is stale: two tracks (80 boxes) carry classes that were later corrected.

Q3 key fact, confirmed: objects are fixed rendered 3D assets with exactly one model per class. helsinki/run_metadata.json has object_totals = 1 for all 16 classes, and the bank crops show the same tank, launcher and helicopter models in validation. Validation differs in yaw, ground texture and shadow, and tall towers lean with perspective. A per-class 2D template therefore does not transfer without rotation handling and new-scene examples.

The scoreboard was used as an oracle on VALIDATION only, to extract class identity, IoU>=0.5 box brackets, class relabels and bbox scale corrections. One recorded probe score, 0.003808073115003808, equals exactly 5/(101*13). That implies the validation score averages over 13 classes, while the team's annotations cover only 11 to 12.

### Answers
- **How are the recognition branches fused in the frozen v3 detector?** [read-from-code]
  FixedAssetDetector.detect() runs all branches on one delivered image. suppress() then chains merge_proposals() with a fixed priority: SIFT features > calibrated_pixels > pose_pixels > tiny_shape > cnn > heatmap. A lower-priority row is dropped only when it has the SAME class and IoU > 0.35 (nms_iou) with a kept row. The cap is 300 rows.

Scores are not calibrated across families. SIFT scores sit around 0.90 to 0.95, while the heatmap small_tower match scored 0.31.

There is no cross-class suppression. One object can be emitted as two classes, for example jet_plane and condor, and the wrong one is a false positive under the competition scorer.

Inside the SIFT family, fuse_localizations() averages box coordinates within the strong group (score >= 0.80). Weak matches (0.70 to 0.80) can only fill gaps and cannot shift strong boxes. In the offline benchmark the same suppress() is applied again after merging 25 overlapping views.
  Evidence: BASE\solution\drone\scratch_objects\fixed_assets.py (FixedAssetDetector.suppress and detect), BASE\solution\drone\scratch_objects\hybrid.py (merge_proposals iterates [('features',features),('cnn',cnn)], so the second argument wins), BASE\solution\drone\template_matching\features.py (fuse_localizations, FeatureEnsemble.suppress), selection-v3.json settings.
- **Masked NCC pixel matching: what is it, what was it calibrated on, and what was measured?** [measured]
  There are two detectors.

1. CalibratedPixelDetector (calibrated_pixels.py) is a GPU FFT masked NCC.
- It runs on gray-0.5 and a high-pass channel (gray minus Gaussian sigma 2), with response 0.45*gray + 0.55*highpass and final score 0.75*peak + 0.25*max(0, masked BGR colour correlation).
- It uses scales 0.85/1.0/1.15 and angles -8/0/+8 degrees, 2 peaks per template, proposal threshold 0.55 and accept threshold 0.70.
- It uses ONLY the 12 bank templates flagged calibration=true. All 12 are validation-scene crops: jet_plane, helicopter, medium_launcher and large_tower from frame 110; large_launcher and tank from frame 150; tank from frame 190; tank, mine_roller and large_launcher from frame 206; large_tower from frame 230; small_tower from frame 245.

2. PosePixelDetector (pose_pixels.py) is a CPU OpenCV masked matchTemplate.
- It covers calibrated jet_plane and medium_launcher poses only.
- Scales are 0.8 to 1.4 for jet and 0.9/1.0/1.1 for medium_launcher, angle 0 only, threshold 0.6.
- It adds a 'lower 70 percent' jet mask for when a roof hides the tail.

Neither detector is gated by the verifier.

Measured:
- The GPU pixel matcher alone found 5/13 fresh targets with 5 proposals at 0.70.
- It found 6/6 with 6 proposals on its own calibration crops, which is retrieval, not a holdout.

With a rotation tolerance of only +-8 degrees it is an instance retriever for objects already seen in this validation flight.

My cross-scene test used reference templates rotated in 15 degree steps at 3 scales, with masked BGR CCOEFF_NORMED, against the 31 validation-scene bank crops.
- The true-class NCC median was 0.51 at native scale, and only 4/31 reached 0.70.
- Top-1 class accuracy was 7/31 excluding the two tiny classes. Tank was 6/6; every other class was 0 or 1.
- The 40-mask-pixel floor did not change the native-scale result: small_launcher still won 25 of 31 crops (top-1 1/31 over all classes). That is why the 7/31 figure excludes the tiny classes.
  Evidence: BASE\solution\drone\template_matching\calibrated_pixels.py, pose_pixels.py, detector.py; BASE\release\asset-precision-20260918-v3\model\bank\manifest.json (the 12 rows with calibration=true, and pose_calibration / additional_calibration blocks); RESULTS.md 'Precision follow-up'; my script and output at C:\Users\edlun\AppData\Local\Temp\claude\c--Users-edlun-Desktop-lucky-shots-NordicCupAI\bf48d8ae-2b16-4039-be3b-c570527b37ee\scratchpad\drone\_work\cross_scene_ncc_v2.py and cross_scene_ncc_v2.json.
- **SIFT geometry branch: what is it and what was measured?** [read-from-code]
  FeatureEnsemble runs two FeatureDetectors on shared scene keypoints: ('sift','similarity') and ('root','affine'). Two per-template 'calibrated fallback' matchers run on the calibrated poses.

Scene processing:
- The scene is upsampled 2x.
- SIFT_create uses nfeatures=18000, contrastThreshold=0.012, edgeThreshold=15, sigma=1.2.
- A global FLANN index covers all bank descriptors, with knn k=16, a cross-class ratio test at 0.8 and a distance limit of 330 (0.65 for RootSIFT).

Geometric verification:
- RANSAC with estimateAffinePartial2D or estimateAffine2D, 3000 iterations, reprojection threshold 2.5 px.
- At least 3 inliers (4 for affine).
- Scale must be in [0.35, 2.8] relative to pixels_per_source_pixel.
- Anisotropy at most 3, inlier covariance check, feature coverage at least 1.5 percent.

Score = 0.5 + 0.045*min(inliers,8) + 0.12*(1 - dist/limit) - 0.02*error. The output box is the projected foreground hull plus 10 percent padding for masked crops, or the projected annotation rectangle for organizer or calibrated crops.

Thresholds: propose at 0.70; accept unverified at 0.80 or above. Scores of 0.70 to 0.80 need verifier argmax equal to the class with probability >= 0.30 AND aligned masked pixel similarity >= 0.25. Those two values were chosen to rescue one tower at frame 220 (score 0.743, pixel 0.289, CNN 0.366).

Measured, SIFT alone at 0.8 ('Original deterministic features'): 18/26 reference with 18 proposals, and 2/19 validation with 4 proposals.

It is the only branch that is rotation invariant. 44 of the 210 reference templates carry mask_fallback=true (large_tower 16/16, helicopter 12/16, medium_launcher 5/5, plus 2 of 22 ta-ta) and are skipped by SIFT. large_tower and medium_launcher therefore have no reference SIFT templates at all, and those two classes rely on validation crops.
  Evidence: BASE\solution\drone\template_matching\features.py (FeatureSettings, FeatureDetector.detect, appearance_similarity, EnsembleSettings); BASE\release\asset-precision-20260918-v3\README.md (frame 220 tower numbers); RESULTS.md evaluation table; bank manifest mask_fallback counts measured by script.
- **Tiny-silhouette colour rules: what are they and how were they calibrated?** [read-from-code]
  TinyShapeDetector handles small_launcher only. From the 22 organizer reference crops (22x30 source px) it learns:
- a Lab a/b centre and spread of the green paint (mask a<126 and b>140);
- 24 rotated 32x32 silhouettes per crop (15 degree steps);
- the median a/b contrast between the crop border and the object. The code comment reads 'distinct from the pink reference ground'.

At run time it thresholds a Gaussian colour likelihood (>0.35) and takes connected components with width 6s to 24s, height 6s to 28s and area 25 s^2 to 350 s^2 (s = pixels per source pixel), not touching the image border.

score = 0.65*shape IoU (with an aspect penalty) + 0.15*colour + 0.20*contrast similarity. The threshold is 0.65, min contrast is 0.80, and then verifier argmax == small_launcher with probability >= 0.80 is required. The box is the component plus a 5s px pad.

Measured:
- 3/3 reference small_launchers matched, with IoU only 0.61 to 0.64.
- 4 validation small_launchers matched, with IoU 0.587 to 0.646 and scores 0.68 to 0.74.

IoU is barely above 0.50, so the box size convention is a risk.

The contrast prior is fitted to the single reference ground colour, which is an overfitting risk on a new scene.
  Evidence: BASE\solution\drone\template_matching\tiny_shapes.py; comparison.json runs.selected_development matches (small_launcher rows); selection-v3.json tiny_threshold 0.65, tiny_min_contrast 0.8, tiny_verifier_threshold 0.8.
- **Scratch CNN proposal branch: what is it, what was it trained on, and what did it contribute?** [read-from-code]
  ScratchDetector wraps an Ultralytics YOLO checkpoint (weights.pt, 44,067,410 bytes as an LFS pointer). train.py defaults to the architecture YAML 'yolo26m.yaml' with random initialisation and no pretrained weights. It predicts at imgsz 960 with conf 0.6 and NMS IoU 0.35.

Training data (prepare.py):
- 768 synthetic 640x640 copy-paste images plus 96 synthetic dev images.
- Bank cutouts are alpha-blended (3x3 Gaussian on the mask) onto 640x640 reference backgrounds. Backgrounds exclude all GT boxes with a 24 px guard, 8 per reference frame.
- 3 to 9 objects per image, every 8th image empty.
- Rotation -180 to 180 degrees, scale log-uniform 0.4 to 1.45, gain 0.85 to 1.15.
- Later additions: native 960x540 reference tiles (native_data.py), detector-mined hard negatives (mine_negatives.py), and a context run on early-validation views from frames <= 90 (prepare_context.py; 96 epochs, batch 8, imgsz 960, lr 0.0005, degrees 15).
- The context continuation was REJECTED at epoch 24 (1/19 development targets). 'The selected runtime retains the earlier proposal CNNs.' Which run weights.pt is cannot be told from the export.

Gate: background_probability < 0.5 AND verifier P(proposal's class) >= 0.95.

Measured contribution:
- precision-v2 with both CNN proposal branches disabled: 23/26 reference, 14/19 validation.
- v3 with them restored: 26/26 and 16/19. The CNN branches add 3 reference and 2 validation targets.
- In v7 (loose gates) the CNN branches took proposals on 8 frames from 26 (geometry/pixel/colour subset, 12/13) to 233 (12/13): about 207 extra proposals and no extra recall.
- Permissive compact-CNN combination: 26/26 with 392 proposals, 17/19 with 739.
  Evidence: BASE\solution\drone\scratch_objects\detector.py, train.py, prepare.py, native_data.py, prepare_context.py, run_context_mypc.py, run_followup_mypc.py; RESULTS.md ('CNN experiments' and 'Evaluation contract' tables); release README.md (precision-v2 numbers).
- **Dense tower CNN: what is it, what was it trained on, and how is it used?** [read-from-code]
  AssetNet (asset_heatmap/model.py) is a CenterNet-style fully convolutional net, randomly initialised.
- Encoder: stem block(3,32,stride 2), s4 block(32,64,2), s8 block(64,128,2), s16 block(128,192,2), then two upsampling fusion blocks to stride 4.
- Heads: a 1x1 heat head with 16 classes (bias -2.19) and a 4-channel box head (log w/32, log h/32, dx, dy).
- 1.61M parameters; heatmap.pt is 6,469,458 bytes.
- Loss: focal heat plus 2x smooth-L1. Partial objects are marked -1 (ignored).

Training (train.py):
- 384x384 patches, batch 8, of which 4 are real tiles and 4 are synthetic.
- AdamW lr 0.001 cosine, AMP, 1500 steps in about 2 minutes (v1, the selected one). v2 and v3 used 3000 steps (v3 about 201 s).

Synthetic scenes (data.py, class Scenes): 1 to 4 bank cutouts on an empty real tile, scale 0.5 to 1.4 with aspect 0.7 to 1.3, rotation -180 to 180, max 200 px, brightness jitter, flips. Masks come from the bank, or grabCut when mask_fallback is set. small_launcher uses the Lab colour mask.

Inference: peaks where maxpool3 equals the probability and probability >= 0.3, top 100, NMS 0.35.

In v3 only class 'small_tower' is kept (heatmap_classes), heatmap_reclassify=False, and it needs verifier argmax == small_tower with P >= 0.40. Its visible effect in the receipts is the small_tower match at frame 160 with score 0.3095 and IoU 0.595.

RESULTS.md says v1 'complements other recognizers' for small towers, and that v2 and v3 gave 'no useful combined improvement'.
  Evidence: BASE\solution\drone\asset_heatmap\model.py, data.py, train.py, detector.py; RESULTS.md CNN table; comparison.json frame 160 match.
- **Crop verifier v5: what exactly is it (architecture, input, classes, training data, synthetic composition, thresholds)?** [read-from-code]
  **Architecture.** PatchCNN (scratch_objects/patch_cnn.py):
- 5 blocks of 3x3 conv + BatchNorm + SiLU, with channels 24/32/64/96/128 and strides 1/2/2/2/2.
- Flatten 128*4*4 = 2048, then Linear 128, SiLU, Dropout 0.15, Linear 17.
- About 457k parameters by my count; verifier.pt is 1,840,190 bytes.
- Classes: 'background' plus the 16 sorted class names; it is a softmax.

**Input.** normalize_crop():
- The proposal box is cropped from the delivered image and resized so its LONGER side is 48 px (INTER_AREA when shrinking).
- It is centred on a 64x64 canvas filled with 127, in BGR scaled to 0..1.
- Absolute object size and surrounding context are therefore discarded.

**Training.** asset_heatmap/patch_train.py with flags --proposal-crops --wide-negatives --neutral-tata:
- Random init, 800 steps, batch 128, exactly 50 percent background.
- AdamW lr 0.002, weight decay 0.0005, cosine to 1e-4, label smoothing 0.02.
- 23 s on an RTX 4080.

**Positives.**
- With probability 0.6 the sample is drawn from the class's calibrated validation-scene poses when the class has any. Only 8 classes do.
- 20 percent are the real crop, re-boxed by foreground_box(). Its margin is uniform 0 to 0.2 for 80 percent of samples, otherwise 0.2 to 0.5, with +-4 percent jitter.
- 80 percent are synthetic:
  - The masked foreground is scaled to 25 to 49 px and rotated -180 to 180 degrees.
  - Colour jitter: saturation x0.3 to 1.5, gamma 0.6 to 1.6, gain 0.45 to 1.5, offset +-0.06.
  - It is alpha-blended with a hard mask (no feathering) onto a random empty real tile crop (35 to 200 px square, resized to 64).
  - It is then re-cropped with the same foreground_box().

**Negatives.** 'wide': log-uniform size 8 to 240 px, aspect exp(+-0.7), cut from empty tiles. The empty tiles are helsinki reference tiles plus reviewed-empty early validation views (frames <= 90). NO detector false positives were mined for v5; the release README says 'No new training images or negative tiles were added'.

**Augmentation.** Horizontal flip 50 percent, brightness x0.8 to 1.2, offset +-10. The neutral-tata flag keeps only ta-ta mask pixels with HSV saturation < 70.

**Thresholds.**
- 0.95: scratch-CNN proposals (plus background P < 0.5).
- 0.80: tiny silhouettes (argmax must equal the class).
- 0.40: dense small_tower proposals (argmax must equal the class).
- 0.30: weak SIFT matches at 0.70 to 0.80 (plus aligned pixel similarity >= 0.25).
- Strong SIFT (>= 0.80), pose_pixels and calibrated_pixels are NEVER verified.
  Evidence: BASE\solution\drone\scratch_objects\patch_cnn.py (normalize_crop, PatchCNN), BASE\solution\drone\asset_heatmap\patch_train.py (foreground_box, patch, main receipt fields), BASE\solution\drone\asset_heatmap\data.py (Scenes.backgrounds / calibrated), BASE\solution\drone\scratch_objects\hybrid.py (PatchVerifier.annotate), fixed_assets.py detect(), selection-v3.json, release README.md.
- **Verifier failure modes v3 to v5, and what would a better verifier need (Q1)?** [inferred]
  Recorded history:
- v3 (800 steps, 35 s): 'synthetic object size mismatched detector proposal crops'. Training objects were 25 to 49 px inside a full 64 canvas, while at inference the crop is renormalised to 48 px.
- v4 (37 s): the crop normalisation was fixed and development recall improved, but proposals were 'excessive'. The v3-dense + v4-verifier combination gave 392 reference and 739 validation proposals at permissive thresholds.
- v5 (23 s): negative crop sizes and aspects were matched to proposals, and orange background was removed from gray ta-ta masks.

Remaining v5 errors on frozen tests:
- Reserved 4 frames: 4 background mistakes (two vegetation regions, pale material stacks, a roof window).
- 8 consumed frames: 6 background mistakes.

The verifier's own confusion matrix and ROC were never reported ('Training-batch accuracy is not detection accuracy'), so its standalone accuracy is unknown.

Structural weaknesses:
1. Size normalisation throws away absolute scale. Scale is a real class cue here (small, medium and large launcher, tower and plane).
2. Negatives are random empty-tile crops, not the proposer's real false positives.
3. Thresholds (0.95/0.80/0.40/0.30) were hand-set on at most 12 frames. One pair (0.30/0.25) was set from a single object.
4. 800 steps trained in 23 seconds from random init, against one reference instance per class plus 12 validation poses for 8 classes.
5. It is only a veto. It cannot fix boxes or classes (heatmap_reclassify=False), and it is bypassed for the highest-volume deterministic branches.
6. It was only used at native resolution.

A better verifier needs:
- Hard negatives mined by running the actual proposer over the 249 reconstructed validation frames, inside reviewed regions only (frame 5 in full, plus the top 540 px band every 4th frame), with everything else ignored.
- Scale kept as an input: a fixed source-pixel window, or an extra scale channel or scalar, plus a context margin that includes the shadow.
- Training at the delivered pixel size for L1 and L0, by rendering at native resolution and then INTER_AREA downsampling by 2 and 4.
- Full-rotation synthetic positives on validation-like backgrounds, with feathered edges and blur. Towers need real off-nadir examples, because a 2D rotation cannot make the perspective lean.
- Thresholds or temperature fitted with a leave-one-TRACK-out split, never frame splits.
- A 17-way output that may also RELABEL, so it resolves the cross-class duplicates that the fusion leaves in.
- A latency budget of a few ms per crop batch. The current PatchCNN is already that small. The slow part of the pipeline is the matchers, not the verifier.
  Evidence: RESULTS.md 'CNN experiments' table and 'Completed precision v3 result'; release README.md 'Bounded experiments'; patch_train.py; fixed_assets.py; validation/annotations/README.md (reviewed-region contract).
- **What recall and proposal counts are recorded per configuration and per class?** [measured]
  All numbers come from the exhaustive 25-view native scan, matched at class-aware IoU >= 0.5 against participant labels.

Frozen v3:
- Reference holdout frames 5/12/20: 26/26 with 26 proposals. Per class: condor 1, helicopter 2, jammer 1, jet_plane 3, large_launcher 2, large_tower 2, small_launcher 3, small_plane 1, small_tower 2, spacecraft 3, ta-ta 3, tank 3. The reference holdout has no hangar, medium_launcher, medium_plane or mine_roller.
- Validation development frames 100/127/140/160: 16/19 with 20 proposals and 4 unmatched. Per class: hangar 2, helicopter 2, jet_plane 2, large_launcher 2, large_tower 1, medium_launcher 1, small_launcher 4, small_tower 1, tank 1. The classes of the 3 misses are not in the export.
- 8 consumed frames: 13/13 with 22 proposals. Per class: large_launcher 4, large_tower 2, mine_roller 2, small_tower 2, tank 3. The 9 unpaired proposals were 3 real unlabelled objects and 6 background.
- 4 reserved frames (187/201/224/247): 7/7 with 12 proposals. Per class: tank 2, large_launcher 2, mine_roller 1, large_tower 1, small_tower 1. The 5 extras were 1 real mine roller and 4 mistakes. Against the ORIGINAL uncorrected boxes the same predictions score only 4/7. The geometry/pixel/shape subset scores 7/7 with 10 proposals.

History:
- SIFT only: 18/26 (18 proposals) and 2/19 (4).
- Original hybrid: 26/26 (37) and 5/19 (39).
- v6: 26/26 (29) and 16/19 (33).
- v7: 26/26 (29) and 16/19 (23); on the 8 frames 12/13 with 233 proposals (8/13 on original boxes); subset 12/13 with 26.
- Permissive compact-CNN combination: 26/26 (392) and 17/19 (739).
- precision-v2: 23/26 and 14/19.
- Baseline hybrid: 4/13 with 799 proposals on the 8 frames, and 0/7 with 330 on the 4 frames.
- First reserved test (frames 190/206/230/245, unseen tracks): selected 0/6, hybrid 1/6.

Matched IoU on non-reference frames: median 0.886, with 25 percent below 0.70 and 14 percent below 0.60. hangar, small_launcher and small_tower sit at 0.57 to 0.65.

Per-BRANCH proposal counts, beyond the subset figures above, are not in the export.

Only 5 classes (tank, mine_roller, large_launcher, large_tower, small_tower) were ever tested on late frames.
  Evidence: BASE\release\asset-precision-20260918-v3\comparison.json (runs.* aggregates and matches, parsed by script), RESULTS.md, release README.md, data-use-ledger.json.
- **What latency is recorded?** [measured]
  v3 full detector, per 960x540 native view, RTX 4080 'mypc' while it also hosted a CPU-heavy simulation:
- Development run: median 4.435 s, p95 6.600 s, max 7.096 s.
- Consumed run: median 4.491 s, p95 6.630 s, max 8.236 s.
- Fresh run: median 4.093 s, p95 4.288 s, max 4.538 s.

Other timings:
- Bundle CLI smoke on CPU: 6.37 s for one call on a 55x43 px tank crop. This includes first-call template-variant and FLANN index building.
- Baseline hybrid on a laptop CPU: median 2.47 s, p95 3.39 s, max 4.91 s per view.
- Earlier YOLO26x detector: p50 20.73 ms, p95 21.12 ms, p99 21.22 ms, including PNG decode, batch 1, RTX 4080.

Competition budget: a frame every 333 ms and a hard response_timeout_ms of 3333. The v3 median per view exceeds even the hard timeout, so it cannot answer a single live request as built.

No per-branch timing is recorded. From the code, the heavy parts are:
- SIFT with 18000 features on a 2x upsampled image, plus FLANN k=16, plus 4 RANSAC matchers.
- CPU matchTemplate over calibrated jet and medium-launcher variants.
- FFT NCC over 12 poses times 9 variants.
  Evidence: comparison.json runs.*.view_seconds; bundle-smoke.json 'seconds'; BASE\training\REPORT-2026-09-17.md; C:\Users\edlun\Desktop\lucky shots\NordicCupAI\drone-flyby\README.md lines 149-150 and 379-391.
- **What was learned about the validation sequence through scoreboard probing?** [inferred]
  All probes are VALIDATION-only 'plans': JSON with predictions_by_frame and a fixed camera target [1920,1080], replayed frame by frame. Every builder writes competition_evaluation=false and 'The live runner refuses a nonzero evaluation count'.

Techniques:
1. **Class identity by group testing.** score_probe_experiment.py proves offline that 4 binary class-group queries identify 1 of 16 classes for a known box.
2. **Blind object discovery.**
   - discovery/planner.py builds box lattices from reference size priors: scales 0.8/1/1.25, both orientations, and 100 sampled hypotheses per class per frame. New objects are assumed to enter from the top or sides.
   - discovery/engine.py (class Search) adaptively bisects positive groups, with a hard cap of 100 runs.
3. **Box geometry.**
   - oracle.scan_boxes ranks 100 boxes that sweep one edge.
   - decode_rank reads the first matching rank from the score ratio.
   - bracket_from_rank gives an IoU=0.5 bracket.
   - fit_box least-squares solves the hidden box, and integer_candidates enumerates the consistent integer boxes.
4. **Per-track probes** (build_track_validation_probes.py). Confidence is set by provenance: 1.0 score_confirmed_match, 0.98 score_anchored_projective, 0.90 reviewed_positive, 0.80 algorithmic_shared_motion. A positive score confirms the class plus IoU >= 0.5 on some frames.
5. **Class hypotheses** (build_track_class_probe.py, build_relabelled_validation_probe.py, apply_validation_class_correction.py). RESULTS.md mentions jet/condor and medium/small-launcher identity corrections.
6. **BBox variant probes** ('scale:F' and 'pad:L,T,R,B'), then apply_class_bbox_scale_correction.py applies a class-wide centred scale with before and after validation scores.
7. **Other probes.**
   - Frame-range probes, union probes and single-class probes.
   - Three class-disjoint shards whose scores add up to the full mAP (prove_score_shard_additivity.py).
   - Frames 1 to 4: a 4-box large_launcher probe scored 0.003808073115003808, and the mine_roller control scored 0.0.
8. **repair_score_anchored_tracks.py** projects each score-confirmed anchor through the calibrated camera motion to rebuild whole tracks (datasets score-anchored-validation v2 to v7).

Decoding the one recorded score: 0.003808073115003808 equals 5/(101*13) to floating-point precision. That is 5 of 101 recall points at precision 1 for one class, averaged over 13 classes. The validation scorer therefore averages over 13 classes. If all 4 boxes were true positives, large_launcher has between 81 and 100 ground-truth boxes in validation.

Only 14 score-confirmed seeds are recorded in the exported ledger: large_launcher frames 140 to 146 and frame 5, helicopter frames 45 and 66, large_tower frame 45, jet_plane frame 66, and medium_plane frame 66 twice. Every other probe score lives in a private artifacts/ folder that is not in this export.
  Evidence: BASE\solution\drone\oracle.py, score_probe_experiment.py, prove_score_shard_additivity.py, discovery\engine.py, discovery\planner.py, build_*_validation_probes.py, apply_class_bbox_scale_correction.py, apply_validation_class_correction.py, apply_initial_frame_extension.py (lines 62, 74, 162), repair_score_anchored_tracks.py; python check 5/(101*13) == 0.003808073115003808; BASE\local_evaluator.py score() averages over classes present in ground truth only.
- **What are the validation annotations (coverage-ledger.json) and how trustworthy are they?** [measured]
  **Content.**
- 1009 candidates: 995 'manual_track' boxes over 230 frames, plus 14 score_confirmed_seed rows that manual boxes already cover.
- Frames 5 to 249 only. Frames 1 to 4 are incomplete reconstructions. Frames 10 to 14 and 172 to 181 have no box.
- 37 track files over 11 classes. Box counts: medium_plane 206, large_launcher 130, tank 123, helicopter 104, small_launcher 99, jet_plane 86, large_tower 86, hangar 76, condor 48, small_tower 22, mine_roller 15.
- No jammer, medium_launcher, small_plane, spacecraft or ta-ta.

**Method.**
- A manual seed was compared to reference crops, CSRT-tracked, and then 'reviewed'. The reviewer fields are 'codex-manual-review' and 'codex-small-object-review', so the reviewer was the Codex agent, not a human.
- The review covered frame 5 in full, and only the TOP 540 px band every 4th frame (69 sheets), under a user-supplied 'objects enter from the top' assumption.
- safe_for_full_frame_negative_training = false.

**Measured problems.**
- The 7 'original' boxes in fresh-annotation-review.json are byte-identical to the ledger rows. Their IoU against the tight re-review is 0.33, 0.37, 0.44, 0.55, 0.57, 0.62 and 0.78. Three of 7 are below 0.50, and tank centres are off by 20 px in x. RESULTS.md reports 7 of 13 below 0.50 on the 8 consumed frames.
- The bank's corrected calibration crops overlap the ledger boxes by only IoU 0.31 to 0.70 on late tracks (tank-c 0.36 and 0.31, large-tower-c 0.45, small-tower-b 0.46).
- Two whole tracks carry the wrong class in the exported ledger:
  - 'condor-103-150' (48 boxes) is the object the bank calls jet_plane (pose-jet_plane-000110, IoU 1.00).
  - 'small-launcher-a-092-123' (32 boxes) is medium_launcher (pose-medium_launcher-000110, IoU 1.00).
  That is 80 of 995 boxes, or 8 percent.
- The corrected set 'score-anchored-validation-v7' is NOT in the export.
- Which convention matches organizer ground truth is unknown. Organizer boxes appear loose: the reference helicopter box is 116x94, with a mask foreground fraction of 0.03.

**Verdict.**
- The ledger is good for locating objects and cutting crops.
- It is not reliable as IoU-0.5 ground truth.
- It is not a complete negative set. The server evaluates 13 classes, while 11 to 12 are annotated.
  Evidence: BASE\validation\annotations\coverage-ledger.json and README.md (parsed by script), BASE\release\asset-precision-20260918-v3\fresh-annotation-review.json, bank manifest.json cross-check script, BASE\training\snapshots\manual-method.md, BASE\STABLE-SNAPSHOT-2026-09-18.md, BASE\solution\drone\search_missing_classes.py (CLASSES = small_plane, ta-ta, jammer, spacecraft 'absent from validation v4').
- **Q1: should a vision model verify what the rule-based/template expert system proposes, and what does the evidence say?** [inferred]
  Yes. The repo already shows the effect. Tightening the verifier gates (v7 to v3) cut proposals from 233 to 22 on the same 8 frames while recall went from 12/13 to 13/13. On the reserved frames v3 produced 12 proposals against 330 for the baseline. Under a scorer with no NMS, where every duplicate is a false positive, that precision gain is where the mAP is.

The existing verifier is not the one to ship:
- It is a 23-second, 457k-parameter, random-init net trained on one reference instance per class plus 12 same-track poses.
- It discards object scale, has no mined hard negatives, and uses hand-set thresholds.
- It is bypassed by the three highest-volume deterministic branches.
- It still passes roofs, vegetation and material stacks.

The proposers it guards are instance retrievers that will not fire on unseen evaluation objects (0/6, 1/6, 2/19, 5/19, 0/7). Proposal RECALL on new instances is therefore the binding problem, not verification.

Recommended shape:
- A cheap high-recall proposer that works on a new scene. That means a learned detector trained on massive synthetic data with validation-scene backgrounds, or a colour or objectness blob proposer, not calibrated NCC.
- One shared 17-way crop verifier with:
  - scale as an input;
  - a context margin;
  - training at the L1 and L2 delivered pixel sizes;
  - hard negatives mined from the 249 reconstructed validation frames, reviewed regions only;
  - leave-one-track-out calibration;
  - the right to relabel as well as veto.

Use the tracker to call the verifier once per track at the best zoom rather than on every frame. The verifier costs milliseconds; the v3 matchers cost 4 s per view.
  Evidence: comparison.json, RESULTS.md, release README.md, patch_cnn.py, patch_train.py, fixed_assets.py, local README no-NMS rule.
- **Q2: which object PIXEL SIZE in the delivered image is recognisable?** [measured]
  Positive evidence exists only at native scale (1 source px = 1 image px). v3 was evaluated exclusively on 960x540 native views.

At native scale it recognises:
- objects whose longer side is about 45 to 190 px. tank about 50, mine_roller 47 to 60, small_tower 33x52 to 57x59, large_tower 57x65, large_launcher 108 to 150, helicopter 116, hangar about 190. IoU is 0.83 to 0.98 against reviewed boxes.
- the 20 to 32 px classes, but only through special handling and with weak localisation:
  - small_launcher 22x30 px via the colour rule, IoU 0.59 to 0.65.
  - ta-ta 32x17, scores 0.72 to 0.84.
  - medium_launcher 30x45, score 0.63.

Median organizer ground-truth delivered sizes (L2 / L1 / L0, in px):
- small_launcher: 22x30 / 11x15 / 5.5x7.5
- ta-ta: 32x17 / 16x8.5 / 8x4
- jammer: 32x44 / 16x22 / 8x11
- medium_launcher: 47x45 / 23.5x22.5 / 12x11
- spacecraft: 44x49 / 22x24.5 / 11x12
- small_plane: 43x50 / 21.5x25 / 11x12.5
- tank: 50x47 / 25x23.5 / 12.5x12
- medium_plane: 56x49 / 28x24.5 / 14x12
- mine_roller: 53x60 / 26.5x30 / 13x15
- small_tower: 57x59 / 28.5x29.5 / 14x15
- large_tower: 62x64 / 31x32 / 16x16
- jet_plane: 77x82 / 38.5x41 / 19x20.5
- helicopter: 116x94 / 58x47 / 29x24
- large_launcher: 150x108 / 75x54 / 37.5x27
- condor: 172x168 / 86x84 / 43x42
- hangar: 188x129 / 94x64.5 / 47x32

The repo's own matcher drops a template variant when a side is below 5 px or it has fewer than 32 mask pixels.
- At L0 that removes small_launcher (about 11 foreground px) and ta-ta (about 24 foreground px, 4 px side).
- At L1 all 16 classes pass.

The CNN branches were trained on synthetic object scales of 0.4 to 1.45 (YOLO) and 0.5 to 1.4 (heatmap) of native. L1 is at the edge of that distribution and L0 is outside it.

A YOLO26x fitted to the reference scene detects even 5.5x7.5 px small_launchers (25/25) and 8x4 px ta-ta (25/25) at L0 ON ITS TRAINING FRAMES. It got 0/71, 1/71 and 0/71 on validation at L0/L1/L2. Blobs are findable at L0, but class recognition at that size is memorisation.

My cross-scene NCC test, with a 40-mask-pixel floor, gave:
- True-class score median 0.51 (L2), 0.58 (L1), 0.69 (L0).
- Top-1 accuracy stayed at 7/31, 8/31 and 8/31. Lower resolution makes everything correlate, so thresholds pass more false matches.
- Without that floor, a tiny-mask small_launcher template beat the true class on 31/31 crops at native, L1 and L0.

Practical reading:
- Treat about 25 to 30 px on the longer side as the floor for class verification.
- L1 is plausible for mine_roller, towers, jet_plane and everything larger (26 px and up).
- tank, medium_plane, spacecraft, small_plane and medium_launcher (22 to 28 px at L1) are borderline and unmeasured.
- small_launcher, ta-ta and jammer need L2.
- L0 is only good for finding candidate blobs of the 7 largest classes (16 px and up).

No measurement of v3 at L1 or L0 exists in this repo.
  Evidence: comparison.json per-class matches; helsinki annotations measured by script (C:\...\scratchpad\drone\drone_oscar-live-tracker\drone-flyby\src\helsinki\annotations); template_matching/detector.py Settings.min_mask_pixels=32 and min side 5; prepare.py scale range; asset_heatmap/data.py factor range; BASE\training\receipts\mypc-detector-evaluation.json per-class per-zoom reference_training_fit; BASE\training\EVALUATION-2026-09-17.md; _work\cross_scene_ncc_v2.json.
- **Q3: one expert per class or one shared model, and what is the overfitting risk? Are objects fixed rendered assets with one or few appearances per class?** [measured]
  **Key fact: yes.** helsinki/run_metadata.json records capture.altitude_m = 600, num_frames = 25, total_objects = 16 and object_totals = 1 for every class. The reference scene has exactly one physical object per class, and at most 1 instance per class per frame over 25 frames.

The contact sheet of the bank's 210 reference and 31 validation crops shows the SAME 3D models in the validation scene: identical tank, launcher truck, helicopter, hangar, jet, mine roller and tower models.

The appearance is NOT one fixed 2D sprite:
- Validation instances have different yaw. Tank tracks a to e sit at several headings, and hangars and launchers are rotated.
- Ground differs: grass, sand and paved, against the reference's single patch per object.
- Shadows differ, and the objects are 3D. Tall towers lean with off-nadir perspective: a large_tower crop is 57x101 in validation against 62x64 in the reference, and a small_tower crop is 31x47 against 57x59.
- Box size drifts 7 to 17 percent along a single reference track.
- Validation has several instances per class: 5 tank tracks, 4 to 5 large_launcher, 3 helicopter, 5 medium_plane, 2 hangar and 3 large_tower.

Per-class template experts are natural in the 'one model per class' sense. v3 is effectively that: a colour expert for small_launcher, a pose expert for jet and medium_launcher, a heatmap expert for small_tower.

The evidence still says per-class experts overfit badly here:
- Every branch that worked did so through same-instance validation crops. Every unseen-instance test failed: 2/19, 5/19, 0/6, 1/6, 0/7, YOLO 0/71, and my rotated-NCC transfer 7/31 with only tank succeeding.
- Per-class data is tiny and single-instance: mine_roller 2 reference crops, hangar 3, medium_plane 4, medium_launcher 5. With condor relabelled, 5 classes have ZERO validation-scene examples: condor, jammer, small_plane, spacecraft and ta-ta. The small_launcher expert's contrast prior is literally the pink reference ground.
- 16 experts mean 16 or more thresholds tuned on a handful of frames. The 0.30/0.25 pair was fitted to one tower.
- Independent experts have no cross-class competition. The v3 fusion already cannot suppress a jet_plane and condor double report.

Recommendation:
- One shared multi-class model, a detector or a 17-way verifier, trained on heavy synthetic composites of the cutouts: full rotation, validation-like backgrounds, blur and edge feathering, and scale matched to delivered pixels.
- Keep hand rules only where there is a physical class-specific cue, such as the small_launcher colour blob, the size prior per class, and the tower lean as a function of image position.
- Validate by leave-one-TRACK-out over the 37 validation tracks, never by frame split. The five zero-example classes stay untestable on real data.
  Evidence: C:\...\scratchpad\drone\drone_oscar-live-tracker\drone-flyby\src\helsinki\run_metadata.json and annotations (script: max 1 instance per class per frame); contact sheets C:\...\scratchpad\drone\_work\bank_sheet_a.jpg and bank_sheet_b.jpg built from BASE\release\asset-precision-20260918-v3\model\bank; bank manifest class/source counts; BASE\training\EVALUATION-2026-09-17.md ('The training scene contains only one physical object per class'); coverage-ledger track list; tiny_shapes.py comment; cross-check of ledger 'condor-103-150' against bank pose-jet_plane-000110.
- **Will the v3 recognizer transfer to the one-shot evaluation sequence?** [inferred]
  Probably not as built. The evaluation is a different 250-frame sequence, and whether it shares the scene or object instances with validation is unknown. There are three independent blockers.

1. **Speed.** 4.1 to 4.5 s per native view, against a 3333 ms hard timeout and a 333 ms frame interval. It was also only measured with 25 views per frame, which is not a legal camera policy.

2. **Instance dependence.** 12 of the 12 calibrated NCC poses and 31 of the 241 bank crops are cut from validation tracks. All four holdout sets 'share physical tracks with calibration' (the data-use ledger's own words). 41 validation frames were consumed for training and calibration. The only unseen-track results are failures:
   - Selected detector 0/6 and hybrid 1/6 on frames 190 to 245, before those frames were absorbed into calibration.
   - SIFT only 2/19, hybrid 5/19, baseline 0/7.
   RESULTS.md itself says the result 'does not establish ... generalization to a new scene, or readiness within a live flight budget'.

3. **Coverage.**
   - Only 5 classes were ever tested on late frames.
   - The reference holdout exercised 12 of the 16 classes. hangar, medium_launcher, medium_plane and mine_roller were checked only on development frames or same-track late frames.
   - 5 classes have no validation-scene example at all, so their behaviour on a new scene is untested.

What is reusable:
- The 241-crop masked cutout bank, as a source for synthetic data.
- The SIFT geometry code for large textured objects.
- The coordinate adapters predict() and tracking_detections().
- The offline evaluation harness.
- The data-use discipline.
  Evidence: data-use-ledger.json, comparison.json limitations and first_reserved_test, RESULTS.md final paragraph, bank manifest pose_calibration/additional_calibration, README timing rules.
- **What runtime adapters exist for plugging the recognizer into a live policy?** [read-from-code]
  TemplateDetector is the base class of every branch and of FixedAssetDetector. It exposes:
- detect(image, pixels_per_source_pixel), with 0.25, 0.5 or 1 for L0, L1 or L2. It returns boxes in delivered-image pixels.
- predict(image, source_region, source_size=(3840,2160)). It returns organizer-style dicts {object_id, confidence, bbox normalised to the full frame} and rejects anisotropic views.
- tracking_detections(image, view). It returns drone.perspective_tracking.revisit.Detection objects for DroneTrackingWorkflow.process().

Loading: load_bundle(manifest, device) verifies SHA-256 of weights, verifier, heatmap and bank/manifest.json, then builds FixedAssetDetector. The CLI is python -m drone.scratch_objects --bundle ... --image ... --source-region x1 y1 x2 y2.

Limits:
- Only fully visible objects are handled. The SIFT branch drops projections that leave the image, and TinyShape drops components touching the border.
- Objects cut by the view edge are a declared limitation. That matters for L1 and L2 views, where many objects straddle the crop.
- The variant cache keeps 3 scales.
- The first call pays template-variant and FLANN index construction.
  Evidence: BASE\solution\drone\template_matching\detector.py (detect, tracking_detections, predict, variants cache), BASE\solution\drone\scratch_objects\bundle.py, __main__.py, RESULTS.md 'Run the frozen bundle'.

### Numbers
- v3 reference holdout (frames 5/12/20) matched / proposals: 26/26, 26 proposals, 0 unmatched (release/asset-precision-20260918-v3/comparison.json runs.selected_development.aggregate.reference)
- v3 validation development (frames 100/127/140/160) matched / proposals: 16/19, 20 proposals, 4 unmatched (comparison.json runs.selected_development.aggregate.validation)
- v3 on 8 consumed frames (183,198,204,212,220,228,241,249): 13/13, 22 proposals; 9 unpaired = 3 real unlabelled + 6 background (comparison.json runs.selected_consumed; RESULTS.md)
- v3 on 4 reserved frames (187,201,224,247), reviewed boxes: 7/7, 12 proposals; 5 extras = 1 real mine roller + 4 mistakes (comparison.json runs.selected_fresh; RESULTS.md)
- v3 on the same 4 frames against ORIGINAL participant boxes: 4/7 (comparison.json runs.selected_fresh_original_labels)
- Baseline (original hybrid) on the 4 reserved frames: 0/7 with 330 proposals (frame 224: 199, frame 247: 93) (comparison.json runs.baseline_fresh)
- v7 on 8 frames: full vs geometry/pixel/colour subset: 12/13 with 233 proposals vs 12/13 with 26 proposals; hybrid 4/13 with 799 (RESULTS.md 'Precision follow-up')
- First reserved test on unseen tracks (frames 190,206,230,245): selected 0/6, original hybrid 1/6 (comparison.json first_reserved_test; data-use-ledger.json)
- SIFT-only ('original deterministic features'): 18/26 reference (18 proposals), 2/19 validation (4 proposals) (RESULTS.md evaluation table)
- Original hybrid: 26/26 (37 proposals), 5/19 (39 proposals) (RESULTS.md evaluation table)
- precision-v2 (both CNN proposal branches disabled): 23/26 reference, 14/19 validation (release/asset-precision-20260918-v3/README.md)
- Permissive compact-CNN combination: 26/26 with 392 proposals, 17/19 with 739 proposals (RESULTS.md)
- GPU calibrated pixel matcher alone at 0.70: 5/13 fresh targets with 5 proposals; 6/6 on its own calibration crops (RESULTS.md)
- v3 per-view latency (960x540 native, RTX 4080, loaded machine): median 4.435 / 4.491 / 4.093 s; p95 6.600 / 6.630 / 4.288 s; max 7.10 / 8.24 / 4.54 s (development / consumed / fresh) (comparison.json runs.*.view_seconds)
- Views per full frame in the offline benchmark: 25 (5x5 native 960x540 windows, stride 720/405); 175 views for 7 development frames (scratch_objects/evaluate.py starts(); RESULTS.md)
- CPU bundle smoke: 6.37 s for one 55x43 px crop, 1 tank detection conf 0.98 (release/asset-precision-20260918-v3/bundle-smoke.json)
- YOLO26x latency: p50 20.73 ms, p95 21.12 ms, p99 21.22 ms (RTX 4080, batch 1, incl. PNG decode) (training/REPORT-2026-09-17.md)
- YOLO26x trained on reference scene: reference training-fit recall L0/L1/L2: 243/259 = 93.8%, 563/600 = 93.8%, 75/77 = 97.4% (training/receipts/mypc-detector-evaluation.json; EVALUATION-2026-09-17.md)
- YOLO26x on 71 validation appearances L0/L1/L2 (conf 0.25, IoU 0.5): 0/71, 1/71, 0/71 correct; localized any class 0, 6, 9 (training/EVALUATION-2026-09-17.md)
- YOLO26x training-fit recall at L0 for the two smallest classes: small_launcher 25/25 (5.5x7.5 px delivered), ta-ta 25/25 (8x4 px) (training/receipts/mypc-detector-evaluation.json thresholds['0.25'].metrics)
- Template bank size: 241 crops = 210 organizer_reference + 31 reviewed_validation; 12 with calibration=true; 44 of the 210 reference crops mask_fallback (large_tower 16, helicopter 12, medium_launcher 5, ta-ta 2) (release/.../model/bank/manifest.json (script))
- Bank crops per class (reference + validation): condor 8+0, hangar 3+3, helicopter 16+4, jammer 11+0, jet_plane 18+2, large_launcher 18+3, large_tower 16+5, medium_launcher 5+1, medium_plane 4+2, mine_roller 2+4, small_launcher 22+0, small_plane 8+0, small_tower 17+1, spacecraft 19+0, ta-ta 22+0, tank 21+6 (bank/manifest.json (script))
- Validation frames consumed for training/calibration: 41 (35 frames <= 89, plus 110, 150, 190, 206, 230, 245); dev frames 100,127,140,160 (release/.../data-use-ledger.json)
- Model file sizes (LFS pointers): weights.pt 44,067,410 B; verifier.pt 1,840,190 B; heatmap.pt 6,469,458 B (release/.../model/*.pt pointer files)
- Verifier PatchCNN parameter count: about 457k (computed from layer shapes; consistent with 1.84 MB float32) (scratch_objects/patch_cnn.py)
- Verifier / dense CNN training cost: verifier v3 800 steps 35 s, v4 37 s, v5 23 s; dense CNN v1 1500 steps ~2 min (1.61M params), v3 3000 steps ~201 s (RESULTS.md CNN table)
- Thresholds in selection-v3.json: cnn 0.6, cnn_verifier 0.95, tiny 0.65, tiny_min_contrast 0.8, tiny_verifier 0.8, heatmap 0.3 (small_tower only), heatmap_verifier 0.4, pixel 0.6, calibrated_pixel 0.7, feature 0.7, verify_features_below 0.8, weak_feature_verifier 0.3, weak_feature_pixel 0.25, nms_iou 0.35 (release/.../selection-v3.json)
- Coverage ledger content: 1009 candidates = 995 manual_track boxes over 230 frames (frames 5-249) + 14 score_confirmed_seed; 69 review sheets; 37 track files; 11 classes (validation/annotations/coverage-ledger.json (script))
- Ledger boxes per class: medium_plane 206, large_launcher 130, tank 123, helicopter 104, small_launcher 99, jet_plane 86, large_tower 86, hangar 76, condor 48, small_tower 22, mine_roller 15 (coverage-ledger.json (script))
- Ledger frames with no box: 10-14 and 172-181 (15 frames); median 3 boxes per frame, max 11 (coverage-ledger.json (script))
- IoU of original ledger boxes vs tight re-review (4 reserved frames): 0.371, 0.329, 0.783, 0.568, 0.553, 0.617, 0.437 (3 of 7 below 0.50); RESULTS.md reports 7 of 13 below 0.50 on the 8 consumed frames (release/.../fresh-annotation-review.json (script) and RESULTS.md)
- Stale class labels in exported ledger: 80 of 995 boxes (condor-103-150: 48 boxes is jet_plane; small-launcher-a-092-123: 32 boxes is medium_launcher) (bank manifest pose rows vs coverage-ledger.json, IoU 1.00 on frame 110 (script))
- Probe score decoding: 0.003808073115003808 = 5/(101*13) exactly, so 13 classes are averaged on validation; large_launcher GT count 81-100 if all 4 probe boxes matched (solution/drone/apply_initial_frame_extension.py lines 62/74/162; python arithmetic)
- Classes the team found absent from validation (v4): small_plane, ta-ta, jammer, spacecraft (solution/drone/search_missing_classes.py CLASSES)
- Reference scene composition: 25 frames, altitude 600 m, step 13.89 m, 16 objects, exactly 1 per class; 8-12 boxes per frame (drone_oscar-live-tracker/drone-flyby/src/helsinki/run_metadata.json and annotations (script))
- Organizer GT median box sizes (source px): small_launcher 22x30, ta-ta 32x17, jammer 32x44, medium_launcher 47x45, spacecraft 44x49, small_plane 43x50, tank 50x47, medium_plane 56x49, mine_roller 53x60, small_tower 57x59, large_tower 62x64, jet_plane 77x82, helicopter 116x94, large_launcher 150x108, condor 172x168, hangar 188x129 (helsinki annotations, fully visible boxes only (script))
- In-track box width variation in reference (perspective): condor 166-178, helicopter 109-124, large_launcher 139-163, tank 46-54, large_tower 58-69 px (7 to 17 percent) (helsinki annotations (script))
- Cross-scene rotated masked NCC (my experiment, 31 validation-scene crops, reference templates only): top-1 excluding tiny classes 7/31 (L2), 8/31 (L1), 8/31 (L0); tank 6/6, all other classes 0-1; true-class score median 0.51 / 0.58 / 0.69; true score >= 0.70 in 4 / 10 / 15 of 31 (scratchpad/drone/_work/cross_scene_ncc_v2.py and cross_scene_ncc_v2.json)
- Cross-scene NCC without the 40-mask-pixel floor: small_launcher template wins 25/31 at native, 27/31 at L1 and 31/31 at L0 without the floor (scratchpad/drone/_work/cross_scene_ncc.py output)
- Matched IoU on non-reference frames (v3, 36 matches): median 0.886; 25 percent below 0.70; 14 percent below 0.60; small_launcher 0.59-0.65, hangar 0.57-0.65, small_tower dev 0.595 (comparison.json (script))
- Local test suite: 44 tests, 2.327 s, OK (release/.../final-tests.txt)

### Gaps
- The private 'preparation project' artifacts/ tree is not in the export: artifacts/drone-fixed-assets-20260918-v1, -v2 and -v3 (development-threshold-diagnostic.json, fresh-detection-gallery.jpg, neutral-tata-audit.jpg, precision-v2-stopped.tar.gz, raw predictions, overlays, verifier and heatmap training receipts). Per-BRANCH and per-class proposal counts, the verifier's standalone accuracy or confusion matrix, and the classes of the 3 missed development targets are therefore unknown.
- docs/drone-fixed-asset-result.md, referenced by model/README.md, is not in the export. RESULTS.md is presumably the same text.
- weights.pt, verifier.pt, heatmap.pt and validation/reconstructed-validation/*.png are Git LFS pointers only. I could not inspect checkpoint receipts (steps, flags, classes) or confirm which YOLO run weights.pt is. 'yolo26m' is inferred from the train.py default and the 44 MB size.
- The corrected annotation set data/drone/training/score-anchored-validation-v7 (class corrections, bbox scale corrections, score-anchored projections) is not exported. The exported coverage-ledger.json is the older 2026-09-17T17:46Z state with known wrong classes and displaced boxes.
- All scoreboard probe results (artifacts/drone-api-tests/score-probes, score-assessment-20260917, discovery state.json, confirmed-seeds.json) are missing. Only one probe score (0.003808...) survives as a constant in code. The per-class bbox scale factors that were applied, and the validation scores before and after, are unknown.
- The inference that the server averages over 13 classes rests on one number matching 5/(101*13) exactly. Which 13th class is present in validation but un-annotated is unknown.
- v3 was never measured at L1 or L0, never under a legal camera policy, never as COCO mAP, and never live. No per-branch latency breakdown exists, and none of the latency figures come from an unloaded machine.
- Whether the evaluation sequence shares the scene or the physical object instances with validation is unknown. This decides whether the 31 validation-scene templates have any value on the one-shot run.
- Which box convention matches organizer ground truth on validation is unknown: loose participant boxes, or tight visual extent excluding shadow. The only organizer boxes available are the 25 helsinki frames.
- drone/overnight/config.json (track_aliases), data/drone/reference, data/drone/mined, the 'mypc' training host and the Runpod follow-up work referenced in RESULTS.md are not in this repo.
- Results of the earlier per-confusion expert experiment (overnight/evaluate_expert.py: a 4-way ResNet50 or ConvNeXt-Tiny expert for large_launcher vs mine_roller) are not in the export.
- My cross-scene NCC experiment is crude: 15 degree rotation steps, 3 scales, 2 templates per class, BGR correlation, and crops with background margin. It shows that naive template transfer fails. It does not put a bound on what a trained model can do.
- No competition rule on pretrained weights was found in drone-flyby/README.md. Why every network was trained from random initialisation is not explained in the export.

## critic
```json
{
 "contradictions": [
  {
   "claim_a": "spec-scorer: validation pseudo-label motion fit is dy = 45.26 + 0.01813y with residual sd 12.6 px ('labels are noisy'), 64.8 px/frame at y=1080, i.e. slightly SLOWER than Helsinki (65.7).",
   "claim_b": "data-inventory: validation dy = 52.38 + 0.01467*cy, residual sd 1.31; live-tracker: validation is 1.037x FASTER than Helsinki.",
   "resolution": "data-inventory and live-tracker are right. Re-measured (scratch _work/critic/measure1.py): plain OLS on all 908 unclipped pairs gives 48.56 + 0.01733y, sd 11.6, because 22 pairs are label outliers (jet-plane-d frames 40-52, helicopter-b-early, tank-b, the two frozen frames). A 3-sigma robust fit gives dy = 52.29 + 0.01476y, sd 0.82 (n=875), dx = 0.00741(x-1924); reviewed-only boxes give 52.39 + 0.01469y. At y=1080 validation is 68.2 px/frame against Helsinki 66.2 (+3.0%). Drop spec-scorer's validation fit."
  },
  {
   "claim_a": "spec-scorer headline 3: 'One fixed map fits every object: x' = 1.00709x - 13.57, y' = 1.01333y + 51.85', used as if universal; also treats the map as constant in time.",
   "claim_b": "data-inventory: the map must come from the same flight (borrowed map: 46% IoU>=0.5 at k=10). live-tracker: Helsinki flow grows x1.037 over 24 ticks while validation is flat.",
   "resolution": "The map is per flight and Helsinki is not stationary. Measured (forecast.py): Helsinki boxes with the robust validation map: 87% IoU>=0.5 at k=4, 49% at k=8, 40% at k=10; validation labels with the Helsinki map: 90% / 69% / 56%; with their own map 97% / 95% / 93%. Temporal trend: Helsinki dy gains +0.0755 px/frame per frame (se 0.007, x1.026 over 24 frames); validation +0.0005 (x1.002 over 248 frames; phase correlation on the real images gives 67.2 px/frame early and 67.7 late). The Helsinki constants must not be hard-coded for evaluation; calibrate on the evaluation flight itself."
  },
  {
   "claim_a": "spec-scorer: with memory and forecast the team cycle (L0, L1-TL, L0, L1-TR) is 'equivalent within 0.01' to the L1-only top sweep at every T (0.952/0.952 at T=8, 0.856/0.863 at T=12, 0.741/0.749 at T=16).",
   "claim_b": "live-tracker: the L1-only sweep beats the deployed cycle at every T>=8 (0.950 vs 0.863 at T=8, 0.863 vs 0.759 at T=12, 0.735 vs 0.570 at T=16).",
   "resolution": "Both simulators agree on always-L0 (0.773/0.774, 0.590/0.587, 0.289/0.290) and on the L1-only sweep (0.952/0.950, 0.863/0.863, 0.749/0.735); they differ only on the cycle. Reading _work/starterkit/policy_sim.py: on validation (mode 'prov') spec-scorer ran ONLY forecast='perfect', i.e. once recognised the GT box is reported for ever, no drift, no retirement. live-tracker runs the real RevisitTracker, where every L0 frame is a 'full opportunity' and visible_misses_before_retirement=3 retires L1-only tracks (with 99 misses the cycle recovers 0.570 -> 0.664 at T=16, still below spec-scorer's 0.741 because of real forecast drift). So spec-scorer's cycle numbers are an upper bound for an ideal tracker; the deployed code behaves as live-tracker says. The cycle is only acceptable after miss counting is made level-aware."
  },
  {
   "claim_a": "spec-scorer: an L2 top-row sweep scores 0.866 (pure L2) and 0.918 (L2/L1 hop) and is the recommended policy when the recogniser needs >= 12 delivered px.",
   "claim_b": "live-tracker: l2_top is capped at 0.63 to 0.66 even with a perfect detector because one sighting at the top cannot be forecast 30 frames.",
   "resolution": "Same cause: spec-scorer's L2 numbers use the perfect-forecast mode (GT box after first recognition). With a real one-frame forecast the share of IoU>=0.5 falls to 85% at k=16 (real SIFT calibration on validation, measured here) and about 32% at k=30 (live-tracker), so a single top-row sighting cannot carry an object for its 33-frame life. The 0.918 hop-sweep figure is unproven: nobody simulated the hop sweep (with its L1 re-anchor frames) through the real tracker. Treat 0.63-0.66 as the measured value for pure l2_top and the hop sweep as untested."
  },
  {
   "claim_a": "data-inventory: one observation + own-flight homography + growth keeps IoU>=0.5 in 99% at k=10 and 96% at k=20.",
   "claim_b": "spec-scorer: 93.9% at k=10, 82% at k=16, 65% at k=20 (held-out affine map). live-tracker: 94% at k=8, 62% at k=16, 41% at k=20 (real SIFT model from Helsinki frames 0,1).",
   "resolution": "All three reproduce; the spread is the quality of the map, not the method. Re-measured on Helsinki GT (forecast.py, forecast_h.py): in-sample 8-DOF homography + growth 100% k=8, 99% k=10, 91% k=16, 94% k=20 (n=17 only); leave-one-class-out homography 97 / 91 / 87 / 76%; separable 4-number affine in-sample 95 / 90 / 77 / 65%; map from the frame 0->1 GT pairs only 91 / 90 / 66-72 / 47-53%. Real live calibration on VALIDATION images (calib_val.py, pair 5,6): 94% k=8, 91% k=12, 85% k=16 on reviewed labels. Planning number: a one-frame forecast is safe for 8 frames (>= 91% under every model), marginal at 12, unreliable beyond 16; data-inventory's 96% at k=20 is an in-sample best case on 17 samples."
  },
  {
   "claim_a": "spec-scorer: for box size 'map the four corners (best)'.",
   "claim_b": "live-tracker: the corner warp inflates box height (x1.10 at k=8, x1.22 at k=16); centre warp plus isotropic width scaling is better. data-inventory: height is flat or shrinking for tall objects.",
   "resolution": "live-tracker is right about the height, and spec-scorer contradicts its own growth numbers (w x1.0081, h x1.0017 per frame, while its map scales y by 1.01333 per frame). Measured: corner-warp predicted/true height 1.04 at k=4, 1.11 at k=8, 1.17 at k=12, 1.24 at k=16, 1.30 at k=20 on Helsinki; 1.04 / 1.08 / 1.12 at k=8/12/16 on validation labels. In share of IoU>=0.5 the corner warp is nevertheless not worse than centre + growth (Helsinki in-sample 77% vs 72% at k=16) because the taller box hides the systematic +0.5 px/frame dy drift. So: no score gain proven either way; fix the drift (pooled speed correction) first, then use centre warp with w x1.008, h x1.002."
  },
  {
   "claim_a": "data-inventory: no box in either dataset is cut by the x=1920 seam.",
   "claim_b": "live-tracker: the two side crops cut the seam, e.g. the medium launcher at x 1890..1920 is 'never complete in either side crop'.",
   "resolution": "Measured: 0 of 259 Helsinki and 0 of 987 validation boxes straddle x=1920. The only seam-hugging object is medium-launcher-a-092-123, whose right edge is x2 = 1920, 1920, 1918, 1916, 1918 in frames 92-96 and 1906-1916 afterwards. By the labels it IS complete in the left L1 crop from frame 97 (13 top-band frames; the deployed cycle looks left at about 4 of them), so 'never' is overstated. It stays within 1 to 7 delivered L1 pixels of the crop edge for its whole top-band life, and the organiser box of medium_launcher is loose (object fills 22% of it), so with a real detector and crop_margin_pixels=1 it is fragile. 78% of entries are in x 960..2880, so an L1 top-centre view is the robust cure; the seam itself is not a measured loss."
  },
  {
   "claim_a": "spec-scorer: validation starts with 1 object, so a one-off bottom-row start sweep has no effect there.",
   "claim_b": "data-inventory: two tracks are already in the frame at sequence start (large_launcher at y=938 and mine-roller-a at y=810).",
   "resolution": "Label frame 1 holds exactly one box (large_launcher, status score_positive_group_projected_track); mine-roller-a first appears in frame 5 at y=810, so it was physically in frame 1 at about y=570. Reason: the reconstructed validation frames 1-3 are blank and frame 4 is 81% covered (manifest native_coverage 0, 0, 0, 0.812), so nothing could be labelled there. The true start population of validation is unknown (at least 2, plus any of the two unlabelled classes). spec-scorer's 'no effect' conclusion rests on missing labels; decide a start sweep from the first live L0 frame as he also suggests."
  },
  {
   "claim_a": "spec-scorer: a whole class is worth 1/16 = 0.0625 mAP (e.g. the medium_launcher pass in validation). history-results: consistency check '11/16 x 0.68 = 0.47'. asset-recognizer NUMBERS: 'classes the team found absent from validation: small_plane, ta-ta, jammer, spacecraft' (4).",
   "claim_b": "asset-recognizer: probe score 0.003808073115003808 = 5/(101*13), so validation GT averages over 13 classes. data-inventory/history: five classes have no label (condor too).",
   "resolution": "K = 13 is solid: score*101*K must be a sum of interpolated precisions with denominators <= 4; K = 10,11,12,14,15,16 give 50/13 ... 80/13, only K = 13 gives 5.000 (source: solution/drone/apply_initial_frame_extension.py lines 62-74). So on validation a class is worth 1/13 = 0.0769, a pure label replay of 11 classes is capped at 11/13 = 0.846, and history's 11/16 arithmetic is wrong (0.470 would mean mean AP 0.555 over 11 labelled classes). 'Absent' in the codex scripts (search_missing_classes.py, run_missing_class_yolo_scan.py) means absent from label set v4, not proven absent from the scene; condor was later relabelled jet_plane, giving five unlabelled classes. Exactly two of condor, jammer, small_plane, spacecraft, ta-ta ARE in the validation GT and nobody has found them (0.154 mAP on validation). K for evaluation is unknown."
  },
  {
   "claim_a": "live-tracker and data-inventory: no validation images exist in this repo / in any export, so validation appearance and calibration could not be checked.",
   "claim_b": "history-results: all 249 reconstructed 4K validation frames and the 3 bundle checkpoints are present as LFS blobs in the local clone.",
   "resolution": "history-results is right. Verified: all 249 pointers under origin/codex/drone-training-baseline:drone-flyby/validation/reconstructed-validation resolve to files in C:\\Users\\edlun\\Desktop\\lucky shots\\NordicCupAI\\.git\\lfs\\objects (map written to _work/critic/lfs_map.json; read directly, no git state touched). They open as 3840x2160 RGBA PNGs. manifest.json: 245 frames complete at native_coverage 1.0 (each stitched from 23-27 captured L2 views), frames 1-3 blank (32 KB), frame 4 81%. Every 'no images' caveat of the other readers can now be lifted."
  },
  {
   "claim_a": "live-tracker: from 0.470, about 0.39-0.43 is detector loss and 0.10-0.14 policy/tracking loss (assumes 0.470 came from the live pipeline). spec-scorer: a stateless cycle with a perfect detector gives 0.49-0.60, which brackets 0.470.",
   "claim_b": "history-results: 0.470 most plausibly came from a fixed-plan replay of pseudo-labels (no detector, no camera); the only portal-facing code in the repo is that replay.",
   "resolution": "Not resolvable from the repo. Searched all four branches plus commit messages of all refs for 0.47 / 0.908 / leaderboard: nothing for the drone case (the only hits are the medical case). All three stories fit 0.470 numerically (label replay at mean AP 0.555 over 11 of 13 classes; size-gated cycle at T=20 = 0.483; stateless cycle 0.49-0.60), so none is evidence. Only Oscar's artifacts/drone-api-tests (portal status JSON, score-probes/*/result.json) can settle it. Until then no gap decomposition from 0.470 should be used for planning."
  },
  {
   "claim_a": "history-results: the one controlled ablation puts about 0.60 in the detector and about 0.02 in camera + tracker + placement (Helsinki 0.978 oracle vs 0.378 YOLO26m).",
   "claim_b": "live-tracker: with a perfect detector the deployed defaults lose 0.10-0.14 on validation (0.864 raw, 0.903 de-duplicated).",
   "resolution": "The 0.02 is optimistic and should not be reused: size-prior.json was fitted on the same 25 Helsinki frames, and Helsinki has one instance per class so the same-class cluster birth bug cannot fire. Use live-tracker's validation figures: 0.903 deployed, 0.932 with extent=detector, 0.961 with the cluster-birth fix as well, i.e. about 0.04 irreducible and about 0.06 fixable in two small changes."
  },
  {
   "claim_a": "sprites-synthetic: helicopter has no sprite, large_tower only an L1 one; 'without a helicopter sprite one of the 16 classes scores 0'.",
   "claim_b": "asset-recognizer: the template bank holds helicopter 16+4 and large_tower 16+5 crops with masks.",
   "resolution": "Two different cutout banks. Oscar's sprite bank (68 RGBA sprites) lacks helicopter. The codex bank (release/asset-precision-20260918-v3/model/bank/manifest.json) has 20 helicopter rows: 12 reference with mask_fallback 'Unreliable foreground fraction', 4 reference with usable masks and 4 reviewed_validation with masks; large_tower has 16 reference (all mask_fallback) + 5 validation. A Q4 generator should merge both banks; helicopter is not a missing class. Also now possible: cut new sprites straight from the 245 native validation frames."
  },
  {
   "claim_a": "asset-recognizer: bank = 241 crops, ledger medium_plane 206, 37 track files. live-tracker/data-inventory: 34 tracks.",
   "claim_b": "history-results: bank = 246 crops + 246 masks. data-inventory: medium_plane 208, 38 source files, 34 track ids = 31 physical objects; run_metadata says 37.",
   "resolution": "All bookkeeping, all reconcilable. Bank folder holds 492 PNGs (246 crop/mask pairs) but manifest.json lists 241 templates (210 organizer_reference + 31 reviewed_validation), so 5 files are orphans. Ledger: 206 = manual_track rows, 208 includes 2 score_confirmed_seed rows; 37 track files + 1 seed file = 38 source_file values. v8: 34 distinct (class, track_id) pairs in the per-frame files, run_metadata.json says 37, three of the 34 are 1-2 frame stubs ('-early', '-extension') of other tracks, hence 31 objects."
  },
  {
   "claim_a": "spec-scorer: at L1 only ta-ta (8.5 px) and small_launcher (11 px) are below 12 px min-side, 'every other class >= 16 px', so the cycle is right if the recogniser works at 10-11 delivered px.",
   "claim_b": "data-inventory/sprites: organiser boxes are loose; the visible object is 21x22 src px for medium_launcher (11 px at L1), 11x14 for small_launcher (7 px at L1), 42x43 for jet_plane (21 px at L1, 10.8 at L0). asset-recognizer: floor about 25-30 px on the longer side; yet an in-domain YOLO26x finds 5.5x7.5 px boxes 25/25.",
   "resolution": "The T in both policy simulations is the BOX min side, while recognisability depends on OBJECT pixels, which are 0.47x (medium_launcher, small_launcher), 0.54x (jet_plane) and 0.6-0.9x (others) of the box side. Re-read the policy tables with that shift: medium_launcher joins the classes that need L2 if the object needs >= 12 px. None of the three thresholds (10-11 px, 25-30 px, 'any size in-domain') is a measured recall-versus-size curve for an unseen instance; T is still unknown and is the single number that decides Q5."
  },
  {
   "claim_a": "README-live.md / history: tracker about 10 ms per frame, calibration about 200-277 ms, 'skips no frames'.",
   "claim_b": "live-tracker: tracking median 21 ms (63 ms on L0->L0), calibration frame 331 ms total.",
   "resolution": "live-tracker's measurement stands for this laptop; my runs of the same calibrate_images on validation frames took 504-563 ms per pair (machine was loaded, 3240-4430 matches). Expect the calibration response to miss at least one 333 ms slot on CPU; harmless (one frame, 1/249) but the retry path (CalibrationError -> stay in warm-up) should be exercised."
  }
 ],
 "settled": [
  {
   "claim": "(live-tracker, data-inventory: 'unknown, no images') Whether validation 4K frames are available for measurement.",
   "verdict": "SETTLED: available locally. 245 complete native-resolution frames (5..249), frames 1-3 blank, frame 4 81%.",
   "evidence": "All 249 LFS pointers of origin/codex/drone-training-baseline:drone-flyby/validation/reconstructed-validation/frame_*.png resolve in C:\\Users\\edlun\\Desktop\\lucky shots\\NordicCupAI\\.git\\lfs\\objects (3.8 GB, 250 objects); index in scratchpad drone\\_work\\critic\\lfs_map.json, capture manifest copy in _work\\critic\\val_manifest.json. cv2 opens them as (2160, 3840, 4). Each frame was stitched from 23-27 captured views, so the served validation sequence is deterministic across runs."
  },
  {
   "claim": "(live-tracker, 'unknown here': no images) Does the two-frame SIFT calibration on the validation flight fit a near-zero trace, or does the model flow inflate (x1.62 at tick 248 as implied by the Helsinki trace -1.54e-3)?",
   "verdict": "SETTLED for validation: trace is near zero in 3 of 4 tested pairs, but it is a lottery per pair; clamp or average it.",
   "evidence": "Ran the real tracking.motion.calibrate_images on INTER_AREA 960x540 views of validation frames (script _work\\critic\\calib_val.py). Pair (5,6): trace +1.6e-5, 3239 matches, median residual 0.40 px, flow 54.2/68.1/83.6 px at y=100/1080/2060 (label fit 53.8/68.2/82.7), inflation 0.996 at tick 243; forecast vs reviewed labels IoU>=0.5 97% k=1, 96% k=4, 94% k=8, 91% k=12, 85% k=16. Pair (6,7): -3.3e-6, same quality. Pair (200,201): -1.3e-5. Pair (100,101): trace -2.6e-4, inflation x1.069 at tick 243, forecasts only 89% k=8, 66% k=12, 48% k=16. True flow is flat on validation (x1.002 over 248 frames from labels; 67.2 -> 67.7 px/frame by phase correlation) but really grows on Helsinki (+0.0755 px/frame per frame, x1.026 over 24). So the frozen-geometry design works on validation, yet one bad pair costs about 25-35 points of k>=12 forecast share; calibrate on two or three consecutive pairs and take the median, or re-scale online with the motion-clock ratio that is already computed."
  },
  {
   "claim": "(data-inventory, live-tracker: from labels only) Frozen frames 8->9 and 238->239 followed by a double step are real and belong to the served sequence.",
   "verdict": "CONFIRMED from pixels; exactly two such events in 245 frames.",
   "evidence": "Mean absolute difference of consecutive frames (1/8 scale): median 40.8; pairs (8,9) = 2.7 and (238,239) = 1.2, no other pair below 25% of the median. Phase-correlation shift: median 67 px/frame, 0 px for the two frozen pairs, 127 px for (9,10) and 123 px for (239,240), no other pair above 1.5x median. Since every reconstructed frame merges about 25 separate validation runs, the freeze is in the organiser's sequence, not a capture glitch; expect the same kind of event in evaluation and keep DRONE_OBSERVE_MOTION on (live-tracker: 0.903 with the clock vs 0.894 without)."
  },
  {
   "claim": "(asset-recognizer, inferred) The validation scorer averages over 13 classes; large_launcher has 81-100 GT boxes.",
   "verdict": "CONFIRMED K = 13; and it exposes label noise: organiser large_launcher count is <= 100 while v8 labels 149 boxes as large_launcher.",
   "evidence": "0.003808073115003808 * 101 * K equals 5.000 only for K = 13 (other K in 10..16 give multiples of 1/13 that no mix of precisions 1, 3/4, 2/3, 1/2, 1/3, 1/4 can produce). For every TP/FP pattern that yields exactly 5 points, N_large_launcher <= 100 ((80,100] if all 4 probe boxes matched, (60,75], (40,50], (20,25] otherwise). v8 has 5 large_launcher tracks (17 + 33 + 32 + 33 + 34 = 149 boxes; median sizes 84x62, 150x136, 110x61, 60x44, 179x103 vs Helsinki 149x108). The 'annotations' track (84x62) and large-launcher-b (150x136) are score-confirmed, so at least one, probably two, of large-launcher-c (110x61), c2 (60x44) and d (179x103) are another class in the organiser GT: at least 49 boxes (33%) of that class are wrong. c2 at 60x44 is a medium_launcher/mine_roller sized object. Assumes the portal uses the kit's scorer."
  },
  {
   "claim": "(spec-scorer, headline 5) Policy table on validation 'with memory plus forecast'.",
   "verdict": "It is a perfect-forecast, never-retire upper bound, not a forecast simulation.",
   "evidence": "_work\\starterkit\\policy_sim.py: 'for forecast in (['perfect','H'] if mode=='class' else ['perfect'])'; validation runs use mode 'prov', so after the first recognising frame the GT box is emitted in every later frame. The one-frame homography forecast ('H') was only run on Helsinki. Use live-tracker's numbers for what the deployed tracker does and spec-scorer's as the ceiling that a fixed tracker could approach."
  },
  {
   "claim": "(spec-scorer, data-inventory, live-tracker) How long does a one-frame forecast hold, and does it survive a real live calibration?",
   "verdict": "8 frames safely, 12 marginal, 16+ unreliable; one frame beats two-frame constant velocity in every reader's data (no contradiction on Q6 itself).",
   "evidence": "See forecast.py / forecast_h.py / calib_val.py in _work\\critic. IoU>=0.5 share at k=8 / 12 / 16: Helsinki in-sample homography 100 / 95 / 91, leave-class-out homography 97 / 89 / 87, separable affine 95 / 82 / 77, map from frame 0->1 pairs only 91 / 87 / 66-72, live SIFT model on validation 94 / 91 / 85, wrong-flight map 49 / 27 / 17. Per class at k=16 (Helsinki, homography): large_tower 0% (n=2, height shrinks), ta-ta 78%, all others 100%."
  },
  {
   "claim": "(history-results, inferred) Which configuration produced the team's 0.470.",
   "verdict": "STILL UNKNOWN after search; not recorded anywhere reachable.",
   "evidence": "git grep for 0.47x / 0.908 / leaderboard over drone-flyby on all four branches: no hit except a YOLO label coordinate. git log --all --grep for portal/leaderboard: only medical-appointment commits. origin/drone-flyby holds only the starter-kit import. A local branch drone/elias-verifier (2fbda67, merges the sprite and codex branches) exists but adds no score record. Must be asked of Oscar."
  },
  {
   "claim": "(sprites-synthetic, inferred) Helicopter cannot be synthesised from existing cutouts.",
   "verdict": "REFUTED: usable helicopter cutouts exist in the codex template bank.",
   "evidence": "release/asset-precision-20260918-v3/model/bank/manifest.json: 20 helicopter templates, of which 4 organizer_reference and 4 reviewed_validation have masks without the mask_fallback flag."
  }
 ],
 "still_missing": [
  {
   "question": "Q1 verifier on top of the expert-system proposals",
   "missing": "(1) No standalone accuracy of any verifier exists (verifier v5 has no confusion matrix or ROC; thresholds 0.95/0.80/0.40/0.30 were hand-set on <= 12 frames). (2) No proposer with measured recall on UNSEEN instances: every success of the fixed-asset bundle shares physical tracks with its calibration crops (unseen results 0/6, 1/6, 2/19, 5/19, 0/7), and it takes 4.1-4.5 s per view against the 3333 ms timeout; per-branch timing and per-branch proposal counts are not recorded, so it is unknown whether a fast subset (geometry only: 69 hits / 3 false alarms at L0) fits in 333 ms. (3) No hard-negative set: can now be mined from the 245 native validation frames, but only inside reviewed regions (frame 5 in full, top 540 px every 4th frame). (4) Clean labels to score a verifier: v8 has >= 33% class noise in large_launcher, 13 wrong jet-plane-d boxes (frames 40-52), a tank-like mine-roller-a track, 25% of boxes disagree with the codex set at IoU < 0.5, and two GT classes are entirely unlabelled. (5) Missing artefacts: private artifacts/drone-fixed-assets-*, the weights behind the 0.378 run, ultralytics is not installed here; torch usability unconfirmed."
  },
  {
   "question": "Q2 recognisable object pixel size in the delivered image",
   "missing": "The key number does not exist: recall (and class accuracy) as a function of delivered OBJECT pixels for a model tested on instances it was not trained on. What exists is only in-domain (YOLO26x 25/25 at 5.5x7.5 px on its training frames, then 0/71, 1/71, 0/71 on validation), by-eye readings of one Helsinki frame, and template NCC where lower resolution raises false matches. Both policy simulators gate on BOX min side although the object is 0.47x to 0.9x of the box. Now measurable because validation frames are local: render L0/L1/L2 views with cv2.resize INTER_AREA from the 245 native frames and score any candidate model per class and per size bin, split by track. Still impossible on real data for the 5 unlabelled classes (condor, jammer, small_plane, spacecraft, ta-ta), of which jammer, ta-ta are among the three smallest. Also unknown: whether the live server downsamples like local_evaluator.render_view (INTER_AREA box average)."
  },
  {
   "question": "Q3 one expert per class versus one shared model, overfitting risk",
   "missing": "No experiment compares the two; the only trained 'expert' (4-way large_launcher/mine_roller/other/background) has no result in the repo, nor do the overnight RunPod runs (tile detectors, ResNet50/ConvNeXt proposal classifiers, ensembles). Data limits are hard: one Helsinki instance and heading per class, 1-5 validation instances for 11 classes, none for 5 classes, so a leave-one-track-out split leaves 5 classes untestable and medium_launcher with a single (disputed: small_launcher in codex, medium_launcher in v8) track. Unknown whether evaluation reuses the validation scene, the same instances, or a third background; unknown K (number of classes present) for evaluation. Cross-class duplicate suppression (jet_plane/condor double reports) is absent in both the bundle fusion and tracker birth, and its cost is unmeasured."
  },
  {
   "question": "Q4 massive synthetic data from the cutouts, validated on real frames",
   "missing": "No model has ever been trained on Oscar's 180-image set, and no synthetic-to-real transfer number exists for any generator (the scratch YOLO26m on copy-paste data scored 0.378 on Helsinki, its training scene). The shipped composer cannot run here (needs private data/drone/grid-comparison-20260918-v1/256 tiles and manifest); a new composer is needed. Missing design inputs: validated alpha-edge softness (the sigma measurement failed), per-class box-to-object ratio table (boxes are loose: 0.22-0.85 fill) and its dependence on heading, arbitrary-yaw realism (objects are 3-D renders: towers lean, 2-D rotation cannot reproduce that), cutouts with native detail for helicopter and large_tower in the sprite bank (they exist in the codex bank; more can now be cut from the native validation frames). Hold-out rules must change: validation-derived sprites (11 in the sprite bank, 31 in the codex bank) and the 41 frames already consumed must stay out of what is scored. No GPU training environment is confirmed on this machine."
  },
  {
   "question": "Q5 is L0 -> L1 top-left -> L0 -> L1 top-right the right camera policy",
   "missing": "Decided almost entirely by the unmeasured T from Q2. Beyond that: (1) no simulation of the L2/L1 hop sweep, of the L0-L1-L2-L1 excursion on candidates, or of DRONE_REVISIT_EVERY with an imperfect detector THROUGH THE REAL TRACKER (spec-scorer's 0.918/0.912 are perfect-forecast ceilings; live-tracker only ran pure l2_top and found revisits never trigger with a perfect detector); (2) no run of any policy with a real detector on validation is recorded in the repo (Oscar's replay logs are on RunPod /workspace volumes, pods 2-5 EXITED, pod 1 RUNNING idle); (3) level-aware miss counting (the 3-visible-miss retirement by L0 frames) is identified but not implemented or tested; (4) detector latency at the chosen policy is unknown, and local --realtime is pessimistic by about 190 ms per frame of harness overhead, while the real service's behaviour when the endpoint is faster than 333 ms is unknown; (5) the start population of a sequence is unknown for validation (frames 1-4 not reconstructed), which decides whether a one-off bottom sweep pays."
  },
  {
   "question": "Q6 trajectory and box size from ONE recognising frame or TWO",
   "missing": "The direction is settled (one frame plus the shared per-flight motion model; all three readers and my re-measurement agree, two-frame constant velocity collapses by k=6-8). Still open: (1) robustness of the two-frame SIFT calibration at the start of an unseen flight: 1 of 4 validation pairs gave a poor model (48% at k=16 vs 85%), and the first frames are where it happens; no multi-pair or online refinement exists (adapt_edges is off, geometry is frozen); (2) the pooled global speed correction proposed by live-tracker (signed drift +0.47 px/frame) is not implemented; (3) box size: Helsinki size prior does not transfer (heading changes the box, large_launcher AP 0.66 under blend vs 0.98 with detector extents), corner warp inflates height x1.11 at k=8, and with a real detector the first box is noisy, so the value of a second look for SIZE and for re-anchoring small classes (ta-ta 59% at k=8, shift tolerance 5.7 px) is unmeasured with real detections; (4) whether the evaluation flight has constant flow like validation or accelerating flow like Helsinki (+2.6% over 24 frames) is unknown."
  },
  {
   "question": "Cross-cutting facts the plan depends on",
   "missing": "Origin of 0.470 (ask Oscar for artifacts/drone-api-tests portal status and score-probes results); what the leader does; which two of the five unlabelled classes are present in validation GT (worth 2/13 = 0.154 there; findable now by searching the native frames, or by class-group probes on the portal, which this agent must not run); corrected label set score-anchored-validation-v7/v8 per-track JSON and the codex class corrections are not in the export (the exported coverage-ledger.json still carries 80 boxes with stale classes); identity of the weights behind the 0.378 Helsinki run; all private docs (docs/drone-box-placement.md, drone-fixed-asset-result.md, drone-revisit-workflow.md) and the grid dataset."
  }
 ]
}
```
