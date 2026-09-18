# Tilt study: how far may a sprite be rotated in the image plane?

Written 2026-09-18, revised 2026-09-19 after a review (see "What changed in the revision" at the end).
Code: `elias/data/tilt_study.py` (subcommands `camera`, `rotation`, `lean`, `sun`), sprites from
`elias/sprites/bank.json` (167 GrabCut sprites, see `elias/data/sprites.py`). Every number below was
measured by those scripts; raw output is in `tilt_camera.json`, `tilt_rotation.json`, `tilt_rotation.log`,
`tilt_lean.json`, `tilt_lean.log`, `tilt_sun.json`, `tilt_sun.log`. Pair contact sheets: `tilt_pairs_*.png`.
Camera plot: `tilt_camera.png`.

## Short answer

1. The camera is not looking straight down. It is pitched forward by about 18 degrees. The point straight
   below the drone sits at the bottom edge of the frame (x 1921, y 2116), and the off-nadir viewing angle
   runs from 0 at the bottom centre to 37 degrees at the top centre and 46 degrees in the top corners.
2. Low objects (tank, planes, jammer, mine roller, condor) keep their image orientation along a whole
   track and match another heading after a plain in-plane rotation about as well as they match themselves
   at another image row. Rotate them freely, 0 to 360.
3. Tall objects (large tower above all, then small tower, ta-ta, medium launcher, hangar, the erected large
   launcher) lean away from the nadir point. Their lean always points into the upper half of the image.
   For the two tower classes the best rotation between two tracks follows the lean prediction within
   13 degrees on all 9 pairs with matched view angle (section 2c). A rotation that makes the lean point
   downwards produces a view that the real footage never contains. Use the rule in the last section
   instead of a blind 360.
4. No cast shadows and no measurable fixed-sun shading on the objects, so rotation cannot break a sun
   direction. What does differ is brightness between scenes: the same class is 1.4 to 1.9 times brighter in
   helsinki than in validation. That matters more for cross-scene training than any rotation limit.

## 1. Camera tilt from helsinki

Input: box centres of the 15 helsinki tracks with at least 3 boxes that do not touch the frame border
(233 points, 218 frame-to-frame displacements), altitude 600 m, step 13.889 m per frame.

Straight-line summaries, no camera model:

| quantity | value |
|---|---|
| dy per frame at y = 0 | 51.5 px |
| dy per frame at y = 2159 | 80.1 px |
| dy slope | 0.01322 px per px |
| dx per frame at x = 0 | -13.5 px |
| dx per frame at x = 3839 | +13.6 px |
| dx slope | 0.00706 px per px |
| dx is zero at | x = 1914 (frame centre is 1920) |

Objects also grow while they move down the frame (large launcher box width 139 px at y 359, 163 px at
y 1580), which is the same perspective effect: the bottom of the frame is closer to the drone.

Pinhole model: optical axis pitched forward by p from straight down, focal length f, image yaw, one
ground point per track, least squares on all 233 points. For a ground point the model gives
dy(v) = f (d/H) (cos p + sin p v / f)^2 with v = y - 1080, so the dy slope is twice the dx slope at
the centre, which the data shows (0.01322 against 2 x 0.00706 = 0.01411).

| parameter | fit | leave one track out |
|---|---|---|
| focal length | 3140 px | 3133 to 3146 |
| forward pitch | 18.25 degrees | 18.06 to 18.38 |
| image yaw | -0.03 degrees | |
| rms residual | 2.97 px (1.80 px on the 10 low objects only) | |
| horizontal / vertical field of view | 62.9 / 38.0 degrees | |
| ground sampling at nadir | 0.191 m per px | |
| nadir pixel | (1921, 2116) | |

Off-nadir angle of the viewing ray:

| place in the frame | off-nadir |
|---|---|
| bottom centre | 0.7 |
| y = 1620, centre column | 8.5 |
| frame centre | 18.3 |
| y = 540, centre column | 28.0 |
| top centre | 37.2 |
| left or right middle | 35.9 |
| bottom corners | 30.0 |
| top corners | 46.4 |

Validation: the pseudo-label boxes jitter too much for the joint fit (a soft-L1 fit kept only 12 % of the
residuals below 10 px), so I used the closed form of the same model on a trimmed line through dy(y)
(535 of 556 displacements kept, f fixed at 3140): dy 52.0 px at the top, 83.6 px at the bottom, pitch
18.7 degrees, step over height 0.0241 against 0.0231 in helsinki. The same closed form on helsinki
gives 17.5 degrees, so treat the two scenes as the same camera within about 1 degree. (This section was
not rerun in the revision; the camera fit does not use the sprite bank.)

Consequence for a tall object of height h metres at off-nadir angle a: its top is displaced from its
foot by roughly 5.2 x h x tan(a) source pixels (times 0.8 to 1.05 for the row scale), radially away from
the nadir pixel. At 37 degrees that is about 3.9 px per metre of height; at level 0 divide by 4.

## 2. Does an in-plane rotation reproduce another heading?

Method: sprite A is rotated over 0 to 359 degrees in 1 degree steps, with scale in
{0.85, 0.92, 1.0, 1.08, 1.17} and a shift search of +-6 px, and compared with sprite B by masked NCC on
gray: a = (gray - mean) inside the mask and 0 outside, ncc = sum(ab) / sqrt(sum(aa) sum(bb)). A silhouette
mismatch lowers the score because pixels covered by one mask only add to the denominator. Mask IoU at the
best pose is reported as well. 110 pairs, about 70 s on CPU.

Pair choice. Two sprites of the same track: neighbouring frames, and first against last kept frame.
Two tracks (or two scenes): the one sprite from each track with the SMALLEST difference in off-nadir
angle, so that a heading change is not mixed up with a change of view angle. The first version of this
study paired the middle sprite of each track whatever its view angle, which for the large tower compared
15 degrees off-nadir with 32 and produced a meaningless 209 degrees.

"Lean turn" below is view_azimuth(B) - view_azimuth(A): how far a purely vertical structure turns in the
image between the two positions, counter-clockwise, the same sense as the rotation search.

Reference levels over all classes:

| pair kind | n | median best NCC | range |
|---|---|---|---|
| same track, neighbouring frames (ceiling) | 33 | 0.745 | 0.21 to 0.94 |
| same track, top against bottom of the frame (view angle only) | 32 | 0.604 | 0.21 to 0.88 |
| different tracks, same scene, matched view angle (heading change) | 13 | 0.604 | 0.22 to 0.79 |
| helsinki against validation, matched view angle | 16 | 0.509 | 0.20 to 0.73 |
| different classes (what a 360 degree search reaches by chance) | 16 | 0.254 | 0.05 to 0.42 |

So: above about 0.5 is a real match, 0.25 to 0.4 is what unrelated objects reach.

### 2a. View angle alone (same object, same heading, top against bottom of the frame)

Sorted by NCC. "Measured" is the best NCC angle, signed.

| class | track and frames | off-nadir A -> B | lean turn predicted | measured | NCC | IoU | NCC neighbour frames |
|---|---|---|---|---|---|---|---|
| small_tower | val c f157 -> f164 | 37.1 -> 31.0 | +5 | +1 | 0.88 | 0.85 | 0.94 |
| hangar | hel f22 -> f24 | 43.7 -> 42.7 | +2 | 0 | 0.87 | 0.97 | 0.85 |
| small_tower | val b f240 -> f249 | 36.0 -> 27.2 | -3 | 0 | 0.86 | 0.84 | 0.86 |
| tank | val c f195 -> f206 | 25.3 -> 13.4 | +20 | 0 | 0.80 | 0.80 | 0.80 |
| small_plane | hel f0 -> f7 | 10.4 -> 2.6 | -52 | -1 | 0.74 | 0.83 | 0.74 |
| condor | hel f0 -> f8 | 18.5 -> 14.9 | -35 | -1 | 0.73 | 0.85 | 0.91 |
| medium_plane | hel f21 -> f24 | 41.3 -> 39.4 | +3 | -1 | 0.72 | 0.83 | 0.74 |
| tank | hel f1 -> f24 | 39.8 -> 23.6 | +35 | -2 | 0.69 | 0.78 | 0.85 |
| tank | val 015-036 f15 -> f36 | 37.1 -> 15.4 | -15 | 0 | 0.67 | 0.85 | 0.74 |
| small_tower | hel f8 -> f18 | 19.1 -> 14.1 | -43 | -14 | 0.66 | 0.69 | 0.92 |
| spacecraft | hel f0 -> f21 | 26.8 -> 7.6 | +73 | -1 | 0.66 | 0.75 | 0.83 |
| tank | val d f30 -> f61 | 36.7 -> 6.4 | -81 | +5 | 0.64 | 0.67 | 0.75 |
| helicopter | val b f107 -> f137 | 40.9 -> 25.6 | +57 | +2 | 0.62 | 0.43 | 0.78 |
| jet_plane | hel f4 -> f24 | 36.9 -> 16.8 | -14 | +3 | 0.62 | 0.75 | 0.79 |
| jammer | hel f0 -> f11 | 16.5 -> 7.7 | +59 | +1 | 0.61 | 0.87 | 0.79 |
| mine_roller | val b f197 -> f206 | 36.7 -> 28.1 | +1 | +3 | 0.61 | 0.87 | 0.62 |
| large_launcher | hel f0 -> f18 | 39.4 -> 31.2 | -30 | -3 | 0.57 | 0.75 | 0.55 |
| jet_plane | val d f54 -> f82 | 37.1 -> 13.5 | +59 | +4 | 0.54 | 0.68 | 0.75 |
| hangar | val 103-148 f112 -> f131 | 37.2 -> 20.5 | -25 | 0 | 0.53 | 0.88 | 0.63 |
| tank | val b f129 -> f160 | 37.9 -> 14.1 | +73 | 0 | 0.52 | 0.62 | 0.70 |
| large_tower | hel f7 -> f24 | 43.0 -> 33.5 | -22 | -12 | 0.47 | 0.59 | 0.80 |
| helicopter | val b-early f85 -> f104 | 35.2 -> 14.6 | +9 | -6 | 0.45 | 0.44 | 0.70 |
| ta-ta | hel f0 -> f24 | 34.8 -> 17.0 | +51 | +2 | 0.45 | 0.54 | 0.75 |
| medium_launcher | val a f92 -> f123 | 36.4 -> 0.6 | +87 | no match (+167) | 0.37 | 0.30 | 0.57 |
| large_tower | val c f218 -> f239 | 41.5 -> 29.1 | +30 | +26 | 0.32 | 0.64 | 0.51 |
| large_tower | val b f93 -> f122 | 36.1 -> 2.2 | +33 | no match (-173) | 0.27 | 0.54 | 0.21 |
| hangar | val 053-082 f56 -> f82 | 37.2 -> 15.7 | +50 | -4 | 0.25 | 0.78 | 0.65 |
| large_launcher | val d f196 -> f226 | 36.0 -> 3.3 | -74 | -2 | 0.21 | 0.49 | 0.77 |
| small_launcher | 4 tracks | | | -19 to +153 | 0.41 to 0.60 | 0.39 to 0.57 | 0.52 to 0.72 |

Reading: low objects do not turn at all (measured within +-6 degrees although the view direction changes
by up to 81 degrees) and lose 0.05 to 0.2 NCC from top to bottom. The large tower does turn with the view
direction (-12 and +26 measured against -22 and +30 predicted) and at the bottom of the frame it is seen
from straight above and looks like a different object (val b: side view with legs at y 37, platform and
antenna seen from the top at y 1972, NCC 0.27, which is the chance level). The medium launcher behaves
the same way. Ta-ta shows legs and flank at the top of the frame and only its back at the bottom.
Hangar: the open mouth with its two grey end wall stubs is visible at 37 degrees and nearly closed at
16 degrees (val 053-082: NCC 0.25 with IoU 0.78, so the roof outline stays and the bright details go).
Large launcher, validation track d: at 36 degrees the raised canister and the cab are seen from the side,
at 3 degrees the vehicle is a flat long strip with outriggers (NCC 0.21, IoU 0.49, against 0.77 between
neighbouring frames). The helsinki launcher only covers 39 to 31 degrees and keeps 0.57.

### 2b. Heading change (different tracks, and helsinki against validation), view angle matched

| class | pair (frames) | off-nadir A / B | best angle | NCC | IoU | NCC at 0 | best NCC 30+ away | lean turn predicted | NCC at lean turn |
|---|---|---|---|---|---|---|---|---|---|
| tank | val 015-036 f15 -> val b f129 | 37.1 / 37.9 | 343 | 0.62 | 0.87 | 0.37 | 0.32 | 26 | 0.22 |
| tank | val 015-036 f36 -> val c f206 | 15.4 / 13.4 | 219 | 0.64 | 0.81 | 0.20 | 0.36 | 60 | 0.36 |
| tank | val 015-036 f15 -> val d f30 | 37.1 / 36.7 | 140 | 0.56 | 0.72 | 0.23 | 0.34 | 0 | 0.23 |
| tank | val b f160 -> val c f206 | 14.1 / 13.4 | 241 | 0.62 | 0.84 | 0.18 | 0.37 | 306 | 0.13 |
| tank | val b f129 -> val d f30 | 37.9 / 36.7 | 161 | 0.50 | 0.67 | 0.19 | 0.35 | 334 | 0.24 |
| tank | val c f195 -> val d f46 | 25.3 / 20.6 | 285 | 0.68 | 0.79 | 0.16 | 0.35 | 326 | 0.33 |
| tank | hel f17 -> val 015-036 f25 | 28.0 / 27.7 | 223 | 0.60 | 0.80 | 0.28 | 0.33 | 304 | 0.12 |
| tank | hel f24 -> val b f145 | 23.6 / 23.5 | 198 | 0.63 | 0.87 | 0.20 | 0.32 | 333 | 0.22 |
| tank | hel f21 -> val c f195 | 25.3 / 25.3 | 76 | 0.60 | 0.80 | 0.20 | 0.41 | 324 | 0.23 |
| tank | hel f4 -> val d f30 | 37.7 / 36.7 | 0 | 0.69 | 0.73 | 0.69 | 0.43 | 324 | 0.34 |
| hangar | val 053-082 f56 -> val 103-148 f112 | 37.2 / 37.2 | 342 | 0.67 | 0.86 | 0.21 | 0.27 | 321 | 0.18 |
| hangar | hel f24 -> val 053-082 f56 | 42.7 / 37.2 | 289 | 0.27 | 0.74 | 0.08 | 0.20 | 337 | 0.18 |
| hangar | hel f24 -> val 103-148 f112 | 42.7 / 37.2 | 266 | 0.40 | 0.72 | 0.07 | 0.18 | 298 | 0.11 |
| small_tower | val b f240 -> val c f159 | 36.0 / 35.4 | 16 | 0.79 | 0.79 | 0.72 | 0.68 | 26 | 0.78 |
| small_tower | hel f8 -> val b f249 | 19.1 / 27.2 | 36 | 0.72 | 0.75 | 0.58 | 0.61 | 35 | 0.71 |
| small_tower | hel f8 -> val c f164 | 19.1 / 31.0 | 53 | 0.73 | 0.69 | 0.41 | 0.60 | 67 | 0.66 |
| jet_plane | hel f4 -> val d f54 | 36.9 / 37.1 | 355 | 0.51 | 0.81 | 0.34 | 0.26 | 26 | 0.16 |
| mine_roller | hel f0 -> val b f206 | 1.9 / 28.1 | 91 | 0.38 | 0.77 | 0.13 | 0.24 | 51 | 0.22 |
| helicopter | val b f117 -> val b-early f85 | 34.3 / 35.2 | 64 | 0.34 | 0.45 | 0.22 | 0.29 | 322 | 0.11 |
| large_tower | val b f93 -> val c f225 | 36.1 / 36.9 | 28 | 0.22 | 0.68 | 0.17 | 0.20 | 38 | 0.21 |
| large_tower | hel f24 -> val b f93 | 33.5 / 36.1 | 222 | 0.20 | 0.35 | 0.13 | 0.17 | 60 | 0.14 |
| large_tower | hel f9 -> val c f218 | 41.9 / 41.5 | 248 | 0.27 | 0.54 | 0.07 | 0.25 | 71 | 0.14 |
| large_launcher | hel f15 -> val d f200 | 32.1 / 32.3 | 344 | 0.23 | 0.65 | 0.08 | 0.13 | 62 | 0.03 |
| small_launcher | 6 pairs | 30 to 39 | 6 to 345 | 0.42 to 0.60 | 0.31 to 0.75 | 0.27 to 0.56 | 0.37 to 0.54 | | |

What I saw on the contact sheets (`tilt_pairs_cross_track_0.png`, `tilt_pairs_cross_scene_0.png`):

- Tank: the rotated tank has the barrel, the hull outline and the camouflage bands where B has them, at
  every one of the 8 large angles (76 to 285 degrees). Median NCC 0.63 across headings against 0.67 for
  the same heading at another image row and 0.42 for the best different class (jammer). The peak is sharp
  (0.32 to 0.43 elsewhere). In-plane rotation is a faithful augmentation for the tank.
- Hangar: between the two validation tracks the match is good at 342 degrees (NCC 0.67, IoU 0.86): the
  mouth with its two stubs lands on the mouth. Helsinki against validation matches in outline only
  (IoU 0.72 to 0.74, NCC 0.27 to 0.40): helsinki sees the hangar from the side at 43 degrees with the
  arch front as a light band, validation looks into the mouth. Apart from the stubs and the band the
  hangar is nearly black (mean gray 10 to 18), so NCC rests on few pixels.
- Small tower: good match (0.72 to 0.79), but the available heading differences are only 16 to 53 degrees
  and they coincide with the lean turn (section 2c), so heading and lean cannot be told apart here.
- Mine roller: the outline matches at 91 degrees (IoU 0.77); texture NCC is low because the helsinki
  roller is seen from straight above (off-nadir 2, no closer pair exists) and the validation one at 28,
  and the scenes differ in brightness.
- Large tower, validation b against c at the same view angle (36.1 and 36.9 degrees): the best angle is
  28 degrees and the lean predicts 38. On the sheet the rotated tower leans the way B leans. The NCC itself
  (0.22, against 0.20 elsewhere and a same-track ceiling of only 0.21 to 0.51 for this lattice object
  with terrain showing through) is at the chance level, so one pair proves nothing; section 2c has the
  evidence. Helsinki against validation does not match at any angle (0.20 and 0.27,
  lean prediction 60 and 71): the helsinki tower is a light camouflaged lattice, the validation towers are
  black with an orange band, so this is a paint difference on top of the geometry (angles 222 and 248).
- Large launcher: helsinki and validation track d are the same asset: long chassis, cab, canister, four
  outriggers, white stripe. The silhouettes agree (IoU 0.65) at 344 degrees, so the two scenes show it at
  nearly the same heading (16 degrees apart) and there is no heading change to test rotation on. Texture
  NCC is low (0.23) because the paint differs: light grey-green in helsinki (mean gray 107), dark green
  camouflage in validation (68). The first version of this report said the two assets do not match at any
  angle and recommended not to mix the scenes. That was wrong: it compared the helsinki launcher with
  track large-launcher-c2-080-105, which is a 50 px vehicle with roller arms (a pseudo-label error, now
  excluded from the bank together with track c-047-078).
- Jet plane: both scenes have the same heading (best angle 355). With the view angle matched the NCC is
  0.51 (IoU 0.81) although the paint differs (white in helsinki, grey with a dark nose in validation).
- Helicopter: thin rotor blades make NCC fragile (0.70 to 0.78 between neighbouring frames, 0.45 to 0.62
  top against bottom); the cross-track pair reaches 0.34 at 64 degrees. Section 2c shows that the angle is
  stable over 12 pairs (60 to 68 degrees on 11 of them, 44 on one), so it is the heading difference, but the NCC is too low to call
  the match good: the rotor is at a different phase in the two tracks.
- Small launcher: a 12 to 15 px blob. The NCC curve is flat over angle (best elsewhere is within 0.06 of
  the best), so the test cannot tell anything, and rotation cannot hurt much either.

### 2c. Does the best angle follow the object's heading or its lean?

`tilt_study.py lean --max-off-nadir-difference 6`: ALL sprite pairs across two tracks of one class in one
scene whose off-nadir angles differ by at most 6 degrees (angle step 2). If the heading rules, the best
angle is the same for every pair of the same two tracks. If the lean rules, it moves with the predicted
lean turn. An angle drawn at random has a median error of 90 degrees against the prediction and lands
within 20 degrees in 11 % of draws.

| class | pairs | best angles | lean turn predicted | median error | within 20 degrees | median NCC |
|---|---|---|---|---|---|---|
| large_tower b -> c | 4 | 24, 28, 36, 60 | 31, 38, 48, 60 | 8.5 (max 12.3) | 4 of 4 | 0.23 |
| small_tower b -> c | 5 | 14, 16, 22, 24, 26 | 24, 26, 27, 29, 32 | 9.5 (max 10.3) | 5 of 5 | 0.78 |
| tank, 6 track pairs | 12 | constant per track pair: 338 to 342, 218 to 220, 140 to 142, 242, 162 to 164, 286 | 0 to 344 | 135.8 | 0 of 12 | 0.62 |
| hangar 053-082 -> 103-148 | 4 | 342 to 344 | 245 to 321 | 53.8 | 0 of 4 | 0.61 |
| helicopter b -> b-early | 12 | 60 to 68 on 11 pairs, 44 on one | 277 to 333 | 117.6 | 0 of 12 | 0.29 |
| small_launcher | 9 | flat curve | | 19.2 | 5 of 9 | 0.55 |

Large tower: the best angle rises from 24 to 60 degrees as the predicted lean turn rises from 31 to 60.
A fixed heading difference would give one angle for all four pairs. So for the large tower the image
orientation is set by where in the frame it stands, not by its heading. The four pairs share sprites
(b f93 appears three times), so this is 4 measurements on 2 objects, not 4 independent objects.
Small tower: the same, with a high NCC. Tank, hangar and helicopter: the angle is the same for every pair
of the same two tracks while the lean prediction moves by up to 90 degrees, so their orientation is their
heading and the lean plays no part. Small launcher: lean turn near 0 and a flat NCC curve, no information.

Too thin to tell from data: condor, jammer, small_plane, spacecraft, ta-ta and medium_plane exist at one
heading only (helsinki, one instance each), medium_launcher at one heading (one validation track; the
helsinki cut failed), jet_plane and large_launcher at nearly the same heading in both scenes. For these the
recommendation below rests on 2a (how much the view angle changes them) and on section 3, not on a
measured heading change.

## 3. Is shading baked with a fixed sun direction?

- Cast shadows: for each of 158 sprites I compared the mean gray of a 6 px band just outside the mask, in
  eight 45 degree sectors, with the terrain further out. On quiet terrain (terrain std / median below 0.08,
  13 sprites) the darkest sector is at 0.98 (validation, 12 sprites) and 1.02 (helsinki, 1 sprite) of the
  surroundings: no shadow. Over all sprites 14 have a sector below 0.80, and every group traces back to
  terrain next to one object seen in several frames (4 x the dark tree patch left of the helsinki jet,
  3 x the trees beside the helsinki hangar, 3 x the hedge and building edge at the upper left of validation
  large tower c, 2 x validation hangar 103-148, and two singles); the directions do not agree between
  objects (157.5, 337.5, 112.5, 67.5 degrees). I looked at large tower c in its four frames: the pale sand
  pad around its base is evenly pale on all sides, so this tall object throws no shadow. The terrain
  texture itself does contain tree shadows.
- Bright side: a brightness plane fitted inside each mask explains 0 to 25 % of the gray variance for
  every class except the small tower, so there is no sprite-wide shading gradient to compare. For the
  small tower (r2 0.3 to 0.74) the bright side is the white slab against the dark roof, and it turns with
  the object: 3.5, 16.1 and 10.4 degrees off for best angles of 16, 36 and 53 degrees, against 19.5, 19.9
  and 42.6 if it were fixed in the image.
- Hangar roof as a probe: a half cylinder would show a highlight under a directional light. It is uniformly
  near black at all three headings (mean gray with the grey end wall parts included: 10.2 helsinki, 11.0
  and 17.8 validation).
- Tanks rotated onto each other match in texture (2b), which they would not if one side were sunlit.

Conclusion: at sprite level there is no visible fixed-sun shading and there are no cast shadows, so
rotation does not break lighting. I cannot rule out a weak directional term below what the camouflage
texture hides.

Brightness between scenes (mean gray inside the mask):

| class | helsinki | validation | ratio |
|---|---|---|---|
| tank | 99.3 | 52.5 (tracks 44.5 to 59.0) | 1.9 |
| large_tower | 91.0 | 50.3 | 1.8 |
| jet_plane | 174.3 | 107.7 | 1.6 |
| large_launcher | 106.5 | 68.4 (track d) | 1.6 |
| mine_roller | 104.0 | 71.0 | 1.5 |
| small_tower | 177.6 | 125.2 | 1.4 |
| small_launcher | 174.5 | 129.4 | 1.3 |
| hangar | 10.2 | 13.9 | 0.7 |

For the tank the camouflage pattern is the same in both scenes (cross-scene NCC 0.60 to 0.69), so the
factor 1.9 is lighting or exposure, not paint. For the large launcher and the jet the paint itself also
differs. A sprite generator that trains on one scene and tests on the other needs a brightness jitter
that spans at least 0.5x to 1x (helsinki sprites) or 1x to 2x (validation sprites).

## 4. Recommended rotation per class

Two rules, then the table.

Free rotation: any angle, plus flips.

Lean rule for tall classes. Every bank entry stores `view_azimuth_deg` (direction from the nadir pixel
(1921, 2116) to the object, counter-clockwise from image right; this is the direction the object leans in)
and `off_nadir_deg`. In real footage the azimuth is always between 0 and 180, because the nadir is at the
bottom edge: tall things never lean towards the bottom of the image.
- If the generator knows where in the frame the window sits: rotate the sprite counter-clockwise by
  (azimuth of the paste position - sprite azimuth), within about +-15 degrees (2c measured errors up to 12),
  and paste at a row whose off-nadir angle is within about 8 degrees of the sprite's.
- If it does not (a free 64x64 window): choose the rotation so that sprite azimuth + rotation stays inside
  0 to 180. That is a 180 degree window per sprite, not 360. A horizontal flip is always allowed (it maps
  azimuth a to 180 - a); a vertical flip and a 180 degree turn are not.
Oscar's synthetic set v1 used quarter turns and flips on every class (his README), so some of its tall sprites
should lean downwards. I did not open that set to count them.

| class | recommendation | evidence |
|---|---|---|
| tank | full 360 | measured: 8 pairs at 76 to 285 degrees, NCC 0.50 to 0.69, same as view-angle-only change; angle constant per track pair over 12 pairs while the lean prediction moves |
| mine_roller | full 360 | one pair at 91 degrees, outline IoU 0.77, texture NCC only 0.38, view angle not matched (2 against 28): thin |
| jammer | full 360 | no second heading; view-angle change costs 0.17 NCC, no turn (+1): thin |
| condor | full 360 | no second heading; flattest object in the set (0.73 top to bottom): thin |
| small_plane | full 360 | no second heading; 0.72 top to bottom, but only seen at off-nadir 3 to 10: thin |
| medium_plane | full 360 | 4 sprites from 4 consecutive helsinki frames, one heading, off-nadir 39 to 41 only; 0.72 to 0.74 between them; by analogy with the other planes: thin |
| jet_plane | full 360 | same heading in both scenes (0.51 across scenes); 0.54 to 0.62 top to bottom, no turn: thin |
| spacecraft | full 360, lean rule if it is free | no second heading; upright wing panels, 0.67 top to bottom: thin |
| helicopter | full 360 | angle constant (60 to 68 on 11 of 12 pairs), so orientation is the heading; NCC only 0.25 to 0.34 because of rotor phase; body is low: thin |
| small_launcher | full 360 | NCC cannot discriminate at 12 to 15 px; nothing to break |
| hangar | full 360 for the footprint, keep off-nadir within about 10 degrees | two validation headings match at 342 degrees (NCC 0.67, IoU 0.86), angle constant over 4 pairs; mouth and end wall stubs visible at 37 degrees, nearly gone at 16 (NCC 0.26 on the same track) |
| large_launcher | lean rule, off-nadir matched within about 8 degrees; scenes may be mixed with brightness jitter | same asset in both scenes (IoU 0.65 at 344 degrees) with different paint; no second heading to test; top against bottom of the frame on validation track d drops to 0.21 (raised canister seen from the side at 36 degrees, from above at 3) |
| small_tower | lean rule (180 degree window) | 5 cross-track pairs follow the lean within 10.3 degrees (NCC 0.78); turns -14 with the view direction along its own helsinki track; nothing measured beyond 53 degrees |
| ta-ta | lean rule (180 degree window) | no second heading; side view at the top of the frame, back view at the bottom (0.46): thin |
| medium_launcher | lean rule, off-nadir matched | one track; top against bottom of the frame does not match (0.37 at +167 degrees) |
| large_tower | lean rule, off-nadir matched within 8 degrees, prefer the position-aware form | with matched view angle the best angle follows the lean on 4 of 4 validation pairs (24, 28, 36, 60 measured for 31, 38, 48, 60 predicted); along one track it turns -12 and +26 for predicted -22 and +30; NCC stays at 0.22 to 0.35, the chance level, so the evidence is the angle, not the score; near nadir it is a different picture; helsinki and validation towers differ in paint and do not match |

At small pixel sizes the lean shrinks with everything else (divide by 4 at level 0), so for the low
classes none of this is visible below about 16 delivered pixels. For the large tower it stays visible: most
of what the camera sees of it is the leaning shaft.

## Limits of this study

- Masks are GrabCut masks. Lattice towers carry see-through terrain, which pushes their NCC to the chance
  level even between neighbouring frames (val b: 0.21). Thin parts are handled per class now (tank barrels
  through a line detector, see `sprites.py`), rotor blades come from native-size GrabCut.
- One instance per class in helsinki and at most five headings in validation (tank). Anything marked thin
  above is an argument from geometry, not a measurement.
- The lean check reuses sprites across pairs, so its pairs are not independent.
- The validation camera was estimated with f borrowed from helsinki.
- NCC on gray ignores colour; none of the conclusions depend on colour.

## What changed in the revision (2026-09-19)

- Bank: tracks large-launcher-c2-080-105 and c-047-078 excluded (not launchers); validation large launcher
  now comes from track d-194-227 (8 sprites, sand pad removed with a pad-colour seed). Hangar sprites carry
  the end wall stubs. Three validation tanks got their barrel back and one was discarded. medium_plane has 4
  sprites (was 0). GrabCut now runs from a fixed random seed, so a rerun gives the same pixels; with the seed
  two helsinki jet frames grew a terrain patch and were discarded. 167 sprites (was 156).
- `pick_pairs` matches the off-nadir angle for cross-track and cross-scene pairs; new subcommand `lean`.
- Rewritten: the large launcher rows and bullets (2a, 2b, brightness 53.8 -> 67.8, section 4), the large
  tower bullets and table row, the hangar rows, the reference table, and the counts in section 3. All numbers in
  sections 2 and 3 come from the final 167 sprite bank.
