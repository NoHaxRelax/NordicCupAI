# Drone flyby: overnight plan, 2026-09-18 to 19

Branch `drone/elias-verifier` (worktree `../NordicCupAI-drone`), built from Oscar's live tracker with his
sprite branch and the codex baseline merged in. Venv `~/venvs/nordic-drone` (torch 2.11 cu128, RTX 5070 8 GB).

## What is measured so far

- Score today: 0.470 on validation, 5th in Denmark. Leader 0.908.
- Oscar's tracker and camera cycle with a perfect detector: 0.978 on Helsinki, 0.923 on the 249-frame
  validation scene. Tracking, forecasting and box placement are not the leak.
- With a perfect detector that can only recognise objects of at least N delivered pixels (long side):

  | N | L1/L0 cycle | L2 top sweep |
  |---:|---:|---:|
  | 0 | 0.923 | 0.729 |
  | 16 | 0.750 | 0.727 |
  | 24 | 0.517 | 0.726 |
  | 32 | 0.380 | 0.679 |

  The binding constraint is the pixel size at which recognition works, and the camera policy that feeds it.
  0.470 sits on the 24 px ceiling of the L1 cycle.
- Object sizes (source px, long side): large four 123 to 189, medium nine 43 to 89, small three 19 to 44.
  Divide by 4, 2, 1 for L0, L1, L2. The evaluator downsamples with cv2.INTER_AREA.
- Motion: objects enter at the top, about 65 px per frame, 33 frames per crossing, perspective flow
  (faster low in the frame, diverging left and right). A new object about every 6.7 frames on validation.

## Streams

1. **Policy simulator and search.** Pure-Python replay of annotations, the evaluator's Camera class and
   Oscar's sweep code, with a size-gated recognition model. Accept it only if it reproduces the 16 harness
   numbers above within 0.03. Then search hybrid policies (L1 coverage plus cued L2 looks).
2. **Data.** Real crop sets from Helsinki (organiser labels) and validation frames (participant labels).
   Missing sprites (helicopter, large_tower). Synthetic generator v2: composite at native resolution,
   arbitrary rotation, feathered edges, photometric jitter, then INTER_AREA by 1, 2, 4; hard negatives
   from empty terrain; 50k to 100k samples; backgrounds split by scene.
3. **Models.** (a) Scale-preserving crop classifier and verifier, 16 classes plus background, zoom level
   as input. Deliverable: accuracy per class versus delivered pixel size on REAL crops, cross-scene
   (train with Helsinki backgrounds, test on validation crops, and the reverse). (b) If that curve says
   L0/L1 recognition is feasible, a dense heatmap detector on delivered views trained on synthetic views.
4. **Integration.** Plug into Oscar's endpoint through the `module:factory` detector hook, run the real
   local harness on both scenes, offline and --realtime, against the current baseline.
5. **Morning report.** Numbers table, what to deploy, what is still open.

## Guardrails

- Never the evaluation attempt (hook in place). Portal validation runs only if Elias signs off on them.
- Only this branch and worktree. The medical checkout stays on tag serve-2026-09-20, untouched.
- RunPod: at most 10 USD, only pods created by the agent, terminated when done, never Oscar's.
  Default is local; spend only when a specific training run does not fit 8 GB or the night.
- Validation labels are participant-made from score probing: treat as noisy, never as ground truth.
- Every claimed gain is measured on a scene the model did not see backgrounds from.
