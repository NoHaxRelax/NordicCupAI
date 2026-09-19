# Released checkpoints (Git LFS)

All are yolo26m trained at 1280 for 5 to 8 epochs on synthetic full views (`elias/data/synth_yolo.py`) with 40 % UC Merced
aerial tiles as extra backgrounds. Scores are from the validation portal, organiser ground truth, three concealed thirds.

| file | trained on | portal |
|---|---|---:|
| **`both_m1280.pt`** | sprites and backgrounds from helsinki AND validation, boxes in the organisers' convention only (`SYNTH_ORGANISER_BOXES=1`: validation frames give backgrounds and sprites, never their pseudo-label boxes) | **0.694** (0.250 + 0.331 + 0.112). Deploy this one. |
| `helsinki_only_m1280.pt` | helsinki sprites and backgrounds only: it has never seen the validation scene | **0.601** (0.263 + 0.251 + 0.087). The honest estimate for an unseen flight. |
| `both_m1280_pseudoboxes.pt` | as the first, but real validation objects kept with the team's pseudo-label boxes | 0.558. Kept to document the cost of those boxes: small_tower 0.07 AP against 0.86, tank 0.47 against 0.93. |

All expect `DRONE_IMGSZ=1280` and the settings in `../README.md`.
