# Released checkpoints (Git LFS)

All are yolo26m trained at 1280 for 5 to 8 epochs on synthetic full views (`elias/data/synth_yolo.py`) with 40 % UC Merced
aerial tiles as extra backgrounds. Scores are from the validation portal, organiser ground truth, three concealed thirds.

| file | trained on | portal |
|---|---|---:|
| **`both_m1280.pt`** | sprites and backgrounds from helsinki AND validation, boxes in the organisers' convention only (`SYNTH_ORGANISER_BOXES=1`: validation frames give backgrounds and sprites, never their pseudo-label boxes) | **0.694** (0.250 + 0.331 + 0.112). Deploy this one. |
| `F5_fixed_m1280.pt` | as `both_m1280.pt` plus the objects the team labels missed (keep-out zones for 19 unlabelled tracks, 18 shadow-free walker sprites, second small_tower and medium_launcher sprites), batch 16 on a 4090 | 0.614 from the Oslo pod on 2026-09-19 (0.315 + 0.196 + 0.103) against 0.678 for `both_m1280.pt` measured the same afternoon; ta-ta 0.50 where the first scores 0. Served only for the classes where it beat the first on a same-day one-class run (`ELIAS_ROUTE`, `elias/ensemble.py`). |
| `helsinki_only_m1280.pt` | helsinki sprites and backgrounds only: it has never seen the validation scene | **0.601** (0.263 + 0.251 + 0.087). The honest estimate for an unseen flight. |
| `both_m1280_pseudoboxes.pt` | as the first, but real validation objects kept with the team's pseudo-label boxes | 0.558. Kept to document the cost of those boxes: small_tower 0.07 AP against 0.86, tank 0.47 against 0.93. |

All expect `DRONE_IMGSZ=1280` and the settings in `../README.md`.
