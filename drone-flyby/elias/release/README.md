# Released checkpoints (Git LFS)

| file | trained on | evidence |
|---|---|---|
| `both_m1280.pt` | yolo26m at 1280, 5 epochs, 12k synthetic views: sprites and backgrounds from helsinki AND validation, 40 % UC Merced backgrounds (`elias/pod_final.sh` with `GEN_EXTRA`) | validation portal, organiser ground truth, three concealed thirds: 0.220 + 0.257 + 0.080 = 0.558. **Deploy this one.** |
| `helsinki_only_m1280.pt` | same recipe, helsinki sprites and backgrounds only (it has never seen the validation scene) | 0.583 mAP50 on 500 real validation-scene views; 0.457 end to end on the local validation harness. Kept as the honest generalisation reference. |

Both expect `DRONE_IMGSZ=1280` and the settings in `../README.md`.
