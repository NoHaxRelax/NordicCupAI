# First drone training run

This branch contains the first reproducible detector and crop-classifier training run. It starts from `drone-flyby`; exploratory experiments belong on separate `codex/drone-experiment-<name>` branches. Do not merge unreviewed experimental changes into this baseline.

The first detector's independent-scene check failed: it found only 1 of 71 known L1 validation targets at confidence 0.25 and IoU 0.50. See [EVALUATION-2026-09-17.md](EVALUATION-2026-09-17.md) for evidence and limitations. Successful training and fast inference did not establish usable detection quality.

## Models and data

- Detector: COCO-pretrained YOLO26x, all layers fine-tuned, 960 input, batch 8, AMP, 50 epochs. Its 250 camera views come only from the complete organizer Helsinki labels. Zoom sampling is 70% L1, 20% L2, 10% L0. Every intersecting known object is labeled, including clipped visible boxes. No mosaic or aggressive scale augmentation that would obscure this first comparison.
- Classifier: ImageNet-pretrained ResNet50, all layers fine-tuned, 224 padded crops, batch 64, AMP, 40 epochs. Camera rendering occurs at L0/L1/L2 before cropping, so upsampling does not restore lost image detail. Sampling balances classes and uses the same 10/70/20 zoom weighting.
- Reviewed manual positive crops supplement the classifier only. Manual full frames are not complete, so they cannot safely supply detector negatives. Every third track frame is sampled to reduce redundant adjacent views. All zooms of a physical track stay in one split. The classifier is an appearance experiment; it is not yet connected as a detector cascade.

The snapshot excludes the actively corrected second helicopter. It also excludes all three manual medium-plane tracks after native crop QA failed to establish a reference-sprite match. These files remain in the annotation snapshot for provenance, but the builder explicitly rejects them. Dataset v1 was a preflight candidate and was rejected before training. Only v2 is used.

## What the results can establish

The reference scene has one physical object per class. There is no independent, fully labeled detector holdout yet. Ultralytics requires a validation path; it points to the training views and any resulting mAP is a **training-fit diagnostic only**. The chosen detector checkpoint is the fixed final epoch, not `best.pt` selected from this diagnostic.

The classifier holdout contains different manual physical tracks, but training and holdout share a scene/background and the labels are participant pseudo-labels. It covers only six of the sixteen classes. Report per-zoom and per-class results; do not present this accuracy as competition mAP or as a 16-class generalization result. Using manual validation examples for training makes the official validation scene development data, not an untouched test set. Do not invoke the single final evaluation attempt as part of this run.

## Reproducibility and storage

`snapshots/` freezes the source annotations and their method. `manifests/first-run-v2.json` records source hashes, rendered image hashes, label hashes, groups, splits, class mapping, and exclusions. `requirements.lock.txt` records the actual HPC environment. The README, scripts, frozen annotations, manifests, metrics, and job receipts are committed; generated images and checkpoint binaries stay out of Git.

`receipts/storage.json` records the archive locations and SHA256. Data is copied to persistent HPC home storage, not scratch. Completed checkpoints are copied back locally and their hashes/locations committed. DTU's announced 18–21 September service window makes local backup especially important.

Build from the preparation workspace containing the source reference and reconstructed validation frames:

```bash
python prepare.py --source-root /path/to/preparation-workspace --output /path/to/first-run-v2
```

On a DTU application node, prepare Python 3.11 and the pinned environment before submission. PyTorch and torchvision use CUDA 12.4 wheels (`torch==2.6.0`, `torchvision==0.21.0`); install the remaining versions from the lock file. Download official `yolo26x.pt` and torchvision ResNet50 IMAGENET1K_V2 weights ahead of time. The training job must not install packages or depend on a queued interactive session.

From `$HOME/nordic-drone`, create `logs/`, set `DRONE_COMMIT`, `DRONE_CODE_ROOT`, and `DRONE_TASK`, then submit the versioned `submit.lsf`. The classifier can use `bsub -q gpua100 -W 00:20`; the detector defaults to a dedicated L40S for up to one hour. Do not override scheduler GPU visibility. Use shared `a100sh` only for short profiling/debugging.

`smoke.py` measures a warmed batch-one pretrained checkpoint, including PNG decoding, model preprocessing/inference/postprocessing and conversion to JSON. It excludes network, HTTP handler, tracking, and camera control. Repeat on the trained artifact and real endpoint before claiming the response-time target is met.

Publish meaningful checkpoints: prepared recipe and data manifest; verified job submission; first successful epochs; final metrics and artifact hashes. Never commit passwords, tokens, raw environment files, or GitHub credentials to the HPC.

## Tracking future runs

Future code snapshots default to local W&B offline tracking in the `nordic-ai-cup-drone` project. Install `requirements-tracking.txt` before those runs. Online logging needs an explicit personal workspace/entity and authentication. The original runs stay unchanged with tracking disabled. See [TRACKING.md](TRACKING.md) for metrics, launch flags, offline sync and verification.

## RTX 4080 run on mypc

The personal Windows machine has an RTX 4080 with 16 GB VRAM. Its separate detector run keeps the same 960 input, pretrained weight checksum, data v2, seed, and 50 epochs, using batch 2 and two data-loader workers. Ultralytics nominal batch remains 64 through gradient accumulation. Different batch sizes, warmup behavior, operating systems, and GPU kernels mean this is a practical second run, not a bitwise reproduction of the HPC trajectory.

`setup-windows.ps1` installs a dedicated Python 3.11.11 environment under `C:\Users\oscar\nordic-drone` without changing global PATH. The platform-specific dependency lock is separate. `preflight-mypc.py` verifies every image/label hash and the exact YOLO26x pretrained checksum before launch.

Upload an immutable committed code snapshot, then use `launch-mypc.ps1` with its code directory, full commit, and a unique run ID. Its independent noninteractive Windows process survives SSH disconnection; `run-mypc.ps1` writes its PID/status, logs, and persistent checkpoints. It requests system wakefulness only for the lifetime of training, then releases that request. It never shuts down the PC. Reusing a run ID is rejected. Stop a run by identifying its recorded PID and command first; do not kill unrelated Python processes.

The `receipts/mypc-*` files track the actual launch, versions, speed, and progress. Model binaries remain under the remote `runs/` directory and local `data/drone/mypc-runs/` backups, outside Git.

Sources: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/), [training options](https://docs.ultralytics.com/modes/train/), [DTU HPC](https://www.hpc.dtu.dk/).
