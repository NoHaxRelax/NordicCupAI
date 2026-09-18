# Tracking future drone runs

The project is [nordic-ai-cup-drone](https://wandb.ai/badecar-danmarks-tekniske-universitet-dtu/nordic-ai-cup-drone) in the academic workspace `badecar-danmarks-tekniske-universitet-dtu`. It was created and verified with team-only (`PRIVATE`) visibility on 17 September 2026, using the signed-in `badecar` account. Browser authentication is separate from SDK authentication on each training machine.

These changes apply to newly deployed code snapshots only. The initial HPC jobs and `mypc-yolo26x-20260917-v2` keep tracking disabled; do not restart or retrofit them for W&B.

Future launches default to **offline** tracking. This records W&B history locally without login or uploading anything. Online mode requires an explicit entity, preventing accidental logging to a default account. `--tracking disabled` remains available without the W&B dependency.

## What is recorded

| Information | Location in W&B |
| --- | --- |
| Detector loss, classifier loss and learning rates | `train/*`, `lr/*` |
| Classifier accuracy, macro accuracy and sample counts for each zoom and represented class | `holdout/L0/*`, `holdout/L1/*`, `holdout/L2/*` |
| Detector diagnostics on its training views | `train_fit/*`, never independent holdout accuracy |
| Epoch duration and cumulative peak PyTorch GPU allocation | `time/epoch_seconds`, `gpu/peak_allocated_gib` |
| Model, seed, hyperparameters, GPU/runtime versions, code commit/source hashes, data counts and manifest hash | Run config |
| Frozen manifest with file hashes, splits and evaluation limitations | Dataset metadata artifact |
| Final results, checkpoint path/hash, runtime, progress and full classifier history including confusion matrices | Run metadata artifact |

The classifier's development holdout covers six classes from one scene. Its accuracy cannot establish detector recall or competition mAP. The detector has no independent, fully labeled holdout in this first dataset. Run configuration preserves these limitations.

No raw training images or checkpoint binaries are uploaded by default. `--wandb-upload-checkpoints` explicitly includes the final checkpoint as a model artifact. Metadata does include local artifact paths and host name for provenance. Console/code auto-capture and Ultralytics' separate W&B callback are disabled; there is one explicitly managed run. W&B may also collect its standard system metrics.

## Install and run

Install the pinned extra in a future run's environment before submission, never in an environment with active training:

```bash
python -m pip install -r requirements-tracking.txt
```

The updated Windows setup script installs this extra from its own code directory. The two existing platform lock files describe the original runs and intentionally remain unchanged. For future runs, save a new dependency freeze alongside the run receipt.

Direct invocation:

```bash
python train.py detector --data /path/to/first-run-v2 --weights /path/to/weights \
  --output /path/to/runs/unique-run-name --epochs 50 \
  --tracking offline --wandb-project nordic-ai-cup-drone --wandb-group baseline-v2
```

For DTU `submit.lsf`, `WANDB_MODE` defaults to `offline` and `WANDB_PROJECT` defaults to the project above. Set `WANDB_ENTITY=badecar-danmarks-tekniske-universitet-dtu` explicitly before selecting `WANDB_MODE=online`. `WANDB_RUN_GROUP` is optional. Preserve `DRONE_COMMIT` and the immutable `DRONE_CODE_ROOT` as usual. A replacement of an original failed job should explicitly retain `WANDB_MODE=disabled`.

For future Windows launches:

```powershell
.\launch-mypc.ps1 -CodeRoot 'C:\path\to\committed-code' -RunId 'unique-run-name' `
  -Commit 'FULL_COMMIT_HASH' -TrackingMode offline -WandbGroup 'baseline-v2'
```

After SDK authentication is configured, use `-TrackingMode online -WandbEntity 'badecar-danmarks-tekniske-universitet-dtu'`. The launcher passes these settings to the detached process explicitly. Group and project/entity arguments accept letters, numbers, dots, underscores and hyphens. Direct `train.py` has the full W&B flags, including `--wandb-entity badecar-danmarks-tekniske-universitet-dtu`.

Authenticate locally using W&B's interactive login flow. Never paste API keys into chat, source files, command-line arguments, job scripts or receipts. No credentials are copied between machines by these scripts.

## Offline files and later upload

Each output folder contains `tracking.json` plus `wandb/offline-run-*`. Back up the complete run folder, including W&B's referenced artifact cache, if it will be moved before syncing. Keep the original machine's W&B cache until sync succeeds. Once the destination and authentication are configured, upload a chosen run explicitly:

```bash
wandb sync --entity badecar-danmarks-tekniske-universitet-dtu --project nordic-ai-cup-drone /path/to/run/wandb/offline-run-TIMESTAMP-ID
```

Do not sync the synthetic local test runs. The first training run has no W&B log and is intentionally kept in its original JSON/Git receipts.

If online initialization fails, the tracker attempts an offline run and records the actual mode. Subsequent provider exceptions are caught, recorded by error type in `tracking-errors.jsonl`, and do not abort training. Local JSON/checkpoints remain the training record. An unavailable SDK is a setup error detected before training; install the extra or choose disabled mode. Normal SDK network retries may add delay, so offline mode is appropriate for restricted HPC compute nodes.

## Verification

```bash
python -m unittest -v test_tracking.py
python -m py_compile train.py tracking.py
bash -n submit.lsf
```

The CPU-only suite exercises the real pinned W&B SDK in offline mode using synthetic metrics and a synthetic manifest. It checks persisted output, zoom namespaces, explicit entity selection, provider-error isolation, disabled mode without the SDK, and optional checkpoint uploads. It does not launch GPU training or validate access to an online workspace.

References: [W&B initialization and modes](https://docs.wandb.ai/models/ref/python/functions/init), [W&B settings](https://docs.wandb.ai/models/ref/python/experiments/settings), [Ultralytics W&B integration](https://docs.ultralytics.com/integrations/weights-biases/).
