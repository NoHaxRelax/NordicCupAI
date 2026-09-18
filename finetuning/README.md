# Medical extractive QA fine-tuning

Trains a yes/no evidence-span model on the supplied medical-appointment
transcripts. The default is `deepset/deberta-v3-large-squad2`; BERT and RoBERTa
are also supported. There are 390 questions from 39 conversations; four
unaligned positive examples are excluded by default. Folds split by
conversation, so the same transcript never appears in both training and
validation within a fold.

## Windows setup and commands

Install `uv`, then run these commands from the repository root in PowerShell:

```powershell
# Check command-line setup without loading a model.
.\finetuning\run.cmd python scripts/finetune_qa.py --help

# One-fold, one-epoch check with 16 training examples.
# This loads the full pretrained model and still needs model memory/downloads.
.\finetuning\run.cmd python scripts/finetune_qa.py --smoke

# Full default run, including saved weights and held-out predictions.
.\finetuning\run.cmd python scripts/finetune_qa.py --out runs/deberta-01 --save-dir checkpoints/deberta-01 --dump-preds

# Train BERT instead.
.\finetuning\run.cmd python scripts/finetune_qa.py --model deepset/bert-large-uncased-whole-word-masking-squad2 --out runs/bert-01 --save-dir checkpoints/bert-01 --dump-preds
```

`run.cmd` changes into this folder, uses the locked dependencies, and keeps
the environment at `%USERPROFILE%\.venvs\nordicmed` outside OneDrive. Set
`UV_PROJECT_ENVIRONMENT` to choose another location. The environment is a CUDA
12.8 PyTorch build; CPU execution is available with `--cpu`, but large-model
training will be slow. `--optim adamw8bit` additionally requires the optional
dependency: `run.cmd --extra bnb python scripts/finetune_qa.py ...`.

Each run produces metrics JSON/CSV and a plot. Weights are saved only when
`--save-dir` is supplied. Use a fresh output directory each time. Runs do not
resume mid-epoch; `--only-folds 3 --folds 3 --out runs/retry-fold3` reruns a
selected fold into a separate directory without erasing earlier results.
The epoch-by-epoch held-out measurements are validation results, not an
independent final test after selecting an epoch or committee rule.

## Data and lightweight checks

The prepared data is included; rebuilding is optional and reads the adjacent
`medical-appointment/data` and `medical-appointment/transcripts` directories:

```powershell
.\finetuning\run.cmd python scripts/build_qa_dataset.py --out runs/rebuilt-data
.\finetuning\run.cmd python -m unittest discover -s tests -v
```

The tests use a tiny CPU model and require no pretrained-model download.
See [data/DATASET.md](data/DATASET.md) for alignment details. Committee training
and evaluation use `scripts/run_committee.py` and `scripts/committee_eval.py`.
