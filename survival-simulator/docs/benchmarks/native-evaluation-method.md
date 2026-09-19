# Native policy evaluation method

`native_policy/evaluate.py` runs the C++ policy inside the native simulator. The
Python process schedules independent games and records results; it does not make
policy decisions. Use one worker per available core, capped at 10 on the free
12-core SSH host.

The final 100-game seed range must be chosen once, after the controller variant
is frozen. It must not overlap the earlier range beginning at `930001` or known
development seeds such as `1883894846` and `204871`. Pass the agreed first seed
explicitly; there is deliberately no default:

```bash
cd survival-simulator/native_policy
PYTHONPATH=. python evaluate.py \
  --config configs/entrapment-final.json \
  --config-name entrapment_final \
  --seed-start 2026091900 \
  --games 100 --workers 10 --horizon 3000 \
  --out-dir ../logs/native-final-reproduction
```

Each completed game is appended and synced to `games.jsonl`. Repeating the same
command resumes missing seeds. A resume is rejected if the commit, source,
configuration, native extension, runtime, seed range, horizon, or worker count
differs from the manifest. `summary.json` reports mean, median, minimum, maximum,
sample standard deviation, p10, p90, and a deterministic percentile-bootstrap
95% confidence interval for score and simulated duration. It also reports the
number and rate reaching the horizon. `score-histogram.svg` is a standalone
score histogram.

Score, duration, population, and death cause come from native engine state and
events. Controller diagnostics are read from `engine.evaluation()` when the
compiled engine provides it. Metrics absent from that API remain explicitly
`unknown`. In particular, arrival or proximity is never reported as a predator
capture or confirmed retention.

## Final frozen comparison

Controller and evaluator commit: `04173f8ee9d957ff947e6f7d5e73f76ef27750b0`.
Both configurations use the same 100 seeds, `2026091900..2026091999`, with a
3000-second horizon, predators, normal energy, aging and reproduction. The
control uses `configs/with-predators-best.json`; trapping is disabled there.
The final comparison uses one Runpod environment for both configurations.
The independent PC baseline is retained separately and is not pooled with it.

On the 32-core pod, four disjoint 25-game shards use eight workers each. Their
first seeds are 2026091900, 2026091925, 2026091950 and 2026091975. Each completed
entrapment shard frees its eight worker slots for the matching baseline shard.
No strategy changes or seed substitutions are made during the batch.

`scripts/aggregate_native_evaluation.py` requires exactly the declared 100
seeds, verifies result hashes and source/config hashes against frozen Git
blobs, and rejects mixed builds or runtimes. Original shard manifests are
preserved. The pod's minimal checkout marks Git dirty because unrelated root
files are absent and build products are untracked; `source-provenance.json`
records this and verifies every tracked native input against the frozen commit.

`scripts/compare_native_evaluations.py BASELINE_DIR ENTRAPMENT_DIR` compares
each seed with its counterpart and reports a deterministic paired bootstrap
interval for the mean score difference. The two presets also differ in
exploration and avoidance settings, so this compares the complete strategies,
not the isolated causal effect of turning trapping on.

Raw top-level guide counters are legacy debug fields. Use the authoritative
`native_evaluation` fields and aggregate summary for handoff counts.

## Frozen run on `pc`

The configured `pc` host is WSL2 Linux with 12 logical CPUs, 932 GiB free disk,
and passwordless `sudo`. The existing
`/home/lucas/entrapment-research-20260918/.venv` has Python 3.13.14, NumPy
2.5.3, and setuptools 84.0.0. GCC/G++ 13.3 and Make 4.3 are installed, and a
compile probe including `Python.h` and NumPy headers passes. After the final
variant is committed and pushed, fetch only the required directory:

```bash
commit=FROZEN_COMMIT
remote=/home/lucas/NordicCupAI-$commit
ssh pc "git clone --filter=blob:none --no-checkout --depth 1 --single-branch \
  --branch survival-simulator/lucas-experimental \
  https://github.com/NoHaxRelax/NordicCupAI.git '$remote' && \
  cd '$remote' && git sparse-checkout set survival-simulator/native_policy && \
  git checkout -q && test \"\$(git rev-parse HEAD)\" = '$commit'"
```

Compile on the target host so the extension matches its Python ABI and glibc:

```bash
python=/home/lucas/entrapment-research-20260918/.venv/bin/python
ssh pc "cd '$remote/survival-simulator/native_policy' && '$python' nightsim/build.py"
```

After checking that `git rev-parse HEAD` on `pc` equals the recorded frozen
source commit and reviewing the final config name, start the batch in a durable
SSH session:

```bash
ssh pc "cd '$remote/survival-simulator/native_policy' && \
  nohup env PYTHONPATH=. '$python' evaluate.py \
    --config configs/FINAL.json --config-name FINAL_VARIANT \
    --seed-start 2026091900 --games 100 --workers 10 --horizon 3000 \
    --out-dir ../docs/benchmarks/native-final-20260919 \
    > ../docs/benchmarks/native-final-20260919.log 2>&1 & echo \$!"
```

This fixes the inclusive seed range to `2026091900..2026091999`. Copy results
back only after `summary.json` exists, then compare SHA-256 values for
`games.jsonl`, `manifest.json`, `summary.json`, and `score-histogram.svg` on
both machines with `sha256sum`.
