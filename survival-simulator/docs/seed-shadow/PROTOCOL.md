# Seed recovery and shadow simulation

Worktree: `/home/Ucals/.codex/worktrees/seed-shadow`, branch `codex/seed-shadow`.
Base: `0682a4c` (finished engine 3ccd187 plus fast policy and completed late-gated5 work).

User objective: recover the unknown game seed from normal observations, retain own
actions, replay from the recovered seed, then develop a stronger model-assisted
controller. Test unmodified Python and the fastest native engine. Up to ten Runpod
pods authorized; no SSH PC. Keep CPU pods running. No official validation attempt.
User clarified the seed finder has **600 seconds of computation**, excluding time
gathering observations. Count search, parsing and candidate replay. Runtime figures
are elapsed compute-phase wall time at the stated CPU allocation, not aggregate CPU
seconds. Current scalar full-domain case also records aggregate worker seconds.
If the interpretation should instead be ten aggregate CPU-minutes, the scalar
fallback does NOT meet it (about 4.57 aggregate CPU-hours in this case).

Requested development cadence: evaluate latest working controller on 1000 full games
every 15 minutes. Do not call supplied-seed/oracle experiments end-to-end recovery.
Do not run repeated unchanged baseline evaluations to pretend the new bot exists.
Before large policy tuning, retain the existing mean-full-game-runtime <20s gate;
report seed search/replay compute separately and overall latency honestly.

## Upstream

Oscar's `origin/codex/seed-aware-policy-2200` at
`5638ee77a25d8f6b1af3d9901cf7df742da2836b` contains a terrain filter and absolute
boundary anchoring, but its score benchmark supplies current engine truth. Its own
README states this is an upper bound, not live recovery. Copied README in UPSTREAM.md
and original helpers in oscar-seed-helpers.cpp. User does not know another location.

## Implemented

- `models/seed_shadow/public_terrain.py`: only public observations enter. Boundary
  anchoring, shared agent poses and non-river terrain constraints. Age advancement
  filters stale observation caches after an agent-list removal skips an update.
- `models/seed_shadow/scan.cpp`: exhaustive candidate filtering over any uint32 range.
  `scripts/build_seed_scan.py` extracts PyRandom from the exact engine implementation.
- `scripts/scan_seed_domain.py`: parallel, resumable, input-hashed chunks and explicit
  complete-domain metadata. New runs default to 540s search allowance, preserving
  time for parsing and replay within 600s. Partial results are never marked complete.
- `models/seed_shadow/replay.py`: own-action/public-response journal, independent
  replay, unique candidate acceptance, continuous public consistency checks and
  revocation on mismatch. Compute clock starts when searching, not while gathering.
- `scripts/verify_seed_search.py`: requires complete uint32 coverage and proves the
  sampled terrain was derived from the supplied public transcript. No evaluator seed
  or hidden state is passed to recovery.
- `models/seed_shadow/rng_linear.py`: experimental GF(2) MT19937 state reconstruction
  from partial outputs at KNOWN stream positions. Supports rounded uniform values
  and skipped draws. It returns a consistent state hypothesis: withheld outputs must
  verify it. This is NOT yet wired to game observations or original-seed inversion.

## Evidence so far

1. Bounded range [0,65536) recovered seed12345 in native and seed1 in unmodified Python
   from 50 simulated seconds. These only validate plumbing; they are not full-domain
   recovery. Absent seed1234567 correctly returned no match.
2. Full range [0,2^32) on one natural native map, after 180 simulated seconds:
   2965 public terrain samples, 10736 pose checks, maximum position reconstruction
   error 1.82e-12. Exactly one candidate: 1894581302. Search521.583s, candidate replay
   14.370s, search + parsing + replay542.719s (~9m03s). 32vCPU pod; worker-seconds16457.641.
   No hidden test seed entered the scanner or verifier. This is ONE case, not a success
   rate estimate. Full-domain-result.json contains recorded result/provenance.
3. Unmodified Python vs native, identical orchard-generated commands:
   seed1 and seed2 matched full projected world within absolute1e-8 throughout games
   lasting2185.2s and1746.3s. Seed3 diverged in predator position at2060.5s despite
   identical RNG state. Public observation ordering differs much earlier because
   Python uses identity-hashed sets. Exact replay is therefore not guaranteed.
   Existing fastsim/verify.py patches Python identity hashes; these tests do not.
4. Journal checks passed: persistent actions, wrong seed rejection, unfinished search
   rejection, catch-up, ongoing shadow advancement, mismatch revocation and deadline.
5. Algebraic inversion SYNTHETIC controls:700 exact random() values ->1000/1000 held-out
   outputs in0.223s.1600 uniform(30,100) values with1e-10 uncertainty ->1000/1000 in0.735s.
   The1000-rounded-value test got966/1000 and is retained as a negative control: fitting
   observed values alone is insufficient. No game seed was recovered by this method.

## Next work, in priority order

1. User asks for more creative, consistent recovery: investigate recovering usable RNG
   bits from wall lengths/positions, initial fruit IDs/positions, and observed birth
   mutations/headings. The hard parts are stream ordering, missing draws, numeric
   uncertainty and reaching sufficient information. Synthetic raw-RNG tests are not
   evidence of solving these. A direct six-output seed inversion needs specific early
   outputs; game rendering consumes many draws before obstacles, so cannot apply it
   directly to arbitrary observed coordinates.
2. Audit terrain extraction across many random maps and improve observation gathering
   to sample biome boundaries. Keep a strong exhaustive fallback; vectorized seed
   initialization and sharing seed-prefix work across sample sets could accelerate it.
3. Measure full-domain recovery on multiple fresh uint32 seeds under600 compute seconds,
   including unsuccessful/ambiguous cases. Do not constrain evaluation seeds to the
   search prefix or pass the seed to policy code. Keep every result.
4. Build actual live controller: normal strongest orchard before recovery, inferred
   independent model afterward, immediate observation-only fallback on divergence.
   All actions that actually execute must enter the journal. Current files are recovery
   infrastructure, not this finished controller. Address memory: retaining every public
   frame in Python RAM is too expensive for large full-game batches; journal actions
   durably and use streaming/checkpoints for observations.
5. Only then launch honest1000-game comparisons at requested cadence with score CIs,
   paired baseline, recovery rates, acquisition time, compute time, desynchronization
   frequency, policy CPU time per tick and full-game runtime. Conditional known-seed
   oracle ablations may be useful but must be clearly separate.

## Runpod / reproduction

Created this session: `ft7k34t881e55j` (`lucas-seed-shadow-01`),32vCPU64GB cpu3c,
`runpod/base:1.4.0-rc.164-ubuntu2404`,20GB disk, $0.96/hour compute. Leave running.
SSH host213.173.105.94 port17138, key `/home/Ucals/.ssh/runpod_codex_team`.
Remote `/workspace/lucas-seed-shadow/`: scan, scan.cpp, seed_random.inc,
scan_seed_domain.py, samples.txt, full-domain/ with1024 chunk results and manifest.
The initial full-domain run predates the540s scanner timeout patch and completed
normally. Local full-domain-search/ has summary files; all public transcript frames
are local `full-domain-case/public-journal.jsonl.gz` (17MB, deliberately Git-ignored).
The current pod only has standalone scanner tooling; game/policy deployment still
needed for future evaluations. Other running Oscar pod rejected this SSH key after
restart; do not assume access or interrupt its work. Re-read endpoints after restart.

Project Python: `/home/Ucals/projects/NordicCupAI/.venv/bin/python`.
Build engine/policy in this worktree using fastsim/build.py and build_policy.py.
Full recovery command:

```
python survival-simulator/scripts/verify_seed_search.py \
 --journal survival-simulator/docs/seed-shadow/full-domain-case/public-journal.jsonl.gz \
 --search survival-simulator/docs/seed-shadow/full-domain-search \
 --samples survival-simulator/docs/seed-shadow/full-domain-case/samples.txt
```

Commit/push with `PATH=/tmp/lucas-lfs/git-lfs-3.8.0:$PATH`.

## Research sources

- CPython algorithm: https://github.com/python/cpython/blob/main/Modules/_randommodule.c
- Original six-output attack and its early-output assumption:
  https://stackered.com/blog/python-random-prediction/
- Prior truncated-output reconstruction: https://github.com/fx5/not_random

The GF(2) implementation here was written independently against the native RNG
transition/tempering constants, not vendored from these repositories.


## Update: Oscar V4 and Hetzner reference CPU (20 September)

This section supersedes earlier infrastructure status above. The user's runtime
reference is the existing Hetzner server, **2 vCPUs AMD EPYC Milan**. Target under
10 minutes of compute, preferably under one minute. The 32-vCPU Runpod result
above does not establish either target on Hetzner.

Oscar pushed `codex/seed-recovery-v4-handover-20260920` at `ad55ea0`. The pristine
`survival-simulator/seed-recovery-v4/` package was imported using git archive,
without merging unrelated branch changes. Read HANDOVER.md and retain its hashes.
Its historical 19.18-second full-domain live recovery used six processes with
24 threads each. It includes 5,598 matching mid-run observations but lacks final
whole-game parity evidence. This is Oscar's historical evidence, not our fresh test.

Fresh tests: package preflight passes all hashes, 762 public terrain samples and
historical coverage receipts. Standalone terrain scanner compiled on Hetzner with
strict floating point and retains positive seed 1854492595. Its first 1,048,576-seed
scan took 1.21689 seconds on one vCPU (861,683 seeds/sec). Our eight-lane scanner
measured 1,284,740 seeds/sec on the same host previously. Different public sample
sets were used; these are preliminary throughput figures, not a controlled
head-to-head comparison. Neither rate supports the desired full-domain latency.
The fixed-thread V4 coordinator has NOT been launched on the two-vCPU server.
No production service was changed or competition validation submitted.

New public-observation audit: 1,000 random maps, first 180 simulated seconds each,
retained the true seed in 1,000/1,000 cases. Maximum public pose error was
1.154e-11; no false candidates appeared in the tested 65,536-seed prefix.
This is **not 1,000 full-game recoveries**, nor proof of global uniqueness.
The audit exposed and fixed zero-distance parent/child heading propagation;
atan2(0,0) cannot infer the relative heading. Reports: audit-1000/.

New public geometry and birth extractors retain uncertainty rather than inventing
stream ordering. Python birth test: 82 blocks, 85 births, 1,075 inferred draw
constraints, zero mismatches. RNG tracing is harness-only, never extractor input.
Wall generation order and inter-block RNG offsets remain unknown. Synthetic
linear inversion does not yet solve live seed recovery.

The Runpod now has engine/policy dependencies and the completed public audit in
/workspace/lucas-seed-shadow/observation-audit-1000-v2. Another task also uses this
pod under /workspace/seed-validation-live; do not interrupt or alter its process.
Hetzner test files are only in /tmp/codex-seed-benchmark; leave live services alone.

Next: benchmark V4 wall verification and replay with its pinned Python 3.12 /
NumPy 2.3.5 environment. Investigate reducing candidate work (inversion or explicit
one-time indexed preprocessing), measuring preprocessing separately from online
recovery. Do not imply SIMD alone satisfies the small-CPU target. Oscar's updated
seed-aware-policy branch c717732 reports mode46 improved oracle-model scores;
that still requires verified shadow state before honest controller integration.
