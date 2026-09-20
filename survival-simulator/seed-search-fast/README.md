# Faster terrain prefix scan for seed recovery

20 September 2026. A drop-in replacement for the `terrain_filter` in
`seed-recovery-v4` (branch `codex/seed-recovery-v4-handover-20260920`), with the same
CLI, the same stdout contract and the same candidate file format. C++ only; no Python
in the scan path.

**Measured on this laptop: 2.16x single-threaded, 1.39x on 20 threads.** Verified
bit-identical to CPython's own `random.Random` on 274 pinned vectors and 98,578 swept
seeds, with zero mismatches.

Read [the caveat about what this actually buys](#what-this-does-not-fix) before
planning around it — the prefix scan is roughly 15% of V4's live end-to-end time.

## Why the old kernel was slow

V4's `terrain_filter` reseeds Mersenne Twister per candidate and reads the first ~50
output words to recover the 10 Voronoi sites and 10 biome types. CPython's
`init_by_array` seeding is 1,247 steps over a 624-word state, and each step is

```
mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * K)) + seed
```

which is a **serial dependency chain**: shift, xor, multiply, xor, add, with the
multiply alone costing 3-4 cycles. The critical path is ~8 cycles per step, about
10,000 cycles per 16-lane group — while the actual work is only ~5 vector ops per step.
The kernel was **latency-bound, not throughput-bound**, and spent most of its cycles
waiting on that chain. On top of that it streamed a 40 KB `vector_initial[624]` array
and materialised a 40 KB seeded state per group, neither of which fits L1.

## What changed

1. **Interleave independent chains.** `GROUPS` seed groups advance in one loop body, so
   the ~8-cycle latency of each chain is filled with the other groups' work. This is
   the single biggest win — measured in isolation, 8 lanes went from 0.78 to 1.79 M
   seeds/s going from 1 group to 8.
2. **Broadcast `initial[]` from scalars.** It is the same value in every lane, so the
   40 KB vector copy was 16x wasted bandwidth. The scalar array is 2.5 KB and stays in L1.
3. **Never materialise the seeded state.** Pass A carries the loop-1 chain in a register
   to obtain `mt[1]`; pass B re-runs loop 1 *fused* with loop 2. The two chains are
   independent within a step, so pass B costs one chain's latency while doing two chains'
   work — the recompute is free. Only the 129 low words the twist needs are stored.
4. **Window the output.** Only 128 output words are produced, twisted and tempered on
   the fly, instead of a 256-word cache over a full 624-word state.

Net effect on the working set: **880 words/lane to 257 words/lane**.

### How many output words are actually needed

`randbelow(1600)` and `randbelow(1200)` take 11 bits with rejection, `randbelow(4)` takes
3 bits with rejection, so the 20 coordinates plus 10 types consume a variable number of
words. Measured over 2,000,000 seeds (`probe_words` methodology, reproduced in the test):

| window | seeds covered | max observed |
|---|---|---|
| 64 words | 98.54% | |
| 96 words | 99.99995% | |
| **128 words** | **100%** | **97** |

The kernel uses 128 and keeps an exact scalar CPython fallback for any lane that would
overrun it, so a window overflow changes throughput, never a result. No fallback fired in
any test run.

## Measurements

Intel Core i7-13800H (6 P-cores + 8 E-cores, 20 logical), Windows 11, MinGW g++ 13.2.0,
`-O3 -march=native -ffp-contract=off`. **This CPU has AVX2 but no AVX-512**, so
`LANES=8` is one `ymm`; `LANES=16` builds and passes correctness but is emulated here and
measured slower. Best of 3-4 alternating rounds, because the laptop throttles.

| config | baseline | fast | speedup |
|---|---|---|---|
| 1 thread, `LANES=8 GROUPS=10` | 0.906 M seeds/s | **1.954 M seeds/s** | **2.16x** |
| 20 threads, `LANES=8 GROUPS=4` | 8.085 M seeds/s | **11.209 M seeds/s** | **1.39x** |

Projected full 2^32 scan at 20 threads: **531 s to 383 s**.

`GROUPS` has a different optimum depending on load, because each group adds ~8 KB of
per-thread working set:

| GROUPS | 1 thread | 20 threads |
|---|---|---|
| 2 | 1.09 | 10.37 |
| 3 | 1.46 | 11.93 |
| **4** | 1.60 | **12.54** |
| 6 | 1.76 | 12.10 |
| 8 | 1.79 | 11.88 |
| **10** | **1.94** | — |

**Tune `GROUPS` on the target host.** Use ~4 when every core is busy, ~10 when the scan
has a machine to itself. The 20-thread speedup is lower than the single-thread figure
because this laptop's E-cores and memory bandwidth saturate first; a 96-core EPYC with
AVX-512 and 12-channel DDR5 has not been measured and could land anywhere between the two.

## Localisation uncertainty: declare a radius

The sample file is `x y label [radius]`. The fourth column is optional and defaults to 0.

- **radius omitted (0)** — exact test: the observed label must equal the nearest site's type.
  Integer squared distances. This is what `reliability.py` uses, because its sample
  positions come from the engine.
- **radius > 0** — relaxed test, the same one the Rust filter this replaces implements
  (`../scripts/seed_search_rust/src/lib.rs:240-254`) and the Python reference mirrors
  (`../scripts/seed_survey_probe.py:100-117`):

  ```
  d(p, nearest site of the observed type)  <=  d(p, nearest site of any type) + 2r
  ```

The receipt reports which test ran (`"test":"exact"` or `"relaxed"`) and the `max_radius`.

**Why this column exists.** An earlier version of this filter had no radius at all, which is
a false-negative channel — the one failure mode that actually breaks a recovery claim. A
live agent's estimated position carries r ≈ 1.5 px (geometric registration) to 12 px
(shared world estimator), and the exact test then rejects the **true** seed on any sample
whose integer pixel lookup flipped. Measured here, 40 boundary-sited samples under a rigid
10 px bias:

| test | candidates | true seed retained |
|---|---:|---|
| exact (no radius) | **0** | **no** |
| relaxed (r declared) | 1 | **yes** |

The exact run reports `"complete":true, "hits":0` — **indistinguishable from "the seed is
not in this range"**. Boundary-sited samples are exactly what a good acquisition policy
collects, since biome transitions are the most informative observations.

**The relaxed test is sound, and tight.** If the estimate `p` is within `r` of the pixel `q`
whose label was read, then `d(p,S_lab) <= d(q,S_lab)+r = d(q,S_any)+r <= d(p,S_any)+2r`. So
a correctly declared radius can **never** reject the true seed. An *understated* one
silently can, and the candidate yield barely moves when it does — yield does not detect it.

**So do not guess the radius tightly.** Survivor sets are nested (`survivors(r) ⊆
survivors(r')` for `r <= r'`), so scan once at a generous radius and refine downward with
`refine_candidates`, which costs milliseconds. If the list empties at some level, that
identifies the radius at which the evidence became self-contradictory — a bad anchor —
rather than reporting "seed not in range".

## Time to seed — the number that matters

`T_seed = T_acquire + T_scan + T_refine`. V4's live recovery was 132 s, of which the search
was 19.2 s, so acquisition was ~113 s and nobody had optimised it. Both terms were measured
this session.

### T_scan: 9.9 s for the whole domain, on one GPU

`terrain_filter_cuda.cu`, full `[0, 2^32)`, one RTX 4090 (RunPod, CUDA 12.4, `-arch=sm_89`):

| kernel | full 2^32 | rate | notes |
|---|---:|---:|---|
| this laptop, 20 threads | 324 s | 13.2 M/s | |
| RunPod 128 vCPU | 361 s | 11.9 M/s | shared vCPUs, worse per-thread than the laptop |
| V4 production, 6 shards x 24 threads | 19.2 s | 224 M/s | historical receipt |
| **RTX 4090, `PFX_WINDOW=128`** | **9.86 s** | **436 M/s** | **no window overflow possible** |
| RTX 4090, `PFX_WINDOW=96` | 7.21 s | 596 M/s | 83 overflow seeds emitted for exact CPU re-check |

**One consumer GPU is 2x faster than the entire six-machine cluster, and 33x this laptop.**

Verified, not assumed: the GPU reproduced the CPU filter's candidate list **exactly** on a
4M-seed range containing the live-recovered seed, and the two window sizes agree on the
full domain. The kernel arithmetic lives in `prefix96.hpp` and is checked on the host by
`tests/test_prefix96.cpp` against CPython vectors (49,426 seeds, 0 mismatches) — that test
is what vouches for the GPU math on a machine that cannot build CUDA.

**Use `PFX_WINDOW=128` (the default).** The 96-word window is 2.6 s faster but leaves 83
seeds in 4.29 billion unresolved. An overflow is **not** a rejection: those seeds are
emitted to `<candidates>.overflow` and must be re-checked exactly, or the true seed could
be missed. 2.6 s is not worth a caveat on a reliability claim.

```sh
nvcc -O3 -arch=sm_89 -o terrain_filter_cuda terrain_filter_cuda.cu   # sm_89 = Ada/4090
./terrain_filter_cuda samples.txt 0 4294967296 1 candidates.txt
```

### T_acquire: ~15 simulated seconds, not 113

`acquisition_curve.py` drives the real engine with five agents fanning outward and measures
what the pooled public `biome` observations are worth. Median over 8 seeds, projecting
full-domain candidates from a 1% probe:

| sim seconds | samples | biomes | span px | projected candidates over 2^32 |
|---:|---:|---:|---:|---:|
| 2 | 11 | 3 | 992 | 558,700 |
| 5 | 20 | 3 | 1050 | 33,200 |
| **10** | **34** | **3** | **1066** | **400** |
| **15** | **46** | **3** | **1353** | **~0 (unique)** |
| 30+ | 63-71 | 3 | 1409 | ~0 |

**Five agents walking outward have enough evidence to pin the seed at ~15 simulated
seconds.** V4's threshold of 128 samples across 3 biomes (or 400 across 2 after 180 s) is
roughly 8x more than a scan needs. The curve also saturates by t=30: walking longer adds
samples but no information, because the agents have already crossed the biome boundaries
that matter.

**This is a floor, not a live estimate.** It uses the engine's true positions. A live agent
must localise itself: relative displacement is cheap (movement is deterministic given our
commands and the observed biome penalty), but the absolute offset needs wall anchoring, and
V4's relaxed radius test needs more samples for the same discriminating power.

### Where that leaves the number

| | V4, measured | with GPU + early dispatch |
|---|---:|---:|
| acquire | ~113 s | ~15-20 s **[floor]** |
| scan | 19.2 s | **9.9 s** |
| refine | ~0 | ~0 |
| **total** | **132 s** | **~25-30 s [E]** |

The scan is no longer the bottleneck and, per the algebraic analysis, cannot be made faster
by better mathematics — all 1,247 seeding steps are load-bearing. **Everything left is in
acquisition and localisation.**

## Reliability: recovering seeds we did not choose

`reliability.py` is the harness for the claim that actually matters — not "it is fast" but
**"given only public terrain observations from a map we did not generate, we recover the
exact seed, every time, with no false positives, having searched the whole 2^32 domain."**

How the harness keeps itself honest:

- Targets are drawn **uniformly from the full uint32 range** by a reproducible RNG. Not
  small, not sequential, not hand-picked.
- Observations come from the **real engine** (`fastsim.Engine.biome_map()`), not from a
  reimplementation of the generator. Verified separately: the engine's rendered map agrees
  with the filter's Voronoi model on **every** non-river pixel across seeds 1, 614466944
  and 4000000000 (11,277 land samples, 0 disagreements). River pixels are skipped — the
  river overwrites the land map and carries no information about it.
- Samples follow a **random walk**, so they are clustered and connected the way a moving
  agent's observations would be, not conveniently spread over the map.
- The seed is passed to the recovery pipeline **nowhere**. It builds the world, and at the
  very end it marks the answer right or wrong. Nothing in between sees it.
- The scan covers `[0, 2^32)` **in full**, and a run that does not complete is reported as
  incomplete rather than counted as a success.

```sh
python reliability.py --targets 32 --scan-samples 24 --out runs/reliability
```

Measured, 24 targets drawn uniformly from the full uint32 domain, full 2^32 scan with 64
walked samples each and 192 for refinement:

```
dispatched              : 24/24  (held by the gate: 0)
uniquely recovered      : 23/24 of dispatched
true seed retained      : 24/24
false positives (total) : 3
complete domain scan    : True
```

**The true seed survived in 24 of 24 — no false negatives.** One target retained 4
survivors rather than 1; the correct seed was among them, so it resolves with more samples.
That is the failure mode to expect: ambiguity, never loss.

Exit status is 0 only if the domain was fully scanned, every target was uniquely
recovered, and there were zero false positives. `summary.json` records every target, its
candidate count after the scan, its survivors, and the receipt from the scan itself.

Because the filter accepts several sample files in one pass, all targets are tested in a
**single** sweep of the domain. That amortises well for a reliability study, but note what
it does and does not say: in a live game there is one target, so the cost is the full sweep
(~324 s here, ~19 s on V4's 6-shard cluster), not the amortised per-target figure.

## How fast is seed finding, end to end

Measured, not projected: a complete run over the **entire 2^32 domain** on one laptop
(i7-13800H, 20 threads), samples taken from the live-recovered seed `614466944`.

| stage | input | output | time |
|---|---:|---:|---|
| terrain scan, full 2^32, 24 samples | 4,294,967,296 seeds | 1,552 candidates | **324.0 s** |
| refine to 32 samples | 1,552 | 895 | 0.0032 s |
| refine to 64 samples | 895 | 101 | 0.0027 s |
| refine to 128 samples | 101 | **1 — correct** | 0.0026 s |
| *(upper bound)* materialise all 1,552 candidate worlds | 1,552 | — | 3.3 s |

**Total: 324 s — about 5.4 minutes on a single laptop, from nothing to a unique seed.**
Sustained scan rate 13.25 M seeds/s. The refine stages together cost **8.5 ms**, which is
0.003% of the run, so the scan *is* the cost.

Per-candidate verification is bounded above by a full `Engine` construction (map
generation, biome render, 80 rocks, starting entities), measured at **42.1 ms**. V4's
`fast_walls` is cheaper still — it skips entity creation and bit-tricks the render — so
3.3 s is a ceiling, not an estimate.

### Against V4 in production

V4's live run used 6 shards x 24 threads and scanned the full domain in **19.2 s**, at
36.2 M seeds/s per 20-thread worker (`seed-recovery-v4/evidence/coverage-audit.json`).
Applying the 1.39x measured under full thread load projects **~14 s** for the same
cluster. That is a projection across CPU families — the EPYC 9654 has AVX-512 and far more
memory bandwidth than this laptop, and `LANES=16` has not been measured there.

Which keeps the point from [What this does not fix](#what-this-does-not-fix) in view:
V4's end-to-end was 132 s of which only 19.2 s was search. Going 19.2 s -> 14 s changes
almost nothing. **Acquisition is the thing to attack**, and the refine numbers above are
what makes attacking it practical: once an early, weak-evidence scan has produced a
candidate list, every later observation narrows it for free.

## Correctness

The authority is CPython itself, not a second C++ implementation.

`tests/gen_vectors.py` mirrors the exact call sequence in
[`src/elements/biome.py`](../src/elements/biome.py) `Map_generator.generate`
(`num_biomes=10`, 1600x1200, as instantiated at
[`src/elements/environment.py:38`](../src/elements/environment.py#L38)) using a plain
`random.Random`, and emits ground-truth prefixes.

```
cpython vectors  : 274 checked
sweep domain start          : ok
sweep live seed 614466944   : ok
sweep domain end            : ok
seeds checked    : 98578
window fallbacks : 0 (0.000000%, exact scalar path)
ALL CHECKS PASSED
```

Vectors pin the domain edges (0, 2^32-1, 2^31), small integers, V4's live-recovered seed
`614466944`, V4's regression fixture seed `1854492595`, and a spread of large values.
Sweeps run contiguous ranges at several batch alignments so each seed is exercised in
multiple lane and group positions.

End-to-end, against 40 terrain samples taken from the live-recovered seed:

```
$ ./terrain_filter_fast tests/terrain_sample.txt 605000000 20000000 20 tests/cands.txt
{"tested":20000000,"hits":1,"seconds":1.85927,"seeds_per_second":1.07569e+07,"threads":20}
$ cat tests/cands.txt
614466944
```

## Build and run

```sh
g++ -O3 -march=native -ffp-contract=off -std=c++17 -DLANES=8 -DGROUPS=4 \
    -o terrain_filter_fast terrain_filter_fast.cpp

g++ -O3 -march=native -ffp-contract=off -std=c++17 -DLANES=8 -DGROUPS=4     -o refine_candidates refine_candidates.cpp

python tests/gen_vectors.py > tests/vectors.txt
g++ -O2 -march=native -std=c++17 -DLANES=8 -DGROUPS=4 -I. -o tests/test_prefix tests/test_prefix.cpp
./tests/test_prefix tests/vectors.txt
```

Always pass `-DLANES`; the header defaults to 16, which warns about the AVX-512 ABI on
a machine without AVX-512.

```
refine_candidates CANDIDATES_IN SAMPLES THREADS CANDIDATES_OUT
```

Same CLI as V4, so the frozen streaming coordinator can call it unchanged:

```
terrain_filter_fast SAMPLES START COUNT THREADS CANDIDATES_OUT [STOP_FILE]
```

Keep `-ffp-contract=off` and never add `-ffast-math`, per the V4 handover. Build on the
CPU that will run it; `-march=native` binaries are not portable.

On Linux with AVX-512 (the EPYC production target) try `-DLANES=16` and re-tune `GROUPS`.
On Windows, `LANES=16` needs the `aligned(16)` fallback already in `mt_prefix.hpp`,
because MinGW cannot over-align stack locals past 16 bytes.

## What this does not fix

V4's live recovery was **132.0 s end-to-end**, of which the search was **19.2 s**
(`seed-recovery-v4/evidence/last-verified-snapshot.json`). The rest is observation
acquisition — V4 waits for 128 anchored samples across three biomes, or 400 across two
after 180 simulated seconds, and recovery landed at 84.9 simulated seconds.

So a 2x prefix scan saves roughly 10 s of 132 s. **If live recovery needs to be
meaningfully earlier, the work is in acquisition, not in this kernel.**

Also unchanged: the ~24,671 candidates that survive the terrain filter across the full
domain still need native wall verification and replay. That stage was not touched here.

## Scan early, then refine - measured

V4 waits for its acquisition threshold and then sweeps the domain. It does not have to.
Survivors collapse very fast in the sample count, so a scan started on partial evidence
already produces a list small enough to carry forward.

Survivors over a fixed 100 M-seed range, samples taken from the live-recovered seed
(`uniform` = spread over the map, `clustered` = a Gaussian patch, closer to what agents
actually walk):

| samples | clustered | proj. over 2^32 | uniform | proj. over 2^32 |
|---:|---:|---:|---:|---:|
| 4 | 2,558,604 | 110 M | 1,018,053 | 44 M |
| 8 | 49,269 | 2.1 M | 26,014 | 1.1 M |
| **12** | **8,955** | **385 k** | 450 | 19 k |
| 16 | 1,093 | 47 k | 21 | 902 |
| 24 | 161 | 6.9 k | 1 | unique |
| 32 | 12 | 515 | 1 | unique |
| 48 | 2 | 86 | 1 | unique |
| 64 | 1 | unique | 1 | unique |

`refine_candidates` re-tests a retained list against new samples instead of re-sweeping.
End to end, starting from a 12-sample scan and refining as more samples arrive:

```
scan  12 samples, 100M seeds        8955 kept   7.75   s
refine to 16 samples   8955 ->   1093           0.0070 s
refine to 24 samples   1093 ->    161           0.0033 s
refine to 32 samples    161 ->     12           0.0027 s
refine to 48 samples     12 ->      2           0.0026 s
refine to 64 samples      2 ->      1           0.0024 s   -> 614466944
```

**Each refinement costs 2-7 ms against 7.75 s for the equivalent rescan**, and the
sequence converges on exactly the live-recovered seed. Scaled to the full domain, the
first scan retains ~385 k candidates at 12 samples, which refine in a few hundred ms.

The consequence for the live pipeline: the scan can start as soon as ~12 anchored samples
exist, rather than at V4's 128-sample / 180-second threshold, and every later sample is
then applied for free.

**Caveat, and it matters.** These samples are exact positions. Real acquisition carries
localisation uncertainty, and V4's filter correspondingly uses a relaxed radius test,
which is why it needs many more samples for the same discriminating power - V4 retains
24,671 candidates over the full domain with 128 real samples, far more than this table
would suggest. Treat the curve as the *shape* of the collapse, not as a sample budget for
live use. The sample-count axis has to be re-measured with real radii before it drives a
live threshold.
