# Seed recovery and seed-aware policy — status report

20 September 2026. Written for the team working on the Nordic Cup survival-simulator entry.

Every number is labelled **[M] measured**, **[E] estimated from measured inputs**, or
**[S] speculative**. Results that contradict something we previously believed are called
out explicitly, because several of them do.

---

## 1. Headline

**Seed recovery works, it is reliable, and it is fast.** Given only public terrain
observations from a map we did not generate, we recover the exact 32-bit seed by searching
the entire 2^32 domain. On 24 targets drawn uniformly from the full domain: **23/24
uniquely recovered, and the true seed retained in 24/24 — zero losses.** The full-domain
scan takes **9.86 s on one RTX 4090**, which is faster than the six-machine cluster that
did it in 19.2 s.

**Seed-aware policy is a much weaker story, and honesty requires saying so.** The largest
credible legitimate-play result is a search policy measuring **+163%** — but against a
deliberately weak baseline, and the blocker to measuring it against the real one is
unresolved. Several seed exploits that sound compelling have been measured at
approximately zero.

**Where the remaining time goes: acquisition, not search.** That is now the whole game.

---

## 2. Seed recovery — where the time goes

`T_seed = T_acquire + T_scan + T_refine`. V4's live recovery was 132 s, of which the search
was 19.2 s. **86% of it was acquisition**, and nobody had optimised it.

### 2.1 Scan [M]

Full `[0, 2^32)`, all verified to reproduce the same candidate list:

| platform | time | rate |
|---|---:|---:|
| laptop, 20 threads | 324 s | 13.2 M/s |
| RunPod 128 vCPU | 361 s | 11.9 M/s (shared vCPUs; worse per-thread than the laptop) |
| V4 production, 6 machines x 24 threads | 19.2 s | 224 M/s |
| **RTX 4090, `PFX_WINDOW=128`** | **9.86 s** | **436 M/s** |
| RTX 4090, `PFX_WINDOW=96` | 7.21 s | 596 M/s (83 seeds unresolved, re-checked on CPU) |

The 96-word window is 2.6 s faster but leaves 83 seeds in 4.29 billion unresolved. **An
overflow is not a rejection** — those seeds are emitted to a `.overflow` file and must be
re-checked, or the true seed could be silently missed. Default is 128, which cannot
overflow. 2.6 s is not worth a caveat on a reliability claim.

### 2.2 Acquisition [M, floor]

Five agents fanning outward, median over 8 seeds, projecting full-domain candidates:

| sim seconds | samples | biomes | projected candidates over 2^32 |
|---:|---:|---:|---:|
| 2 | 11 | 3 | 558,700 |
| 5 | 20 | 3 | 33,200 |
| 10 | 34 | 3 | 400 |
| **15** | **46** | **3** | **unique** |

V4 waits for 128 samples across 3 biomes, or 400 across 2 after 180 s. That is **~8x more
than a scan needs.** The curve saturates by t=30: walking longer adds samples but no
information, because the agents have already crossed the boundaries that matter.

This is a **floor** — it uses the engine's true positions. Accounting for real localisation
(§2.5), the realistic budget is **20-25 simulated seconds, 24-32 samples, ≥3 distinct
labels, ≥1,000 px span** — still ~3.5x cheaper than V4.

### 2.3 Where that leaves time-to-seed

| | V4 [M] | now [E] |
|---|---:|---:|
| acquire | ~113 s | 15-25 s |
| scan | 19.2 s | 9.9 s |
| refine | ~0 | ~0 |
| **total** | **132 s** | **~25-35 s** |

### 2.4 Start early, narrow continuously [M]

The scan costs ~10 s regardless of how good the evidence is, so there is no reason to wait
for conclusive evidence before dispatching it. `stream_filter.cpp` implements this: it
derives each candidate's terrain prefix **once**, then each new observation costs ten
integer distance computations per surviving candidate.

Measured end to end on seed `3141592653`, dispatching at **t = 2 simulated seconds** with
only **14 samples spanning 2 biomes**:

```
full 2^32 scan on that evidence  -> 26,165,409 candidates retained (0.6% of the domain)
cache their prefixes             -> 22.1 s, 1.46 GB
sample  1 : 26,165,409 -> 25,226,090   3.2 s
sample  9 : 17,393,477 ->    449,543   1.7 s      <- one high-information observation
sample 13 :    350,568 ->        947   0.2 s
sample 14 :        947 ->        513   56 microseconds
sample 21 :         45 ->         37   2 microseconds
final                  ->          1   = 3141592653
```

Two things to take from this. **Samples are wildly unequal** — #9 removed 97% of the list
and #13 removed 99.7%, while others removed 3%. That is what makes sample *selection* worth
optimising. And **once the list is small, narrowing is free**: below ~1,000 candidates an
observation costs microseconds, so the seed converges the moment the evidence arrives.

**The new bottleneck is the 22 s cache step**, which is pure MT seeding that the GPU scan
*already did* and threw away. Emitting each candidate's prefix alongside its seed would
remove it entirely. **[E]** That is the single clearest next optimisation.

### 2.5 Reliability [M]

24 targets drawn uniformly from the full uint32 domain, observations from the **real
engine**, seed never passed to the recovery pipeline, complete 2^32 scan:

```
dispatched              : 24/24
uniquely recovered      : 23/24
true seed retained      : 24/24
false positives         : 3
complete domain scan    : True
```

**The true seed survived every time.** One target kept 4 survivors with the correct seed
among them. **The failure mode is ambiguity, never loss** — which is the property you want,
because ambiguity resolves with more samples and loss does not.

### 2.6 Two correctness bugs found and fixed [M]

Both were mine, both were silent, and both are the kind that make a reliability claim false
without making it *look* false.

**The filter had no uncertainty radius.** It did an exact nearest-site test; the Rust filter
it replaces implements the localisation-tolerant `+2r` test. A live agent's position
estimate carries r ≈ 1.5 px (geometric registration) to 12 px (shared estimator), so the
exact test rejects the **true** seed on any sample whose integer pixel lookup flipped.
Demonstrated with 40 boundary-sited samples under a 10 px bias:

| test | candidates | true seed retained |
|---|---:|---|
| exact (no radius) | **0** | **no** |
| relaxed (radius declared) | 1 | **yes** |

The exact run reports `"complete":true, "hits":0` — **indistinguishable from "the seed is
not in this range"**. Boundary samples are exactly what a good acquisition policy collects.
Fixed: the sample file is now `x y label [radius]`, and the receipt says which test ran.

The relaxed test is **provably sound and tight** — a correctly declared radius can never
reject the true seed. An *understated* one silently can, and the candidate yield barely
moves when it does, so yield cannot detect it. Therefore: **scan at a generous radius and
refine downward**, since survivor sets are nested and refinement costs milliseconds.

**A threading crash that was hiding in plain sight.** A refactor made GCC spill `ymm`
registers to the stack, and MinGW's thread entry only guarantees 16-byte stack alignment,
so `vmovdqa %ymm0,0x70(%rsp)` faulted in worker threads non-deterministically. Found under
gdb; no `-mstackrealign`-class flag fixed it reliably. The fix was to restore the original
register-resident hot path. Worth knowing: **the original code was only accidentally
correct** — the same latent fault was always there and earlier full-domain runs got lucky
on frame layout.

### 2.7 Silent-failure guards now in the filter [M]

Three failure modes that previously returned "complete, 0 candidates" — indistinguishable
from "seed not in range" — now fail loudly: a river label (which can never match any seed),
a malformed sample line (which used to silently truncate the sample set), and a
single-biome sample set (which now warns, since it is consistent with roughly a third of
the domain). The scanner also handles multi-word CPython keys, so a seed at or above 2^32
becomes a bounded window to scan rather than an invalidated claim.

### 2.8 Algorithmic efficiency: there is none left in the scan [M]

Verified three independent ways:

- seed → first output is **not GF(2)-affine** (200/200 violations of linearity)
- **no 2-adic low-bit closure**, so no Minecraft-style structure-seed/world-seed split
- truncating the seeding by a **single step** gives 0/400 matches — all 1,247 steps are
  load-bearing, so there is no cheap pre-filter

Algebraic inversion needs six *complete* 32-bit outputs at stream positions
{0,1,2,227,228,229}; we observe 11-bit-truncated words at rejection-shifted offsets, and
227-230 land inside the river walk. Adapted cost: **2^167**. SAT/SMT's best published
`init_by_array` case is ≈ our runtime and assumes full outputs.

For calibration, a published paper reports ~140 minutes for this sweep on Xeon Golds. We
are at 9.86 s — **~850x ahead of the literature.** Brute force is the right algorithm here,
not a stopgap.

---

## 3. Seed-aware policy — what it is actually worth

### 3.1 The reframe that matters [M]

Score is almost entirely survival time: `+dt` every tick, `+fruit.energy/1000` on eating,
`−agent.energy/100` when eaten. Over 128 baseline games: time 1720, food +163, predation
−157, total 1726, and **0 of 128 runs reached the 3000 s horizon.**

Three consequences no heuristic author would encode:

1. **Population is worth exactly zero.** One agent alive scores the same per tick as thirty.
2. **Fruit is worth almost nothing as score** (163 points a game) but is the entire
   instrumental currency.
3. **Agents are cheap to lose** — an agent eaten at 20 energy costs 0.2 points, two ticks of
   colony survival. A low-energy agent is a near-free decoy.

Headroom is ~+1400, against the +84 that heuristic peeking delivered.

### 3.2 Results, ranked by confidence

| what | result | confidence |
|---|---|---|
| **Rollout search** (40 s tail, high frequency) | **830 → 2185, +163%** | **[M]**, n=3 seeds, weak baseline |
| **A\* routing** vs engine reflex | **2.15x survival** | **[M]**, n=16, but see below |
| **Harvest scheduling** on predicted spawns | +40% at current survival; +96% if surviving the full game | **[E]** from measured inputs |
| **Birth-phase predator dodging** | predators 1.7→0.8, score +51% | **[M]** but **confounded** |
| Arrival-time-aware foraging | **+131 ± 164 ticks (t=0.80) — nothing** | **[M]** |

### 3.3 The findings that cost the most to learn

**Truncated-horizon search is unreliable in sign, and that explains the prior +4.9%.** At a
40-second lookahead you see **0.8%** of the available value discrimination and pick the true
best action 30% of the time. At one tick — which is what the earlier effort did — you see
0.0%. Per-seed deltas for low-frequency truncated search: **+799, −575, +235, +302**.

**Beam search loses at equal compute.** 1704 vs 1158 on an identical 399-rollout budget. Its
deeper plies refine the *second* macro, which receding-horizon commitment then discards.

**Most of the A\* win is not about the map.** A map-free Bug2 (wall-following with one
memory bit) captures **64-68%** of the reflex→A* gap. The map plus planner is worth a
further **~24%** — significant without predators (t=2.83), **not significant with them**
(t=1.72). And the deployed policy already has a trap-escape detector, so it sits between
reflex and Bug2, not at reflex.

**The exploits that sound best are worth nothing.** Exact predator position: agents outrun
predators and the predator refuses to approach an agent facing it. Exact agent `max_age`:
crossing it causes a 10x drain spike the policy sees in the next observation. Early fruit
arrival: optimised the wrong end of the window — a fruit is worth 20 energy on arrival and
60 twenty seconds later.

**The pattern:** foreknowledge only pays when the information is genuinely hidden, acted on
*before* it would become visible anyway, and changes a decision that matters.

### 3.4 Machinery that now exists [M]

`survival-simulator/mirror/` adds `clone()`, `snapshot()` and `restore()` to the fastsim
engine **without touching `_engine.cpp`** (it is provenance-pinned; the `#include`
arrangement `_policy.cpp` documents is used instead). Measured: **snapshot 2 µs, restore
1 µs**, against an engine step of 60-200 µs — 476x cheaper than `clone()`. Branch-and-replay
is verified bit-identical to full game horizon (13k-20k ticks).

Also verified: **exact branch-and-replay cannot desync**, because it is the same function of
the same inputs. The "shared RNG stream" concern that shaped earlier designs is not a
blocker. Only two things in the whole engine are action-coupled: births (which we control
exactly) and predator wander (2 draws, ~7% of awake-predator-ticks). **The entire food
supply curve of a seed is exactly forecastable given a committed action plan.**

**CORRECTION.** An earlier version of this claimed the food supply is action-INDEPENDENT.
That is wrong, and Lucas caught it (`codex/seed-shadow:docs/seed-shadow/NIKOLAJ_REVIEW.md`).
The tree and fruit *mechanisms* depend only on tree count, age, biome and time, and no agent
action touches those quantities. But they draw from the same `e.rng` as `spawn_agent`, so a
birth re-phases the stream and changes which draws the tree and fruit checks consume. The
schedule is exactly forecastable *conditional on a committed action plan*; it is not
invariant to the plan. The original wording also contradicted our own birth-phase result,
which works precisely by re-phasing that stream.

---

## 4. Open risks

1. **Is the seed even in [0, 2^32)?** `src/core.py:22` defaults to `randint(0, 2**32-1)`,
   but the README says evaluation uses *preset* seeds, and `random.Random` accepts arbitrary
   ints, floats, strings and bytes. A string seed produces a 17-word key and is
   unrecoverable by enumeration. **Partly mitigated** — the scanner now takes multi-word
   keys, so a clock-derived seed is a bounded window. **This is the only risk that
   invalidates the approach rather than degrading it; worth asking the organisers.**
2. **The shipped policy is not snapshot-safe.** It keeps a `shared_ptr` graph of
   `Mind`/`Group` objects, so a rewind leaks branch state into the trunk. **Every policy
   number above is therefore measured against a weaker substrate than we would ship.** Two
   independent investigations named fixing this as the top follow-up.
3. **Replay parity against the live server is not guaranteed.** The terrain prefix is pure
   integer arithmetic and ports exactly; full replay involves NumPy float loops and
   identity-hashed observation sets and is not even process-stable. Any shadow model needs a
   resync tripwire. Useful calibration **[M]**: floating-point-level drift perturbs a branch
   value by sd ≈ 8 against a between-branch signal of ≈ 90 and is survivable; an
   event-level desync is not.
4. **Acquisition soundness fails silently, discrimination fails safe.** A weak sample set
   returns a huge candidate list and you keep collecting — safe. An understated radius
   returns zero or a confident wrong answer — silent. All reliability effort should go to
   the soundness side.

---

## 5. What to do next, in order

1. **Emit candidate prefixes from the GPU scan.** Removes the 22 s cache step from the
   streaming path — the clearest remaining win on time-to-seed. **[E]**
2. **Attack acquisition, not scan speed.** Going 9.9 s → 5 s on a ~30 s pipeline is noise.
   Sample *selection* is the lever: some observations removed 97% of the candidate list and
   others removed 3%.
3. **Make the shipped policy snapshot-safe.** It is the prerequisite for any deployable
   policy number.
4. **Settle the seed-range question** with the organisers.
5. **Resolve the birth-phase confound** (equal births in both arms) before trusting +51%.

---

## 6. Where the code is

| path | what |
|---|---|
| `seed-search-fast/terrain_filter_fast.cpp` | CPU scanner, drop-in for V4's `terrain_filter` |
| `seed-search-fast/terrain_filter_cuda.cu` | GPU scanner, 9.86 s full domain |
| `seed-search-fast/stream_filter.cpp` | cache-once streaming narrower |
| `seed-search-fast/refine_candidates.cpp` | one-shot refinement of a retained list |
| `seed-search-fast/reliability.py` | the blind recovery harness |
| `seed-search-fast/acquisition_curve.py` | evidence-vs-time measurement |
| `mirror/` | `clone`/`snapshot`/`restore` for shadow models |
| `docs/seed_advantage_preliminary_2026-09-20.md` | the earlier policy-exploit analysis |

All of it is untracked. Nothing has been committed.

---

## 7. The lookup-table question, and why the first API call is the wrong target

Asked: can we build the landmark lookup table once, store it on a Hetzner box, and query it
the moment the first API call arrives?

**Build and storage: yes, easily. Query at the first API call: no, and nothing can.**

### 7.1 The build is cheap; moving it is not [M/E]

The index is "for each of 4.29 B seeds, the land label at K landmark pixels". Computing that
is the same work as one scan — the 10 Voronoi sites and types — plus ~10 integer ops per
landmark.

| | K=12 | K=16 |
|---|---:|---:|
| bits/seed | 24 | 32 |
| size | **12.9 GB** | **17.2 GB** |
| compute | ~10 s on a 4090 **[M]**, ~4 min on 8 CPU threads **[E]** | same |

Lucas measured the same order independently: 56.8 s per 67 M-seed shard on one Hetzner vCPU,
16 GiB for the full index, and a warm query of 0.111 s
(`origin/codex/seed-shadow`, `docs/seed-shadow/fingerprint-pilot.json`).

Uploading 17 GB at 100 Mbit/s is ~23 minutes; **rebuilding it in place on the Hetzner box
costs ~4-5 minutes of its own CPU.** So this artefact should be built where it is used and
never shipped.

A design constraint that is easy to miss: a `seed -> labels` table does not help, because
answering "which seeds match this pattern" would mean scanning all 17 GB — no faster than
rescanning the domain. It must be **inverted**, with seeds bucketed by landmark signature,
so an observation maps straight to a bucket.

### 7.2 Why it cannot answer the first API call [M]

The index keys on labels at **exact** pixels. A live agent does not know where it is. It has
to anchor itself first, by registering rock-edge constellations and pinning them against map
boundary walls.

Measured, from the held-out seed-78431 survey
(`docs/seed_survey_heldout_2026-09-19.json`, four checkpoints at 0.7 / 10 / 20 / 30 s):

| sim time | anchored agents | eligible samples | selected samples |
|---:|---:|---:|---:|
| **0.7 s** | **0** | **0** | **0** |
| 10 s | 1 | 14 | 14 |
| 20 s | 5 | 70 | 64 |
| 30 s | 5 | 106 | 64 |

`docs/seed_recovery_survey_2026-09-19.md:56` states it directly: *"at 0.7 seconds the shared
model had no anchored agent, so the filter correctly rejected nothing."*

**Beware a contradiction in our own docs.** `docs/seed_evidence_2026-09-20.md:54` says the
survey "recovered three spawn positions by 0.7 seconds". That is **retrospective** —
`seed_survey_evidence.py` reconstructs where an agent *was* at 0.7 s using rock geometry
observed much later. It is not a live capability, and reading it as one would put a
position-keyed index in the plan on false grounds.

So at the first API call there is no position, therefore no landmark label, therefore no
index key. This is not an implementation gap; it is the physical situation. **The binding
constraint is that agents must walk far enough to anchor and to cross biome boundaries**, and
no amount of precomputation moves it.

### 7.3 What the Hetzner box is genuinely good for [E]

Reframed, the hardware question has a clear answer that runs the *other* way from the GPU
result:

- **A warm always-on box beats an on-demand GPU for live play.** Provisioning a RunPod
  instance took ~2-3 minutes in this session. That dwarfs a 9.86 s scan. For a live game you
  want compute that is already running.
- **17 GB fits in a GEX44's 64 GB RAM.** Pre-warm it into page cache (`vmtouch`, or simply
  `cat index > /dev/null` at boot) and the lookup is memory-speed whenever it does become
  queryable.
- Rent GPUs for offline and batch work — building the index, sweeping parameters, regression
  runs. Use the persistent box for anything on the live clock.

### 7.4 What the index actually saves, honestly [E]

Using the anchoring timeline above:

| stage | without index | with warm index |
|---|---:|---:|
| acquire to a usable frame | ~20 s | ~20 s |
| scan | 9.9 s (GPU) / 234 s (8-thread CPU) | ~0.1 s |
| **total** | **~30 s** (GPU) | **~20 s** |

**It saves roughly 10 s of a ~30 s pipeline, and zero of the acquisition that dominates it.**
That is worth having if the index is free to keep around, and it is not worth reorganising
the plan around. On a CPU-only box the saving is much larger (234 s to 0.1 s), which is the
strongest argument for it.

### 7.5 The tension worth understanding

There are two classes of evidence and they trade off exactly against each other:

- **Cheap to index, needs a position.** The Voronoi prefix is the first ~50 MT words, so any
  seed's biome map is computable in microseconds — but querying it needs absolute
  coordinates we do not have early.
- **Position-free, expensive to index.** Rock rectangle dimensions are invariant to
  translation and rotation, so they are queryable from the very first frame with no
  localisation at all. But rocks are drawn *after* the 1.92 M-draw biome render, so
  generating them per candidate costs milliseconds rather than microseconds. A full-domain
  rock index is ~1.4 TB raw and days of CPU **[E]** — not buildable.

That trade-off is why V4 is built the way it is, and it is the thing to attack if anyone
wants recovery meaningfully earlier than ~20 simulated seconds.

### 7.6 Recommendation

1. **Do not plan around answering at the first API call.** Plan around ~20 s.
2. Build the index **on** the Hetzner box, never upload it. ~5 minutes, in place.
3. Pre-warm it into page cache at boot so it is hot before the game starts.
4. Keep the streaming filter as the primary mechanism regardless — it already delivers the
   "start immediately, narrow continuously" behaviour, and unlike the index it needs no
   absolute position to begin.
5. If someone wants recovery before ~20 s, the only lever is §7.5: find a pose-invariant
   observable that is cheap to index. That is a research question, not an engineering one.

---

## 8. Can we beat 400 seconds? Measured: yes, by roughly 10x

The 400 s benchmark is beaten by every configuration we have, including CPU-only. This
section replaces the estimate in §2.3 with a measurement of the piece that estimate was
resting on: **does realistic-quality acquisition actually resolve the seed?**

### 8.1 The experiment [M]

Four targets drawn uniformly from the full uint32 domain. Five agents fanning outward.
Samples taken at the agents' positions, then **perturbed by 8 px and declaring a 9.41 px
radius** (8 px is the shared world estimator's measured accuracy of 6 + uncertainty, observed
7.0-8.3 px; the extra sqrt(2) is the integer-pixel lookup margin). Complete `[0, 2^32)`
scan using the relaxed `+2r` test.

**At t = 20 simulated seconds** (46-64 samples, 4 distinct biomes each):

| target | candidates | truth retained |
|---|---:|---|
| 3048614886 | 24 | yes |
| 548576473 | 10 | yes |
| 1760855471 | **1 — recovered** | yes |
| 1881884024 | 7 | yes |

**4.29 billion seeds down to between 1 and 24, and the true seed retained in 4/4.**

**Then extend the same walk to t = 30 s** and refine the retained lists:

```
target 0:  24 ->  2 candidates in   5.0 ms
target 1:  10 ->  1 candidate  in  12.6 ms
target 2:   1 ->  1 candidate  in  32.6 ms
target 3:   7 ->  1 candidate  in  16.8 ms
```

**3 of 4 uniquely recovered at t=30 s, 4 of 4 retained, worst case 2 candidates.** Refinement
costs **milliseconds**. Two candidates can be separated by full wall generation at 42.1 ms
each — about 0.1 s.

### 8.2 The resulting budget [M components, E total]

Acquisition runs on the game clock; the scan can overlap it because the scan is dispatched
on partial evidence (§2.4). V4's live run gives the wall-per-sim ratio: 132.03 s end-to-end
for 84.9 sim s, minus 19.18 s of search and 18.59 s of deliberate pacing, so ~1.1-1.3x.

| | 8-thread CPU | warm RTX 4090 |
|---|---:|---:|
| acquire to t=20 s sim | ~22-26 s | ~22-26 s |
| full-domain scan (overlaps further acquisition) | 234 s | **9.9 s** |
| refine with the samples that arrived meanwhile | ~0.03 s | ~0.03 s |
| separate any residual pair by wall generation | ~0.1 s | ~0.1 s |
| **seed known at** | **~260 s** | **~35 s** |

**Both beat 400 s.** The GPU path beats it by ~11x and beats V4's own 132 s live run by ~3.5x.

### 8.3 Where the remaining time actually is

Of that ~35 s, **~25 s is acquisition and ~10 s is the scan**. Two consequences:

- Further scan optimisation is nearly worthless. Halving 9.9 s buys 5 s of 35.
- **Provisioning latency would dominate everything.** Spinning up a RunPod instance took 2-3
  minutes in this session. Whatever runs the scan must already be running when the game
  starts — which is the real argument for a persistent box, independent of any lookup table
  (§7.3).

### 8.4 What this does and does not establish

**Does:** the filter resolves the full 2^32 domain to 1-2 candidates from the sample quality
a real 20-30 second survey produces, at the real estimator's positional accuracy, with the
true seed never lost. The relaxed radius test behaves exactly as its proof says.

**Does not:** this perturbs true positions by a realistic amount rather than running the
localisation chain. It models *how accurate* positions are, not *whether they can be
obtained*. The end-to-end observation-only harness is still unbuilt — it was the workflow
that died on the API spend limit.

The 20 s acquisition figure therefore still rests on V4's held-out survey (5 anchored agents,
64 selected samples at t=20 s, `docs/seed_survey_heldout_2026-09-19.json`), which is genuine
observation-only measurement but a single seed. **That is the one soft term in the 35 s
budget, and closing it is the highest-value remaining work.**
