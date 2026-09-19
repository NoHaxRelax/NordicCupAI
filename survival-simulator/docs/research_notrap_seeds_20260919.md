# No-trapping seed selection (2026-09-19)

Status: **branch survey complete, head-to-head screening in progress, nothing
launched**. Written for the `sim-optimization-no-trapping` campaign: optimize
colony survival/score without predator trapping, keeping the shared-map
coordination logic. No paid job has been started; the campaign plan at the end
is awaiting operator confirmation.

Every number labelled *measured here* comes from this repository's reference
Python engine (`src/core.py`), natural predator spawning left enabled
(`starting_predators=0`, the engine's own time-increasing spawn chance, never
disabled and never forced), policies receiving only public per-agent
observations and simulation time. Numbers labelled *claimed upstream* come from
other branches' own reports on their own harnesses and are not reproduced here.

## Operator priority

1. The starting point this branch shipped with (`OptimizationPolicy`) went
   **extinct in 2 of 2 completed screening games** at the 900 s horizon. It is
   not a safe default and must not be treated as the seed by inertia.
2. Oscar's orchard forager, which contains **no predator handling whatsoever**,
   survived 3 of 3 at the same horizon on the same seeds. The colony-economy
   half of the problem dominates the predator half at this horizon.
3. Trapping is not the bar to beat. The repository's own full-horizon trapping
   benchmark reached 3,000 s in **0 of 991 games**.

## Branch survey

Surveyed: `survival-simulator/entrapment`, `entrapment-9059-benchmark`,
`entrapment-AI-attempt`, `lucas-experimental`, `lucas-trap-slopsesh1`,
`oscar-fastsim`, `oscar-orchard-population`, `oscar-overnight-cpp`,
`oscar-trapper`, `codex/survival-trap-handoff-2026-09-17`, `challenge1_nikolaj`,
`prep`.

| Branch | Verdict |
| --- | --- |
| `entrapment`, `entrapment-AI-attempt`, `lucas-trap-slopsesh1`, `oscar-trapper`, `codex/survival-trap-handoff-2026-09-17` | Trapping strategies and trap research only. Out of scope. Their own handoffs flag results as 60–300 s proxies with full-game behaviour unestablished. |
| `entrapment-9059-benchmark`, `lucas-experimental` | Hybrid. Contain the separable non-trapping exploration stack that this branch already carries as `models/exploration`. Source of the 991-game full-horizon trapping benchmark. |
| `oscar-orchard-population` | **Non-trapping candidate.** `OrchardPolicy` foraging/population/lifecycle play with lineage- and contact-merged shared frames. |
| `oscar-overnight-cpp` | Infrastructure plus the evasion evidence below. Carries trap machinery that its own notes measure as negative. |
| `oscar-fastsim` | C++ engine port, no new strategy. Relevant only as a possible throughput multiplier. |
| `challenge1_nikolaj` | Identical commit to this branch's fork point; nothing extra. |
| `prep` | Unrelated subproject. |

Two correction to the premises this branch started from:

- The handoff described `models/survival/oscar_orchard.py` as tracking "its OWN
  independent per-agent position estimate" and never touching a shared map.
  Its own docstring contradicts that: lineages share a frame and a map of
  trees, fruit and coverage cells, and groups merge on mutual sighting. It is
  not globally coordinated like `ExpertPolicy`, but it is not per-agent either.
- The in-repo `models/survival/oscar_orchard.py` is an **older revision** than
  `oscar-orchard-population`'s. The tuned configs from that line address
  parameters the older file swallows through `**_`, so applying them to the
  in-repo copy would silently no-op. The newer revision is therefore vendored
  as `models/survival/orchard_population.py`.

## Measured here: 900 s screening, seeds 0/1/2, natural predators

| Policy | Survived | Mean score | Population arc |
| --- | --- | --- | --- |
| `OptimizationPolicy` (branch default: ExpertPolicy + harvest) | **0 / 3** — extinct at 733 s, 858 s, 795 s | 833.4 | Peaks at 41–44 agents, then collapses to zero. |
| `OrchardPolicy`, stock defaults, no predator logic | **3 / 3 to horizon** | 670.6 | Peak 32–47, decays to 7–10. |
| `OrchardEvasionPolicy`, tuned, evasion disabled | 2 / 2 completed | 731.1 | Peak 66–101, decays to 3–6. Seed 2 pending. |
| `OrchardEvasionPolicy`, tuned + evasion | 1 / 2 completed, extinct at 828 s on seed 1 | 781.7 | Peak 71–84. Seed 2 pending. |

`OptimizationPolicy` scores higher per unit time while alive and still loses,
because extinction ends scoring. Under this repository's established ranking
(survival first, score as tiebreak) it is behind a policy that cannot even see
predators.

**The 900 s screening horizon cannot settle the evasion question.** Predator
spawn chance rises with elapsed time, so a 900 s game contains only a handful of
predators; the evasion layer has little to respond to while still paying for
fleeing and sprinting. Evasion-on scored higher (781.7 vs 731.1) but lost one
game the evasion-off ablation survived, on two seeds. That is noise, not a
result. Whatever separates these variants should appear between roughly 1,500 s
and 3,000 s, as predator pressure accumulates. Campaign games therefore run the
full 3,000 s horizon; 900 s was a screening device for choosing seeds, and its
verdict on evasion parameters specifically should not be trusted.

## Claimed upstream, not reproduced here

- `oscar-overnight-cpp` `notes/BEST.md`: on its own C++ engine port, with
  predators, tuned foraging plus a small-radius own-sighting evasion layer
  reached ~1,507–1,533 s mean survival against ~888 s for the same forager with
  no predator logic. Wide-radius evasion using shared sightings measured
  *worse* (923 s) than narrow-radius own-sighting evasion (1,239 s). Every trap
  variant it tested was negative against that baseline (baits alone −57 ± 27 s;
  baits + guides −140 ± 27 s).
- `oscar-orchard-population` `HANDOFF.md`: ~2,530 s mean survival, 5 of 32 games
  reaching 3,000 s — **with no predator spawning**. Not a comparable condition.
- `entrapment_9059` full-horizon benchmark: 0 of 991 games reached 3,000 s,
  median survival ~621 s, median score ~639, for the integrated trapping policy.

The evasion parameter defaults adopted here (`pred_r` 70, `pred_face_r` 80,
`pred_sprint_r` 40, `pred_dodge_r` 80, `pred_dodge_ang` 1.4, own sightings only)
are taken from that campaign's `with_predators_best` config. They are a
starting point chosen from someone else's measurement on a different engine,
not a result established in this repository.

## Selected seeds

**Family A — `orchard_evasion` (primary).** `models/orchard_evasion_policy.py`:
the vendored newer orchard forager plus a new own-sighting evasion layer that
flees inside `pred_r`, breaks across the predator's approach inside
`pred_dodge_r`, sprints inside `pred_sprint_r`, and otherwise turns to keep a
watched predator in vision. Evidence: the only non-trapping code in the
repository that survives the screening horizon, plus the strongest upstream
full-horizon evidence. Group-shared predator sightings are deliberately not
implemented; upstream measured sharing worse, and `pred_share` rejects nonzero
values rather than silently ignoring them.

**Family B — `expert_harvest` (secondary).** `models/optimization_policy.py`:
the shared-map exploration, planning, crowding and territory stack with
coordinated harvest enabled. Retained despite losing the screening because it is
the only family that actually builds the central shared map the branch mandate
is about, its collapse looks like a population/energy-economy failure rather
than a mapping failure, and 198 of its parameters have never been searched with
harvest enabled. It is a genuine research question, not a favourite.

Oscar's `OrchardPolicy` is not proposed as a standalone seed: with zero predator
awareness it is Family A with `pred_mode=0`, which the search covers as an
ablation.

## What is unverified

- No candidate has been run to the real 3,000 s horizon in this repository.
  All screening here is 900 s, chosen for turnaround, and 900 s survival does
  not establish 3,000 s survival.
- Three seeds is a very small sample; seeds 0/1/2 differ by >90 score points
  between neighbouring seeds for the same policy.
- The evasion layer has had one 300 s functional run. Its parameters are
  inherited, not fitted here.
- Native/Python engine equivalence for any `fastsim` throughput shortcut has
  not been rechecked for these families.

## Campaign launched (2026-09-19 09:39 UTC)

Twelve `cpu3c` Pods (32 vCPU, 64 GB, $0.96/h) in `EU-RO-1`, six per family, each
running an independent `tune` study over training seeds 0–3 at the full 3,000 s
horizon with a distinct search seed. All Pods share the network volume, so they
read one source tree at `/workspace/NordicCupAI-notrap` and their source hashes
agree; the previous campaign's `/workspace/NordicCupAI` and
`/workspace/research-night-1` were left untouched. The guard is armed with a
verified **$77.24** ceiling and a 5.75 h deadline per Pod.

**Provisioning error worth recording.** The first twelve Pods came up with an
RTX 4090 each and 16 vCPU. Runpod's `POST /v1/pods` defaults `computeType` to
`GPU`, and `cpuFlavorIds`/`vcpuCount` are ignored unless it is explicitly set to
`CPU`, so the request silently produced GPU Pods for a workload with no CUDA
path. They were terminated after about twenty minutes, roughly **$3**, which is
carried in the guard's `prior_spend_usd`. The corrected request doubles the
compute for 1.3x the price: 384 vCPU at $11.52/h against 192 vCPU at $8.88/h.
Any future launcher must set `computeType` explicitly.

Authorized envelope: **~$80**, up to **12 Pods**, speed preferred over compute
savings. Independent-shard design: each Pod runs its own `tune` study with a
distinct search seed over a shared training seed set, because the optimizer
takes a study lock and does not support concurrent coordinators on one study.
Finalists from every shard are then re-evaluated on a common held-out seed set
before anything is selected.

Search space is `models/notrap_config.py`: 288 configuration leaves, 269
tunable — 71 in Family A (64 orchard, 7 evasion) and 198 in Family B (expert and
planner). Fixed entries carry explicit reasons: simulator mechanics, rendering
options, coordinator invariants, one categorical, and five orchard constructor
arguments the source never reads.

Reuse of existing machinery: the campaign runs through the existing `run.py tune`
block evolutionary search (controls, then one probe per tunable scalar
round-robined across blocks, then evolutionary candidates with parentage from
the best five), the existing spawned JSON policy boundary in
`models/experiment_actor.py`, and the existing case cache, study lock, resume
and phase structure. New code is confined to the parameter space
(`models/notrap_config.py`), the two policy entry points, a `--space` /
`--family` selector, and reporting. The `trapping` space remains the default and
its plan is unchanged at 328 entries / 273 tunable.

Reporting adds `progress.png` to `report.md` on every write, rendered by
`scripts/research_progress_plot.py`: primary objective, mean survival, mean
score, and objective against cumulative game wall-hours, each showing completed
trials behind the incumbent best. The incumbent is selected by the optimizer's
own lexicographic rank rather than an independent per-metric maximum, which
would otherwise draw a champion no single candidate achieved. The score panel
exists because survival saturates once candidates reach the horizon, at which
point score is the only thing still separating them.

Spend is bounded by `scripts/runpod_notrap_guard.py`, a separate guard holding
this campaign's authorization ($80 total, $8 reserve, at most 12 Pods, at most
12 h and $1.50/h per Pod). The trapping campaign's `runpod_fleet_guard.py` and
its $40/four-Pod constants are untouched, so the two campaigns cannot consume
each other's ceiling.

Related: [policy optimization](policy_optimization.md),
[Runpod policy search](runpod_policy_search.md),
[Runpod research launch](runpod_research_launch.md).
