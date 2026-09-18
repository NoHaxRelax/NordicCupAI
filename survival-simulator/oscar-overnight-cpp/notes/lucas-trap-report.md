# Lucas's predator trap: technical report for the C++ port

Prepared 19 Sep 2026 from a read-only clone of `NoHaxRelax/NordicCupAI`
(clone: `/private/tmp/claude-501/-Users-bumblebee-Github-repos-Projects-nordic-ai-cup-2026/4d2b387e-14fb-4694-9f6e-3eac348e1ba8/scratchpad/lucas`,
sparse checkout of `survival-simulator/` at `origin/survival-simulator/lucas-experimental`, commit `53f1a4c`).
Nothing was pushed or committed, no pod was touched, no endpoint was called and no simulation was run.
All paths below are relative to `survival-simulator/` on that branch unless stated otherwise.
**[inferred]** marks conclusions I derived myself. Everything else was read in code or docs.

---

## 0. Which branch is Lucas's latest working trap

| Branch | Last commit (CEST) | Author | What it is |
|---|---|---|---|
| **`survival-simulator/lucas-experimental`** | **2026-09-18 23:06** `53f1a4c` | Lucas | **Latest.** Contains all of `entrapment` and `entrapment-9059-benchmark` as ancestors, then about 30 more commits (18:29 to 23:06) that split the policy into `models/{exploration,survival,entrapment}` + `models/core.py` and iterate on delivery. |
| `survival-simulator/entrapment-9059-benchmark` | 09-18 18:24 `dfef3c5` | Lucas | Frozen baseline `9b0c3e8` plus the 1000-game native benchmark results. It is an ancestor of lucas-experimental (merge-base = `dfef3c5`). |
| `survival-simulator/entrapment` | 09-18 15:24 `91437ef` | Lucas | First guiding policy + 1000-map single-delivery and 30+1 results. Ancestor of the above. |
| `survival-simulator/lucas-trap-slopsesh1` = `entrapment-AI-attempt` (identical tips) | 09-18 08:22 `df3ef96` | Lucas | Separate, older lineage built on Oscar's `codex/survival-trap-handoff-2026-09-17` (`oscar-trap-research/`). Arranged-geometry capacity tests plus real-map delivery attempts. Handover says **zero demonstrated real-map deliveries**. Superseded. |
| `codex/survival-trap-handoff-2026-09-17` | 09-17 16:05 | Badecar (Oscar) | Oscar's original wall-bait / narrow-gap research handoff that the slopsesh1 lineage grew from. |
| `survival-simulator/oscar-trapper` | 09-18 12:41 | Badecar | Oscar's trapper. Lucas copied only its viewer scripts (`scripts/game_viewer.py`, `recording_agent_server.py`, `tick_viewer.html`; blob hashes identical). No trapper logic is imported. |

**Chosen: `survival-simulator/lucas-experimental` @ `53f1a4c`.** It is the most recent by commit date, every commit is by Lucas, and it strictly contains the other two `entrapment*` branches. Its `docs/entrapment_iteration/README.md:3` names it as the working branch, and `overnight_plan.md:4` says to push checkpoints there. The last commits (22:44 to 23:06) are docs and results only. The last *code* change is `eb2ee46` (22:56), which only adds debug fields to `guide_steering.py`. So the policy code is effectively the "selected default" described in `docs/entrapment_iteration/final-summary.json`.

Engine check: `src/elements/{predator,environment,creature,agent,biome}.py`, `src/utils/{sensing,simulation}.py` on this branch are **byte-identical** to our `survival/vendor/survival-simulator/src/...` (cmp run locally). So every engine fact below also holds for our bit-exact C++ port `survival/fastsim/_engine.cpp`.

---

## 1. Language, structure, entry points, results

### 1.1 Language and layout
Pure **Python** (numpy, shapely, pydantic; pygame only for rendering). No Rust or C++. The overnight plan notes Sol trialled our C++ engine in lockstep but kept Python for fixtures (`docs/entrapment_iteration/overnight_plan.md:68-72`).

```
models/core.py                      EntrapmentPolicy: the colony coordinator (379 lines)
models/entrapment/
  entrapment_sites.py               static geometry site enumerator (228)
  observed_trap_sites.py            observed edges -> rectangles -> sites (85)
  my_guide.py                       guide controller (223)
  guide_pathfinding.py              A* with predator-width clearance (191)
  guide_steering.py                 local safety/contact sampler for guides (113)
  predator_following.py             "is the predator following me?" model (138)
  bystander_avoidance.py            avoidance for everyone else (73)
models/exploration/*                Nikolaj's explorer + WorldEstimator (localisation), imported from
                                    challenge1_nikolaj @77eb2a2 (PROVENANCE.md)
models/survival/oscar_orchard.py    copy of Oscar's orchard.py, OLD version (see 5.1)
src/                                vendored native engine (unchanged)
scripts/                            runners, batch evaluators, viewers
```

### 1.2 How a game is run
- Full native game: `python survival-simulator/scripts/entrapment_game.py --seed S --seconds 3000 --out DIR`. It builds `SimulationCore(seed)` and `EntrapmentPolicy(seed)` (`scripts/entrapment_game.py:66-67`) and calls `policy(states_list, sim_time)` each tick. `models/core.py:316` `__call__` returns `[(aid, ActionRequest)]`.
- Viewer: `scripts/entrapment_viewer.py DIR --port 9059`.
- Isolated guide lab (arranged encounter, bait fixture): `scripts/guide_lab.py --seed N [--encounter-seed E]` (`docs/guide_lab.md:70-95`).
- 30 preloaded + N deliveries fixture: `scripts/guide_multi.py --seed M --encounter-seed E --deliveries 1`. It disables ambient spawning, preloads 30 awake predators at the handoff point with energy U(80,200), and refills the bait's energy and age every tick (`scripts/guide_multi.py:60-73, 217, 238`).
- Batches: `scripts/guide_batch.py [--multi] --maps 1000 --workers 32 --seed 9182026`. Full-game benchmark: `scripts/entrapment_benchmark.py --count 1000`.

### 1.3 Measured results (best first, with setup)

**Arranged capacity (older slopsesh1 lineage).** From `oscar-trap-research/LUCAS_TRAP_SLOPSESH1_HANDOVER.md`:
- Gap 15, bait depth 5, seed 483, guided arrivals every 90 s: **33/33 predators acquired and retained for 3000 s**, bait alive, 33 guides sacrificed.
- Bait energy was refilled indefinitely and geometry and arrivals were arranged.
- Direct-arrival controls at gap 19 and gap 11 also held 33/33 for 3000 s.
- Closest predator-bait centre distance seen: **15.0167**. Capture needs < 15, so the margin is tiny.
- Depth 4.9 at gap 19 failed the crowd tests.
- Real-map deliveries in that lineage: **zero**.

**Isolated single delivery, 1000 random maps (`docs/guide_batch_1000_results.md`).**
- **794/1000 delivered (79.4%)**, or 794/953 maps that had a site (83.3%).
- Guide alive in 690 of those, dead in 104. Another 112 guides died without delivering, 47 runs had no delivery by 60 s, and 47 maps had no site.
- Delivery proxy: the tracked predator stays within 40 units of the bait while sensing it for 10 s continuously.
- Setup: bait fixed at full energy with no aging. The guide starts at full energy with an awake, visible predator 80-160 units away, at least 250 from the bait. Static edges are known exactly (perfect static localisation).
- Batch seed 9182026. This is an older policy (single guide, before the 55-unit stop).

**30 preloaded + 1 new delivery, 1000 maps (`docs/guide_multi_1000_results.md`).**
- **682/1000 (68.2%)**, or 682/953 eligible (71.6%). Wilson 95% intervals: 65.2-71.0% overall, 68.6-74.3% eligible.
- Pass = the newcomer is delivered, all 31 are held for the final 30 s, all original 30 are continuously retained, the bait is alive and the rear is clear.
- Failures: 141 delivery or retention, 124 originals left, 6 rear, 47 no site.

**Selected default on fresh maps (`docs/entrapment_iteration/README.md:7-41`, `final-summary.json`).**
- Default = observed-motion association + 55-unit stop + route recovery.
- **74/100 fresh maps (74/98 eligible)**, Wilson 64.6-81.6%.
- Failures: 12 original-group retention, 10 delivery or retention, 2 rear side, 2 no site.
- Development set: 72/100.

**Per-map site validation (`site-coverage-baseline20-report.md`).**
- 19/20 maps have a candidate, and **319/380 validation attempts succeeded (83.9%)**.
- 15/20 maps have a per-site Wilson lower bound above 50%.
- The site is picked offline from simulation outcomes, so this measures site potential, not live selection.

**Whole native games (the only honest end-to-end numbers).**
- **Frozen 9059 baseline, 991/1000 completed games** (`docs/entrapment_9059_benchmark_results/README.md`, `summary.json`), seeds 0-999, native defaults:

  | Metric | Result |
  |---|---|
  | Survived to 3000 s | **0/991** |
  | Survival mean / median | **621.4 / 622.1 s** (range 130-1433) |
  | Score mean / median | **637.5 / 639.4** (max 1491.6) |
  | Physical bait established | 708/991 |
  | Guide delivery arrivals / assignments | 2676 / 14556 (18.4%) |
  | Held30 predators: mean max / median / max | 2.07 / 1 / 11 |
  | Rear occupation | 335/708 baited games |

- **Latest selected default, full-game pilot** (`docs/entrapment_iteration/fullgame-pilot-results.json`), seed 1266967689: **extinction at 284.2 s, score 306.6**.
  - Peak 22 agents, 3 predators, 4 guide assignments / 3 guide deaths, 2 delivery arrivals, 0 predators held for 30 s.
  - 55.9 s bait gap, 558 ticks with no viable bait.
- **Selected default, seed 3, 300 s** (`native-default-seed3-300-summary.json`): 17 agents alive, score 330.7, 3 predators, all 3 near the bait for at least 30 s, 6 guide assignments / 4 deaths, 0 guide releases, 0 bait gap.
- **Release variant, seed 3, 300 s**: 26 assignments / 24 guide deaths, 21 alive.
- **Renewal probe** (`coordinator-renewal-probe.md:27-33`):
  - Pilot seed 1266967689: 284.2 s baseline, 376.9 s with the "spawn guard", 325.4 s with "scarcity + guard".
  - Seed 3: extinct at 208.8 s with the spawn guard; "scarcity + guard" reached the 300 s horizon with 20 agents.
- For comparison, Oscar's no-predator orchard: about 2530 s mean survival, score about 2680 (our docs; Lucas's `overnight_plan.md:56-57` quotes the same about 2680).

**Bottom line [inferred from the above]:** the trap mechanism works when it is arranged. Delivery works about 70-80% of the time when isolated. The integrated colony has never survived past about 1433 s, and the latest integrated default died at 284 s in its one full-game pilot. The colony collapses before predator numbers get large (about 6 predators at 620 s, see 4.3), and Lucas's own analysis blames population renewal (role assignment + age-55 spawn veto + old orchard), not the trap.

---

## 2. The trap

### 2.1 What physically holds a predator
- The trap is a **bait-maintained crevice, not a cage**. Two axis-aligned obstacle faces are parallel, 10.1-19.9 apart, and overlap along the channel by at least 10.3. A radius-5 agent fits. A radius-10 predator cannot, because native collision rejects any endpoint within `radius` of an obstacle AABB (`src/elements/environment.py:781-806`, expanded box `x-r < px < x+w+r`, strict).
- The bait sits `goal = mouth + 5*inward`, **5 units inside the front mouth** (`models/entrapment/entrapment_sites.py:128-131`, `bait_depth=5.0`).
- The predator's closest reachable centre is about 10 outside the mouth, so the distance to the bait is at least about 15. Capture needs centre distance < 15 (`environment.py:717-718`, `distances < size_sum`).
- Why it holds: every awake predator chases the **closest agent it senses** (`src/elements/predator.py:31-44`). The predator's hearing (60) is omnidirectional and ignores walls (`src/elements/creature.py:131, 135-143`). Inside 90 units (`hearing_radius*1.5`) it always direct-chases (`predator.py:37`). A predator at the mouth therefore keeps pressing toward the bait forever. Each tick native collision handling rotates its step 10°, 20°, ... left/right until an endpoint is free, or it stays put (`environment.py:541-556`).
- **Energy is charged before collision** (`environment.py:512-518` then `541`), so a held predator burns energy even when it cannot move. It then rests at 0 energy and wakes above 100 (`environment.py:676-681, 726-728`), and does this forever. **Predators never die or starve.** No code removes a predator; `env.predators` only grows (grep: no removal in `src/`).
- A held predator escapes when:
  - the bait dies or leaves;
  - another agent becomes its closest sensed agent (for example a passing forager or a guide);
  - the geometry lets it slide around the channel end.
- Oscar's handoff (codex branch, `HANDOFF.md` section 1): "These are bait-maintained traps, not permanent cages. A bait that leaves, dies, loses attention or is displaced by a closer agent can release the predator."
- **Tunnelling:** collision is checked only at the step endpoint, and a sprinting predator moves up to 15 per tick. A zero-depth diagonal gate between two corners can be tunnelled through. A channel needs a traversal depth of more than one step (`docs/entrapment_iteration/corner-dogleg-probe.md:27-40`). With overlap ≥ 10.3 plus the 10-unit expansions at both ends, the forbidden stretch is more than 30, so ordinary crevices are safe **[inferred]**.
- Bait safety from the rear: the goal is 5 from the front mouth, so it is `overlap-5 ≥ 5.3` from the rear mouth, giving a centre distance of at least 15.3 from a predator at the rear **[inferred]**. The real risk at the rear is blocking replacement access, not capture.
- Plain inside corners fail: 16/16 bait caught. The 15-unit crevice control: 0/16 caught in 30 s (`entrapment_iteration/README.md:51-52`).

### 2.2 Capacity
- There is no engine limit, because **predators overlap freely**: there is no predator-predator collision (`scripts/guide_multi.py:80` assumption; engine only collides with obstacles).
- Arranged tests held **33** (slopsesh1) and **30+1** (1000-map fixture).
- In native games the most held at once for 30 s was **11**, mean 2.07 (9059 benchmark).
- A crowd increases the chance that one predator's collision-rotated step lands closer. The 15.0167 minimum distance and the depth-4.9 failures show the margin is thin (slopsesh1 handover).

### 2.3 Finding trap sites
Pipeline: observed edges → closed rectangles → site enumerator → handoff filter → choose.

1. **`observed_rectangles(group)`** (`models/entrapment/observed_trap_sites.py:7-48`)
   - Needs an anchored group with a known world size.
   - Splits observed edges into horizontal and vertical segments (tolerance 1.5).
   - An interior rectangle is accepted **only when all four sides have been seen** (two matching horizontals plus both verticals).
   - Arena walls are added once their inner face has been seen, using the public thickness 30.
2. **`enumerate_sites(static, min_gap=10.1, max_gap=19.9, min_overlap=10.3)`** (`entrapment_sites.py:86-191`, called with 10.3 from `observed_trap_sites.py:56`). For every ordered rectangle pair and both axes:
   - Compute `gap = b.x - (a.x+a.w)` and the overlap interval `[low, high]`. Require `10.1 ≤ gap ≤ 19.9` and `overlap ≥ 10.3` (`L113-117`; gap range widened from 19.1 in `b027919`).
   - Try both channel ends as the front `mouth`. Set `goal = mouth + 5*inward` and require it to be free for r = 5.01 (`L123-131`).
   - Approach lane (`L136-161`): try cross-offsets 0, ±2, ... ±26. Require:
     - `far` = 125 out and `hold` = 75 out, both free for r = 11.01, with a clear segment between them;
     - at least one clear runup point at 145-250 out;
     - a clear `hold → mouth` segment (r = 11, ignoring the two channel walls).
   - Rear access: `replacement_entry = other_mouth + 20*inward` must have a straight r = 5.01 path to the goal (`L166-170`).
   - Score = `overlap + 0.05*margin` (`L188`). Records `boundary_indices` (arena-wall-supported channels).
3. **`our_sites`** (`observed_trap_sites.py:51-73`):
   - Add a `handoff` point at 25, 20 or 30 units outside the mouth along the lane. It must be free for r = 11 and at most 44 from the goal.
   - **Corner pockets** (`enumerate_corner_sites`, `entrapment_sites.py:204-228`, uses shapely) are only a fallback when no crevice exists (`available_sites`, `observed_trap_sites.py:76-85`).
   - A corner pocket needs overlap 8-10.3 and a goal at least 15.05 from every predator-valid centre.
4. **Choice:** in `models/core.py:124-129` `_find_site`, the site with the **largest overlap** wins.
   - Rechecked every 2 s (`L98-99`).
   - If re-mapping moves the goal by more than 2 units, the site is **invalidated** and the bait, incoming bait and all tracks are dropped (`L111-123`).
   - Lucas's screen found boundary-supported sites pass 85% vs 63.5% for interior-only (`site-ranking-screen.md:11-17`). That ranking is **not** in production.
5. **Availability:**
   - 953/1000 maps have a site with perfect geometry (`guide_batch_1000_results.md:17`).
   - In live games the policy reported a trap on 966/991 maps, but physical bait was established on only 708 (9059 benchmark; it had a boundary-reconstruction bug).

---

## 3. Bait and lure, step by step

### 3.1 Roles (coordinator `models/core.py`)
- `bait` stands at the goal. `replacement_bait` is walking in. `retired_bait` is an old bait left standing at the goal until it dies.
- `guide` leads one predator track. Everyone else is `explorer` (before a site is known) or `gatherer` (after), wrapped by `avoid_predators`.

### 3.2 Which agent is bait, and replenishment (`core.py:136-192`)
- **Dispatch rule (rolling replacement):** a replacement is dispatched when there is no bait, or when `remaining_life(active bait) < candidate_travel + 20 s` (`L175-186`). It walks in, then **both stay** at the goal. On arrival the old bait becomes `retired_bait` and keeps standing there (`L165-172, 187, 256`).
- **Candidate filter:** in the site's frame group, pose uncertainty ≤ 8, and a route exists to the **rear** `replacement_entry` (never through the predator crowd).
  - Route: `Navigator(clearance=5.05)` from Nikolaj's code, then `rear → goal`.
  - Conservative travel time = distance / (speed × 0.3 × 10) s, which assumes river speed everywhere.
  - Walking cost = distance / 0.3 × 0.05 + 6. Requires `remaining_life(energy - walk_cost, age) ≥ travel + 15` (`L140-149`).
- **Preference (lexicographic max, `L152`):**
  1. arrives before the current bait dies;
  2. is "old" (orchard senescence flag, or age ≥ 55);
  3. higher age if old;
  4. more spare life;
  5. shorter distance.

  Old, dying agents are preferred as bait.
- **`remaining_life(energy, age)`** (`core.py:28-40`): the idle lifetime with no food.
  - It assumes senescence starts at the earliest possible age, 60 s.
  - Young drain is 1/s. Old drain per tick is `0.01*age + 0.1`, solved in closed form with a 2-energy and half-tick margin.
  - Example **[inferred]**: age 20 with 400 energy lasts about 80 s. It spends 40 s young, then goes senescent at age 60 and dies around age 100.
- **Bait movement** (`_bait_action`, `core.py:254-269`):
  - Walk to `replacement_entry`. Once within 3 units, target the goal. Walking speed only (`min(speed, dist/terrain)`), turn capped ±0.3.
  - At the goal (≤ 2 units by estimated pose) it stands still with `action_for(aid)` = all zeros.
  - If blocked it turns 0.2 in place and releases the navigator plan every 3 s.
- **Energy:** the bait never eats and never spawns (spawn is suppressed for all non-explorer/gatherer roles, `core.py:347-348`). It only pays the passive drain, plus the old-age drain once senescent. There is no feeding logic. Replenishment works only by replacement.
- In the benchmark fixtures the bait is refilled to full energy every tick. In native games it is not.

### 3.3 How a predator is attracted and guided (`models/entrapment/my_guide.py`, `guide_steering.py`, `guide_pathfinding.py`, `predator_following.py`)

**Guide assignment** (`core.py:194-252`).
- **Tracks:** every agent projects each Predator observation into the shared group frame using its estimated pose (`L203-221`). It joins the nearest existing track within `min(90, 20 + 150*(now - seen))`, otherwise a new track is made. Track expiry is 8 s, or 30 s while it has a guide (`L195-196`).
- Assignment runs only when the bait has arrived. It applies to tracks more than 60 units from the goal (`L239`) that are seen by an agent with no role.
- **Guide choice:** prefer old or age ≥ 55, then highest energy, then nearest (`L242-243`).
- There is **no energy floor**. 36/113 assigned guides were already below 20% energy, so they were walk-capped and could not sprint (`native-guide-start-audit.md:7-10`).
- At assignment the guide gets a **snapshot of the group's known edges** (`L245-247`).

**Per-tick guide logic** (`my_guide.guide`, `L84-103` → `_guide`, `L156-223`).
1. **Target predator.** In native games it is the coordinator track's observation from this guide (`core.py:277-280`). In the lab it is `_select_target`, a motion association: reject predators within 40 of the bait and gate to 15.75 units per elapsed tick (`my_guide.py:128-153`).
2. **Always look at the predator:** `turn = predator['angle']` (`L164`).
   - This matters in the engine. The predator's `rel_dir` test (`predator.py:37`) treats the agent as "watching" when the predator is in front of the agent's half-plane, `|rel_dir| ≤ π/2`, and ignores the real vision cone and walls (`docs/game_mechanics.md:48-51`).
   - While watched **and** farther than 90, the predator does not charge straight in. It moves at ±45° to the line of sight at sprint 15 (`predator.py:46-62`). That still closes about 0.7 × 15 per tick.
   - Inside 90 it direct-chases regardless of facing.
3. **Moving backwards:** movement direction is relative to the old heading and applied before the turn. Speed does not depend on facing (`environment.py:520-524`; `guide_lab.md:146-150`). The guide walks or sprints **backwards along an A\* route** while facing the predator.
4. **Pace:** `sprint_speed` if the predator is within 100, else walking `speed` (`my_guide.py:205`). Both are capped at the distance to the next waypoint divided by the terrain factor (`L217-222`). There is no circling. The 0.3 rad/tick turn limit is modelled only in the following check (below), not actively exploited. `guide_lab.md:22-23` notes: "A feasible route does not guarantee that the predator follows around corners."
5. **Route** (`guide_pathfinding.py`):
   - A\* on 20-unit, then 10-unit grids. Obstacles are the static edges buffered by **11 units** (predator radius 10 + 1) with square caps (`L13, L44-53`), so the predator can physically follow and agent-only gaps are excluded.
   - Smoothing uses collision-clear segments (`L122-132`).
   - Routes are cached in a "fixed frame": origin at the bait, x-axis along the longest edge (`L16-41`), so they survive guide motion.
   - If a collision nudge leaves the guide inside the 11-unit buffer, it takes an agent-width (5.01) step back into a predator-width lane (`L142-170`).
6. **Following check** (`predator_following.py`):
   - Given two consecutive observations of the (single, unambiguous) predator, it enumerates every move the native `Predator.step` could have produced toward the guide's previous pose (`L20-74`). This covers direct chase or watched pivot, speed caps 11 or 15, all terrain factors, and native 10° collision rotation.
   - Two consecutive incompatible moves set `not_following`. One compatible *translating* move clears it (`L77-138`).
   - `not_following` or predator lost triggers `_return_to_predator` (`my_guide.py:44-82`): wait 5 ticks, then walk back toward the last seen predator position (fallback: our last contact position) until within 12 units.
7. **Safety/contact sampler** (`guide_steering.prioritize`, `L19-113`), applied to every non-delivery action.
   - It samples stationary plus {requested, speed/2, speed, cap} × (requested direction, straight-away, 72 directions).
   - It drops moves whose segment crosses a wall (5.5 clearance).
   - Rank, lexicographically:
     1. affordable;
     2. safe: every predator at least `SAFE_DISTANCE = 48` = 15 contact + 15 stale step + 15 next step + 3 (`L12`), and the path never passes closer than that;
     3. contact: stay within 55 hearing, or 235 range and 25° of its vision cone with a clear line of sight;
     4. route progress.

     On the final approach (bait within 55 + sprint) progress outranks contact (`L88-92`).
   - Predators already within 40 of the bait (the held crowd) get a relaxed clearance of `max(18, dist_to_bait + 5)` (`L29-31`).
   - The low-energy walk cap is respected (`L46-47`).
8. **Delivery = deliberate sacrifice:**
   - Condition: the bait is within **`HEARING_TARGET = 55`** of the guide, or the guide is within 1 unit of the handoff (`my_guide.py:183-194`, `guide_steering.py:9`).
   - The guide then **stops**, turns toward the predator and bypasses the safety sampler: "Being caught here is allowed" (`my_guide.py:99-102`).
   - The predator catches the guide within about 55-59 of the bait. After eating, the bait is its nearest agent and inside its 60-unit hearing, so it attaches to the bait (`guide_lab.md:171-174`).
   - If the predator switches to the bait before eating the guide, the guide survives.

### 3.4 How the guide gets away afterwards
- `core.py:224-234`: once a track has been within 40 of the goal for at least 10 s (seen within the last 0.2 s), its guide is **released** back to normal work.
- **This never happened in any native game** (`guide_releases: 0` in every summary). In practice guides die: 24/26 in the release variant and 4/6 in the default.
- Each held predator gains energy from the guide it ate (capped at 200), and the score takes `-guide_energy/100` (`environment.py:722-723`).
- There is no escape manoeuvre after handoff.

### 3.5 Energy costs [engine, `environment.py:496-565`]
- Walking: 0.05 per unit.
- Sprinting: `0.05*speed + 0.5*(d-speed)`. For a default agent sprinting 20 that is 5.5/tick. For a predator sprinting 15 it is 2.55/tick.
- Turning: `min(π,|a|)/(2π)`.
- Below 20% of max energy, movement is capped at walking speed (`L512-513`).
- A guide sprinting 3 s costs about 165 energy, so a 150-energy founder can barely afford one chase **[inferred from costs]**.

---

## 4. Predator mechanics (cited to the vendored engine)

All line numbers are in `src/elements/*.py` on the branch. They are identical in `survival/vendor/survival-simulator/src`.

### 4.1 Stats and senses

| | Predator | Default agent |
|---|---|---|
| Radius | 10 (`predator.py:10`) | 5 (`agent.py:13`) |
| Walk / sprint per tick | 11 / 15 (`predator.py:10`) | 10 / 20 (`agent.py:14-15`) |
| Max energy | 200; spawns at **0**, resting (`predator.py:10-12`, `environment.py:487` uses the defaults) | 500; founders 150, children 75 (`environment.py:383`, `agent.py:16`) |
| Hearing (omni, through walls) | 60 (`predator.py:13`) | 50 |
| Vision | 250, cone π/3 (Creature default), occluded by walls (`creature.py:131-160`, `sensing.py:3`) | 200, π/3 |

### 4.2 Behaviour
- **Target selection:** it considers only agents (from its 3×3 neighbourhood of 400-unit chunks) and chases the **closest** observed one. There is no memory or commitment (`predator.py:27-32`; `environment.py:683-694`).
- **Direct chase:** used if the agent faces away (`|rel_dir| > π/2`) **or** is within `hearing*1.5 = 90` (`predator.py:37`).
  - `turn = clamp(0.5*angle, ±0.3)`. It moves `min(15, dist)` in direction `heading + turn`, then turns by `turn` (`L38-44`; move is applied before turn in `environment.py:699-707`).
  - So the **turn rate is 0.3 rad/tick**, and the turning radius at 15/tick is about 50 units **[inferred]**.
- **Watched pivot:** the agent faces the predator and is more than 90 away. The predator moves at sprint 15 along `angle ± π/4`, away from the side the agent looks toward, then turns to face the agent **with no turn cap** (`L46-62`).
- **No agent sensed:** it steers away from the nearest edge at walking speed, or wanders with ±0.1 turns (`L64-99`).
- **Energy:** no passive drain. Movement and turns cost as for agents. Below 40 energy (20%) it is capped at walking speed 11 (`environment.py:512`).
  - At energy ≤ 0 it rests (`L726-728`) and regains **30/s (3/tick)**. It wakes when energy > 100 (`L676-681`), about 3.4 s.
  - **Resting predators neither move nor kill** (`continue` before the contact check, `L681`).
  - Eating gives the predator the agent's energy, capped at 200 (`L722`).
  - After waking at 100 it can sprint about 24 ticks before dropping to 40 (`docs/game_mechanics.md:42`).
  - A continuously chasing predator cycles about 24 sprint + about 65 walk + 34 rest ticks, roughly 12 s **[inferred]**.
- **Kills:** a non-resting predator kills every agent whose centre is closer than `10 + agent radius` = 15 after its own move (`L709-724`).
- **Other agent deaths:**
  - energy ≤ 0 after the passive drain `dt * 1.0` per tick = 1/s (`L639-644`);
  - old age: after the hidden `max_age = 60 + U(0,60)` s (`agent.py:27`), an extra `0.01*age` per tick (`L646-647`).
- **Observation staleness:** agent observations are computed inside the agent loop, **before** predators move that tick (`L649-660`, then predators at `L674+`). Every predator observation is one predator step (up to 15 units) stale. Lucas's 48-unit safety distance accounts for this. The slopsesh1 handover recorded a DTO distance of 33.5 vs an actual 24.7 at death.
- **What agents see of predators:** `distance`, `angle`, and `rel_dir` = bearing(predator → agent) − predator heading, so 0 means the predator faces you. No id, energy or rest flag (`creature.py:136-139, 167-168`).

### 4.3 Spawn schedule
- Each tick, after `time += dt`, the spawn probability is `p = dt * time * 1e-4 / max(1, n_predators)` (`environment.py:756-762`).
- The spawn point is uniform. **The spawn is silently skipped** if a 10×10 box at that point overlaps an obstacle (`L478-490`, `_is_position_free`). There is no cap.
- The mean-field ODE `n dn = 1e-4 t dt` gives **n ≈ 0.01·t**.
- I propagated the exact tick-level count distribution for this Bernoulli rule. This is a numeric Markov-chain calculation, not a game run. `q` = fraction of spawn attempts that land on free space; `q = 1` is an upper bound, and I estimate `q` at about 0.8 **[inferred]** (obstacles plus 30-unit walls cover roughly 20-25% of the map).

  | t (s) | mean n (q=1) | 5-95% | mean n (q=0.8) |
  |---|---|---|---|
  | 300 | 3.0 | 1-5 | 2.6 |
  | 500 | 5.1 | 3-7 | 4.6 |
  | 1000 | **10.2** | 7-13 | 9.2 |
  | 2000 | **20.3** | 16-25 | 18.2 |
  | 3000 | **30.3** | 25-36 | 27.1 |

- First predator: median 118 s, mean 125 s (q = 1).
- New predators spawn **resting** and wake about 3.4 s later. That window can be used to lead them **[inferred; Oscar's handoff calls it the rest-window acquisition]**.

---

## 5. The rest of the population, and known failure modes

### 5.1 Non-role agents (`core.py:316-351`)
- **Before a site is found** (`self.site is None`): Nikolaj's `ExpertPolicy` explorer, with survey-gap harvesting disabled and `population_after_alignment=False` (`core.py:63-68`). It builds the map and localisation.
- **After a site is found:** everyone without a role follows `OrchardPolicy(seed, extra_old=False, heir_age=1e9)` (`core.py:69`).
  - This is **the old orchard**: `models/survival/oscar_orchard.py` has blob `7c2b4b4…`, identical to Oscar's handoff `f6ab035`, not the tuned `a7a7d63`/`dab7960` that our `fastsim/policy/orchard_ref.py` is.
  - `heir_age=1e9` disables heir births.
- **Every explorer or gatherer action** passes through `avoid_predators` (`models/entrapment/bystander_avoidance.py`). It samples 48 directions × lengths and ranks lexicographically:
  1. keep 31 units of clearance to any predator path;
  2. do not cross within 105 of the bait;
  3. keep out of 76-unit hearing distance and out of 105 of the bait;
  4. stay out of a 40° half-cone to 265 range of each predator.

  It uses the agent's own predator sightings plus teammates' fresh sightings (≤ 0.15 s old) transformed into its own frame (`core.py:291-314`). When it overrides, it also cancels spawning.
- **Breeding veto:** spawning is disabled for anyone with a role, and for anyone the orchard marks `old` **or with age ≥ 55** (`core.py:346-348`).
  - Lucas later found this contradicts the tuned orchard's heir age of 55.9 and helps explain the renewal collapse (`coordinator-renewal-probe.md:53-64`, `overnight_plan.md:63-67`).
  - A low-energy clamp to walking speed is applied to all actions (`L349-351`).
- **The actions actually executed are fed back** into the orchard's dead-reckoning (`m.last_action`, `m.spawned_ok`, `last_spawners`) and into the explorer maps (`core.py:352-366`). This is essential when a layer overrides the orchard.

### 5.2 Failure modes and TODOs Lucas recorded
- **Colony extinction:** 0/991 full games survived. Mean survival 621 s; the latest default died at 284 s.
- **Renewal collapse.** Recommended fixes (`coordinator-renewal-probe.md:51-69`):
  - integrate the tuned orchard;
  - keep a "spawn guard" (never select as bait an agent whose orchard action is spawning this tick);
  - remove the broad age-55 spawn veto.

  None of these is in production on the branch.
- **Guide losses:** assignments far exceed deliveries (18.4% arrivals in the benchmark), guides die in most assignments, and release has never been demonstrated.
- **Walk-capped guides:** 36/113 were assigned below 20% energy, because the coordinator prefers old agents over energetic ones (`native-guide-start-audit.md`). A viability-aware coordinator was "being evaluated", with no result.
- **Predator identity:** there are no IDs. Association is ambiguous when predators overlap and after occlusion (`entrapment_iteration/README.md:114-116`). A nearby held predator can be mistaken for the newcomer.
- **Original-group retention:** 12 of 26 fresh failures were original predators leaving the 40-unit zone, often because the guide's final approach perturbed them (`staged-front-probe.md:33-38`).
- **Rear occupation** in 335/708 baited games blocks replacement access.
- **Site invalidation** when mapping refines a wall drops all bait and track state (`core.py:111-123`).
- **Missing sites:** about 5% of maps have no crevice (47/1000). Corner pockets add a few; the generic corner detector (not in production) found candidates on 98/100 maps but was never benchmarked.
- **Route churn:** the following-check false positives while turning around walls cause backtracking. The "visible contact" variant fixed 2 of 7 selected failures but was rejected after a paired-100 comparison (commit `3a03582`).
- **Rejected tuning:** smaller stop distances, a front-lane-only stop, bounded waiting, central-site preference, the sight-checked distant sacrifice (`--vision-delivery`, 5/12) and side-offset handoff all regressed (`entrapment_iteration/README.md:125-153`).
- **Other TODOs:** the `my_guide.py:206` TODO says "Also don't move if predator is more than 120 away". Route cost ignores terrain (`guide_pathfinding.py:1-5`). The A\* can be slow on no-route cases (`guide_batch_1000_results.md:57-58`).
- Oscar's handoff flags the same bottleneck: sustaining bait food and replacements while predators accumulate.

---

## 6. Observation-only constraints and localisation

- The controller receives only the per-agent DTOs and sim time (`core.py:1-5`, "No engine imports").
  - DTO fields: observations, energy, biome, age, speed, sprint_speed, hearing_radius, vision_angle, vision_range, max_energy (`environment.py:586-598`).
  - Observations include relative Edge segments (`creature.py:172-177`), Agent observations with ids, and Predator observations without ids.
- Evaluator-only data (true positions, predator identities) is kept out of policy inputs in every harness (`guide_lab.md:137-143`, `entrapment_9059_benchmark_protocol.md:24`).
- **Localisation in native games** is Nikolaj's `WorldEstimator` (`models/exploration/world_estimator.py`, 1031 lines):
  - dead reckoning from the executed actions (heading exact, position times the known own-biome factor);
  - corrections from remembered trees and edges;
  - frame groups merged by agent-agent sightings (ids + rel_dir);
  - **anchoring to absolute coordinates** once boundary walls are recognised (`_anchor_groups`, `L520-595`; `boundary_wall_thickness=30`).
  - Each pose has an `uncertainty`. Lucas skips agents with uncertainty above 8 (bait candidates, shared sightings) or above 12 (predator tracking) (`core.py:141, 205, 294`).
- Site coordinates live in one group frame (`site_group`, `site_frame` revision). Goal, handoff, mouth and edges are transformed into each guide's local frame through its estimated pose every tick (`core.py:271-283`, `local()` at `L43-44`). Arrival is "estimated pose within 2 of goal" (`L131-134`), so occupancy is an estimate.
- The guide lab and 1000-map batches assume **perfect static-map localisation** as a convenience (`guide_lab.md:141-143`). The native games do not.
- Our orchard (`_orchard.hpp`) already has an equivalent pose system: exact heading, dead reckoning, 10° collision-deflection correction, lineage groups, merges and boundary anchoring (`orchard_ref.py` docstring, `_orchard.hpp:546-575`).
- What ours lacks:
  - a **persistent static-obstacle map** (per-mind `edges` are pruned after 40 s, `_orchard.hpp:630-638`);
  - pose uncertainty.

---

## 7. What must change to port this into our C++ policy (`survival/fastsim/_orchard.hpp` inside `_engine.cpp`)

1. **Add a coordinator layer after `orchard::Policy::call()`.**
   - Our loop is `acts = pol->call(policy_states(e), time)`, then `agent_step` for each (`_engine.cpp:1865-1886`).
   - Insert a `trap::Coordinator` that takes the same `AState`s plus the orchard's minds and groups. It assigns roles (bait / replacement / retired / guide) and overrides those agents' `Act`s.
   - Then **write the executed action back** into `Mind::last_action`, `spawned_ok` and `last_spawners`, as `core.py:352-366` does. Otherwise odometry and heir bookkeeping drift.
   - Also stop `assign_posts` / `assign_fruits` from giving tree posts and fruit claims to role agents **[recommended]**. In Lucas's Python they still consume orchard slots.
2. **Keep a persistent static map per anchored group.**
   - Accumulate observed Edge segments in group coordinates (they are exact relative to a pose; our heading is exact).
   - Reconstruct closed axis-aligned rectangles when all four sides are seen, plus the arena walls (thickness 30). Port `observed_rectangles` (`observed_trap_sites.py:7-48`).
   - Only anchored groups can host a site. Reuse our boundary anchoring.
3. **Port the site enumerator** (`entrapment_sites.py:86-191` + `observed_trap_sites.py:51-85`) as plain AABB math: pair loop, gap/overlap tests, depth-5 goal, approach lane, runup, rear access, handoff.
   - `_Geometry.free` / `clear` are already expanded-AABB tests plus Liang–Barsky (`_segment_rect`, `L31-48`). They port directly and match native `_in_obstacle` semantics exactly (strict inequalities).
   - Skip shapely corner pockets at first (fallback only; small coverage gain).
   - Choose by overlap, or better, with boundary support first (Lucas's screen, `site-ranking-screen.md`, unvalidated).
4. **Replace the shapely A\*** (`guide_pathfinding.py`) with a C++ grid A\* over rectangles expanded by 11 (predator lane, for guides) and by about 5.05 (agent lane, for bait/replacement via the rear entry; Lucas uses Nikolaj's `Navigator(clearance=5.05)`).
   - Square-capped buffered segments are exactly expanded AABBs for axis-aligned obstacles, so no polygon library is needed.
   - With absolute anchored coordinates, drop `fixed_frame()` and plan in group coordinates.
5. **Port the guide:**
   - `my_guide._guide`: face the predator; sprint if within 100, otherwise walk; back along the route; stop at 55 from the bait; hold and sacrifice.
   - `_return_to_predator` recovery.
   - `predator_following.possible_follow_moves` (a direct mirror of `Predator.step`; our engine already has exact `predator_step` code to reuse).
   - `guide_steering.prioritize`: a sampler of 1 + 4×74 candidates. Replace shapely line distances with point-segment distance and segment-vs-edge crossing, both of which `_orchard.hpp` already has (`point_segment`, `segments_cross`).
   - Bit-exact parity with Lucas's Python is **not realistic** (shapely/GEOS, numpy). Validate statistically on the 30+1 fixture instead.
6. **Port predator tracks and assignment** (`core.py:194-252`):
   - project sightings with our poses;
   - gate at `min(90, 20 + 150*dt)`;
   - expire after 8 s (30 s with a guide);
   - assign only once the bait is established, and only to tracks more than 60 from the goal.
   - **Fixes Lucas identified:** require guide energy ≥ 20% max plus a sprint budget (36/113 guides were walk-capped); never pick the last young parent (`coordinator-renewal-probe.md:15-18`).
7. **Port bait logistics:**
   - `remaining_life` (`core.py:28-40`, closed form);
   - `_select_bait` preference;
   - the rolling-replacement trigger (`life < travel + 20`);
   - the rear-entry approach;
   - retired baits standing at the goal.
   - Add Lucas's recommended **spawn guard**: never turn an agent that is spawning this tick into bait.
   - Use our tuned orchard's heir logic instead of the `heir_age=1e9` + age-55 veto that he identified as the likely cause of extinction.
8. **Port `bystander_avoidance.avoid_predators`** for all orchard agents:
   - 31 clearance, 76 hearing margin, 105 bait keep-out, 40° cone to 265;
   - teammate sightings shared via the group frame.
   - Our orchard currently ignores predators entirely.
9. **Engine and test hooks for fixtures.** Lucas's numbers come from fixtures that mutate engine state: preload 30 awake predators, refill bait energy, disable ambient spawning (`scripts/guide_multi.py:57-73, 217, 238`). Sol noted these "require mutation APIs absent from C++ snapshots" (`overnight_plan.md:68-72`).
   - Add `set_predator(...)` / `add_predator(x, y, dir, energy, resting)` / `set_agent_energy(id, e)` / a spawn toggle to `_engine.cpp` to reproduce the 30+1 benchmark natively.
   - `predators_enabled` exists but is all-or-nothing (`_engine.cpp:1028-1040`).
10. **Economics check before investing [inferred].**
    - By 3000 s there are about 30 predators. Each needs a guide, and guides usually die.
    - Each bait lasts at most about 60-100 s unfed, so a full game consumes roughly 30-50 baits plus about 30 or more guides.
    - Our orchard's economy is about one agent per tree, with trees halving every about 600 s.
    - Use the C++ engine to measure trap-on vs trap-off on the same seeds before tuning delivery details. Lucas never ran that control (the 9059 benchmark has no no-trap baseline).
