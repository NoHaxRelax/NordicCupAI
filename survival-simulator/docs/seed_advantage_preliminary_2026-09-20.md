# What knowing the seed actually buys, and how a policy uses it

Preliminary, 20 September 2026. Written for the team working on the survival-simulator
entry. Every number is labelled **[M] measured**, **[E] estimated from measured inputs**,
or **[S] speculative**. Several of the most attractive-sounding advantages are worth much
less than they look, and those are called out first so nobody spends a week on them.

---

## 0. The one-paragraph version

Recovering the seed converts the game from partially observed to **fully observed and
exactly simulable**. The whole map is known at t=0, and the entire future food supply is
known and **cannot be changed by anything we do**. Against that, the obvious exploits —
knowing where predators are, what an agent's hidden max age is, when fruit appears — are
worth almost nothing, because each is either already visible one tick later or optimised
against the wrong end of the reward function. That has been measured: a prior effort with
a fully exact model got **+4.9%**. The value is in three places instead: **routing and
scheduling against a known future**, **search with an exact forward model**, and a genuinely
new one we found this session — **steering the shared RNG stream by timing births**, which
lets a policy push predator spawns away from itself.

---

## 1. What recovery gives you, concretely

**[M]** The scan recovers the exact 32-bit seed from public terrain observations, searching
the whole 2^32 domain. On one 20-thread laptop that is 324 s at 13.25 M seeds/s; V4's
6-shard cluster did it in 19.2 s. Reliability harness: targets drawn uniformly from the
full domain, observations from the real engine, seed never given to the pipeline —
**11/11 dispatched targets uniquely recovered, 0 false positives** on the current run.

**[M]** With the seed plus our own action history, a private engine reproduces the live
game **bit-identically to full horizon** (verified at 13,176 / 19,825 / 13,221 ticks on
three seeds: identical RNG state, score and sim time). Branch-and-replay cannot desync,
because it is the same function of the same inputs.

**[M]** `survival-simulator/mirror/` makes that cheap: `snapshot()` 0.002 ms, `restore()`
0.001 ms, against an engine step of 0.06–0.2 ms. So a rewind costs **under 1/60th of a
tick**, and search is affordable.

---

## 2. The advantages, ranked by measured or estimated value

### 2.1 Tier 1 — large, and the evidence is in hand

| # | Advantage | Value | Evidence |
|---|---|---|---|
| 1 | **Predator-spawn avoidance by birth timing** | predators 1.7 → 0.8/game, score +51%, survival +50% | **[M]**, confounded — see below |
| 2 | **Exact predator spawn schedule for harvest scheduling** | ~52% of predators currently harvested; ceiling is 0.3 x seconds-remaining each | **[M]** current rate, **[E]** ceiling |
| 3 | **The entire static map at t=0** | no exploration cost, no collision-retry waste, correct biome-aware routing | **[M]** map exactness |

**On #1, the honest statement.** With 1.2 s of scheduling slack, choosing *when* to breed
pushed the next predator spawn past a 90 s horizon in **146/147** decisions, against
**70/147** for blind breeding; mean gain **+20.9 s per decision** **[M]**. Over full games
on 20 seeds it gave predators 1.7 → 0.8, score 267.5 → 404.8, survival 2677 → 4005 ticks
**[M]** — **but births also rose 6.2 → 9.2, so the arms are not matched and the effect is
confounded.** A subagent is resolving that now. Treat +51% as an upper bound until it does.

The mechanism is solid and worth understanding because it generalises: agent actions apply
before `non_agent_step` (`fastsim/_engine.cpp:1735-1736`), the predator-spawn coin is the
**last** draw of the tick (`:1544`), and a birth consumes 24–36 MT words (`:942-975`).
So choosing to breed on tick T re-phases every world draw from T onward, by an amount our
model computes exactly. A blind birth is a coin flip (it delays the spawn only 57% of the
time); **the value comes entirely from being able to evaluate both branches and pick.**

### 2.2 Tier 2 — structurally valuable, not yet measured here

**The whole food supply is known and is immune to us. [M]** Tree births and deaths, fruit
spawns, and their positions depend only on tree count, tree age, biome and time. Agents
never touch them: eating a fruit consumes no RNG. So at t=0 we know every fruit that will
ever exist, where, and when. Combined with deterministic ripening (spawns at energy 20,
grows +2/s to a cap of 60 at ~20 s, rots at ~50 s) each fruit is a **known time window with
a known value curve**. That turns foraging into a prize-collecting routing problem with
time windows rather than a reactive chase.

**Exact travel cost. [M]** `move_distance` is charged energy *before* the biome penalty
scales the distance actually travelled, so crossing swamp (0.5) costs full energy for half
the distance and river (0.3) is worse. The engine's obstacle avoidance is a blind ±10°
sweep, not a planner. Knowing the obstacle map and biome raster turns routing into A* on a
known graph. **[S]** how much this is worth is unmeasured; an agent is on it.

**Hidden state becomes visible. [M]** Agent `max_age` (`60 + uniform(0,60)`), fruit energy
and age, and predator `resting` are all absent from observations but exact in the model.
The most useful of these is `resting`: a fresh predator is helpless for exactly 34 ticks
(3.4 s) and the public observation cannot distinguish a sleeping predator from an awake one.

### 2.3 Tier 3 — sounds valuable, measured to be worth little

This is the part most worth reading, because each of these has already cost someone time.

| Exploit | Why it fails |
|---|---|
| **Exact predator position** | Agents outrun predators (sprint 20 vs 15), and the predator *refuses* to approach an agent that is facing it. "Turn to face it" already neutralises most of the threat from public data alone. Failed 128-seed confirmation. **[M]** |
| **One-step predator action search** | The predator is a deterministic automaton. Its only stochastic act is a `uniform(-0.1, 0.1)` turn firing on ~6.6% of awake-predator-ticks, deflecting it ~1 px. There is nothing to search. Failed 16 paired seeds. **[M]** |
| **Exact agent max_age** | Crossing it costs `energy -= 0.01*age` per tick — a 10x drain jump the policy *sees in the next observation*. Foreknowledge buys a fraction of a tick. Failed. **[M]** |
| **Early fruit arrival** | Optimised the wrong end of the window: arriving early gets energy 20, arriving 20 s later gets 60. The valuable version is the opposite. Failed. **[M]** |
| **Algebraic seed inversion instead of brute force** | Needs six *complete* 32-bit outputs at stream positions {0,1,2,227,228,229}. We observe 11-bit-truncated words at rejection-shifted offsets, and positions 227–230 land inside the river walk where they survive only as a pixel polyline. Adapted cost **2^167**. **[M]** |

**The pattern:** every failed exploit predicted something that was either already observable
one tick later for free, behaviourally trivial, or optimised against the wrong shape of the
reward. Foreknowledge is only worth something when the information is *hidden*, *acted on
before it would become visible*, and *changes a decision that matters*.

---

## 3. What the score function actually rewards

This reframes everything, so it goes before the policy designs.

```
score += dt                    every tick, unconditionally      -> +0.1 / tick
score += fruit.energy / 1000   on eating                        -> +0.06 for a ripe fruit
score -= agent.energy / 100    when an agent is eaten           -> -1.0 at 100 energy
```
(`src/elements/environment.py:671,723,757`)

**[M]** Over 128 baseline games: time 1720.1, food +162.9, predation −157.3, total 1725.7,
and **0 of 128 runs reached the 3000 s horizon**. Three consequences:

1. **Population is worth exactly zero.** One agent alive scores the same per tick as thirty.
   The only requirement is *at least one alive*.
2. **Fruit is worth almost nothing as score** (163 points a game) but is the entire
   instrumental currency. A search that maximises score over a short window will chase
   fruit points and ignore survival.
3. **Agents are cheap to lose.** An agent eaten at 20 energy costs 0.2 points — two ticks of
   colony survival. **A low-energy agent is a near-free decoy.** No heuristic author would
   write that; a search finds it immediately.

Headroom is therefore roughly **+1400** (dying at ~1720 s out of 3000), against the **+84**
that heuristic peeking delivered.

---

## 4. How a policy uses this — four designs, in build order

All of these must be **C++ for engine and policy** with no per-tick Python boundary. Budget
is **1200 s accumulated latency** over 30,000 ticks ≈ **40 ms/tick**, with a 10 s
per-response cap that permits bursting. (`README.md:1162` says 600 s and is stale.)

### Design A — Birth-phase predator dodging *(cheapest, most novel, start here)*

Each time a birth is possible, roll the mirror forward under both "breed now" and "breed in
k ticks" for k in a small window, and pick the phase that pushes the next predator spawn
furthest away.

- Cost **[M]**: 12 candidates × 900 rollout ticks × ~0.1 ms ≈ **1.1 s per decision**, with
  decisions every few hundred ticks. Comfortably inside 1200 s.
- Risk **[S]**: the spawn *rate* is a function of time and predator count, not RNG phase, so
  re-phasing cannot change the long-run expectation — only repeated *selection* can bias the
  realised outcome. Whether that compounds over a game is exactly the open question.
- Verification: both arms must take the same number of births.

### Design B — Scheduled predator harvesting *(largest raw score, gated on rules)*

Predators spawn uniformly at random with no dependence on world state, and rest exactly
3.4 s. With the schedule known, pre-position a sacrificial pair at the spawn point *before
the predator exists* and harvest at wake, instead of reactively hunting.

- Current **[M]**: ~52% of predators harvested, mean delay 46.6 s, mean score 8046.
- Ceiling **[E]**: 0.3 × seconds-remaining per predator.
- **Rules caveat:** this rides on engine bugs. An admin was asked whether exploitation is
  permitted and has not replied. Design and measurement only.

### Design C — Rollout planning in the endgame *(biggest headroom, least certain)*

Activate late, when population is 1–5 and rollouts are cheap, designate 1–3 survivors, give
each a small macro-action set (hold / forage / camp / flee / breed), roll each to the
horizon with the **exact** score function, and pick the best.

- Why late: the horizon self-truncates, so rollouts get cheaper exactly as the score is
  being lost. Population being worth zero licenses planning only the designated agents.
- **The trap [M]:** within a short rollout the score is nearly flat, so all discrimination
  comes from the terminal value. Either roll to the horizon, or calibrate the value function
  against measured remaining survival time. A rollout search with a bad terminal value is a
  heuristic wearing a costume — which is a fair description of the modes that already failed.
- **Run the ablation.** If "stop breeding after t=1400 and designate a survivor" captures
  most of the gain, ship that instead. It is a 20-line heuristic.

### Design D — A* routing and time-windowed foraging *(safe, incremental)*

Precompute a distance field over the known obstacle map with biome-weighted costs once at
recovery, then route to the fruit whose *value at arrival time* is highest. Per-tick cost is
a table lookup. **[S]** value unmeasured.

---

## 5. Risks that would make all of this worthless

1. **The seed may not be in [0, 2^32).** `src/core.py:22` defaults to `randint(0, 2**32-1)`,
   but the README says evaluation uses **preset** seeds, and `random.Random` accepts
   arbitrary ints, floats, strings and bytes. A string seed produces a 17-word key and is
   unrecoverable by enumeration. **Mitigated:** the scanner now takes multi-word keys
   (`SEED_HIGH_WORDS`), so a clock-derived seed becomes a bounded window; verified against
   CPython vectors for seeds ≥ 2^32.
2. **A complete scan returning nothing is ambiguous.** It looks identical to bad samples, a
   stray river label, or a generator-config change. **Mitigated:** the filter now rejects
   out-of-range labels and truncated sample files loudly, reports distinct label counts, and
   supports a candidate cap.
3. **Weak acquisition, not slow search, is the real bottleneck.** V4's live recovery was
   132 s end-to-end of which only 19.2 s was search. And sample **diversity dominates sample
   count**: two of our own harness runs died leaving 1.4M and 33M candidates because the
   sampling walk never left one biome. **Mitigated:** an acquisition gate now holds back
   low-diversity targets instead of burning the search budget on them.
4. **Replay parity against the live server is not guaranteed.** The terrain prefix is pure
   integer arithmetic and ports exactly; full replay involves NumPy float loops and
   identity-hashed observation sets, and is not even process-stable. Any shadow model needs a
   resync tripwire and a fallback to observation-only play.

---

## 6. What to do next

1. **Resolve the birth-phase confound** (equal births in both arms). It decides whether the
   cheapest and most novel exploit is real. *In progress.*
2. **Run the ablation for Design C** before building it. The heuristic may win.
3. **Settle the seed-range question** with the organisers if possible. It is the only risk
   that invalidates the whole approach rather than degrading it.
4. **Attack acquisition, not search speed.** Going 19 s → 14 s on a 132 s pipeline changes
   nothing; halving acquisition time changes a lot.
