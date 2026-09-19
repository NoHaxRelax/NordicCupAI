# Fuzzing the agent action interface: results, 2026-09-19

Every tick a policy returns `(move_distance, move_direction, turn_angle,
spawn_agent)` per agent. This is what happens when those fields carry values a
policy should never send.

**Summary.** The clamping is mostly correct and the energy ledger has no
exploitable hole - across 126 sign/magnitude combinations the minimum cost was
0.0 and never negative. But **NaN defeats every guard**, because every comparison
against NaN is false. One NaN action teleports an agent to a fixed corner and
makes it **immortal**, scoring 3000 where the control scores 95. It is
reproducible, identical in both engines, and **not usable** - it cannot cross a
standards-compliant JSON boundary, has exactly one destination, and the agent is
frozen there permanently.

The finding that matters for our own work is unrelated to exploits: the native
engine coerces `np.float16`/`np.float32` and **silently breaks Python-vs-C++
lockstep**, so our parity guarantee holds only while the policy emits plain
Python floats.

---

## 1. Method

A toy world, because a full game is far too slow to fuzz: `Environment(200, 160)`
with 1 agent, no predators, fruit or trees, and only the four auto-created
boundary walls. **Build 93 ms, 1453 steps/s.** Larger builds for cross-checks:
400x320 in 0.18 s, 800x600 in 0.49 s.

Lockstep was verified before comparing anything: `fastsim.SimulationCore` against
`src.core.SimulationCore` at 400x320 / seed 11 - start state and five legal steps
**bit-identical**. A normal step was confirmed first (move 5 at direction 0.1 ->
displacement 1.5 after the 0.3 river penalty, 0.25 energy).

---

## 2. Clean negatives - the clamping that works

| input | behaviour |
|---|---|
| `move_distance` = -1, -1e6, -1e300, -inf | clamped to 0, **0 energy charged**, no backward movement |
| `move_distance` = 20.000001, 1e6, 1e15, 1e300, +inf | clamped to `sprint_speed` = 20, charged 5.5 |
| `move_direction` = +/-1e6, 2*pi*k, 1e300 | ordinary move, 0.25 energy |
| `turn_angle` = +/-1e6, 2*pi*1e15, 1e300 | heading accumulates unbounded; cost capped at 0.5 |
| `spawn_agent` = 2, -0.5, `"false"`, `[0]`, `object()` | engine is truthiness-based, but **pydantic `ActionRequest` rejects all of them** |
| repeated `spawn_agent=True` | 1 child/tick, -100 each, strict `>100` gate |

Citations: negative and over-sprint clamps at `environment.py:505-510` /
`_engine.cpp:1070-1071`; low-energy walk cap `:512-513` / `:1072`; turn cost
`environment.py:564` as `min(pi, abs(turn_angle))/(2*pi)`, capped at 0.5 and using
`abs()`, so there is no sign exploit.

Two ledger asymmetries, both of which **cost** the agent and never pay it: the
charge is computed from the *clamped* distance, and it is applied even when a
collision blocks the move.

**Large finite values are harmless.** `cos`/`sin` of any finite double is in
[-1, 1], so displacement stays under the sprint cap. There is no modulo or
integer-overflow path. The `turn_angle` accumulator at `environment.py:563` has no
clamp, no wrap and no finiteness check, but that only matters via NaN/inf below.

---

## 3. The NaN defect

### 3.1 Teleport (F1)

`environment.py:537-538` computes `new_x/new_y` from `cos`/`sin` of the direction.
A NaN anywhere makes both NaN. Then:

1. `_in_obstacle` returns **False** - every comparison against NaN is false
   (`environment.py:802-804`), so the obstacle guard is bypassed.
2. `:556` **assigns** rather than increments: `entity.x, entity.y = new_x, new_y`.
3. `_keep_agent_in_bounds` (`environment.py:777-779`) does
   `max(size, min(W-size, NaN))`. Python's `min` returns its first argument when
   the comparison is false, so this yields `W-size`.

The agent lands on exactly `(W-size, H-size)`. On the real 1600x1200 map:
`(100, 80) -> (1595, 1195)`, a **displacement of 1865 against a sprint cap of 20 -
93x - for 0.25 energy**. Identical in C++ (`_engine.cpp:1063-1065`; `std::min`
returns `a` when `b` is NaN).

Reproduction: `env.agent_step(aid, 5.0, float('nan'), 0.0, False)`.

Note `move_distance = 0` does not protect you: `0 * NaN = NaN`. And a NaN or inf
`turn_angle` poisons `direction` **permanently** (`inf + x = inf`), so every later
move re-teleports - a one-way trip.

### 3.2 Immortality (F2) - the serious one

`environment.py:505` (`distance < 0`) and `:509` (`distance > sprint_speed`) are
both false for NaN, so a NaN `move_distance` reaches the charge at `:515-518` and
`entity.energy -= NaN`. The death test `if agent.energy <= 0`
(`environment.py:642`, C++ `_engine.cpp:1408`) is then **false forever**.

Measured, toy world: **30000 ticks (the full 3000 s horizon) survived, score
3000.0**, against a control that starved at 76.3 s with score 76.4. Native engine,
full-size map, seed 5: one NaN action on tick 1 -> survives all 30000 ticks, score
**3000.12**; the same seed without it goes extinct at tick 957, score **94.71**.

### 3.3 The corner is inescapable and predator-proof (F3)

The landing point is *inside* the boundary obstacle (walls are 30 thick, agent
size 5), so:

- **No escape.** 16 distances (1e-12 to 1e6) x 72 angles = **1152 attempts, one
  distinct resulting position**. The collision-rotation loop
  (`environment.py:546-552`) only retries the *same* distance at rotated angles,
  and escaping needs >= 30 units against a 20 cap.
- **No children escape either.** On 200x160, 400x320 and 1600x1200, every child
  spawns at the parent's **exact position** and is itself immobile. Cause:
  `environment.py:292-294` clamps the child into `[20, W-20]`,
  `_is_position_free` fails at `:372`, and `:375` falls through to "spawn at
  parent's position". Children get the default finite 75 energy, so they are
  mortal and starve.
- **Predators cannot reach it.** Same `_in_obstacle` buffer at radius 10. Closest
  approach over 600 ticks with 4 predators: **49.5** against a 15-unit touch
  threshold.
- The boundary is unconditional - `_create_boundaries(thickness=30)` runs in
  `Environment.__init__` (`environment.py:61`) and in the C++ constructor
  (`_engine.cpp:651-654`), so no real map lacks it.

Since the game ends on extinction (`simulation_server.py:46`,
`scripts/experiment_case.py:84`) while `score += dt` runs unconditionally
(`environment.py:757`), one immortal anchor guarantees a full-horizon score.

### 3.4 One destination only - it is not steerable

Injected coordinates *do* reach different corners, confirming the clamp
arithmetic:

| injected (x, y) | lands at |
|---|---|
| (+inf, +inf) | (195, 155) bottom-right |
| (-inf, -inf) | **(5, 5) top-left** |
| (+inf, -inf) | (195, 5) top-right |
| (-inf, +inf) | (5, 155) bottom-left |
| (nan, nan) | (195, 155) - NaN behaves like +inf |

But **no action can produce an infinite coordinate**. `move_distance` is clamped
into `[0, sprint_speed]` *before* use, `prev_x/prev_y` are re-clamped finite every
tick, and `cos`/`sin` are bounded - so `new = finite + finite*bounded` is finite,
and the only other outcome is NaN. A 5x5 sweep of
`{nan, +/-inf, 1e300, 20} x {0, nan, +/-inf, 1e300}` produced exactly two classes,
finite and NaN, **never inf**.

### 3.5 Latent, not reachable (F4)

`environment.py:670` does `min(max_energy, NaN)`, which returns `max_energy` - so
a 20-energy fruit would refill a NaN tank to 500.0. Unreachable in practice: NaN
energy only arises from a NaN `move_distance`, which in the same call teleports
the agent into the wall, where no fruit can spawn or be touched (fruit needs
x <= 165; the agent sits at 195 with a 10-unit touch radius).

### 3.6 Minor

`environment.py:530` does `distance *= biome_movement_modifier`. Passing a 0-d
`np.ndarray` mutates the caller's object in place (`array(5.)` -> `array(1.5)`).

---

## 4. Complex numbers

| field | value | Python engine | native engine |
|---|---|---|---|
| `move_distance` | `complex(1,2)`, `(0,1)`, `(5,0)`, `nan+nanj`, `inf+infj` | clean `TypeError` @ `environment.py:505` | `TypeError: must be real number, not complex` |
| `move_distance` | `np.complex128(1+2j)` | propagates through clamp, charge, position and bounds clamp, then `TypeError` @ `environment.py:85` | **accepted - silently uses the real part** |
| `move_direction` | `complex('nan+nanj')`, `complex('inf+infj')` | **accepted -> corner teleport**, game continues | `TypeError` |
| `move_direction` | `complex(1,2)` | `TypeError` @ `environment.py:85` | `TypeError` |
| `turn_angle` | any complex | accepted, cost stays real via `abs()`, heading becomes complex; **next `non_agent_step` raises** `ufunc 'remainder'` @ `creature.py:91` | `np.complex128` accepted as real part; plain complex raises |

Zero-imaginary complex - `(1j)**2 == (-1+0j)`, `complex(3,4)*complex(3,-4) ==
(25+0j)`, `complex(5,0)` - is **rejected cleanly by both engines**. `abs(complex)`
returns a plain float and takes the ordinary path identically.

Complex energy would also not fire the death test (`np.complex128(-5+0j) <= 0`
returns True, builtins raise), but no input reaches complex energy.

---

## 5. The type-coercion defect class

`get_double` / `PyFloat_AsDouble` (`_engine.cpp:1687-1701`) coerces **anything
with `__float__` or `__index__`**, while the Python engine requires a type
supporting both ordering and float arithmetic. From 27 values x 3 fields:

| value | Python engine | native engine |
|---|---|---|
| `np.complex128(5+0j)` | TypeError | accepts real part 5.0 |
| `Decimal("5")` | TypeError (`Decimal * float` @ `:516`) | accepts 5.0 |
| **`Decimal("NaN")`** | `InvalidOperation` | **accepts -> full NaN exploit** |
| `Decimal("Infinity")`, `Decimal("-5")` as distance | accepted (clamped to float before arithmetic) | accepted - agree |
| `HasFloat(5.5)` / `(nan)` / `(inf)` | TypeError | **accepts, including the NaN exploit** |
| `HasIndex(7)` (no `__float__`) | TypeError | accepts 7.0 |
| `np.float16(5)`, `np.float32(5.5)` | accepted, **different result** | accepted |
| Fraction, np.float64, np ints, bool, int | agree | agree |
| `"5"`, `None`, `__float__` that raises | both reject | both reject |

Two consequences worth separating.

**A second door to the NaN defect.** `Decimal("NaN")` and any object whose
`__float__` returns NaN reach it through the native engine while Python rejects
them.

**A real risk to our own verification.** `move_direction = np.float16(5)` gives
x = **269.5** in Python against **269.678585586** native - 0.18 units apart on
tick one, which diverges chaotically over a game. Python propagates the reduced
dtype through its arithmetic; the native engine converts to `double` at the
boundary. So the lockstep claims in `fastsim/verify.py` and
`fastsim/verify_evasion.py` hold **only while the policy emits plain Python
floats**. In the DTO path `ActionRequest.model_validate`
(`scripts/experiment_case.py:104`) normalises to Python float, but the native
`run_policy` path never crosses Python at all. A numpy-based policy returning
`np.float32` would silently break parity.

Also fragile: `Decimal("Infinity")` is accepted by the Python engine while
`Decimal("5")` is rejected - the same type accepted or refused depending on its
*value*, because out-of-range values are replaced by a float at the clamp before
any Decimal arithmetic runs.

---

## 6. Predators

### 6.1 They cannot be trapped by a policy

Call path confirmed: `environment.py:696` `predator.step()` -> `:702`
`update_entity_position` (the **same** function, same `_in_obstacle` guard at
`:541-552`, same bounds clamp at `:557`) -> `:707` `update_entity_direction`. The
guard tests the new position *before* assigning, so a predator can never enter
geometry. Over 6000 predator-ticks of ordinary play: zero entries.

One predator was found permanently stuck, but the cause is engine-internal and not
reachable by a policy: `spawn_predator` (`environment.py:486`) tests
`_is_position_free(x, y, size, size)` - an axis-aligned box anchored at (x,y) -
while movement tests `_in_obstacle(point, radius=size)`, a symmetric buffer. A
predator spawned at (30.9, 146.7) with `_is_position_free=True` but
`_in_obstacle=True` had **total displacement 0.00 after 600 ticks**.

### 6.2 The anchor does not occupy predators - hypothesis refuted

`predator.step()` is a pure function of the current observation
(`predator.py:16-100`); the only carried state is `self.resting`. There is no
stale target to exploit. Measured anyway:

- Predator mid-charge at 24 units when its target NaN-teleports: over the next
  3000 ticks it spent **0 ticks** within hearing range (60) of the corner, 40
  ticks (1%) within 100, and ended **779 units away**.
- Predator *starting* adjacent to the corner: **0/3000 ticks** within 60, ended
  457 away.
- Sleep fraction 15% in both - identical to a predator with no agent present.

The edge-avoidance branch (`predator.py:87-93`) actively steers predators *away*
from walls whenever no agent is perceived, and the corner agent is invisible (the
wall blocks the vision polygon) and outside hearing range.

### 6.3 The legal pocket - this one works

Agent radius 5, predator radius 10, touch range 15 (`environment.py:717-718`)
leaves geometry an agent can occupy and a predator cannot. Constructed: two
80-wide obstacles with a **14-wide slot**, agent at the centre.

- agent-legal (r=5) True, predator-legal (r=10) False
- **3000 ticks (300 s): agent survived, closest predator approach 50.0** against
  a 15-unit touch threshold
- predator burned 2512 energy and was asleep **27%** of the time against **15%**
  for an unbaited control, ending at 34 energy - below the 40-energy walk cap
- the agent issued no move at all: **zero movement cost**

Chase overrides wall avoidance exactly as expected - `predator.py:31-44` returns
before the `if edges:` branch - so a hooked predator jams indefinitely.

Entry must be at sprint:

| approach | result |
|---|---|
| sprint 20 from 390 / 360 / 345 | safe in 2-6 ticks, holds 1500 ticks, predator drained to ~20 energy, asleep 26% |
| **walk 10 from 390** | **caught at tick 9** - the predator closes at 15 against 10 |

Two limits on the record: natural pockets are **rare** - only **0.18%-0.27%** of
agent-legal cells on real maps (seeds 1, 5, 11) are further than 15 from any
predator-legal cell, and reachability was not verified. And the test topped the
bait's energy up every tick; a real agent still pays ~1 energy/s and starves in
~66 s from 400. **The pocket buys predator safety, not food.**

### 6.4 The predator energy ledger

Defaults (`predator.py:10-14`): speed 11, sprint 15, size 10, `max_energy` 200,
**spawns at energy 0 and resting**, hearing 60, vision 250.

| state | cost/tick | citation |
|---|---|---|
| pursuit (sprint 15) | **2.5500** = 11*0.05 + (15-11)*0.5 | `environment.py:518` |
| pursuit with turn | +<= 0.0477 (turn capped at 0.3) | `environment.py:564`, `predator.py:38` |
| walk / wander / edge-avoid (11) | 0.55-0.57 | `predator.py:93, 99` |
| below 40 energy (`max_energy/5`) | forced to walk -> 0.55 | `environment.py:512-513` |
| resting | **+3.0/tick**, wakes above 100 | `environment.py:676-681` |
| eating an agent | **gains** `min(200, energy + agent.energy)` | `environment.py:722` |

From a full tank: 200 -> 40 at 2.55 = 63 ticks (6.3 s), then 40 -> 0 at 0.55 = 73
ticks (7.3 s), so **~13.6 s of continuous pursuit forces a sleep**, followed by
3.4 s to regain >100 and wake. An unbaited wanderer cycles in ~18 s, so baiting
burns 1.3-2x faster - and occupies the predator throughout.

**Kiting is a bad trade and is not recommended.** 60 ticks of sprint-fleeing cost
the agent **5.600/tick** against the predator's **2.025/tick** - **2.77x worse for
us** - and the agent's 500 max energy has no regeneration while the predator
regains 3.0/tick asleep. Draining a predator by running from it loses.

---

## 7. Legitimacy

| mechanism | inside the normal action interface? |
|---|---|
| NaN corner anchor (immortality + invulnerability) | **No.** Engine defect, and it cannot cross a standards-compliant JSON boundary. |
| `Decimal("NaN")` / `__float__` returning NaN, via native | **No.** Same defect, different door; blocked by pydantic and by JSON. |
| np.float16/32 lockstep divergence | Not an exploit - a correctness risk in **our own** pipeline, worth fixing regardless. |
| Predator stuck at spawn | Not reachable or targetable by a policy. |
| **Pocket-holding / predator baiting** | **Yes - fully legal.** Ordinary movement values and public geometry, no defect. |

The transport was **verified, not assumed**. `ActionRequest` accepts NaN and inf
(pydantic's `allow_inf_nan` defaults to True) and `json.dumps`/`loads` round-trip
them - but **starlette's `JSONResponse` uses `allow_nan=False`**, so the stock
`agent_server.py` endpoint cannot emit a NaN action: it raises
`ValueError: Out of range float values are not JSON compliant: nan`, returns 500,
and `simulation_server.py` treats that as a failed game. Shipping it would require
deliberately bypassing the framework's serializer to put a non-standard `NaN`
token on the wire. It also poisons the observation stream with a NaN `energy`
field, which will crash any stricter runner.

**Recommendation: report the defect, do not exploit it.** An organiser would patch
it with a two-line finiteness guard and would very likely disqualify a submission
that depended on it.

---

## 8. Suggested patches

For the engine, if we report upstream - **both engines must be patched together
or lockstep breaks**:

1. Finiteness guard in `update_entity_direction` (`environment.py:563` /
   `_engine.cpp:1096`).
2. NaN check in `_keep_agent_in_bounds` (`environment.py:777-779` /
   `_engine.cpp:1063-1066`).
3. Finiteness guard on `move_distance` before the clamp (`environment.py:505`),
   which closes the immortality path at `:642`.

For our own pipeline, independent of any of the above:

4. Reject non-`float` action values at the native boundary
   (`_engine.cpp:1687-1701`), or coerce through `float()` explicitly, so
   `np.float16`/`np.float32` cannot silently break the Python-vs-C++ guarantee
   that `verify_evasion.py` claims.

One incidental and fully legitimate observation worth keeping: `move_direction` is
a **free relative heading applied per step** (`environment.py:524`), so an agent
can move in any direction regardless of facing - facing only affects vision - and
the `turn_angle` cost caps at 0.5 energy. Orientation is never a movement
constraint. That is designed behaviour in both engines and is already the right
thing to build policy around.
