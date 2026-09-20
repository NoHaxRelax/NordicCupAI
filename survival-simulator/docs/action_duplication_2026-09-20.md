# Duplicate actions: one tick, many moves

20 September 2026. Simulator revision `282446421263e7bb11081d692f0ac443b99f9e20`.

**The action list is never deduplicated. Listing one `agent_id` N times makes that agent take N
turns inside a single tick.** Energy is charged per action from that action's own distance, so N
walk-sized actions bypass both the `sprint_speed` distance cap and the sprint surcharge. Every
value involved is an ordinary finite float that survives `json.dumps`, pydantic and starlette —
the only unusual thing about the payload is the repetition.

Measured value: **+36% score and 57% lower predation loss** over 6 seeds at full horizon when used
as a travel and escape budget. Used naively on every action it is catastrophic (**score 4–10 against
a 723 baseline**).

## Mechanism

`simulation_server.py` takes the action list straight from the policy's HTTP response and hands it
through unmodified:

```python
actions = [ActionRequest(**a) for a in resp.json().get("actions", [])]
parsed_actions = [(a.agent_id, a) for a in actions]
state = sim.step(parsed_actions)
```

[simulation.py](../src/utils/simulation.py), line 57, then iterates it:

```python
for agent_id, action in actions:
    env.agent_step(agent_id, ...)
```

No dedup, no length cap, no per-agent limit. Each entry is a full `agent_step`: clamp, energy
charge, biome-scaled displacement, collision check, direction update, spawn gate.

A policy emits duplicates by repeating the dict in its response:

```python
action = {"agent_id": 3, "move_distance": 10.0, "move_direction": 0.0,
          "turn_angle": 0.0, "spawn_agent": False}
return {"actions": [action] * 200}
```

## Why the repetition is worth anything

[environment.py](../src/elements/environment.py), lines 505–518. The distance clamp and the
walk/sprint branch are both evaluated **per action**:

- `distance > sprint_speed` → clamped to `sprint_speed`
- `distance <= speed` → charge `distance * 0.05`
- otherwise → charge `speed * 0.05 + (distance - speed) * 0.5`

Keep every individual action at or below `speed` and you always pay the walking rate, however many
you send. Measured, founder traits (speed 10, sprint 20), flat terrain:

| Scheme | Distance | Energy | Per pixel |
| --- | ---: | ---: | ---: |
| 1 × sprint 20 | 20 px | 5.500 | 0.2750 |
| 2 × walk 10 | 20 px | **1.000** | **0.0500** |
| 10 × sprint 20 | 200 px | 55.000 | 0.2750 |
| 20 × walk 10 | 200 px | **10.000** | **0.0500** |
| 200 × walk 10 | 2000 px | 100.000 | **0.0500** |
| 1000 × walk 1 | 1000 px | 50.000 | **0.0500** |

Cost is exactly linear in N with **no per-action overhead** (0.500000 per action measured at
N = 1, 10, 100, 1000 and 10000). Granularity below `speed` is free — a thousand one-pixel steps
cost what a hundred ten-pixel steps cost.

Granularity is not behaviourally neutral, though. Each action runs its own collision check, so small
steps path around obstacles unaided, while only steps of **≥ 14.142 px** can clip a rock corner.

One counterintuitive interaction: the low-energy clamp at `:512` silently downgrades sprint actions
to walk speed once `energy < max_energy / 5`, which is a **5.5× efficiency gain** per pixel. Measured
mid-sequence: action 140 cost 5.500 for 20 px (0.2750/px), action 150 cost 0.500 for 10 px
(0.0500/px). A nearly-empty agent is the cheapest traveller in the game.

## Use 1 — travel and escape

Predators close 11–15 px/tick against a walking agent's 10, so without duplicates an agent cannot
outrun one at a price it can afford. Measured over 6 seeds, full horizon, identical forager core:

| Policy | Mean score | Mean predation loss |
| --- | ---: | ---: |
| 1 walk-step per tick (baseline) | 594.58 | 112.21 |
| **hop ≤ 4 steps, flee 2 steps** | **806.33** | **48.61** |
| hop ≤ 8, flee 3 | 737.50 | 27.24 |
| hop ≤ 20, flee 4 | 764.65 | 42.90 |

Naive use is fatal: `dup=8` on every action scored **4–10** against the 723 baseline, because agents
burn out in seconds. Treat it as a budget, gated on need.

## Use 2 — score farming

Requires combining with a second defect. [environment.py](../src/elements/environment.py) line 635
is `for agent in self.agents:` while `kill_agent` at `:643` calls `self.agents.remove(agent)` — the
comment claiming a copy is made is false. Removing element *i* **skips element *i+1*** for the
whole tick, so that agent never reaches the `energy <= 0` check at `:642`.

`turn_angle = π` costs exactly 0.5 energy and moves **0.0000 px** (measured), making it a pure drain
primitive that keeps the agent in place beside a predator. The payout is `score -= agent.energy/100`
at `:723`, which inverts once energy is negative:

| Energy when eaten | Score delta |
| ---: | ---: |
| 400.0 | −4.0000 |
| 75.0 | −0.7500 |
| 0.5 | −0.0050 |
| −100.0 | **+1.0000** |
| −5000.0 | **+50.0000** |

Measured scaling, one farm agent starting at 400 energy:

| Duplicate actions | Farm energy | Score | Engine time | Payload |
| ---: | ---: | ---: | ---: | ---: |
| 2 000 | −600.0 | +6.10 | 0.19 s | 0.23 MB |
| 10 000 | −4 600.0 | +46.10 | 0.83 s | 1.15 MB |
| 20 000 | −9 600.0 | +96.10 | 1.57 s | 2.30 MB |
| 60 000 | −29 600.0 | **+296.10** | 4.97 s | 6.90 MB |

The sacrificial agent is required — the same 2000-action drain without one scored **+0.10** (clock
only) against **+6.10** with one.

Harvests **stack within a tick**, and one predator harvests every agent touching it, since the eat
loop at `:721` iterates all of them:

| Doomed/farm pairs | Predators | Score |
| ---: | ---: | ---: |
| 1 | 1 | +16.10 |
| 2 | 1 | +32.10 |
| 4 | 1 | +64.10 |
| 8 | 1 | **+128.10** |

Payout is `N × F / 200` for N duplicates per agent and F farm agents harvested. Both scale linearly
and they multiply.

**Use newborns, not full tanks.** The payout is exactly `−energy/100`, so the agent's entire starting
balance burns off before anything earns. A child at 75 needs 150 actions to reach zero; a 400-energy
agent wastes 800.

## Use 3 — predator removal

`predator.energy = min(max_energy, predator.energy + agent.energy)` at `:722`, so a negative-energy
victim **drains** the predator. Below zero it sets `resting = True` and the resting branch `continue`s
past observation, movement and the eat check.

| Drain | Predator after | Resting | Ticks to wake | Seconds |
| ---: | ---: | --- | ---: | ---: |
| −100 | 99.8 | no | 0 | 0.0 |
| −1 600 | −1 400.2 | yes | 501 | 50.1 |
| −9 600 | −9 400.2 | yes | 3 167 | 316.7 |
| −29 600 | −29 400.2 | yes | **9 834** | **983.4** |

Verified inert, not merely not-hungry: 500 ticks with live bait 20 u away — inside the 30 u effective
kill radius — left the bait **alive** with predator displacement **0.00 px**.

A −100 drain leaves the predator at 99.8 and still awake; the harvest must exceed its current energy
(at most 200) to knock it out.

## Limits and costs

- **One harvest per predator.** Repeat attempts in following ticks paid clock only, because the
  predator is asleep. It is a series of one-shot detonations, not a pump.
- **Small harvests lose score.** 40 ticks of repeated sub-zero-threshold harvests paid **0 times**
  and finished at **−5.30** against +4.00 for the clock alone.
- **All sub-actions run against the same stale observation.** The world does not step between them,
  so a long sequence commits blind. The observation is already one predator-move (15.00 u) behind,
  giving a 45 u reaction budget against the 30 u kill radius.
- **Payload and compute**: 102 bytes and ~1 ms per action. The sim step is untimed — only the
  policy's response carries `timeout=10` — so a 200 000-action tick would run ~3 minutes server-side
  unchecked.
- **A one-line fix removes it.** Any dedup on the action list reduces this to a single action per
  tick, so policies should degrade gracefully without it.

## Classification

| Element | Verdict |
| --- | --- |
| Duplicate action list | **DEFECT** in the engine; the payload is entirely ordinary finite floats |
| Bypassing `sprint_speed` and the sprint surcharge | Direct consequence of per-action charging |
| List-mutation skip (`:635` + `:643`) | **DEFECT**, independent of duplication |
| Score increase on death | **DEFECT** — `score -= energy/100` was not written for negative energy |
| Predator immobilisation | Consequence of `min(max_energy, energy + agent.energy)` with a negative addend |

Related: the stale-observation and phantom-observation defects are documented in
[action_interface_fuzzing_20260919.md](action_interface_fuzzing_20260919.md).

## Interactive walkthrough

An interactive page running the engine's real arithmetic — distance clamp, walk/sprint branch,
biome multiplier applied after the charge, low-energy cap — is published at
<https://claude.ai/artifact/TEXF6PJjHA6on4B6xy6HHV> (private; ask Nikolaj for access).
