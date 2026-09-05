# speculative_preparedness

Preparation for a DM i AI 2026 survival-simulator task that we have seen
exactly one screenshot of, from an endpoint that no-ops every action and
returns a frozen example state.

The organising principle is that **code is stratified by how much it assumes**,
so that when the real game contradicts us we know exactly which layer dies.

| Layer | Module | Assumes | Survives being wrong? |
| --- | --- | --- | --- |
| 0. Contract | `schema.py` | field shapes | yes -- tolerant parse, drift reported |
| 0. Action seam | `actions.py` | nothing | yes -- swap one codec |
| 1. Baseline | `policies.ReactivePolicy` | predators bad, walls solid | yes |
| 1. Belief | `belief.py` | objects persist and move continuously | yes |
| 2. Harness | `mock.py` | everything, deliberately randomised | n/a -- not a target |
| 3. Data | `capture.py` | nothing | yes |
| 4. Forward model | not built | action format + real data | -- |

## Why there is no learned world model in here

A forward model `p(next_state | belief, action)` needs `(state, action,
next_state)` triples. We cannot produce one triple before the competition,
because no action we send has any effect. Anything "trained" now would be
fitted to `mock.py`, which is fiction we wrote ourselves.

So the preparation for the learnable part is the **pipeline**, not the
weights: `capture.transitions()` emits exactly the training triples a forward
model wants, from the first minute of real data. The model gets built when
there is something true to fit it to.

What *is* built now is the low-assumption half of a world model:
`belief.py` maintains object tracks, velocity estimates, and confidence decay
across frames. Under a vision cone plus a hearing radius, most of the world is
unobserved rather than absent, and a stateless policy forgets a predator at
the exact moment it becomes most dangerous. That layer needs no action format,
no simulator, and no game rules -- only the observation stream, which is the
one thing we have a schema for.

## Why the baseline comes first

`ReactivePolicy` is a pure function of the current frame. It is running proof
that parsing, policy, encoding, capture and the HTTP endpoint all work under
real load, and it is the control every later change is measured against. On
the randomised harness it roughly triples no-op survival, and it would still
work if every entry in ASSUMPTIONS.md turned out to be false.

Ship it first. Measure everything else against it.

## The mock is a harness, not a target

`mock.py` draws its unknown rules fresh per episode -- angle sign convention,
half-cone versus full field of view, sprint cost, whether predators pursue at
all. A policy that only scores well under one draw is fitted to our guesses.
Report `mock.evaluate()`, which is a median across draws, and never a single
episode.

Current standing, 25 draws, seed 7 (median survival seconds):

    noop      16.6
    reactive  60.0
    belief    59.0

The belief layer is not currently beating the reactive baseline on this mock.
That is an honest result and a limitation of the mock rather than a verdict:
its predators are naive pursuers, and its hearing radius is large enough that
you always hear an attacker before it reaches you, which is exactly the
regime where memory is worth least. Sweeping the sensing parameters
(`hearing_radius` 1-10, `vision_range` 20-50) moves the belief advantage
between +0.2 and +3.4 seconds. Keep both policies; decide on real data.

## Layout

    schema.py       tolerant payload parsing + schema_drift()
    actions.py      canonical Action intent + interchangeable wire codecs
    belief.py       object tracking, velocity, confidence decay
    policies.py     NoOp / Reactive / Belief, all guesses in PolicyConfig
    mock.py         rule-randomised sim, RuleDraw.sample()
    capture.py      JSONL capture, replay, transitions()
    server.py       FastAPI endpoint (lazy deps)
    ASSUMPTIONS.md  every guess, its cost, and a probe that falsifies it

Core modules are standard library only. `server.py` needs `fastapi` and
`uvicorn`, imported lazily so tests run without them.

## First fifteen minutes of the competition

1. Point the endpoint at the real server with `SP_POLICY=noop`. Confirm
   frames arrive and read the `schema_drift` output. (A1, A9)
2. Run the A2 probe: send an extreme action, watch for any bearing change.
   Cycle codecs until something moves. Nothing else matters until this works.
3. Run the A3 sign probe and the A6 energy probe. Both take under a minute
   and both silently invert policy behaviour if wrong.
4. Switch to `SP_POLICY=reactive`. Record the score. That is the floor.
5. Only now consider anything cleverer -- with `capture.replay()` to check a
   change against real frames before shipping it.

## Scope warning

This rests on one screenshot of a stub. The 2026 task may differ in shape or
may not exist. Nothing here is worth more time than it takes to keep layers
0, 1 and 3 solid; those are useful for *any* streaming-state task the
competition ships, survival simulator or not.
