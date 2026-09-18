# `/predict` payload vs. `src/utils/DTOs.py`

Recorded on 2026-09-17 with `scripts/recording_agent_server.py` from requests sent by
`cases.nordicaicup.com` (`python-httpx/0.28.1`, source IP 46.62.240.126).
Logs: `logs/ticks/game_20260917_144307_368147.jsonl`.

The platform sends two kinds of payload:

1. **The "Test endpoint" sample**: one fixed request, sent each time you press the button.
   Its format **differs** from `StepResponse`.
2. **Validation game ticks**: the real simulation (156 ticks recorded). The fields and types
   **match** `StepResponse`. How the game starts and ends differs from `simulation_server.py`.

Short version: don't validate the request strictly against `StepResponse`, and
treat `sim_time` and `n_agents` as optional. Otherwise the "Test endpoint" check fails.
With the starter `agent_server.py` it returns 422; with our first version it returned 500.

## 1. "Test endpoint" sample

We received it twice, and both copies were byte-identical:

```json
{
  "game_status": "running",
  "score": 123.4,
  "agent_status": [{
    "agent_id": 1,
    "observations": [
      {"type": "tree", "distance": 12.5, "angle": 1.57},
      {"type": "predator", "distance": 30.0, "angle": -1.57, "rel_dir": -2.57},
      {"type": "edge", "coords": [[50.0, 50.0], [100.0, 100.0]]}
    ],
    "energy": 85.0, "biome": "forest", "age": 5.2,
    "speed": 12.5, "sprint_speed": 13.5, "hearing_radius": 10.0,
    "vision_angle": 1.57, "vision_range": 50.0, "max_energy": 500.0
  }]
}
```

| Field | `DTOs.py` / local simulator | Sample | Impact |
|---|---|---|---|
| `sim_time` | required `float` | **missing** | Strict `StepResponse(**body)` fails validation |
| `n_agents` | required `int` | **missing** | Strict `StepResponse(**body)` fails validation |
| `game_status` | `"ok"` / `"game_over"` | **`"running"`** | Code that checks `== "ok"` treats it as not running |
| Observation `type` | `"Tree"`, `"Predator"`, `"Edge"`, `"Fruit"`, `"Agent"` | **lowercase**: `"tree"`, `"predator"`, `"edge"` | Case-sensitive type checks miss every observation |
| Agent traits | Starting traits are 10 / 20 / 50 / 60° / 200 | Placeholder values (e.g. sprint 13.5, hearing 10) | Don't sanity-check the traits |

Everything else matches: the `agent_status` entry has the same 11 keys as
`ObservationResponse`, and the observation keys (`distance`, `angle`, `rel_dir`, `coords`) are the same.
The predator sample has `rel_dir` but no `id`, which is also what the simulator does.

## 2. Validation game ticks

Every top-level and per-agent field, and its type, matches `StepResponse` /
`ObservationResponse`:

- **Top level:** `game_status: str`, `score: float`, `sim_time: float`, `n_agents: int`, `agent_status: list`.
- **Per agent:** all 11 fields are present. All numbers are floats, including traits like `speed: 10.0`.
- **Observation types** are capitalised, as in the simulator: `Tree`, `Fruit`, `Agent`, `Edge`.
  No predator appeared in this short game.
- **Observation keys** match `creature.py`:
  - `Tree` and `Fruit` have `type, distance, angle`.
  - `Agent` has `type, distance, angle, rel_dir, id`.
  - `Edge` has `type, coords`.
- **Edge `coords`** are relative to the agent, rotated so that +x is the facing direction. They are not
  world coordinates, whatever the README says. The same edge is often repeated several times in one
  observation list, so deduplicate edges.

How the game runs, compared with `simulation_server.py` / `local_playground.py`:

| | Local `simulation_server.py` | Platform |
|---|---|---|
| First request | `agent_status: []`, `n_agents: 0`, `sim_time: 0` (empty step to get observations) | Already has observations: `sim_time: 0.1`, `n_agents: 5`, 5 agents |
| `game_status` values seen | `"ok"` | `"ok"` on every game tick |
| End of game | Loop stops without sending a `game_over` tick | No `game_over` request was received. The last tick was `sim_time: 15.6` with 1 agent |
| Request rate | Back-to-back | About 53 ms between ticks (median), 0.56 s at most |
| Payload size | n/a | 0.4 to 24 KB per tick (1 to 10 agents) |

Consequences:

- **Don't rely on a `game_over` tick** to reset per-game state. Detect a new game instead: `sim_time`
  goes backwards, or requests stop for a while. The evaluation runs 3 games back to back on the same server.
- **Detect "Test endpoint" samples** by the missing `sim_time`, and don't let them corrupt game state.
- **Expect agent IDs to appear and disappear** between ticks. Respond only for the IDs in the current request.

## Recommended request handling

```python
@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    agents = body.get("agent_status") or []
    sim_time = body.get("sim_time")          # None for the platform's test sample
    for agent in agents:
        for obs in agent.get("observations") or []:
            obs["type"] = obs["type"].capitalize()   # "tree" -> "Tree"
    ...
    return {"actions": [...]}   # one ActionRequest-shaped dict per agent_id
```

The response format (`{"actions": [ActionRequest, ...]}`) was accepted with no changes.
