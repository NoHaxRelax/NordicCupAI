"""Action-order and newborn-timing probes for the pinned survival simulator.

All claims execute the unmodified engine at commit
acfc31a4003a5f91bf11032a02cd98c178ddbd7e.  Small fixtures isolate ordering;
the generated-map check uses ordinary default maps.  No network calls occur.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

SURVIVAL_ROOT = Path(__file__).resolve().parents[2]
VENDOR = SURVIVAL_ROOT / "vendor" / "survival-simulator"
sys.path.insert(0, str(VENDOR))

from src.core import SimulationCore
from src.elements.biome import Forest_biome
from src.elements.environment import Environment
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment


COMMIT = "acfc31a4003a5f91bf11032a02cd98c178ddbd7e"


def action(agent_id: int = 0, distance: float = 0.0, direction: float = 0.0,
           turn: float = 0.0, spawn: bool = False) -> ActionRequest:
    return ActionRequest(agent_id=agent_id, move_distance=distance,
                         move_direction=direction, turn_angle=turn,
                         spawn_agent=spawn)


def advance(env: Environment, actions: list[ActionRequest]) -> dict:
    return step_environment(env, [(item.agent_id, item) for item in actions])


def fixture(seed: int = 1101, width: int = 400) -> tuple[Environment, object]:
    env = Environment(width, 400, 400, random.Random(seed))
    env.biome_map.fill(Forest_biome())
    parent = env.spawn_agent(x=100, y=200)
    parent.direction = 0.0
    parent.max_age = 1e9
    return env, parent


def state(env: Environment, agent: object) -> dict:
    return {
        "id": agent.agent_id,
        "x": float(agent.x),
        "y": float(agent.y),
        "direction": float(agent.direction),
        "energy": float(agent.energy),
        "age": float(agent.age),
        "alive": agent.agent_id in env.agents_dict,
    }


def probe_duplicate_actions() -> dict:
    cases: dict[str, dict] = {}
    specs = {
        "omitted": [],
        "one_walk_10": [action(distance=10)],
        "one_oversized_1000": [action(distance=1000)],
        "ten_walk_10": [action(distance=10)] * 10,
        "hundred_walk_10": [action(distance=10)] * 100,
    }
    for name, actions in specs.items():
        env, parent = fixture(width=1200)
        before = state(env, parent)
        advance(env, actions)
        after = state(env, parent)
        cases[name] = {
            "action_count": len(actions),
            "distance": math.hypot(after["x"] - before["x"], after["y"] - before["y"]),
            "energy_spent": before["energy"] - after["energy"],
            "after": after,
        }

    assert math.isclose(cases["one_walk_10"]["distance"], 10)
    assert math.isclose(cases["one_oversized_1000"]["distance"], 20)
    assert math.isclose(cases["ten_walk_10"]["distance"], 100)
    assert math.isclose(cases["hundred_walk_10"]["distance"], 1000)
    assert math.isclose(cases["ten_walk_10"]["energy_spent"], 5.1)
    assert math.isclose(cases["hundred_walk_10"]["energy_spent"], 50.1)
    return {
        "classification": "rule-breaking: duplicate IDs violate one action per observed agent",
        "observation_requirement": "ordinary action fields; result magnitude measured with privileged positions",
        "cases": cases,
        "magnitude": "100 walk requests move 1000 units in one 0.1 s tick at walking cost, versus the 20-unit single-action cap",
    }


def probe_newborn_timing() -> dict:
    env_a, parent_a = fixture(seed=1201)
    child_id = env_a._next_agent_id
    response_a = advance(env_a, [
        action(parent_a.agent_id, spawn=True),
        action(child_id, distance=20, turn=1.0),
    ])
    child_a = env_a.agents_dict[child_id]
    child_a_state = state(env_a, child_a)
    parent_saw_child = any(
        obs.get("type") == "Agent" and obs.get("id") == child_id
        for obs in env_a.agent_observations[parent_a.agent_id]
    )
    child_saw_parent = any(
        obs.get("type") == "Agent" and obs.get("id") == parent_a.agent_id
        for obs in env_a.agent_observations[child_id]
    )

    # Negative control: the same future-ID action before the birth is silently skipped.
    env_b, parent_b = fixture(seed=1201)
    response_b = advance(env_b, [
        action(child_id, distance=20, turn=1.0),
        action(parent_b.agent_id, spawn=True),
    ])
    child_b = env_b.agents_dict[child_id]
    child_b_state = state(env_b, child_b)

    spawn_x, spawn_y = child_b.x, child_b.y
    moved = math.hypot(child_a.x - spawn_x, child_a.y - spawn_y)
    facing_delta = (child_a.direction - child_b.direction + math.pi) % (2 * math.pi) - math.pi

    # Newborn energy is 75 < max_energy/5 (100), so a requested sprint is
    # reduced to its 10-unit walking speed before the ordinary energy costs.
    assert math.isclose(moved, 10)
    assert math.isclose(facing_delta, 1.0)
    assert math.isclose(child_b.energy, 74.9)
    assert math.isclose(child_a.energy, 75 - 0.5 - 1 / (2 * math.pi) - 0.1)
    assert child_a.age == child_b.age == 0.1
    assert parent_saw_child and child_saw_parent
    assert any(item["agent_id"] == child_id for item in response_a["observations"])
    assert any(item["agent_id"] == child_id for item in response_b["observations"])

    return {
        "classification": {
            "same_request_action": "rule-breaking: child was not in the preceding observation list",
            "same_tick_aging_observation": "contract-compliant consequence of an ordinary birth",
        },
        "observation_requirement": {
            "exploit": "predictable sequential ID plus privileged/guessed newborn heading for directed movement",
            "ordinary_next_response": "child status and mutual relative observations are returned normally",
        },
        "spawn_then_child_action": child_a_state,
        "child_action_then_spawn_control": child_b_state,
        "same_request_requested_distance": 20,
        "same_request_actual_distance": moved,
        "turn_applied_after_newborn_move": facing_delta,
        "parent_saw_child_same_tick": parent_saw_child,
        "child_saw_parent_same_tick": child_saw_parent,
        "returned_child_age": child_a.age,
        "returned_child_idle_energy_control": child_b.energy,
        "limitation": "the newborn's initial heading is random and unseen, turn_angle is applied after movement, and starting energy 75 is below the 100 sprint threshold, so the blind requested sprint is reduced to a 10-unit walk",
    }


def spawn_case(energy: float, *, distance: float = 0.0,
               turn: float = 0.0) -> dict:
    env, parent = fixture(seed=1301)
    parent.energy = energy
    before_population = len(env.agents)
    before_next_id = env._next_agent_id
    advance(env, [action(distance=distance, turn=turn, spawn=True)])
    return {
        "initial_energy": energy,
        "distance": distance,
        "turn": turn,
        "birth_occurred": env._next_agent_id > before_next_id,
        "parent": state(env, parent),
        "population": len(env.agents),
        "living_ids": [agent.agent_id for agent in env.agents],
    }


def probe_reproduction_order_and_reserve() -> dict:
    cases = {
        "100.05_idle": spawn_case(100.05),
        "100.11_idle": spawn_case(100.11),
        "100.40_idle": spawn_case(100.40),
        "100.40_walk10": spawn_case(100.40, distance=10),
        "100.40_turn_pi": spawn_case(100.40, turn=math.pi),
        "100.61_walk10": spawn_case(100.61, distance=10),
    }
    assert cases["100.05_idle"]["birth_occurred"]
    assert not cases["100.05_idle"]["parent"]["alive"]
    assert cases["100.11_idle"]["parent"]["alive"]
    assert cases["100.40_idle"]["birth_occurred"]
    assert not cases["100.40_walk10"]["birth_occurred"]
    assert not cases["100.40_turn_pi"]["birth_occurred"]
    assert cases["100.61_walk10"]["birth_occurred"]
    assert cases["100.61_walk10"]["parent"]["alive"]
    return {
        "classification": "contract-compliant",
        "observation_requirement": "ordinary energy, age, traits, biome and requested action; the private max-age threshold makes exact elder reserve uncertain",
        "cases": cases,
        "rule": "spawn is checked after movement and turning; parent survival is checked after the 100 birth cost and passive/elder drain",
        "young_forest_thresholds": {
            "birth": "energy - movement_cost - turn_cost > 100",
            "parent_survival": "energy - movement_cost - turn_cost - 100 - 0.1 > 0",
        },
        "practical_effect": "a low-energy birth can preserve the lineage by replacing a doomed parent, but movement or turning can unexpectedly cancel the birth or kill the parent",
    }


def ripe_fruit(env: Environment, x: float, y: float):
    fruit = env.spawn_fruit(x=x, y=y)
    assert fruit is not None
    while fruit.energy < 60:
        fruit.grow(1)
    return fruit


def probe_fruit_and_death_order() -> dict:
    # Fruit under the agent is collected only in non_agent_step, after spawn checks.
    env, parent = fixture(seed=1401)
    parent.energy = 99.0
    fruit = ripe_fruit(env, parent.x, parent.y)
    first = advance(env, [action(spawn=True)])
    first_state = state(env, parent)
    first_population = len(env.agents)
    second = advance(env, [action(spawn=True)])
    second_state = state(env, parent)
    assert first_population == 1 and len(env.agents) == 2
    assert math.isclose(first_state["energy"], 158.9)

    # Reaching food does not rescue an agent if action + passive costs cross zero;
    # the death test precedes fruit collection.
    survival_cases = {}
    for label, energy in (("dies_at_0.59", 0.59), ("survives_at_0.61", 0.61)):
        env_c, agent_c = fixture(seed=1402)
        agent_c.energy = energy
        fruit_c = ripe_fruit(env_c, 110, 200)
        advance(env_c, [action(distance=10)])
        survival_cases[label] = {
            "initial_energy": energy,
            "agent": state(env_c, agent_c),
            "fruit_remaining": fruit_c in env_c.fruits,
        }
    assert not survival_cases["dies_at_0.59"]["agent"]["alive"]
    assert survival_cases["dies_at_0.59"]["fruit_remaining"]
    assert survival_cases["survives_at_0.61"]["agent"]["alive"]
    assert not survival_cases["survives_at_0.61"]["fruit_remaining"]

    return {
        "classification": "contract-compliant",
        "observation_requirement": "ordinary energy/biome/action costs; fruit ripeness is hidden, and exact elder drain can be hidden",
        "fruit_cannot_fund_birth_same_tick": {
            "initial_energy": 99.0,
            "first_tick_birth": first_population > 1,
            "first_tick_after_fruit": first_state,
            "second_tick_birth": len(env.agents) == 2,
            "second_tick_parent": second_state,
            "first_response_ids": [item["agent_id"] for item in first["observations"]],
            "second_response_ids": [item["agent_id"] for item in second["observations"]],
            "fixture": "ripe 60-energy fruit placed under parent",
        },
        "food_arrival_death_boundary": survival_cases,
        "practical_effect": "reserve movement plus passive drain before a food endpoint; spawn only from pre-collection energy",
    }


def probe_two_parent_order() -> dict:
    outputs = {}
    for label, order in (("parent_0_then_1", (0, 1)), ("parent_1_then_0", (1, 0))):
        env, parent0 = fixture(seed=1501)
        parent1 = env.spawn_agent(x=250, y=200)
        parent0.speed = 10.0
        parent1.speed = 18.0
        parent0.energy = parent1.energy = 150.0
        parent0.max_age = parent1.max_age = 1e9
        advance(env, [action(agent_id, spawn=True) for agent_id in order])
        outputs[label] = {
            "action_order": list(order),
            "children": [
                {"id": child.agent_id, "speed": float(child.speed),
                 "x": float(child.x), "y": float(child.y)}
                for child in env.agents if child.agent_id >= 2
            ],
        }
    assert outputs["parent_0_then_1"]["children"][0]["speed"] <= 15
    assert outputs["parent_1_then_0"]["children"][0]["speed"] >= 15
    return {
        "classification": "contract-compliant but not a demonstrated advantage",
        "observation_requirement": "ordinary: controller chooses list order and receives child IDs/traits next response",
        "results": outputs,
        "effect": "birth actions consume the shared RNG and allocate sequential child IDs in request-list order",
        "practical_effect": "ordering changes lineage-to-ID assignment and deterministic replay outcomes; hidden RNG prevents choosing the better mutation prospectively",
    }


def probe_validation_gaps() -> dict:
    env, parent = fixture(seed=1601)
    before = state(env, parent)
    advance(env, [action(agent_id=999, distance=10)])
    unknown_after = state(env, parent)
    assert before["x"] == unknown_after["x"]
    assert unknown_after["age"] == 0.1

    nan_request = action(distance=float("nan"))
    env_nan, parent_nan = fixture(seed=1602)
    advance(env_nan, [nan_request])
    nan_state = state(env_nan, parent_nan)
    assert not math.isfinite(nan_state["energy"])

    return {
        "classification": "malformed/rule-breaking inputs; no useful strategy claim",
        "unknown_id": {
            "result": "silently skipped while the world still advances",
            "parent_after": unknown_after,
        },
        "non_finite_float": {
            "dto_accepted_nan": True,
            "energy_remained_finite": math.isfinite(nan_state["energy"]),
            "x_after": nan_state["x"],
            "effect": "NaN poisons energy and produces invalid state rather than a scoring or survival advantage",
        },
        "negative_control": "omission/unknown IDs do not freeze passive aging",
    }


def probe_generated_maps(seeds: tuple[int, ...] = (1, 42)) -> list[dict]:
    """One-tick realistic-map confirmation of duplicate-action magnitude."""
    rows = []
    for seed in seeds:
        modes = {}
        for name, repeats in (("single", 1), ("duplicate_x10", 10)):
            sim = SimulationCore(seed=seed)
            before = {
                agent.agent_id: (float(agent.x), float(agent.y), float(agent.energy))
                for agent in sim.env.agents
            }
            actions = []
            for agent in sim.env.agents:
                actions.extend([action(agent.agent_id, distance=10)] * repeats)
            sim.step([(item.agent_id, item) for item in actions])
            per_agent = []
            for agent in sim.env.agents:
                bx, by, be = before[agent.agent_id]
                per_agent.append({
                    "id": agent.agent_id,
                    "displacement": math.hypot(agent.x - bx, agent.y - by),
                    "energy_spent": be - agent.energy,
                    "biome": sim.env.get_agent_state(agent.agent_id)["biome"],
                })
            modes[name] = per_agent
        rows.append({"seed": seed, "modes": modes})

    for row in rows:
        assert all(math.isclose(item["energy_spent"], 0.6)
                   for item in row["modes"]["single"])
        assert all(math.isclose(item["energy_spent"], 5.1)
                   for item in row["modes"]["duplicate_x10"])
    return rows


def hypotheses(results: dict) -> list[dict]:
    return [
        {"id": "A1", "hypothesis": "duplicate IDs execute sequentially before one world tick", "status": "confirmed-known-and-extended", "evidence": "duplicate_actions and generated_maps"},
        {"id": "A2", "hypothesis": "one oversized action bypasses sprint cap", "status": "rejected", "evidence": "one_oversized_1000 moved 20"},
        {"id": "A3", "hypothesis": "omitting/unknown IDs freezes the agent", "status": "rejected", "evidence": "world age and passive drain still advance"},
        {"id": "A4", "hypothesis": "a future newborn ID can act after its birth in the same list", "status": "confirmed-known-and-extended", "evidence": "reverse ordering is a negative control; unseen random heading limits aiming"},
        {"id": "A5", "hypothesis": "newborns avoid the birth tick's aging/drain/observation phase", "status": "rejected", "evidence": "returned at age 0.1 and idle energy 74.9 with mutual observations"},
        {"id": "A6", "hypothesis": "turning in an action changes that action's movement heading", "status": "rejected", "evidence": "movement precedes turn, including the newborn action"},
        {"id": "A7", "hypothesis": "energy just over 100 guarantees birth and parent survival", "status": "rejected", "evidence": "action costs can cancel birth; passive drain can kill parent after successful birth"},
        {"id": "A8", "hypothesis": "fruit collected this tick can pay for this tick's birth", "status": "rejected", "evidence": "99-energy parent ate ripe fruit after failed birth and reproduced next tick"},
        {"id": "A9", "hypothesis": "reaching fruit rescues an agent whose movement cost crosses zero", "status": "rejected", "evidence": "death check precedes fruit collection"},
        {"id": "A10", "hypothesis": "two compliant birth actions are order-independent", "status": "rejected-as-determinism-claim", "evidence": "request order assigns RNG sequence and child IDs to different lineages; no prospective advantage"},
        {"id": "A11", "hypothesis": "DTO validation rejects non-finite movement", "status": "rejected", "evidence": "NaN accepted and corrupts state; malformed and not useful"},
    ]


def run(include_generated_maps: bool = True) -> dict:
    results = {
        "duplicate_actions": probe_duplicate_actions(),
        "newborn_timing": probe_newborn_timing(),
        "reproduction_order_and_reserve": probe_reproduction_order_and_reserve(),
        "fruit_and_death_order": probe_fruit_and_death_order(),
        "two_parent_order": probe_two_parent_order(),
        "validation_gaps": probe_validation_gaps(),
    }
    results["generated_maps"] = probe_generated_maps() if include_generated_maps else []
    return {
        "source_commit": COMMIT,
        "scope": "unmodified local engine; isolated fixtures plus default generated maps; no remote calls",
        "fixture_disclosures": [
            "isolated fixtures replace the biome map with forest and set max_age high",
            "threshold probes assign explicit energy and ripe fruit positions",
            "positions and engine state are read only for diagnostic measurement",
            "generated_maps use default dimensions, obstacles, agents, fruits, trees, energy, and seeds",
        ],
        "results": results,
        "hypotheses": hypotheses(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=SURVIVAL_ROOT / "results" / "mechanics_hunt" / "01_actions.json",
    )
    parser.add_argument("--skip-generated-maps", action="store_true")
    args = parser.parse_args()
    payload = run(include_generated_maps=not args.skip_generated_maps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
