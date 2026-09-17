"""Recorded generated-map search, engagement, and intake harness.

The unmodified simulator generates the complete biome and obstacle map. Agent
and initial predator positions come from separate role/slot RNG streams over
the full valid map; there is no prepared guide/predator pairing. The policy is
constructed with a JSON-only static map, role IDs, and nothing else. At every
tick it receives only JSON-round-tripped native observation DTOs and public
simulation time. Hidden creature state is read after actions solely for the
evaluation receipt. Agent energy is refilled; predator energy/rest stay native.
"""
from __future__ import annotations

import argparse
import heapq
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "research"), str(ROOT / "debugger"),
                str(ROOT / "vendor" / "survival-simulator")]

from recorder import ReplayRecorder, json_value
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest, ObservationResponse

from sampling import sample_fixture
from static_map import json_clone, payload_hash, snapshot_environment


OUT = ROOT / "results" / "real_map_intake_sol"
STOP = Path("/tmp/predator-intake-stop")


def _refresh_visualization_manifest():
    """Refresh the native-frame catalog without risking the saved receipt."""
    builder = ROOT / "research" / "real_map_visualization" / "build_manifest.py"
    if not builder.exists():
        return {"status": "builder_missing"}
    try:
        completed = subprocess.run(
            [sys.executable, str(builder)], cwd=ROOT, capture_output=True,
            text=True, timeout=120, check=False,
        )
    except Exception as exc:  # receipt/replay validity must not depend on viewer tooling
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "ok" if completed.returncode == 0 else "nonzero",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-500:],
        "stderr": completed.stderr.strip()[-500:],
    }


def _harness_hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("run.py", "sampling.py", "static_map.py")}


def _load_policy(spec: str):
    try:
        module_name, class_name = spec.split(":", 1)
    except ValueError as exc:
        raise ValueError("policy must be module:Class") from exc
    module = importlib.import_module(module_name)
    policy_class = getattr(module, class_name)
    source = Path(module.__file__)
    return policy_class, source


def _policy_site(policy, static_map: dict) -> dict:
    site = json_clone(policy.site)
    required = {"mouth", "inward", "cross", "goal", "gap", "overlap",
                "obstacle_indices"}
    missing = required - site.keys()
    if missing:
        raise ValueError(f"policy site missing {sorted(missing)}")
    for name in ("mouth", "inward", "cross", "goal"):
        if len(site[name]) != 2 or not all(math.isfinite(float(v)) for v in site[name]):
            raise ValueError(f"site {name} must be two finite numbers")
        site[name] = [float(v) for v in site[name]]
    norm = math.hypot(*site["inward"])
    if not .999 <= norm <= 1.001:
        raise ValueError("site inward must be a unit vector")
    for index in site["obstacle_indices"]:
        if not isinstance(index, int) or not 0 <= index < len(static_map["obstacles"]):
            raise ValueError("site obstacle index outside static map")
    return site


def _site_goal_free(env, site: dict) -> bool:
    goal = tuple(site["goal"])
    return (5 <= goal[0] <= env.width - 5 and 5 <= goal[1] <= env.height - 5
            and not env._in_obstacle(goal, 5, env.obstacles)
            and env._is_position_free(goal[0], goal[1], 20, 20))


def _segment_hits_rect(a, b, rect):
    x, y, width, height = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    low, high = 0.0, 1.0
    for p, q in ((-dx, a[0] - x), (dx, x + width - a[0]),
                 (-dy, a[1] - y), (dy, y + height - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            value = q / p
            if p < 0:
                low = max(low, value)
            else:
                high = min(high, value)
            if low > high:
                return False
    return True


def _oracle_path(static_map, start, target, radius=5.01):
    """Visibility-graph path used only by the disclosed pre-DTO oracle."""
    rects = [(row["x"], row["y"], row["width"], row["height"])
             for row in static_map["obstacles"]]
    width, height = static_map["width"], static_map["height"]

    def free(point, margin=radius):
        if not (margin <= point[0] <= width - margin
                and margin <= point[1] <= height - margin):
            return False
        return not any(x - margin < point[0] < x + w + margin
                       and y - margin < point[1] < y + h + margin
                       for x, y, w, h in rects)

    def clear(a, b):
        return (free(a) and free(b)
                and not any(_segment_hits_rect(
                    a, b, (x - radius, y - radius,
                           w + 2 * radius, h + 2 * radius))
                            for x, y, w, h in rects))

    if clear(start, target):
        return [target]
    corner_margin = radius + .2
    low = (min(start[0], target[0]) - 250,
           min(start[1], target[1]) - 250)
    high = (max(start[0], target[0]) + 250,
            max(start[1], target[1]) + 250)
    corners = []
    for x, y, w, h in rects:
        for point in ((x - corner_margin, y - corner_margin),
                      (x + w + corner_margin, y - corner_margin),
                      (x + w + corner_margin, y + h + corner_margin),
                      (x - corner_margin, y + h + corner_margin)):
            if (low[0] <= point[0] <= high[0]
                    and low[1] <= point[1] <= high[1] and free(point)):
                corners.append(point)
    nodes = [start, target, *corners]
    distances = [math.inf] * len(nodes)
    distances[0] = 0.0
    previous = {}
    queue = [(0.0, 0)]
    while queue:
        cost, index = heapq.heappop(queue)
        if cost != distances[index]:
            continue
        if index == 1:
            break
        for other in range(1, len(nodes)):
            if other == index:
                continue
            step = math.dist(nodes[index], nodes[other])
            if cost + step >= distances[other] or not clear(nodes[index], nodes[other]):
                continue
            distances[other] = cost + step
            previous[other] = index
            heapq.heappush(queue, (cost + step, other))
    if not math.isfinite(distances[1]):
        return []
    path = []
    current = 1
    while current:
        path.append(nodes[current])
        current = previous[current]
    return list(reversed(path))


def _spawn_fixture(env, fixture: dict, *, site: dict, station_bait: bool):
    rows = json_clone(fixture)
    if station_bait:
        if not _site_goal_free(env, site):
            raise ValueError("policy bait goal is not valid for native agent spawn")
        rows["agents"][0].update(x=site["goal"][0], y=site["goal"][1],
                                  arranged_at_site=True)
    agents = []
    for row in rows["agents"]:
        agent = env.spawn_agent(x=row["x"], y=row["y"])
        if agent is None or math.dist((agent.x, agent.y), (row["x"], row["y"])) > 1e-9:
            raise RuntimeError("native agent spawn rejected a prevalidated fixture point")
        agent.direction = row["heading"]
        agents.append(agent)
        row["agent_id"] = agent.agent_id
    predators = []
    for row in rows["predators"]:
        predator = env.spawn_predator(x=row["x"], y=row["y"])
        if predator is None or math.dist((predator.x, predator.y),
                                         (row["x"], row["y"])) > 1e-9:
            raise RuntimeError("native predator spawn rejected a prevalidated fixture point")
        predator.direction = row["heading"]
        predators.append(predator)
    env._update_spatial_grid()
    return agents, predators, rows


def _dto_states(env):
    return [ObservationResponse(**env.get_agent_state(agent.agent_id)).model_dump()
            for agent in env.agents]


def _validate_actions(actions, env):
    if not isinstance(actions, list):
        raise TypeError("policy act must return a list")
    alive = set(env.agents_dict)
    ids = [action.get("agent_id") for action in actions]
    if len(ids) != len(set(ids)) or set(ids) != alive:
        raise ValueError("policy must return exactly one action for each living agent")
    requests = []
    for action in actions:
        request = ActionRequest(**action)
        agent = env.agents_dict[request.agent_id]
        values = (request.move_distance, request.move_direction, request.turn_angle)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("action contains a non-finite number")
        if not 0 <= request.move_distance <= agent.sprint_speed + 1e-6:
            raise ValueError("move_distance outside native limit")
        requests.append(request)
    return requests


def run(*, policy_spec="smoke_policy:SmokePolicy", map_seed=5101,
        fixture_seed=9101, seconds=1.0, predators=1, station_bait=False,
        native_render=True, native_width=800, reproduction_of=None,
        mode="direct_guide"):
    if STOP.exists():
        raise SystemExit("stop sentinel present")
    if seconds <= 0 or predators < 1:
        raise ValueError("seconds must be positive and predators at least one")
    child_modes = {"encounter_child_guide", "oracle_approach_child_guide"}
    if mode not in {"direct_guide", *child_modes}:
        raise ValueError("unknown run mode")

    core = SimulationCore(starting_agents=0, starting_predators=0, seed=map_seed)
    env = core.env
    static_map = snapshot_environment(env)
    static_hash = payload_hash(static_map)
    policy_class, policy_file = _load_policy(policy_spec)
    policy_hash = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    policy_input = json_clone(static_map)
    try:
        policy = policy_class(policy_input, bait_id=0, guide_id=1)
    except ValueError as exc:
        if "no eligible" not in str(exc).lower():
            raise
        tag = (f"real-map-{policy_class.__name__}-m{map_seed}-f{fixture_seed}"
               f"-unsupported-{uuid4().hex[:8]}")
        result = {
            "schema": "real-map-intake-setup-v1",
            "outcome": "unsupported_map_no_eligible_site",
            "policy": policy_spec,
            "policy_hash": policy_hash,
            "harness_hashes": _harness_hashes(),
            "mode": mode,
            "map_seed": map_seed,
            "fixture_seed": fixture_seed,
            "static_map_hash": static_hash,
            "error": str(exc),
            "simulation_started": False,
            "replay": None,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        result_path = OUT / f"{tag}.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"viewer_manifest_refresh":
                          _refresh_visualization_manifest()}), flush=True)
        return result_path, result
    if payload_hash(policy_input) != static_hash:
        raise ValueError("policy mutated its static-map input during construction")
    site = _policy_site(policy, static_map)

    fixture = sample_fixture(env, fixture_seed=fixture_seed, agents=2,
                             predators=predators)
    agents, tracked_predators, starts = _spawn_fixture(
        env, fixture, site=site, station_bait=station_bait,
    )
    bait, guide = agents
    tag = (f"real-map-{policy_class.__name__}-m{map_seed}-f{fixture_seed}"
           f"-p{predators}-{mode}-sitebait{int(station_bait)}"
           f"{'-reproduction' if reproduction_of else ''}-{uuid4().hex[:8]}")
    replay_path = OUT / "replays" / f"{tag}.json.gz"
    recorder = ReplayRecorder(
        env, title=tag, policy=policy_spec, seed=map_seed, every=1,
        scenario="native generated map; independently randomized full-map starts",
        notes=(__doc__ + " Replay interval is exactly one native 0.1-second tick. "
               + ("Bait start is arranged at the policy-selected site; guide and "
                  "predator starts remain independent full-map samples."
                  if station_bait else
                  "Both agents and all initial predators use independent full-map samples.")
               + (" Encounter-child mode: initial agent 1 is a random-start "
                  "fruit gatherer and parent. A child may be born only through "
                  "its native spawn action after an ordinary Predator DTO. The "
                  "policy may assign a lineage member as guide from native "
                  "observations; the receipt records every role change."
                  if mode in child_modes else "")
               + (" Setup-oracle approach mode: only before agent 1 receives its "
                  "first ordinary Predator DTO, the harness replaces that agent's "
                  "action with a direct move toward the initial tracked predator. "
                  "The handoff time is recorded; after it, no privileged dynamic "
                  "position affects policy actions."
                  if mode == "oracle_approach_child_guide" else "")
               + (f" This is a new deterministic reproduction of the prior "
                  f"state-only run {reproduction_of}; it is not reconstructed "
                  "footage of that run." if reproduction_of else "")),
        policy_sha256=policy_hash, native_render=native_render,
        native_width=native_width,
    )
    recorder.capture()
    for agent in env.agents:
        agent.energy = agent.max_energy
    states = _dto_states(env)

    first_seen = None
    first_follow = None
    bait_deployed_at = None
    bait_deployment_streak = 0
    bait_left_after_deployment = None
    seen_by_guide = {predator: False for predator in tracked_predators}
    tracked_encounter_times = {predator: None for predator in tracked_predators}
    followed_guide = {predator: False for predator in tracked_predators}
    physical_entered_at = {}
    physical_streak = {predator: 0 for predator in tracked_predators}
    guided_delivered_at = {}
    delivery_streak = {predator: 0 for predator in tracked_predators}
    physical_losses = {}
    target_losses = {}
    max_distance_after = {predator: 0.0 for predator in tracked_predators}
    trace = []
    physical_tail = []
    contained_tail = []
    spawn_requests = []
    child_births = []
    fruit_consumptions = []
    deaths = []
    target_switches = []
    guide_role_trace = []
    guide_follow_trace = []
    last_active_guide_id = None
    first_active_follow = None
    last_targets = {predator: None for predator in tracked_predators}
    # Agent observations are generated before predator movement. Preserve the
    # matching pre-movement snapshot for the next policy call; using current
    # post-movement coordinates can shift an encounter several seconds.
    observation_snapshot = {predator: (predator.x, predator.y)
                            for predator in tracked_predators}
    observation_guide_pose = (guide.x, guide.y, guide.direction)
    oracle_handoff_time = None
    oracle_ticks = 0
    oracle_route = []
    oracle_route_target = None
    oracle_route_resets = 0
    oracle_policy_reset = False
    ticks_requested = round(seconds / core.dt)

    for tick in range(ticks_requested):
        if STOP.exists():
            break
        for agent in env.agents:
            agent.energy = agent.max_energy
        # Energy is part of the native DTO. Refresh after the permitted refill
        # so the policy input describes the actual state at action time.
        states = _dto_states(env)
        policy_states = json_clone(states)
        guide_state = next((state for state in policy_states
                            if state["agent_id"] == guide.agent_id), None)
        guide_predator_observations = ([] if guide_state is None else [
            observation for observation in guide_state["observations"]
            if observation.get("type") == "Predator"
        ])
        if first_seen is None and guide_predator_observations:
            first_seen = round(env.time, 1)
        # Native predator DTOs intentionally have no IDs. The evaluator may
        # associate them to tracked fixture objects by their exact native
        # distance/angle signature. This mapping is never passed to policy.
        for predator in tracked_predators:
            predator_point = observation_snapshot[predator]
            guide_x, guide_y, guide_direction = observation_guide_pose
            distance = math.dist((guide_x, guide_y), predator_point)
            angle = math.atan2(predator_point[1] - guide_y,
                               predator_point[0] - guide_x) - guide_direction
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            if any(abs(item["distance"] - distance) < 1e-5
                   and abs(((item["angle"] - angle + math.pi) % (2 * math.pi))
                           - math.pi) < 1e-5
                   for item in guide_predator_observations):
                seen_by_guide[predator] = True
                if tracked_encounter_times[predator] is None:
                    tracked_encounter_times[predator] = round(env.time, 1)
        if (mode == "oracle_approach_child_guide"
                and oracle_handoff_time is None and guide_predator_observations):
            oracle_handoff_time = round(env.time, 1)
            reset = getattr(policy, "oracle_handoff", None)
            if reset is None:
                raise RuntimeError("oracle mode policy must implement oracle_handoff()")
            reset()
            oracle_policy_reset = True
        actions = policy.act(policy_states, env.time)
        active_guide_id = getattr(policy, "active_guide_id", guide.agent_id)
        if active_guide_id != last_active_guide_id:
            guide_role_trace.append({"time": round(env.time, 1),
                                     "active_guide_id": active_guide_id})
            # Delivery after a lineage role change needs fresh evidence that
            # the predator pursued this role, not the predecessor.
            followed_guide = {predator: False for predator in tracked_predators}
            first_active_follow = None
            last_active_guide_id = active_guide_id
        if (mode == "oracle_approach_child_guide"
                and oracle_handoff_time is None):
            if guide.agent_id in env.agents_dict:
                # This is the only setup control that reads a dynamic predator
                # coordinate. It is retired permanently at the first native DTO.
                target = tracked_predators[0]
                target_point = (target.x, target.y)
                if (not oracle_route or oracle_route_target is None
                        or math.dist(oracle_route_target, target_point) > 40):
                    oracle_route = _oracle_path(static_map, (guide.x, guide.y),
                                                target_point)
                    oracle_route_target = target_point
                    oracle_route_resets += 1
                while oracle_route and math.dist((guide.x, guide.y),
                                                  oracle_route[0]) < 7:
                    oracle_route.pop(0)
                waypoint = oracle_route[0] if oracle_route else target_point
                relative = (math.atan2(waypoint[1] - guide.y,
                                       waypoint[0] - guide.x)
                            - guide.direction + math.pi) % (2 * math.pi) - math.pi
                oracle_action = {"agent_id": guide.agent_id,
                                 "move_distance": guide.sprint_speed,
                                 "move_direction": relative,
                                 "turn_angle": 0.0, "spawn_agent": False}
                actions = [oracle_action if action.get("agent_id") == guide.agent_id
                           else action for action in actions]
                oracle_ticks += 1
        if payload_hash(policy_input) != static_hash:
            raise ValueError("policy mutated its static-map input")
        requests = _validate_actions(actions, env)
        pairs = [(request.agent_id, request) for request in requests]
        action_t = env.time
        agent_ids_before_actions = set(env.agents_dict)
        requesting_parents = [request.agent_id for request in requests
                              if request.spawn_agent]
        spawn_requests.extend({"time": round(action_t, 1), "parent_id": parent}
                              for parent in requesting_parents)
        for request in requests:
            env.agent_step(**request.model_dump())
        newborns = [agent for agent in env.agents
                    if agent.agent_id not in agent_ids_before_actions]
        for child in newborns:
            child_births.append({
                "time": round(action_t, 1),
                "child_id": child.agent_id,
                "x": round(child.x, 4),
                "y": round(child.y, 4),
                "requesting_parent_ids": list(requesting_parents),
                "native_parent_proximity_spawn": True,
            })
        # Attribute fruit only when a native agent is touching it immediately
        # before the engine's interaction pass and that fruit is then removed.
        touching = {}
        for fruit in env.fruits:
            eater = next((agent for agent in env.agents
                          if math.dist((agent.x, agent.y), (fruit.x, fruit.y))
                          < agent.size + fruit.radius), None)
            if eater is not None:
                touching[fruit] = eater
        agents_before_nonagent = list(env.agents)
        next_observation_snapshot = {predator: (predator.x, predator.y)
                                     for predator in tracked_predators}
        next_observation_guide_pose = (guide.x, guide.y, guide.direction)
        env.non_agent_step(core.dt)
        observation_snapshot = next_observation_snapshot
        observation_guide_pose = next_observation_guide_pose
        remaining_fruits = set(env.fruits)
        for fruit, eater in touching.items():
            if fruit not in remaining_fruits:
                fruit_consumptions.append({"time": round(env.time, 1),
                                           "agent_id": eater.agent_id,
                                           "x": round(fruit.x, 4),
                                           "y": round(fruit.y, 4)})
        deaths.extend({"time": round(env.time, 1), "agent_id": agent.agent_id,
                       "role": ("bait" if agent.agent_id == bait.agent_id else
                                "parent_guide" if agent.agent_id == guide.agent_id else
                                "native_child")}
                      for agent in agents_before_nonagent
                      if agent.agent_id not in env.agents_dict)

        bait_distance = math.dist((bait.x, bait.y), site["goal"])
        bait_at_goal = (bait.agent_id in env.agents_dict and bait_distance <= 6.0)
        bait_deployment_streak = bait_deployment_streak + 1 if bait_at_goal else 0
        if bait_deployment_streak >= 10 and bait_deployed_at is None:
            bait_deployed_at = round(env.time - .9, 1)
        if (bait_deployed_at is not None and not bait_at_goal
                and bait_left_after_deployment is None):
            bait_left_after_deployment = round(env.time, 1)

        mouth = site["mouth"]
        inward = site["inward"]
        physical_count = 0
        contained_count = 0
        follow_count = 0
        for predator in tracked_predators:
            dx, dy = predator.x - mouth[0], predator.y - mouth[1]
            along = dx * inward[0] + dy * inward[1]
            physical = math.hypot(dx, dy) <= 75 and along <= 10
            physical_streak[predator] = physical_streak[predator] + 1 if physical else 0
            if physical_streak[predator] >= 20 and predator not in physical_entered_at:
                physical_entered_at[predator] = round(env.time - 1.9, 1)
            physical_count += int(physical)

            visible_agents = [item for item in predator.observe(
                agents=list(env.agents), edges=list(env.edges))
                if item["type"] == "Agent"]
            chosen_observation = (min(visible_agents,
                                      key=lambda item: item["distance"])
                                  if visible_agents else None)
            chosen = (chosen_observation["id"]
                      if chosen_observation is not None else None)
            target_label = "resting" if predator.resting else chosen
            if target_label != last_targets[predator]:
                target_switches.append({"time": round(env.time, 1),
                                        "tracked_predator_slot": tracked_predators.index(predator),
                                        "target": target_label})
                last_targets[predator] = target_label
            # Match Predator.step's chase branch, not merely its nearest-agent
            # selection. Outside this gate the predator pivots around a target.
            chase_gate = (chosen_observation is not None and
                          (abs(chosen_observation["rel_dir"]) > math.pi / 2
                           or chosen_observation["distance"]
                           < predator.hearing_radius * 1.5))
            following = (not predator.resting and
                         chosen == active_guide_id and chase_gate)
            followed_guide[predator] = followed_guide[predator] or following
            follow_count += int(following)
            if first_follow is None and following:
                first_follow = round(env.time, 1)
            if first_active_follow is None and following:
                first_active_follow = round(env.time, 1)
                guide_follow_trace.append({
                    "time": first_active_follow,
                    "active_guide_id": active_guide_id,
                    "tracked_predator_slot": tracked_predators.index(predator),
                })
            contained = physical and (predator.resting or chosen == bait.agent_id)
            contained_count += int(contained)
            chain_ready = (bait_deployed_at is not None
                           and seen_by_guide[predator]
                           and followed_guide[predator]
                           and (mode not in child_modes or any(
                               guide.agent_id in birth["requesting_parent_ids"]
                               and tracked_encounter_times[predator] is not None
                               and birth["time"] >= tracked_encounter_times[predator]
                               for birth in child_births)))
            delivery_streak[predator] = (delivery_streak[predator] + 1
                                          if chain_ready and contained else 0)
            if (delivery_streak[predator] >= 20
                    and predator not in guided_delivered_at):
                guided_delivered_at[predator] = round(env.time - 1.9, 1)
            if predator in guided_delivered_at:
                max_distance_after[predator] = max(
                    max_distance_after[predator], math.hypot(dx, dy))
                if not physical and predator not in physical_losses:
                    physical_losses[predator] = round(env.time, 1)
                if not contained and predator not in target_losses:
                    target_losses[predator] = round(env.time, 1)

        physical_tail.append(physical_count)
        contained_tail.append(contained_count)

        recorder.capture(
            pairs, getattr(policy, "decisions", None), inputs=states,
            action_t=action_t,
        )
        if tick % 10 == 0:
            trace.append({"time": round(env.time, 1),
                          "agents_alive": len(env.agents),
                          "predators_total": len(env.predators),
                          "bait_distance_to_goal": round(bait_distance, 3),
                          "bait_deployed": bait_deployed_at is not None,
                          "policy_bait_ready": getattr(policy, "bait_ready", None),
                          "tracked_following_guide": follow_count,
                          "tracked_at_site": physical_count,
                          "tracked_contained_by_bait": contained_count})
        if tick % 100 == 99:
            print(json.dumps({"progress": tag, "time": round(env.time, 1),
                              "bait_deployed": bait_deployed_at,
                              "policy_bait_ready": getattr(policy, "bait_ready", None),
                              "encounter": first_seen,
                              "oracle_handoff": oracle_handoff_time,
                              "children": len(child_births),
                              "active_guide_id": getattr(policy, "active_guide_id",
                                                         guide.agent_id),
                              "guide_alive": guide.agent_id in env.agents_dict,
                              "active_guide_alive": getattr(
                                  policy, "active_guide_id", guide.agent_id)
                                  in env.agents_dict,
                              "guided_delivered": len(guided_delivered_at)}),
                  flush=True)
        if not env.agents:
            break

    stopped = STOP.exists()
    reached_horizon = env.time >= seconds - core.dt / 2
    reason = "stop sentinel" if stopped else "horizon" if reached_horizon else "all agents died"
    replay_summary = recorder.save(replay_path, reason=reason)
    final_distances = [math.dist((predator.x, predator.y), site["mouth"])
                       for predator in tracked_predators]
    tail_ticks = min(300, len(contained_tail))
    encountered = [predator for predator in tracked_predators
                   if tracked_encounter_times[predator] is not None]
    qualifying_births = [birth for birth in child_births
                         if guide.agent_id in birth["requesting_parent_ids"]
                         and any(birth["time"] >= tracked_encounter_times[predator]
                                 for predator in encountered)]
    parent_fruit_before_encounter = sum(
        event["agent_id"] == guide.agent_id and
        any(event["time"] < tracked_encounter_times[predator]
            for predator in encountered)
        for event in fruit_consumptions)
    if mode in child_modes:
        conditional_status = (
            "no_tracked_encounter_excluded" if not encountered else
            "encounter_without_native_child" if not qualifying_births else
            "conditional_delivery_success" if len(guided_delivered_at) == predators else
            "conditional_delivery_failure"
        )
    else:
        conditional_status = "not_applicable"
    final_active_guide_id = getattr(policy, "active_guide_id", guide.agent_id)
    active_role_is_child = final_active_guide_id != guide.agent_id
    result = json_value({
        "schema": "real-map-intake-result-v2",
        "policy": policy_spec,
        "policy_hash": policy_hash,
        "harness_hashes": _harness_hashes(),
        "mode": mode,
        "map_seed": map_seed,
        "fixture_seed": fixture_seed,
        "static_map_hash": static_hash,
        "site": site,
        "station_bait": station_bait,
        "requested_seconds": seconds,
        "seconds": env.time,
        "reason": reason,
        "native_render": native_render,
        "reproduction_of": reproduction_of,
        "record_every_ticks": 1,
        "replay_frames": replay_summary["frames"],
        "initial_predators": predators,
        "final_predators_total": len(env.predators),
        "tracked_predators_present_final": sum(
            predator in env.predators for predator in tracked_predators),
        "ambient_predators_spawned": sum(
            predator not in tracked_predators for predator in env.predators),
        "ambient_predators_present_final": sum(
            predator not in tracked_predators for predator in env.predators),
        "agent_energy_refilled": True,
        "predator_state_native": True,
        "starts": starts,
        "bait_id": bait.agent_id,
        "guide_id": guide.agent_id,
        "bait_alive": bait.agent_id in env.agents_dict,
        "guide_alive": guide.agent_id in env.agents_dict,
        "deaths": deaths,
        "spawn_requests": spawn_requests,
        "child_births": child_births,
        "fruit_consumptions": fruit_consumptions,
        "parent_fruit_consumed_before_tracked_encounter": parent_fruit_before_encounter,
        "tracked_encounter_times": [tracked_encounter_times[predator]
                                    for predator in tracked_predators],
        "conditional_delivery_denominator": bool(encountered),
        "conditional_delivery_status": conditional_status,
        "setup_oracle": mode == "oracle_approach_child_guide",
        "setup_oracle_target": ("initial_tracked_predator_slot_0"
                                if mode == "oracle_approach_child_guide" else None),
        "setup_oracle_ticks": oracle_ticks,
        "setup_oracle_routing": ("static_visibility_graph_radius5.01"
                                 if mode == "oracle_approach_child_guide" else None),
        "setup_oracle_route_resets": oracle_route_resets,
        "setup_oracle_policy_reset_before_handoff_act": oracle_policy_reset,
        "setup_oracle_handoff_time": oracle_handoff_time,
        "dynamic_action_privilege_ended_at": oracle_handoff_time,
        "target_switches": target_switches,
        "guide_role_trace": guide_role_trace,
        "guide_follow_trace": guide_follow_trace,
        "guide_role": "child" if active_role_is_child else "parent",
        "parent_role": "gather_evade" if active_role_is_child else "guide",
        "child_role": "guide" if active_role_is_child else "gather_evade",
        "active_guide_id_final": final_active_guide_id,
        "active_guide_alive_final": final_active_guide_id in env.agents_dict,
        "guide_role_ids": sorted(set(getattr(
            policy, "guide_role_ids", {guide.agent_id}))),
        "bait_deployed_at": bait_deployed_at,
        "policy_bait_ready_final": getattr(policy, "bait_ready", None),
        "bait_left_after_deployment": bait_left_after_deployment,
        "final_bait_distance_to_goal": math.dist((bait.x, bait.y), site["goal"]),
        "first_guide_predator_observation": first_seen,
        "first_predator_following_guide": first_follow,
        "first_predator_following_active_guide": first_active_follow,
        "tracked_seen_by_guide": sum(seen_by_guide.values()),
        "tracked_followed_active_guide": sum(followed_guide.values()),
        "physical_entered": len(physical_entered_at),
        "guided_delivered": len(guided_delivered_at),
        "physical_losses_after_delivery": len(physical_losses),
        "target_losses_after_delivery": len(target_losses),
        "physical_entry_times": [physical_entered_at.get(predator)
                                 for predator in tracked_predators],
        "guided_delivery_times": [guided_delivered_at.get(predator)
                           for predator in tracked_predators],
        "loss_times": [physical_losses.get(predator)
                       for predator in tracked_predators],
        "target_loss_times": [target_losses.get(predator)
                              for predator in tracked_predators],
        "tail_seconds": tail_ticks / 10,
        "tail_physical_min": min(physical_tail[-tail_ticks:]) if tail_ticks else 0,
        "tail_contained_min": min(contained_tail[-tail_ticks:]) if tail_ticks else 0,
        "final_distance_from_mouth": final_distances,
        "max_distance_after_delivery": [max_distance_after[predator]
                                        for predator in tracked_predators],
        "policy_events": json_clone(getattr(policy, "events", [])),
        "success": (reached_horizon and bait_deployed_at is not None
                    and len(guided_delivered_at) == predators
                    and (mode not in child_modes or bool(qualifying_births))
                    and not physical_losses and not target_losses
                    and tail_ticks == 300
                    and min(physical_tail[-tail_ticks:]) == predators
                    and min(contained_tail[-tail_ticks:]) == predators),
        "trace": trace,
        "replay": str(replay_path.relative_to(ROOT)),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / f"{tag}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"viewer_manifest_refresh":
                      _refresh_visualization_manifest()}), flush=True)
    return result_path, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", dest="policy_spec",
                        default="smoke_policy:SmokePolicy")
    parser.add_argument("--map-seed", type=int, default=5101)
    parser.add_argument("--fixture-seed", type=int, default=9101)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--predators", type=int, default=1)
    parser.add_argument("--station-bait", action="store_true")
    parser.add_argument("--state-only", dest="native_render", action="store_false")
    parser.add_argument("--native-width", type=int, default=800)
    parser.add_argument("--reproduction-of")
    parser.add_argument("--mode", choices=("direct_guide", "encounter_child_guide",
                                            "oracle_approach_child_guide"),
                        default="direct_guide")
    args = parser.parse_args()
    path, result = run(**vars(args))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"trace", "starts"}}, indent=2))
    print(path)


if __name__ == "__main__":
    main()
