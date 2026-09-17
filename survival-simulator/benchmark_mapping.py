"""Matched-seed mapping benchmark; simulator truth is used only for diagnostics.

Run with --source-root runs/optimization-baseline to load a frozen src/config
pair. Project imports happen only after that root is selected. Coverage credits
the free area of each visited coarse cell, not every point the agent can see.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parent
TRAITS = ("speed", "sprint_speed", "max_energy", "hearing_radius", "vision_radius", "cone_angle")


class Milestone:
    def __init__(self, duration=5.0):
        self.duration = duration
        self.first = self.sustained = self.since = None

    def observe(self, condition, now):
        if not condition:
            self.since = None
            return
        if self.first is None:
            self.first = now
        if self.since is None:
            self.since = now
        if self.sustained is None and now - self.since + 1e-9 >= self.duration:
            self.sustained = self.since


def free_cell_areas(env, cell_size):
    """One-unit raster integration excludes overlapping rectangular obstacles."""
    import numpy as np
    free = np.ones((env.height, env.width), dtype=bool)
    for obstacle in env.obstacles:
        left = max(0, math.ceil(obstacle.x - 0.5))
        right = min(env.width, math.ceil(obstacle.x + obstacle.width - 0.5))
        top = max(0, math.ceil(obstacle.y - 0.5))
        bottom = min(env.height, math.ceil(obstacle.y + obstacle.height - 0.5))
        free[top:bottom, left:right] = False
    result = {}
    for y in range(0, env.height, cell_size):
        for x in range(0, env.width, cell_size):
            area = int(free[y:y + cell_size, x:x + cell_size].sum())
            if area:
                result[(x // cell_size, y // cell_size)] = area
    return result


def coverage_percent(cells, free_areas):
    denominator = sum(free_areas.values())
    return 100 * sum(free_areas.get(cell, 0) for cell in set(cells)) / denominator if denominator else 0.0


def percentile(values, fraction=.95):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


def inferred_biome_metrics(snapshot, env):
    """Diagnostic accuracy at inferred grid centers; never a controller input.

    Unknown cells are excluded from accuracy and included in the support
    denominator. Support describes the model domain, which can have partial
    bounds; it does not claim full-world coverage in that case.
    """
    total = supported = correct = sites = 0
    for group in snapshot.get("groups", []):
        estimate = group.get("biome_estimate")
        if not group.get("anchored") or not estimate:
            continue
        sites += len(estimate["sites"])
        grid = estimate["labels"]
        ny, nx = len(grid), len(grid[0])
        x0, y0, x1, y1 = estimate["bounds"]
        for y, row in enumerate(grid):
            for x, label in enumerate(row):
                px, py = x0 + (x + .5) * (x1 - x0) / nx, y0 + (y + .5) * (y1 - y0) / ny
                if not (0 <= px < env.width and 0 <= py < env.height):
                    continue
                total += 1
                if label >= 0:
                    supported += 1
                    correct += estimate["palette"][label] == env.biome_map[int(px), int(py)].type
    return dict(inferred_biome_grid_accuracy_percent=100 * correct / supported if supported else None,
                inferred_biome_grid_support_percent=100 * supported / total if total else None,
                inferred_biome_sites=sites)


def anchored_pose_errors(poses, groups, actual):
    """Compare already-anchored estimates with truth only in this diagnostic."""
    result = []
    for pose in poses.values():
        agent = actual.get(pose.agent_id)
        if agent is None or not groups[pose.group_id].anchored:
            continue
        error = math.hypot(float(pose.position[0]) - agent.x, float(pose.position[1]) - agent.y)
        result.append((pose.agent_id, error, float(pose.uncertainty)))
    return result


def sprint_cost_estimate(distance, speed, sprint_speed, walking_cost, sprinting_cost):
    extra = max(0.0, min(distance, sprint_speed) - speed)
    # Premium is the excess over walking the SAME additional distance.
    return extra * sprinting_cost, extra * max(0, sprinting_cost - walking_cost)


def frame_conditions(poses, groups, living_ids):
    """A missing newborn or a stale dead pose must not prove a shared frame."""
    living_ids = set(living_ids)
    living_groups = {poses[agent_id].group_id for agent_id in living_ids if agent_id in poses}
    represented = bool(living_ids) and living_ids.issubset(poses)
    anchored = {key for key in living_groups if key in groups and groups[key].anchored}
    all_anchored = represented and living_groups == anchored
    connected = represented and len(living_groups) == 1
    return dict(any_anchored=bool(anchored), all_living_anchored=all_anchored,
                all_living_connected=connected, shared_absolute=all_anchored and connected)


def population_metrics(lineage, shared_time, horizon):
    """Use event times so births and deaths between checkpoints stay exact."""
    def alive_at(when):
        return sum(row["birth_time"] <= when + 1e-8 and
                   (row["death_time"] is None or row["death_time"] > when + 1e-8)
                   for row in lineage)

    def births_at(when):
        return sum(row["parent_id"] is not None and row["birth_time"] <= when + 1e-8 for row in lineage)

    result = {}
    for when in (60, 120):
        result[f"alive_at_{when}_seconds"] = alive_at(when) if horizon + 1e-8 >= when else None
        result[f"births_by_{when}_seconds"] = births_at(when) if horizon + 1e-8 >= when else None
    result["alive_at_shared_absolute"] = alive_at(shared_time) if shared_time is not None else None
    result["births_before_shared_absolute"] = births_at(shared_time) if shared_time is not None else None
    result["deaths_before_shared_absolute"] = (
        sum(row["death_time"] is not None and row["death_time"] <= shared_time + 1e-8 for row in lineage)
        if shared_time is not None else None)
    result["births_after_shared_absolute"] = (
        births_at(horizon) - result["births_before_shared_absolute"] if shared_time is not None else None)
    result["deaths_after_shared_absolute"] = (
        sum(row["death_time"] is not None and row["death_time"] > shared_time + 1e-8 for row in lineage)
        if shared_time is not None else None)
    result["founders_alive_final"] = sum(row["parent_id"] is None and row["death_time"] is None for row in lineage)
    return result


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def source_fingerprint(source_root):
    digest = hashlib.sha256()
    for pattern in ("src/**/*.py", "config/*.json"):
        for path in sorted(source_root.glob(pattern)):
            digest.update(path.relative_to(source_root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def benchmark_seed(seed, seconds, cell_size, source_root, expert_config=None, planner_config=None):
    # Both imports AND explicit configuration paths refer to the chosen source.
    from src.core import SimulationCore
    from src.utils.controllers.expert_policy import ExpertPolicy, load_config
    from src.utils.controllers.global_planner import load_planner_config
    policy = ExpertPolicy(load_config(expert_config or source_root / "config/expert_policy.json"),
                          load_planner_config(planner_config or source_root / "config/global_planner.json"))
    wall_start = time.perf_counter()
    sim = SimulationCore(seed=seed)
    areas = free_cell_areas(sim.env, cell_size)
    initial = {trait: statistics.mean(getattr(a, trait) for a in sim.env.agents) for trait in TRAITS}
    population = {}
    birth_count = death_count = predation_count = 0
    peak_population = len(sim.env.agents)
    event_time = 0.0

    def event(kind, **data):
        nonlocal birth_count, death_count, predation_count, peak_population
        if kind == "birth":
            agent, parent = data["agent"], data.get("parent")
            traits = {trait: float(getattr(agent, trait)) for trait in TRAITS}
            parent_id = parent.agent_id if parent is not None else None
            row = dict(agent_id=agent.agent_id, parent_id=parent_id,
                       founder_id=population[parent_id]["founder_id"] if parent_id is not None else agent.agent_id,
                       generation=population[parent_id]["generation"] + 1 if parent_id is not None else 0,
                       birth_time=event_time, death_time=None, death_cause=None, children=0,
                       initial_normalized_trait_mean=statistics.mean(traits[t] / initial[t] for t in TRAITS), **traits)
            population[agent.agent_id] = row
            if parent_id is not None:
                population[parent_id]["children"] += 1
                birth_count += 1
            peak_population = max(peak_population, sum(row["death_time"] is None for row in population.values()))
        elif kind == "death":
            row = population[data["agent"].agent_id]
            row["death_time"], row["death_cause"] = event_time, data["cause"]
            death_count += 1
            predation_count += data["cause"] == "predation"

    for agent in sim.env.agents:
        event("birth", agent=agent, parent=None)
    sim.env.event_sink = event
    visited = {(int(a.x // cell_size), int(a.y // cell_size)) for a in sim.env.agents}
    milestones = {key: Milestone() for key in (
        "any_anchored", "all_living_anchored", "all_living_connected", "shared_absolute")}
    coverage_milestones = {percent: None for percent in (10, 25, 50, 75, 90, 95)}
    latencies = []
    actions = []
    nonsurvival_sprints = exploration_actions = 0
    scout_distant_food_skip_decisions = 0
    sprint_actions = 0
    sprint_segment_energy = sprint_premium_energy = 0.0
    nonsurvival_sprint_premium_energy = 0.0
    all_position_errors, final_errors = [], []
    worst_pose_examples = []
    confidence1_wrong_examples = []
    confidence1_example_keys = set()
    confidence1_samples = confidence1_wrong_samples = 0
    confidence1_wrong_agents = set()
    max_exploration_walk_ratio = 0.0
    population_phase_started = None
    checkpoints = []
    next_checkpoint = 10.0
    final_snapshot = {}
    while sim.env.time + 1e-9 < seconds and sim.env.agents:
        event_time = sim.env.time + sim.dt
        state = sim.step(actions)
        now = state["sim_time"]
        visited.update((int(a.x // cell_size), int(a.y // cell_size)) for a in sim.env.agents)
        start = time.perf_counter_ns()
        decisions = policy.actions_for_step(state["observations"], sim_time=now)
        latencies.append((time.perf_counter_ns() - start) / 1e6)
        actions = [(action.agent_id, action) for action in decisions]
        states = {item["agent_id"]: item for item in state["observations"]}
        for action in decisions:
            own = states[action.agent_id]
            danger = action.agent_id in getattr(policy, "_escape_memories", {}) or any(
                obj["type"] == "Predator" and obj["distance"] <= policy.config.perception.predator_danger_radius
                for obj in own["observations"])
            sprint_actions += action.move_distance > own["speed"] + 1e-8
            segment, premium = sprint_cost_estimate(
                action.move_distance, own["speed"], own["sprint_speed"],
                policy.config.mechanics.walking_energy_per_unit,
                policy.config.mechanics.sprinting_energy_per_unit)
            sprint_segment_energy += segment
            sprint_premium_energy += premium
            if not danger:
                nonsurvival_sprints += action.move_distance > own["speed"] + 1e-8
                nonsurvival_sprint_premium_energy += premium
                hint = getattr(policy.planner, "exploration_hints", {}).get(action.agent_id)
                food_limit = getattr(hint, "food_distance_limit", None)
                visible_fruit = [obj for obj in own["observations"] if obj["type"] == "Fruit"]
                food_target = any(food_limit is None or obj["distance"] <= food_limit for obj in visible_fruit)
                if not food_target:
                    scout_distant_food_skip_decisions += bool(visible_fruit)
                    exploration_actions += 1
                    walk = min(own["speed"], own["sprint_speed"])
                    max_exploration_walk_ratio = max(max_exploration_walk_ratio,
                                                     action.move_distance / walk if walk else 0)
        estimator = policy.planner.estimator
        if population_phase_started is None and getattr(policy.planner, "population_phase", False):
            population_phase_started = now
        final_errors = anchored_pose_errors(estimator.poses, estimator.groups,
                                             {agent.agent_id: agent for agent in sim.env.agents})
        all_position_errors.extend(error for _, error, _ in final_errors)
        for agent_id, error, uncertainty in final_errors:
            worst = len(worst_pose_examples) < 10 or error > worst_pose_examples[-1]["error"]
            example_key = (agent_id, int(now // 5))
            confident_wrong = uncertainty <= 1 + 1e-8 and error > 5
            sample_bad = confident_wrong and len(confidence1_wrong_examples) < 20 and example_key not in confidence1_example_keys
            if worst or sample_bad:
                actual = sim.env.agents_dict[agent_id]
                pose = estimator.poses[agent_id]
                example = dict(sim_time=now, agent_id=agent_id, group_id=pose.group_id,
                               error=error, uncertainty=uncertainty,
                               actual_position=[float(actual.x), float(actual.y)],
                               estimated_position=[float(value) for value in pose.position],
                               actual_heading=float(actual.direction), estimated_heading=float(pose.heading))
                if worst:
                    worst_pose_examples.append(example)
                    worst_pose_examples.sort(key=lambda item: item["error"], reverse=True)
                    del worst_pose_examples[10:]
                if sample_bad:
                    confidence1_wrong_examples.append(example)
                    confidence1_example_keys.add(example_key)
            if uncertainty <= 1.0 + 1e-8:
                confidence1_samples += 1
                if error > 5.0:
                    confidence1_wrong_samples += 1
                    confidence1_wrong_agents.add(agent_id)
        living_groups = {pose.group_id for pose in estimator.poses.values() if pose.agent_id in states}
        anchored = {key for key in living_groups if estimator.groups[key].anchored}
        for key, condition in frame_conditions(estimator.poses, estimator.groups, sim.env.agents_dict).items():
            milestones[key].observe(condition, now)
        percent = coverage_percent(visited, areas)
        for threshold in coverage_milestones:
            if percent >= threshold and coverage_milestones[threshold] is None:
                coverage_milestones[threshold] = now
        if now + 1e-9 >= next_checkpoint or not sim.env.agents:
            checkpoints.append(dict(sim_time=now, actual_free_area_visited_percent=percent,
                                    alive=len(sim.env.agents), births=birth_count,
                                    map_groups=len(living_groups), anchored_groups=len(anchored), score=sim.env.score,
                                    anchored_position_error_p95=percentile([error for _, error, _ in final_errors]),
                                    confidence1_wrong_samples=confidence1_wrong_samples,
                                    sprint_premium_energy_estimate=sprint_premium_energy))
            print(f"  seed {seed}: {now:.0f}s, coverage {percent:.1f}%, alive {len(sim.env.agents)}, births {birth_count}", flush=True)
            next_checkpoint += 10
        if sim.env.agents:
            final_snapshot = None  # Snapshot construction is excluded from timed policy calls.
    if final_snapshot is None:
        final_snapshot = policy.planner.snapshot()
    # If everyone died, the policy resets; retain no invented estimator coverage.
    groups = final_snapshot.get("groups", []) if final_snapshot else []
    anchored_cells = {(int(sample["position"][0] // cell_size), int(sample["position"][1] // cell_size))
                      for group in groups if group.get("anchored") for sample in group.get("biomes", [])}
    biome_matches, biome_samples = 0, 0
    for group in groups:
        if not group.get("anchored"):
            continue
        for sample in group.get("biomes", []):
            x, y = sample["position"]
            biome_samples += 1
            if 0 <= x < sim.env.width and 0 <= y < sim.env.height:
                biome_matches += sim.env.biome_map[int(x), int(y)].type == sample["biome"]
    sorted_latency = sorted(latencies)
    row = dict(seed=seed, simulated_seconds=sim.env.time, actual_free_area_visited_percent=coverage_percent(visited, areas),
               actual_visited_cells=len(visited), walkable_area_units=sum(areas.values()),
               estimated_biome_cells_total=sum(len(group.get("biomes", [])) for group in groups),
               estimated_anchored_visited_cells=len(anchored_cells),
               estimated_anchored_free_area_percent=coverage_percent(anchored_cells, areas),
               map_groups=len(groups), anchored_groups=sum(bool(group.get("anchored")) for group in groups),
               births=birth_count, peak_population=peak_population, alive=len(sim.env.agents),
               deaths=death_count, predation_deaths=predation_count, score=sim.env.score,
               population_phase_started_seconds=population_phase_started,
               nonsurvival_sprint_actions=nonsurvival_sprints, exploration_actions=exploration_actions,
               scout_distant_food_skip_decisions=scout_distant_food_skip_decisions,
               sprint_actions=sprint_actions, sprint_segment_energy_estimate=sprint_segment_energy,
               sprint_premium_energy_estimate=sprint_premium_energy,
               nonsurvival_sprint_premium_energy_estimate=nonsurvival_sprint_premium_energy,
               max_exploration_walk_ratio=max_exploration_walk_ratio,
               anchored_pose_samples=len(all_position_errors),
               anchored_position_error_mean=statistics.mean(all_position_errors) if all_position_errors else None,
               anchored_position_error_p95=percentile(all_position_errors),
               anchored_position_error_max=max(all_position_errors, default=None),
               final_anchored_position_error_mean=statistics.mean(error for _, error, _ in final_errors) if final_errors else None,
               final_anchored_position_error_p95=percentile([error for _, error, _ in final_errors]),
               final_anchored_position_error_max=max((error for _, error, _ in final_errors), default=None),
               confidence1_samples=confidence1_samples, confidence1_wrong_samples=confidence1_wrong_samples,
               confidence1_wrong_agents=len(confidence1_wrong_agents),
               final_confidence1_wrong_agents=sum(error > 5 and uncertainty <= 1 + 1e-8 for _, error, uncertainty in final_errors),
               anchored_biome_truth_samples=biome_samples,
               anchored_biome_truth_accuracy_percent=100 * biome_matches / biome_samples if biome_samples else None,
               controller_calls=len(latencies), controller_total_ms=sum(latencies),
               controller_mean_ms=statistics.mean(latencies) if latencies else 0,
               controller_p95_ms=sorted_latency[max(0, math.ceil(len(sorted_latency) * .95) - 1)] if latencies else 0,
               wall_seconds=time.perf_counter() - wall_start)
    for name, milestone in milestones.items():
        row[f"first_{name}_seconds"] = milestone.first
        row[f"sustained5_{name}_seconds"] = milestone.sustained
    for percent, when in coverage_milestones.items():
        row[f"first_{percent}pct_coverage_seconds"] = when
    row.update(population_metrics(list(population.values()), milestones["shared_absolute"].sustained, seconds))
    row.update(inferred_biome_metrics(final_snapshot or {}, sim.env))
    return dict(summary=row, checkpoints=checkpoints, initial_trait_baseline=initial,
                lineage=list(population.values()), worst_pose_examples=worst_pose_examples,
                confidence1_wrong_examples=confidence1_wrong_examples)


def comparison_report(results, baseline=None):
    lines = ["Matched-seed mapping benchmark", "",
             "Priority: establish one absolute frame for all living agents, then grow the surviving population.",
             "Coverage: visited coarse cells weighted by free area; obstacle area excluded.",
             "Sustained milestones report the start of a continuous five-second interval.",
             "Trait mean is an equal-weight diagnostic ratio to initial traits, not policy fitness.", ""]
    def summary_with_population(result):
        row = dict(result["summary"])
        if "lineage" in result:
            horizon = row["simulated_seconds"] if row["alive"] else math.inf
            population = population_metrics(result["lineage"], row.get("sustained5_shared_absolute_seconds"), horizon)
            row = {**population, **row}
        return row

    def display(value):
        return "unreached" if value is None else f"{value:.1f}"

    rows = [summary_with_population(result) for result in results]
    old_rows = {item["summary"]["seed"]: summary_with_population(item)
                for item in baseline.get("runs", [])} if baseline else {}
    lines += ["| Seed | Shared absolute (s) | All anchored (s) | Connected (s) | Alive 60s / final | Births | Deaths | Coverage |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        shared = display(row.get("sustained5_shared_absolute_seconds")) if "sustained5_shared_absolute_seconds" in row else "not recorded"
        lines.append(f'| {row["seed"]} | {shared} | {display(row.get("sustained5_all_living_anchored_seconds"))} | '
                     f'{display(row.get("sustained5_all_living_connected_seconds"))} | '
                     f'{display(row.get("alive_at_60_seconds"))} / {row["alive"]} | {row["births"]} | '
                     f'{row["deaths"]} | {row["actual_free_area_visited_percent"]:.1f}% |')
    inferred = [row for row in rows if row.get("inferred_biome_grid_accuracy_percent") is not None]
    if inferred:
        lines += ["", "Inferred biome labels versus simulator terrain at model grid centers:", "",
                  "| Seed | Supported grid | Accuracy on supported grid | Inferred land sites |",
                  "|---|---:|---:|---:|"]
        for row in inferred:
            lines.append(f'| {row["seed"]} | {row["inferred_biome_grid_support_percent"]:.1f}% | '
                         f'{row["inferred_biome_grid_accuracy_percent"]:.1f}% | {row["inferred_biome_sites"]} |')
        lines += ["", "Support covers the inferred model domain; unknown cells are excluded from accuracy. "
                  "These checks include river predictions and do not measure exact border distance."]
    if old_rows:
        pairs = [(old_rows[row["seed"]], row) for row in rows if row["seed"] in old_rows]
        lines += ["", "Paired means for the same seeds:", "",
                  "| Measure | Baseline | Current | Change |", "|---|---:|---:|---:|"]
        measures = {
            "sustained5_shared_absolute_seconds": "One shared absolute frame, sustained start (s)",
            "sustained5_all_living_anchored_seconds": "All living anchored, sustained start (s)",
            "sustained5_all_living_connected_seconds": "All living connected, sustained start (s)",
            "alive_at_shared_absolute": "Living agents when shared frame forms",
            "deaths_before_shared_absolute": "Deaths before shared frame forms",
            "alive_at_60_seconds": "Living agents at 60 seconds",
            "alive_at_120_seconds": "Living agents at 120 seconds",
            "births_by_60_seconds": "Births by 60 seconds",
            "births_after_shared_absolute": "Births after shared frame forms",
            "deaths_after_shared_absolute": "Deaths after shared frame forms",
            "births": "Total births", "alive": "Final living agents", "deaths": "Total deaths",
            "founders_alive_final": "Surviving founders", "score": "Final score",
            "actual_free_area_visited_percent": "Final visited free area (%)",
            "first_25pct_coverage_seconds": "Time to 25% coverage (s)",
            "first_50pct_coverage_seconds": "Time to 50% coverage (s)",
            "first_75pct_coverage_seconds": "Time to 75% coverage (s)",
            "first_90pct_coverage_seconds": "Time to 90% coverage (s)",
            "first_95pct_coverage_seconds": "Time to 95% coverage (s)",
            "controller_mean_ms": "Controller mean call (ms)",
            "anchored_position_error_p95": "Anchored pose error p95 (units)",
            "anchored_position_error_max": "Anchored pose error maximum (units)",
            "final_anchored_position_error_mean": "Final anchored pose error mean (units)",
            "confidence1_wrong_samples": "Error >5 with uncertainty <=1 (agent ticks)",
            "anchored_biome_truth_accuracy_percent": "Stored biome label accuracy (%)",
            "sprint_premium_energy_estimate": "Sprint premium energy estimate",
        }
        for key, label in measures.items():
            observed = [(old[key], new[key]) for old, new in pairs if old.get(key) is not None and new.get(key) is not None]
            if observed:
                old_mean = statistics.mean(old for old, _ in observed)
                new_mean = statistics.mean(new for _, new in observed)
                suffix = f" ({len(observed)} completed pairs)" if len(observed) < len(pairs) else ""
                lines.append(f"| {label}{suffix} | {old_mean:.2f} | {new_mean:.2f} | {new_mean - old_mean:+.2f} |")
        completed_measures = {"sustained5_shared_absolute_seconds": "a shared absolute frame",
                              "sustained5_all_living_anchored_seconds": "all living anchored",
                              "sustained5_all_living_connected_seconds": "all living connected",
                              "first_90pct_coverage_seconds": "90% coverage",
                              "first_95pct_coverage_seconds": "95% coverage"}
        for key, label in completed_measures.items():
            recorded = [(old, new) for old, new in pairs if key in old and key in new]
            if recorded:
                before = sum(old[key] is not None for old, _ in recorded)
                after = sum(new[key] is not None for _, new in recorded)
                suffix = f" ({len(recorded)} recorded pairs)" if len(recorded) < len(pairs) else ""
                lines.append(f"| Runs reaching {label}{suffix} | {before}/{len(recorded)} | {after}/{len(recorded)} | {after - before:+d} |")
        old_runs = {run["summary"]["seed"]: run for run in baseline["runs"]}
        early = []
        for run in results:
            old = old_runs.get(run["summary"]["seed"])
            if old and old["summary"]["simulated_seconds"] >= 40 and run["summary"]["simulated_seconds"] >= 40:
                before = min(old["checkpoints"], key=lambda item: abs(item["sim_time"] - 40))["births"]
                after = min(run["checkpoints"], key=lambda item: abs(item["sim_time"] - 40))["births"]
                early.append((before, after))
        if early:
            old_mean, new_mean = (statistics.mean(pair[index] for pair in early) for index in (0, 1))
            lines.append(f"| Births by 40 seconds | {old_mean:.2f} | {new_mean:.2f} | {new_mean - old_mean:+.2f} |")
    lines += ["", "Unreached milestones remain null in JSON/CSV. Equal seeds share the initial world; policies consume random events differently after their actions diverge."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--expert-config", type=Path, help="Override the selected source's expert policy JSON.")
    parser.add_argument("--planner-config", type=Path, help="Override the selected source's global planner JSON.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42])
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--cell-size", type=int, default=80)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/mapping-benchmark")
    parser.add_argument("--compare", type=Path, help="Baseline results.json for paired comparison.")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.cell_size <= 0:
        parser.error("seconds and cell-size must be positive and finite")
    source_root = args.source_root.resolve()
    if not (source_root / "src/core.py").is_file():
        parser.error("source-root must contain src/core.py and config/")
    sys.path.insert(0, str(source_root))
    import src.core
    if not Path(src.core.__file__).resolve().is_relative_to(source_root):
        raise RuntimeError("A different src package was already imported; run each source in a fresh process")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    expert_config = (args.expert_config or source_root / "config/expert_policy.json").resolve()
    planner_config = (args.planner_config or source_root / "config/global_planner.json").resolve()
    from src.utils.controllers.expert_policy import load_config
    from src.utils.controllers.global_planner import load_planner_config
    configuration = {"expert": load_config(expert_config).model_dump(),
                     "planner": load_planner_config(planner_config).model_dump()}
    payload = dict(source_root=str(source_root), source_sha256=source_fingerprint(source_root),
                   seeds=args.seeds, seconds=args.seconds, cell_size=args.cell_size,
                   expert_config_path=str(expert_config), planner_config_path=str(planner_config),
                   configuration=configuration,
                   configuration_sha256=hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest(),
                   coverage_definition="Visited coarse-cell free area / all free area; one-unit obstacle raster.",
                   accuracy_definition="Anchored agent pose vs simulator truth after each observation; confident wrong means error >5 units with uncertainty <=1.",
                   sprint_energy_definition="Requested extra distance above walking speed; premium subtracts the cost of walking the same extra distance. Includes the final unapplied decision.",
                   exploration_action_definition="Non-danger decisions with no fruit target after applying the active scout hint food_distance_limit, if present.",
                   shared_absolute_definition="All current living IDs have an estimated pose in the same anchored group. Sustained time is the start of a continuous five-second interval; population event metrics use that time.",
                   runs=[])
    baseline = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None
    if baseline and (baseline.get("seconds") != args.seconds or baseline.get("cell_size") != args.cell_size):
        parser.error("comparison requires equal duration and coverage cell size")
    for seed in args.seeds:
        print(f"Benchmark source={source_root}, seed={seed}, duration={args.seconds:g}s", flush=True)
        payload["runs"].append(benchmark_seed(seed, args.seconds, args.cell_size, source_root, expert_config, planner_config))
        (output / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        write_csv(output / "summary.csv", [result["summary"] for result in payload["runs"]])
        write_csv(output / "lineage.csv", [dict(seed=result["summary"]["seed"], **agent)
                                           for result in payload["runs"] for agent in result["lineage"]])
        (output / "report.md").write_text(comparison_report(payload["runs"], baseline), encoding="utf-8")
    print(output / "report.md", flush=True)


if __name__ == "__main__":
    main()
