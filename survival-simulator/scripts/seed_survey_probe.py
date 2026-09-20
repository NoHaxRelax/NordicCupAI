"""Survey for 30 simulated seconds using the shared world model, then search.

Only public frames and chosen actions enter the policy, map and seed filter.
Hidden positions are used for an audit, never to select or repair observations.
Actual biome samples are used; inferred biome predictions are not evidence.
"""

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from seed_biome_prefix_probe import candidate_prefix
from seed_joint_constraints_probe import angle_close, canonical_frame, heading_hypotheses
from seed_recovery_probe import ROOT, atomic_json, distinct, public_features, source_manifest
from seed_survey_evidence import extract_evidence, informative_samples, candidate_geometry_matches, initial_trees_match


def save(path, data):
    for attempt in range(20):
        try:
            atomic_json(path, data)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.2)


def survey_policy():
    from src.utils.controllers.expert_policy import ExpertPolicy, load_config
    from src.utils.controllers.global_planner import load_planner_config

    config = load_config(ROOT / "config/expert_policy.json")
    config = config.model_copy(update={
        "policy_mode": "standard",
        "reproduction": config.reproduction.model_copy(update={"enabled": False}),
        "movement": config.movement.model_copy(update={"walk_speed_fraction": .6,
                                                        "food_sprint_enabled": False}),
        "harvest": config.harvest.model_copy(update={"enabled": False}),
    })
    planner = load_planner_config(ROOT / "config/global_planner.json")
    rapid = planner.exploration.rapid_mapping.model_copy(update={
        "enabled": True, "only_until_anchored": False, "scan_enabled": True,
        "sprint_enabled": False,
    })
    exploration = planner.exploration.model_copy(update={
        "enabled": True, "early_scout_fraction": 1., "sparse_coverage_threshold": 1.,
        "scout_speed_fraction": .6, "rapid_mapping": rapid,
    })
    planner = planner.model_copy(update={
        "enabled": False, "mapping_enabled": True, "population_after_alignment": False,
        "exploration": exploration,
        "biome_inference": planner.biome_inference.model_copy(update={"enabled": False}),
        "predator_belief": planner.predator_belief.model_copy(update={"enabled": False}),
    })
    return ExpertPolicy(config, planner)


def spread_samples(samples, limit=64):
    """Prefer separated locations and different observed labels, without seeds."""
    if not samples:
        return []
    frequencies = Counter(sample["biome"] for sample in samples)
    pool = sorted(samples, key=lambda row: (frequencies[row["biome"]], row["radius"], row["x"], row["y"]))
    selected = [pool.pop(0)]
    distances = [float("inf")] * len(pool)
    while pool and len(selected) < limit:
        last = selected[-1]
        distances = [min(old, (row["x"] - last["x"]) ** 2 + (row["y"] - last["y"]) ** 2)
                     for old, row in zip(distances, pool)]
        labels = {row["biome"] for row in selected}
        index = max(range(len(pool)), key=lambda i: distances[i] * (2 if pool[i]["biome"] not in labels else 1))
        selected.append(pool.pop(index))
        distances.pop(index)
    return selected


def shared_samples(estimator):
    samples = []
    raw_count = 0
    for group in estimator.groups.values():
        raw_count += len(group.biomes)
        if not group.anchored:
            continue
        for sample in group.biomes.values():
            if sample.biome == "river" or sample.uncertainty > 6.:
                continue
            samples.append(dict(x=float(sample.position[0]), y=float(sample.position[1]),
                                biome=sample.biome, uncertainty=float(sample.uncertainty),
                                radius=6. + float(sample.uncertainty),
                                sim_time=float(sample.last_seen), group_id=group.group_id))
    return spread_samples(samples), raw_count, len(samples)


def compatible(sites, types, samples):
    """Necessary land condition when each true point lies within its radius.

    If a site's cell intersects a radius-r disk around p, its distance at p
    cannot exceed the closest site's distance by more than 2r. This deliberately
    accepts some impossible arrangements to avoid overconfident pixel tests.
    """
    for sample in samples:
        closest = desired = float("inf")
        x, y = sample["x"], sample["y"]
        for (sx, sy), biome in zip(sites, types):
            distance_sq = (x - sx) ** 2 + (y - sy) ** 2
            closest = min(closest, distance_sq)
            if biome == sample["biome"]:
                desired = min(desired, distance_sq)
        if desired > (math.sqrt(closest) + 2 * sample["radius"]) ** 2 + 1e-9:
            return False
    return True


def collect(seed, seconds):
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    started = time.perf_counter()
    sim, policy = SimulationCore(seed=seed), survey_policy()
    settings = dict(policy=policy.config.model_dump(mode="json"),
                    planner=policy.planner.config.model_dump(mode="json"))
    turns, ages, headings = {}, {}, {}
    hashes, action_log, lengths, checkpoints, audit_errors, frames = [], [], [], [], [], []
    checkpoint_ticks = {7, 100, 200, round(seconds * 10)}
    total_ticks = round(seconds * 10)
    payload = sim.step([])
    births = 0
    movement_requests = 0
    for tick in range(1, total_ticks + 1):
        payload = json.loads(json.dumps(payload, allow_nan=False))
        states, now = payload["observations"], payload["sim_time"]
        if not states:
            raise RuntimeError("Survey population died before the requested duration")
        hashes.append(hashlib.sha256(canonical_frame(payload).encode()).hexdigest())
        # Preserve the actual protocol, including repeated edges and all traits.
        # Hashes alone cannot be used to extract new evidence after collection.
        frames.append(json.loads(json.dumps(payload, allow_nan=False)))
        actions = policy.actions_for_step(states, now)
        estimator = policy.planner.estimator
        lengths.extend(public_features(states)["rock_lengths"])
        for state in states:
            agent_id = state["agent_id"]
            if ages.get(agent_id) != state["age"]:
                observed = heading_hypotheses(state["observations"], turns.get(agent_id, 0.))
                if observed is not None:
                    old = headings.get(agent_id)
                    merged = observed if old is None else [value for value in old
                              if any(angle_close(value, candidate) for candidate in observed)]
                    if not merged:
                        raise AssertionError("Inconsistent heading constraints during walking")
                    headings[agent_id] = merged
            ages[agent_id] = state["age"]
            pose = estimator.poses.get(agent_id)
            if pose is not None and estimator.groups[pose.group_id].anchored:
                # Audit only: these errors never affect actions, sample choice, or radii.
                truth = sim.env.agents_dict[agent_id]
                audit_errors.append(dict(sim_time=now, agent_id=agent_id,
                                         error=math.dist(pose.position, (truth.x, truth.y)),
                                         declared_uncertainty=float(pose.uncertainty)))
        if tick in checkpoint_ticks:
            samples, raw, eligible = shared_samples(estimator)
            evidence = extract_evidence(frames, action_log)
            samples = informative_samples(evidence, samples, spread_samples)
            checkpoint = dict(sim_time=round(now, 6), shared_samples=samples,
                              retained_biome_cells=raw, eligible_nonriver_samples=eligible,
                              selected_samples=len(samples),
                              observed_biomes=sorted({row["biome"] for row in samples}),
                              distinct_rock_lengths=len(distinct(lengths, 1e-6)),
                              groups=len(estimator.groups),
                              anchored_agents=sum(estimator.groups[pose.group_id].anchored
                                                  for pose in estimator.poses.values()),
                              known_headings=len(headings), living_agents=len(states),
                              recovered_spawn_positions=len(evidence["founder_positions"]),
                              paired_rock_rectangles=len(evidence["rock_rectangles"]),
                              positioned_rock_edges=len(evidence["world_rock_edges"]),
                              biome_transitions=len(evidence["biome_transitions"]),
                              geometry_diagnostics=evidence["diagnostics"])
            checkpoints.append(checkpoint)
            print(json.dumps({"target": seed, **{key: value for key, value in checkpoint.items()
                                                if key != "shared_samples"}}), flush=True)
        if tick == total_ticks:
            break
        if tick <= 6:
            actions = [ActionRequest(agent_id=state["agent_id"], move_distance=0, move_direction=0,
                                     turn_angle=math.pi / 3, spawn_agent=False) for state in states]
            # The estimator must integrate the actual survey turns.
            policy.planner.remember_actions(actions)
            policy.population.remember_actions(actions, now)
            policy.harvest.remember_actions(actions, now)
        births += sum(action.spawn_agent for action in actions)
        movement_requests += sum(action.move_distance > 0 for action in actions)
        action_log.append([action.model_dump(mode="json") for action in actions])
        for action in actions:
            turns[action.agent_id] = turns.get(action.agent_id, 0.) + action.turn_angle
        payload = sim.step([(action.agent_id, action) for action in actions])
    errors = sorted(row["error"] for row in audit_errors)
    low_error = [row for row in audit_errors if row["declared_uncertainty"] <= 6.]
    result = dict(schema_version=2, target_seed=seed, survey_seconds=seconds, collection_wall_seconds=time.perf_counter() - started,
                  checkpoints=checkpoints, action_log=action_log, public_frame_sha256=hashes,
                  public_frames=frames, seed_evidence=evidence,
                  shared_world_snapshot=estimator.snapshot(),
                  founder_headings=[dict(agent_id=key, hypotheses=value) for key, value in sorted(headings.items())],
                  rock_lengths=distinct(lengths, 1e-6), births_requested=births,
                  movement_requests=movement_requests, final_score=payload["score"],
                  final_energies={state["agent_id"]: state["energy"] for state in payload["observations"]},
                  localization_audit=dict(count=len(errors), median=errors[len(errors)//2] if errors else None,
                                          maximum=max(errors, default=None),
                                          low_uncertainty_checks=len(low_error),
                                          low_uncertainty_outside_radius=sum(row["error"] > 6 + row["declared_uncertainty"]
                                                                            for row in low_error)))
    del sim, policy
    gc.collect()
    return result, settings


def verify(seed, row):
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    started = time.perf_counter()
    sim = SimulationCore(seed=seed)
    generated = time.perf_counter() - started
    heading_match = all(any(angle_close(sim.env.agents_dict[item["agent_id"]].direction, value)
                           for value in item["hypotheses"]) for item in row["founder_headings"])
    dimensions = [value for rock in sim.env.obstacles for value in (rock.width, rock.height)]
    rock_match = all(any(abs(value - dimension) <= 1e-6 for dimension in dimensions) for value in row["rock_lengths"])
    evidence = row.get("seed_evidence")
    geometry = candidate_geometry_matches(sim.env, evidence) if evidence is not None else {}
    tree_match = initial_trees_match(sim.env, evidence["initial_tree_sightings"]) if evidence is not None else None
    replay = None
    first_divergence = None
    if heading_match and rock_match and all(geometry.values()) and tree_match is not False:
        replay = True
        payload = sim.step([])
        for index, expected_hash in enumerate(row["public_frame_sha256"]):
            actual_hash = hashlib.sha256(canonical_frame(payload).encode()).hexdigest()
            if actual_hash != expected_hash:
                replay, first_divergence = False, index
                break
            if index < len(row["action_log"]):
                actions = [ActionRequest(**action) for action in row["action_log"][index]]
                payload = sim.step([(action.agent_id, action) for action in actions])
    result = dict(seed=seed, generation_seconds=generated, heading_match=heading_match,
                  rock_match=rock_match, **geometry, initial_tree_match=tree_match,
                  public_replay_match=replay, first_divergence_frame=first_divergence)
    if evidence is not None:
        result["evidence_counts"] = {key: len(evidence[key]) for key in (
            "founder_positions", "founder_distances", "rock_rectangles", "world_rock_edges",
            "initial_tree_sightings", "biome_samples", "biome_transitions")}
    del sim
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, nargs="+", default=[3, 11, 20260919])
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--scan-count", type=int, default=100000)
    parser.add_argument("--max-native-worlds", type=int, default=20)
    parser.add_argument("--collect-only", action="store_true", help="Save public history/evidence without running the Python seed scan")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output filename")
    if args.seconds < .7 or abs(args.seconds * 10 - round(args.seconds * 10)) > 1e-6:
        parser.error("Survey duration must be at least 0.7 seconds in 0.1-second increments")
    if not 0 < args.scan_count <= 2**32 or any(not 0 <= seed < 2**32 for seed in args.targets):
        parser.error("Use a positive scan count and unsigned 32-bit seeds")
    if args.max_native_worlds < 1:
        parser.error("--max-native-worlds must be positive")
    sources = source_manifest()
    for name in ("seed_biome_prefix_probe.py", "seed_joint_constraints_probe.py", "seed_survey_probe.py", "seed_survey_evidence.py"):
        sources[f"scripts/{name}"] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    data = dict(complete=False, scan_count=args.scan_count, sources=sources, targets=[], limitations=[
        "Search starts only after collecting every requested survey.",
        "Original shared world model and exploration coordinator consume public observations only.",
        "Filtering uses observed non-river labels in anchored groups, not inferred biome predictions.",
        "Sample radii are conservative heuristics, not calibrated confidence bounds.",
        "Static geometry registration assumes matching precise edge-pair constellations identify the same landmarks; conflicting components are omitted.",
        "Raw public frames are retained; spawn reconstruction uses static geometry, never hidden poses or inverse walking commands.",
        "Stationary first six turns reproduce the earlier survey baseline; movement follows.",
        "Native default dynamics remain enabled; reproduction and optional scouting sprints are disabled in policy config.",
        "Bounded enumeration; no general seed inversion or full seed-space scan.",
    ])
    save(args.output, data)
    for seed in args.targets:
        row, settings = collect(seed, args.seconds)
        data["survey_settings"] = settings
        data["targets"].append(row)
        save(args.output, data)

    if args.collect_only:
        data.update(complete=True, search_performed=False, verification_complete=False)
        save(args.output, data)
        print(f"Saved public survey to {args.output}; search not requested", flush=True)
        return

    search_started = time.perf_counter()
    for row in data["targets"]:
        true_sites, true_types = candidate_prefix(row["target_seed"])
        for checkpoint in row["checkpoints"]:
            checkpoint["true_seed_passes"] = compatible(true_sites, true_types, checkpoint["shared_samples"])
            checkpoint["prefix_survivors"] = []
    for seed in range(args.scan_count):
        sites, types = candidate_prefix(seed)
        for row in data["targets"]:
            for checkpoint in row["checkpoints"]:
                if compatible(sites, types, checkpoint["shared_samples"]):
                    checkpoint["prefix_survivors"].append(seed)
    data["prefix_search_seconds_all_checkpoints"] = time.perf_counter() - search_started
    for row in data["targets"]:
        print(json.dumps({"target": row["target_seed"], "checkpoints": [
            dict(seconds=checkpoint["sim_time"], survivors=len(checkpoint["prefix_survivors"]),
                 true_seed_passes=checkpoint["true_seed_passes"]) for checkpoint in row["checkpoints"]]}), flush=True)
    save(args.output, data)
    for row in data["targets"]:
        final = row["checkpoints"][-1]
        row["native_verification"] = []
        if not final["true_seed_passes"]:
            row["verification_skipped"] = "True seed failed the map filter; investigate localization before claiming recovery"
        elif len(final["prefix_survivors"]) > args.max_native_worlds:
            row["verification_skipped"] = "Native-world cap exceeded"
        else:
            for seed in final["prefix_survivors"]:
                result = verify(seed, row)
                row["native_verification"].append(result)
                print(json.dumps({"target": row["target_seed"], "verification": result}), flush=True)
                save(args.output, data)
        row["recovered_seeds"] = [result["seed"] for result in row["native_verification"] if result["public_replay_match"]]
    data["complete"] = True
    data["verification_complete"] = all("verification_skipped" not in row for row in data["targets"])
    save(args.output, data)
    print(f"Saved {args.output}; verification complete: {data['verification_complete']}", flush=True)


if __name__ == "__main__":
    main()
