"""Repeated random-map encounters with independent observation-only guides.

Disclosed fixture: stationary depth-five bait; at fixed public times a random
valid-map predator and an already-engaged guide 55 units away are introduced.
Only fixture placement and after-action evaluation read native creature state.
Predator behavior, terrain, collisions and ambient births remain unmodified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research"), str(ROOT / "research/simple_chase"), str(ROOT / "research/reliability_eval")]
from simple_chase import run_streaming_v2 as h
from simple_chase.sampling import sample_start, _free_for_native_spawn
from reliability_eval.run_frozen import policy_dependency_manifest

OUT = ROOT / "results/sequential_real_map"


def introduce(env, site, fixture_seed, slot):
    occupied = [((a.x, a.y), a.size) for a in env.agents]
    occupied += [((p.x, p.y), p.size) for p in env.predators]
    x, y, heading = sample_start(env, fixture_seed=fixture_seed,
                                kind="predator", index=slot, occupied=occupied)
    for k in range(72):
        angle = heading + k * math.tau / 72
        point = (x + 55 * math.cos(angle), y + 55 * math.sin(angle))
        if (_free_for_native_spawn(env, point, 5., "agent")
                and math.dist(point, site["goal"]) > 100
                and all(math.dist(point, p) > 5 + r for p, r in occupied)):
            break
    else:
        raise ValueError("no valid adjacent guide placement")
    guide = env.spawn_agent(x=point[0], y=point[1])
    predator = env.spawn_predator(x=x, y=y)
    if guide is None or predator is None:
        raise RuntimeError("native spawn rejected validated encounter")
    if math.dist((guide.x, guide.y), point) > 1e-8 or math.dist((predator.x, predator.y), (x, y)) > 1e-8:
        raise RuntimeError("native spawn silently changed encounter position")
    guide.direction = (angle + math.pi) % math.tau
    predator.direction = angle
    predator.energy = predator.max_energy
    predator.resting = False
    env._update_spatial_grid()
    return guide, predator, dict(slot=slot, time=round(env.time, 1),
        guide_id=guide.agent_id, guide_start=[guide.x, guide.y, guide.direction],
        predator_start=[predator.x, predator.y, predator.direction])


def run(policy_spec, map_seed, fixture_seed, encounters, interval, seconds):
    if h.STOP.exists():
        raise SystemExit("stop sentinel present")
    core = h.SimulationCore(starting_agents=0, starting_predators=0, seed=map_seed)
    env = core.env
    static_map = h.snapshot_environment(env)
    policy_class, source = h._load_policy(policy_spec)
    dependencies = policy_dependency_manifest(policy_spec.split(":")[0])
    harness_hashes = h._harness_hashes()
    prototype = policy_class(h.json_clone(static_map), bait_id=0, guide_id=1)
    site = h._policy_site(prototype, static_map)
    bx, by, _ = sample_start(env, fixture_seed=fixture_seed, kind="agent", index=0)
    bait = env.spawn_agent(x=bx, y=by)
    if env._in_obstacle(tuple(site["goal"]), radius=5., obstacles=env.obstacles):
        raise ValueError("bait goal collides with native wall")
    bait.x, bait.y = site["goal"]
    bait.direction = math.atan2(-site["inward"][1], -site["inward"][0])
    controllers = {}
    tracked = []
    rows = []
    deaths = []
    trace = []
    tag = f"sequential-real-map-m{map_seed}-f{fixture_seed}-n{encounters}-i{interval:g}-{uuid4().hex[:8]}"
    replay_path = OUT / "replays" / f"{tag}.json.gz"
    recorder = h.ReplayRecorder(env, title=tag, policy=policy_spec, seed=map_seed,
        every=1, native_render=True, native_width=320,
        scenario="native real map; repeated random-position already-engaged encounters",
        notes=__doc__ + " All native frames retained. Guide survival and sacrifice both permitted. "
              "Prepared bait and arranged encounters; no predator discovery or finite-agent-energy claim.",
        policy_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    error = None
    start_time = env.time
    def add_encounter():
        slot = len(rows)
        guide, predator, row = introduce(env, site, fixture_seed, slot)
        controller = policy_class(h.json_clone(static_map), bait_id=bait.agent_id,
                                  guide_id=guide.agent_id)
        if controller.site != site:
            raise ValueError("independent guides selected different dedicated sites")
        controllers[guide.agent_id] = controller
        row.update(first_guide_target=None, last_guide_target=None,
            last_handoff_guide_id=None, handoff_guide_id=None,
            active_since_guide=1e9, candidate_since=None, capture_time=None,
            first_active_bait=None, handoff_active_seconds=None, handoff_wall_seconds=None,
            first_physical_loss=None, first_target_loss=None,
            guide_dead_at=None, last_target=None, target_switches=[])
        rows.append(row)
        tracked.append(predator)
    try:
        add_encounter()
        recorder.capture()
        for tick in range(round(seconds / core.dt)):
            if h.STOP.exists():
                break
            for agent in env.agents:
                agent.energy = agent.max_energy
            states = h._dto_states(env)
            actions = []
            decisions = {}
            for state in h.json_clone(states):
                aid = state["agent_id"]
                if aid == bait.agent_id:
                    actions.append(dict(agent_id=aid, move_distance=0., move_direction=0.,
                                        turn_angle=0., spawn_agent=False))
                    decisions[aid] = dict(rule="stationary_replaceable_bait")
                else:
                    controller = controllers[aid]
                    actions += controller.act([state], env.time)
                    decisions.update(controller.decisions)
            requests = h._validate_actions(actions, env)
            action_t = env.time
            alive_before = set(env.agents_dict)
            for request in requests:
                env.agent_step(**request.model_dump())
            env.non_agent_step(core.dt)
            for aid in alive_before - set(env.agents_dict):
                deaths.append(dict(time=round(env.time, 1), agent_id=aid))
                for row in rows:
                    if row["guide_id"] == aid:
                        row["guide_dead_at"] = round(env.time, 1)
            held = 0
            for predator, row in zip(tracked, rows):
                dx, dy = predator.x - site["mouth"][0], predator.y - site["mouth"][1]
                physical = math.hypot(dx, dy) <= 75 and dx * site["inward"][0] + dy * site["inward"][1] <= 10
                visible = [o for o in predator.observe(agents=list(env.agents), edges=list(env.edges)) if o["type"] == "Agent"]
                chosen = min(visible, key=lambda o:o["distance"]) if visible else None
                target = None if chosen is None else chosen["id"]
                label = "resting" if predator.resting else target
                if label != row["last_target"]:
                    row["target_switches"].append(dict(time=round(env.time, 1), target=label))
                    row["last_target"] = label
                following = (not predator.resting and target in controllers and chosen is not None
                             and (abs(chosen["rel_dir"]) > math.pi / 2
                                  or chosen["distance"] < predator.hearing_radius * 1.5))
                if following:
                    if row["first_guide_target"] is None:
                        row["first_guide_target"] = round(env.time, 1)
                    row["last_guide_target"] = round(env.time, 1)
                    row["last_handoff_guide_id"] = target
                    row["active_since_guide"] = 0.
                elif not predator.resting:
                    row["active_since_guide"] += core.dt
                contained = physical and (predator.resting or target == bait.agent_id)
                active_bait = physical and not predator.resting and target == bait.agent_id
                if row["capture_time"] is None:
                    if row["candidate_since"] is None and active_bait and row["active_since_guide"] <= 3. + 1e-8:
                        row["candidate_since"] = round(env.time, 1)
                        row["first_active_bait"] = row["candidate_since"]
                        row["handoff_active_seconds"] = round(row["active_since_guide"], 1)
                        row["handoff_wall_seconds"] = round(env.time - row["last_guide_target"], 1)
                        row["handoff_guide_id"] = row["last_handoff_guide_id"]
                    if not contained:
                        row["candidate_since"] = None
                    if row["candidate_since"] is not None and env.time - row["candidate_since"] >= 1.9 - 1e-8:
                        row["capture_time"] = row["candidate_since"]
                if row["capture_time"] is not None:
                    if not physical and row["first_physical_loss"] is None:
                        row["first_physical_loss"] = round(env.time, 1)
                    if not contained and row["first_target_loss"] is None:
                        row["first_target_loss"] = round(env.time, 1)
                held += int(contained)
            # Insert at the scheduled public instant AFTER native movement and
            # BEFORE that instant's frame. Thus every birth position has an
            # exact native image before its first action, with no duplicate t.
            elapsed = env.time - start_time
            if (len(rows) < encounters and elapsed + 1e-6 >= len(rows) * interval
                    and elapsed < seconds - .05):
                add_encounter()
            recorder.capture([(r.agent_id, r) for r in requests], decisions,
                             inputs=states, action_t=action_t)
            if tick % 10 == 0:
                trace.append(dict(time=round(env.time, 1), introduced=len(rows),
                    captured=sum(r["capture_time"] is not None for r in rows), held=held,
                    guides_alive=len(env.agents) - int(bait.agent_id in env.agents_dict),
                    predators_total=len(env.predators)))
            if tick % 300 == 299:
                print(json.dumps(dict(progress=tag, **trace[-1])), flush=True)
            if bait.agent_id not in env.agents_dict:
                break
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        reason = "exception" if error else "stop sentinel" if h.STOP.exists() else "horizon" if env.time - start_time >= seconds - .05 else "bait died"
        summary = recorder.save(replay_path, reason=reason)
        for row in rows:
            row["guide_alive_final"] = row["guide_id"] in env.agents_dict
            row["policy_events"] = controllers[row["guide_id"]].events
            row["pass"] = (row["capture_time"] is not None
                           and env.time - row["capture_time"] >= 30. - 1e-8
                           and row["first_physical_loss"] is None
                           and row["first_target_loss"] is None)
        result = dict(schema="sequential-real-map-encounter-v2", policy=policy_spec,
            map_seed=map_seed, fixture_seed=fixture_seed, requested_encounters=encounters,
            interval=interval, requested_seconds=seconds, seconds=env.time, reason=reason,
            error=error, site=site, bait_id=bait.agent_id, bait_alive=bait.agent_id in env.agents_dict,
            guide_ids=sorted(controllers), encounters=rows, deaths=deaths, trace=trace,
            successful_encounters=sum(r["pass"] for r in rows),
            success=reason == "horizon" and len(rows) == encounters and all(r["pass"] for r in rows),
            native_render=True, record_every_ticks=1, replay_frames=summary["frames"],
            replay=str(replay_path.relative_to(ROOT)),
            limitations="Prepared stationary bait; random predator plus adjacent awake engaged guide; infinite agent energy; native ambient predators; single native map; not a confidence-bound reliability result.",
            attribution="colony guide-to-bait transfer; any guide may assist; actual handoff guide ID recorded separately from assigned encounter guide",
            native_birth_frames=True, policy_dependencies=dependencies, harness_hashes=harness_hashes,
            dependencies_unchanged=dependencies == policy_dependency_manifest(policy_spec.split(":")[0]),
            source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            policy_hash=hashlib.sha256(source.read_bytes()).hexdigest())
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{tag}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(receipt=str(path), success=result["success"],
                              successful_encounters=result["successful_encounters"],
                              reason=reason)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", dest="policy_spec", default="release_validation.policy_v24_release:ReleaseGuide")
    p.add_argument("--map-seed", type=int, default=10040)
    p.add_argument("--fixture-seed", type=int, default=22040)
    p.add_argument("--encounters", type=int, default=5)
    p.add_argument("--interval", type=float, default=75.)
    p.add_argument("--seconds", type=float, default=600.)
    run(**vars(p.parse_args()))
