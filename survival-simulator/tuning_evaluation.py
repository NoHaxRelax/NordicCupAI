"""Headless, fixed-horizon evaluation for the policy tuner.

Only public observations enter the policy. Simulator truth is used for the
score and diagnostic metrics, never for decisions. Keep world initialization
intact: even biome surface construction consumes the simulator RNG.
"""

import json
import math
import os
from pathlib import Path
import signal
import statistics
import time


STOP_EVENT = None


def configure_process():
    # Must happen before NumPy/SciPy/Pygame imports, including in spawned workers.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "0")


def initialize_worker(stop_event):
    global STOP_EVENT
    STOP_EVENT = stop_event
    configure_process()
    # The coordinator handles Ctrl-C and asks workers to finish cooperatively.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluate_seed(expert, planner, seed, seconds, destination, deadline):
    """Return a complete episode, or None on interruption; never score a timeout.

    The destination belongs to a configuration hash within a manifest-checked
    study. Only complete episodes are cached. An extinct episode is complete,
    with its actual (shorter) survival score.
    """
    configure_process()
    destination = Path(destination)
    if destination.exists():
        result = json.loads(destination.read_text(encoding="utf-8"))
        if result["seed"] != seed or result["horizon_seconds"] != seconds:
            raise ValueError(f"Incompatible cached episode: {destination}")
        return result

    def stopping():
        return time.monotonic() >= deadline or (STOP_EVENT is not None and STOP_EVENT.is_set())

    if stopping():
        return None
    from src.core import SimulationCore
    from src.utils.controllers.expert_policy import ExpertConfig, ExpertPolicy
    from src.utils.controllers.global_planner import PlannerConfig
    from src.utils.controllers.population import TRAITS

    started = time.monotonic()
    policy = ExpertPolicy(ExpertConfig.model_validate(expert), PlannerConfig.model_validate(planner))
    sim = SimulationCore(seed=seed, predators_enabled=False)
    attributes = {"vision_range": "vision_radius", "vision_angle": "cone_angle"}
    founders = {trait: statistics.mean(getattr(a, attributes.get(trait, trait)) for a in sim.env.agents)
                for trait in TRAITS}

    def trait_score(agent):
        return statistics.mean(getattr(agent, attributes.get(trait, trait)) / founders[trait] for trait in TRAITS)

    metrics = dict(births=0, fruit_eaten=0, fruit_rotted=0, fruit_energy=0., ripe_fruit=0,
                   child_trait_total=0., parent_trait_total=0., parent_count=0, deaths={})

    def event(kind, **data):
        if kind == "fruit_eaten":
            energy = data["fruit"].energy
            metrics["fruit_eaten"] += 1
            metrics["fruit_energy"] += energy
            metrics["ripe_fruit"] += int(energy >= 56)
        elif kind == "fruit_rot":
            metrics["fruit_rotted"] += 1
        elif kind == "birth":
            metrics["births"] += 1
            metrics["child_trait_total"] += trait_score(data["agent"])
            if data.get("parent") is not None:
                metrics["parent_trait_total"] += trait_score(data["parent"])
                metrics["parent_count"] += 1
        elif kind == "death":
            cause = data["cause"]
            metrics["deaths"][cause] = metrics["deaths"].get(cause, 0) + 1

    sim.env.event_sink = event
    actions, trace = [], []
    next_sample = 0.
    aligned_at = None
    controller_seconds = 0.
    steps = 0
    while sim.env.time < seconds - 1e-9 and sim.env.agents:
        if steps % 10 == 0 and stopping():
            return None
        state = sim.step(actions)
        before = time.monotonic()
        decisions = policy.actions_for_step(state["observations"], state["sim_time"])
        controller_seconds += time.monotonic() - before
        actions = [(action.agent_id, action) for action in decisions]
        steps += 1
        if aligned_at is None and policy.planner.population_phase:
            aligned_at = sim.env.time
        if sim.env.time >= next_sample - 1e-9:
            trace.append(dict(time=sim.env.time, score=sim.env.score, population=len(sim.env.agents),
                              population_target=policy.harvest.population_target))
            next_sample += 60
    if not math.isfinite(sim.env.score):
        raise ValueError("Simulator returned a non-finite score")
    result = dict(seed=seed, horizon_seconds=seconds, seconds=sim.env.time, score=sim.env.score,
                  extinct=not bool(sim.env.agents), population=len(sim.env.agents),
                  shared_map_seconds=aligned_at, births=metrics["births"], deaths=metrics["deaths"],
                  fruit_eaten=metrics["fruit_eaten"], fruit_rotted=metrics["fruit_rotted"],
                  fruit_score=metrics["fruit_energy"] / 1000,
                  mean_fruit_energy=metrics["fruit_energy"] / max(1, metrics["fruit_eaten"]),
                  ripe_fruit_percent=100 * metrics["ripe_fruit"] / max(1, metrics["fruit_eaten"]),
                  living_trait_score=statistics.mean(trait_score(a) for a in sim.env.agents) if sim.env.agents else None,
                  mean_child_trait_score=metrics["child_trait_total"] / metrics["births"] if metrics["births"] else None,
                  mean_parent_trait_score=metrics["parent_trait_total"] / metrics["parent_count"] if metrics["parent_count"] else None,
                  controller_mean_ms=1000 * controller_seconds / max(1, steps),
                  wall_seconds=time.monotonic() - started, trace=trace)
    atomic_json(destination, result)
    return result
