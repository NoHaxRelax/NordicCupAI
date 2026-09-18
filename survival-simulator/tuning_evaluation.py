"""Headless, fixed-horizon evaluation for the policy tuner.

Only public observations enter the policy. Simulator truth is used for the
score and diagnostic metrics, never for decisions. Keep world initialization
intact: even biome surface construction consumes the simulator RNG.
"""

import copyreg
from contextlib import contextmanager
import gzip
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
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


def stopping(deadline):
    parent = multiprocessing.parent_process()
    return (time.monotonic() >= deadline or (parent is not None and not parent.is_alive())
            or (STOP_EVENT is not None and STOP_EVENT.is_set()))


@contextmanager
def episode_lock(destination, deadline):
    """Let orphaned workers save and exit before a resumed worker reads state."""
    path = Path(destination).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        if os.name == "nt":
            import msvcrt
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        while True:
            try:
                handle.seek(0)
                acquire()
                break
            except OSError:
                if stopping(deadline):
                    yield False
                    return
                time.sleep(.1)
        try:
            yield True
        finally:
            handle.seek(0)
            release()


def atomic_replace(temporary, destination):
    # Windows readers/indexers can briefly hold a file without delete sharing.
    # Keep the previous complete checkpoint intact and retry the rename.
    for attempt in range(12):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 11:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, path)


def headless_surface(size, flags, depth):
    import pygame
    return pygame.Surface(size, flags, depth)


def reduce_surface(surface):
    # Pixels are only used by rendering, never sensing or physics. Recreate
    # blank surfaces on resume; fresh world generation still runs unchanged.
    import pygame
    return headless_surface, (surface.get_size(), surface.get_flags() & pygame.SRCALPHA,
                              surface.get_bitsize())


def checkpoint_path(destination):
    return Path(destination).with_suffix(".checkpoint.pkl.gz")


def save_checkpoint(destination, state):
    """Atomically replace a private, local episode snapshot, including its RNG."""
    import pygame
    path = checkpoint_path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    env = state["sim"].env
    event_sink = env.event_sink
    try:
        env.event_sink = None  # Reattached to restored metrics in evaluate_seed.
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1) as compressed:
                writer = pickle.Pickler(compressed, protocol=pickle.HIGHEST_PROTOCOL)
                writer.dispatch_table = {**copyreg.dispatch_table, pygame.Surface: reduce_surface}
                writer.dump(state)
            raw.flush()
            os.fsync(raw.fileno())
        atomic_replace(temporary, path)
    finally:
        env.event_sink = event_sink
        temporary.unlink(missing_ok=True)
    atomic_json(Path(destination).with_suffix(".progress.json"), dict(
        seed=state["identity"]["seed"], horizon_seconds=state["identity"]["seconds"],
        simulated_seconds=env.time, population=len(env.agents), steps=state["steps"],
        active_wall_seconds=state["wall_seconds"], saved_at_unix=time.time()))


def evaluate_seed(expert, planner, seed, seconds, destination, deadline, checkpoint_seconds=60.):
    with episode_lock(destination, deadline) as acquired:
        if acquired:
            return _evaluate_seed(expert, planner, seed, seconds, destination, deadline, checkpoint_seconds)
    return None


def _evaluate_seed(expert, planner, seed, seconds, destination, deadline, checkpoint_seconds):
    """Return a complete episode, or None on interruption; never score a timeout.

    The destination belongs to a configuration hash within a manifest-checked
    study. Partial episodes have separate snapshots and never become scores.
    An extinct episode is complete, with its actual (shorter) survival score.
    """
    configure_process()
    destination = Path(destination)
    if destination.exists():
        result = json.loads(destination.read_text(encoding="utf-8"))
        if result["seed"] != seed or result["horizon_seconds"] != seconds:
            raise ValueError(f"Incompatible cached episode: {destination}")
        return result

    if stopping(deadline):
        return None
    from src.core import SimulationCore
    from src.utils.controllers.expert_policy import ExpertConfig, ExpertPolicy
    from src.utils.controllers.global_planner import PlannerConfig
    from src.utils.controllers.population import TRAITS

    started = time.monotonic()
    attributes = {"vision_range": "vision_radius", "vision_angle": "cone_angle"}
    identity = dict(version=1, expert=expert, planner=planner, seed=seed, seconds=seconds)
    checkpoint = checkpoint_path(destination)
    if checkpoint.exists():
        # Only our own snapshots inside the manifest-checked study are loaded.
        with gzip.open(checkpoint, "rb") as handle:
            saved = pickle.load(handle)
        if saved["identity"] != identity:
            raise ValueError(f"Incompatible episode checkpoint: {checkpoint}")
        sim, policy = saved["sim"], saved["policy"]
        founders, metrics = saved["founders"], saved["metrics"]
        actions, trace = saved["actions"], saved["trace"]
        next_sample, aligned_at = saved["next_sample"], saved["aligned_at"]
        controller_seconds, steps = saved["controller_seconds"], saved["steps"]
        previous_wall_seconds = saved["wall_seconds"]
        print(f"Resuming {destination.parent.name} seed {seed} at {sim.env.time:.1f}/{seconds:g}s", flush=True)
    else:
        policy = ExpertPolicy(ExpertConfig.model_validate(expert), PlannerConfig.model_validate(planner))
        sim = SimulationCore(seed=seed, predators_enabled=False)
        founders = {trait: statistics.mean(getattr(a, attributes.get(trait, trait)) for a in sim.env.agents)
                    for trait in TRAITS}
        metrics = dict(births=0, fruit_eaten=0, fruit_rotted=0, fruit_energy=0., ripe_fruit=0,
                       child_trait_total=0., parent_trait_total=0., parent_count=0, deaths={})
        actions, trace = [], []
        next_sample, aligned_at = 0., None
        controller_seconds, steps, previous_wall_seconds = 0., 0, 0.

    def checkpoint_now():
        save_checkpoint(destination, dict(identity=identity, sim=sim, policy=policy,
            founders=founders, metrics=metrics, actions=actions, trace=trace,
            next_sample=next_sample, aligned_at=aligned_at, controller_seconds=controller_seconds,
            steps=steps, wall_seconds=previous_wall_seconds + time.monotonic() - started))

    def trait_score(agent):
        return statistics.mean(getattr(agent, attributes.get(trait, trait)) / founders[trait] for trait in TRAITS)

    def event(kind, **data):
        if kind == "fruit_eaten":
            energy = data["fruit"].energy
            metrics["fruit_eaten"] += 1
            metrics["fruit_energy"] += energy
            if data["agent"].age > data["agent"].max_age:
                metrics["senescent_fruit_energy"] = metrics.get("senescent_fruit_energy", 0.) + energy
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
            agent = data["agent"]
            closest = min((math.hypot(f.x - agent.x, f.y - agent.y) for f in sim.env.fruits), default=None)
            # Simulator truth is diagnostic only; none of these values enter policy decisions.
            metrics.setdefault("death_details", []).append(dict(
                time=sim.env.time, agent_id=agent.agent_id, age=agent.age, energy=agent.energy,
                position=[agent.x, agent.y], biome=sim.env.get_agent_state(agent.agent_id)["biome"],
                onset_age=agent.max_age, population=len(sim.env.agents),
                trees=len(sim.env.trees), fruits=len(sim.env.fruits), nearest_fruit_distance=closest,
                last_birth_request=policy.population.last_birth_request.get(agent.agent_id),
                task=policy.harvest.tasks.get(agent.agent_id)))

    sim.env.event_sink = event
    next_checkpoint = time.monotonic() + checkpoint_seconds
    # Integer step count avoids an extra tick from accumulated float drift at
    # the 30,000-step full-game horizon (including after a checkpoint resume).
    while steps * sim.dt < seconds - 1e-9 and sim.env.agents:
        if steps % 10 == 0:
            if stopping(deadline):
                checkpoint_now()
                return None
            if time.monotonic() >= next_checkpoint:
                checkpoint_now()
                next_checkpoint = time.monotonic() + checkpoint_seconds
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
                population_target=policy.harvest.population_target,
                reproduction_plan=policy.harvest.population_plan.copy(),
                trees=len(sim.env.trees), fruits=len(sim.env.fruits),
                fruit_energy=sum(f.energy for f in sim.env.fruits),
                births=metrics["births"], fruit_eaten=metrics["fruit_eaten"],
                agents=[dict(agent_id=a.agent_id, age=a.age, energy=a.energy,
                             position=[a.x, a.y], biome=sim.env.get_agent_state(a.agent_id)["biome"],
                             task=policy.harvest.tasks.get(a.agent_id)) for a in sim.env.agents]))
            next_sample += 60
    if not math.isfinite(sim.env.score):
        raise ValueError("Simulator returned a non-finite score")
    result = dict(seed=seed, horizon_seconds=seconds, seconds=sim.env.time, score=sim.env.score,
                  extinct=not bool(sim.env.agents), population=len(sim.env.agents),
                  shared_map_seconds=aligned_at, births=metrics["births"], deaths=metrics["deaths"],
                  death_details=metrics.get("death_details", []),
                  fruit_eaten=metrics["fruit_eaten"], fruit_rotted=metrics["fruit_rotted"],
                  fruit_score=metrics["fruit_energy"] / 1000,
                  mean_fruit_energy=metrics["fruit_energy"] / max(1, metrics["fruit_eaten"]),
                  senescent_fruit_percent=100 * metrics.get("senescent_fruit_energy", 0.) / max(1., metrics["fruit_energy"]),
                  ripe_fruit_percent=100 * metrics["ripe_fruit"] / max(1, metrics["fruit_eaten"]),
                  living_trait_score=statistics.mean(trait_score(a) for a in sim.env.agents) if sim.env.agents else None,
                  mean_child_trait_score=metrics["child_trait_total"] / metrics["births"] if metrics["births"] else None,
                  mean_parent_trait_score=metrics["parent_trait_total"] / metrics["parent_count"] if metrics["parent_count"] else None,
                  controller_mean_ms=1000 * controller_seconds / max(1, steps),
                  wall_seconds=previous_wall_seconds + time.monotonic() - started, trace=trace)
    atomic_json(destination, result)
    checkpoint.unlink(missing_ok=True)
    destination.with_suffix(".progress.json").unlink(missing_ok=True)
    return result
