"""Find native predators confined to a small area with no prey present.

Defaults: 10,000 seeded games, 100 starting predators, 600 seconds per game.
Run from the repository root:
    python survival-simulator/scripts/predator_stuck_scan.py --workers 4
    python survival-simulator/scripts/predator_stuck_scan.py --games 2 --seconds 65 --output survival-simulator/runs/stuck-smoke

Completed games resume automatically. Findings include all native-tick positions
in a qualifying interval, map geometry, and a local PNG. No agents are created;
normal map generation, predator movement, rest and ambient spawns are retained.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time
import traceback

# Set before numpy/scipy imports, including in spawned Windows workers.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DT = 0.1
STOP = None


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class ResidenceWindow:
    """Exact rolling radius test, anchored at each window's first position.

    Each sample is (tick, x, y, heading, energy, resting). Monotonic extrema
    queues reject moving trajectories cheaply. Ambiguous bounding boxes fall
    back to checking every sample, never just the interval endpoints.
    """

    def __init__(self, steps, radius):
        if steps < 1 or not math.isfinite(radius) or radius <= 0:
            raise ValueError("Positive window length and radius required")
        self.steps, self.radius = steps, radius
        self.history = deque()
        self.extrema = [deque() for _ in range(4)]

    def add(self, sample):
        tick, x, y = sample[:3]
        if self.history and tick != self.history[-1][0] + 1:
            raise ValueError("ResidenceWindow requires consecutive ticks")
        if not all(math.isfinite(v) for v in sample[1:5]):
            raise ValueError("Non-finite predator state")
        self.history.append(sample)
        oldest = tick - self.steps
        while self.history[0][0] < oldest:
            self.history.popleft()
        for queue, value, is_min in zip(self.extrema, (x, x, y, y), (True, False, True, False)):
            while queue and (queue[-1][1] >= value if is_min else queue[-1][1] <= value):
                queue.pop()
            queue.append((tick, value))
            while queue[0][0] < oldest:
                queue.popleft()
        if len(self.history) < self.steps + 1:
            return False
        anchor_x, anchor_y = self.history[0][1:3]
        min_x, max_x, min_y, max_y = (q[0][1] for q in self.extrema)
        dx = max(abs(min_x - anchor_x), abs(max_x - anchor_x))
        dy = max(abs(min_y - anchor_y), abs(max_y - anchor_y))
        if dx > self.radius or dy > self.radius:
            return False
        if dx * dx + dy * dy <= self.radius * self.radius:
            return True
        return all((s[1] - anchor_x) ** 2 + (s[2] - anchor_y) ** 2 <= self.radius ** 2
                   for s in self.history)


def init_worker(stop):
    global STOP
    STOP = stop
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def stop_requested():
    return STOP is not None and STOP.is_set()


def event_image(env, event, path):
    """A 160-by-160 world-unit crop, enlarged with the measured radius marked."""
    import pygame
    pygame.font.init()
    span, scale, footer = 160, 4, 100
    cx, cy = event["anchor"]
    left, top = round(cx - span / 2), round(cy - span / 2)
    crop = pygame.Surface((span, span))
    crop.fill((17, 24, 33))
    crop.blit(env.biome_surface, (-left, -top))
    for obstacle in env.obstacles:
        pygame.draw.rect(crop, (88, 95, 105),
                         (obstacle.x - left, obstacle.y - top, obstacle.width, obstacle.height))
    image = pygame.Surface((span * scale, span * scale + footer))
    image.fill((17, 24, 33))
    image.blit(pygame.transform.scale(crop, (span * scale, span * scale)), (0, 0))

    def screen(x, y):
        return round((x - left) * scale), round((y - top) * scale)

    points = [screen(s[1], s[2]) for s in event["samples"]]
    pygame.draw.circle(image, (255, 200, 110), screen(cx, cy), round(event["radius"] * scale), 2)
    pygame.draw.lines(image, (80, 230, 170), False, points, 2)
    pygame.draw.circle(image, (255, 105, 115), points[-1], round(event["predator_size"] * scale), 2)
    font = pygame.font.SysFont("segoeui", 18)
    labels = [f"Seed {event['seed']} | Predator {event['predator_id']} | {event['biome']}",
              f"Confined {event['start_seconds']:g} - {event['detected_seconds']:g}s; radius {event['radius']:g}",
              f"Max distance: {event['max_distance']:.3f} | Amber: test radius | Green: path"]
    for i, label in enumerate(labels):
        image.blit(font.render(label, True, (235, 242, 248)), (14, span * scale + 10 + i * 28))
    pygame.image.save(image, str(path))


def run_game(seed, config, output):
    """One native game. Result files are the durable unit of resumption."""
    if config.get("backend") == "cpp":
        from predator_stuck_cpp import run_game as run_cpp_game
        return run_cpp_game(seed, config, output)
    from src.core import SimulationCore

    folder = Path(output) / "games" / f"seed-{seed:08d}"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(seed=seed, status="running", findings=[], simulated_seconds=0.0)
    try:
        core = SimulationCore(seed=seed, starting_agents=0,
                              starting_predators=config["predators"], dt=DT)
        env = core.env
        first_attempt_count = len(env.predators)
        attempts = 0
        while len(env.predators) < config["predators"]:
            if stop_requested():
                result["status"] = "interrupted"
                return result
            attempts += 1
            if attempts > config["predators"] * 1000:
                raise RuntimeError("Unable to place the requested starting population")
            env.spawn_predator()
        result.update(initial_predators=len(env.predators), native_initial_successes=first_attempt_count,
                      additional_spawn_attempts=attempts, initial_overlaps=[])
        trackers, catalog, reported = {}, [], set()
        steps = round(config["stuck_seconds"] / DT)

        def register(tick):
            for predator in env.predators[len(catalog):]:
                pid = len(catalog)
                overlapping = bool(env._in_obstacle((predator.x, predator.y), predator.size, env.obstacles))
                info = dict(predator_id=pid, born_seconds=round(tick * DT, 6),
                            initial_position=[predator.x, predator.y], initial_heading=predator.direction,
                            spawned_overlapping_obstacle=overlapping)
                catalog.append(info)
                trackers[pid] = ResidenceWindow(steps, config["radius"])
                if tick == 0 and overlapping:
                    result["initial_overlaps"].append(pid)

        register(0)
        final_tick = round(config["seconds"] / DT)
        last_progress = 0.0
        for tick in range(final_tick + 1):
            if stop_requested():
                result["status"] = "interrupted"
                break
            if tick:
                env.non_agent_step(DT)  # Never stop simply because the agent list is empty.
                register(tick)
            if env.agents:
                raise RuntimeError("Unexpected agent in an agent-free game")
            for pid, tracker in list(trackers.items()):
                predator = env.predators[pid]
                sample = (tick, predator.x, predator.y, predator.direction, predator.energy, predator.resting)
                if not tracker.add(sample):
                    continue
                history = list(tracker.history)
                anchor = history[0][1:3]
                event = dict(seed=seed, predator_id=pid, radius=config["radius"],
                             start_seconds=round(history[0][0] * DT, 6),
                             detected_seconds=round(tick * DT, 6),
                             duration_seconds=round((tick - history[0][0]) * DT, 6),
                             anchor=list(anchor), end_position=[predator.x, predator.y],
                             max_distance=max(math.hypot(s[1] - anchor[0], s[2] - anchor[1]) for s in history),
                             awake_samples=sum(not s[5] for s in history),
                             sample_count=len(history), predator_size=predator.size,
                             biome=env.biome_map[int(predator.x), int(predator.y)].type,
                             **{k: v for k, v in catalog[pid].items() if k != "predator_id"})
                events_dir = folder / "events"
                events_dir.mkdir(exist_ok=True)
                if not result["findings"]:
                    write_json(folder / "map.json", dict(width=env.width, height=env.height,
                               obstacles=[[o.x, o.y, o.width, o.height] for o in env.obstacles]))
                    print(f"FOUND seed {seed}, predator {pid}, at {tick * DT:.1f}s", flush=True)
                trace_path = events_dir / f"predator-{pid:04d}.json.gz"
                event["trace"] = str(trace_path.relative_to(Path(output)))
                trace = event | {"dt": DT,
                                "sample_columns": ["tick", "x", "y", "heading", "energy", "resting"],
                                "samples": history}
                temporary = trace_path.with_suffix(".gz.tmp")
                with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                    json.dump(trace, handle, allow_nan=False)
                temporary.replace(trace_path)
                if config["images"]:
                    png_path = trace_path.with_suffix("").with_suffix(".png")
                    event_image(env, trace, png_path)
                    event["image"] = str(png_path.relative_to(Path(output)))
                result["findings"].append(event)
                reported.add(pid)
                del trackers[pid]  # First qualifying interval per predator is enough.
            result["simulated_seconds"] = round(tick * DT, 6)
            now = time.monotonic()
            if now - last_progress >= 10 or tick == final_tick:
                write_json(folder / "progress.json", dict(seed=seed, tick=tick,
                           simulated_seconds=result["simulated_seconds"],
                           predators=len(env.predators), findings=len(reported),
                           wall_seconds=round(now - started, 3)))
                last_progress = now
        else:
            result["status"] = "complete"
        result.update(final_predators=len(env.predators), total_tracked=len(catalog), predators=catalog)
    except Exception:
        result.update(status="error", error=traceback.format_exc())
    finally:
        result["wall_seconds"] = round(time.monotonic() - started, 3)
        write_json(folder / "result.json", result)
    return result


def source_hashes(backend="python"):
    # These are the engine modules used by this agent-free experiment. Changes
    # to unrelated agent policies should not invalidate a multi-day scan.
    paths = [Path(__file__).resolve(), ROOT / "src/core.py",
             ROOT / "src/utils/simulation.py", ROOT / "src/utils/sensing.py",
             *sorted((ROOT / "src/elements").rglob("*.py"))]
    if backend == "cpp":
        native = ROOT / "scripts/predator_stuck_cpp"
        paths = [Path(__file__).resolve(), native / "__init__.py", native / "_stuck.cpp",
                 native / "vendor_engine.cpp", native / "build.py", native / "diagnostics.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def compact_result(result):
    return {key: value for key, value in result.items() if key != "predators"}


def report(output, results, total, config, wall_seconds):
    completed = [r for r in results.values() if r["status"] == "complete"]
    errors = [r for r in results.values() if r["status"] == "error"]
    all_findings = [event for r in results.values() for event in r["findings"]]
    complete_findings = [event for r in completed for event in r["findings"]]
    summary = dict(requested_games=total, completed_games=len(completed), failed_games=len(errors),
                   games_with_findings=sum(bool(r["findings"]) for r in completed),
                   flagged_predators_in_complete_games=len(complete_findings),
                   starting_predators_in_complete_games=sum(r["initial_predators"] for r in completed),
                   total_predators_tracked_in_complete_games=sum(r["total_tracked"] for r in completed),
                   flagged_predators_spawned_overlapping=sum(e["spawned_overlapping_obstacle"] for e in complete_findings),
                   total_initial_overlaps=sum(len(r["initial_overlaps"]) for r in completed),
                   findings_in_failed_or_interrupted_games=len(all_findings) - len(complete_findings),
                   current_invocation_wall_seconds=round(wall_seconds, 3), config=config,
                   conclusion="A finding proves only the measured confinement interval, not permanent entrapment.")
    write_json(output / "summary.json", summary)
    columns = ["seed", "predator_id", "start_seconds", "detected_seconds", "duration_seconds",
               "radius", "max_distance", "awake_samples", "sample_count", "biome",
               "spawned_overlapping_obstacle", "anchor", "end_position", "trace", "image"]
    temporary = output / "findings.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["game_status", *columns], extrasaction="ignore")
        writer.writeheader()
        for seed in sorted(results):
            for event in results[seed]["findings"]:
                writer.writerow({"game_status": results[seed]["status"], **event})
    temporary.replace(output / "findings.csv")
    return summary


def run_batch(args, config):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".batch.lock"
    try:
        lock_handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(f"Output is locked: {lock}. If an earlier process crashed, remove this file after checking it is stopped.")
    try:
        with lock_handle:
            lock_handle.write(f"pid={os.getpid()}\n")
        manifest_path = output / "manifest.json"
        manifest = dict(version=1, config=config, start_seed=args.start_seed,
                        sources=source_hashes(config.get("backend", "python")))
        if config.get("backend") == "cpp":
            from predator_stuck_cpp import _stuck
            import numpy
            manifest["native_runtime"] = dict(python=sys.version, numpy=numpy.__version__,
                binary_sha256=hashlib.sha256(Path(_stuck.__file__).read_bytes()).hexdigest())
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError("Output contains a different configuration or simulator version; choose a new --output directory")
        else:
            if (output / "games").exists():
                raise ValueError("Output contains games without a manifest; choose a new --output directory")
            write_json(manifest_path, manifest)
        results, pending_seeds = {}, []
        for seed in range(args.start_seed, args.start_seed + args.games):
            path = output / "games" / f"seed-{seed:08d}" / "result.json"
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous["seed"] != seed:
                    raise ValueError(f"Unexpected seed in {path}")
                if previous["status"] == "complete":
                    results[seed] = compact_result(previous)
                    continue
            pending_seeds.append(seed)
        print(f"{len(results)}/{args.games} games already complete; "
              f"{len(pending_seeds)} to run with {args.workers} workers. Output: {output}", flush=True)
        started = time.monotonic()
        report(output, results, args.games, config, 0)
        if not pending_seeds:
            return 0
        context = multiprocessing.get_context("spawn")
        stop = context.Event()
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                   initializer=init_worker, initargs=(stop,))
        futures = {}
        jobs = iter(pending_seeds)
        interrupted = False
        try:
            def submit_one():
                seed = next(jobs, None)
                if seed is not None:
                    futures[pool.submit(run_game, seed, config, str(output))] = seed

            for _ in range(min(len(pending_seeds), args.workers * 2)):
                submit_one()
            last_report = time.monotonic()
            while futures:
                done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    seed = futures.pop(future)
                    result = future.result()
                    results[seed] = compact_result(result)
                    print(f"Seed {seed}: {result['status']}, {len(result['findings'])} flagged, "
                          f"{result['wall_seconds']:.1f}s wall time", flush=True)
                    if result["status"] == "error":
                        print(result["error"], file=sys.stderr, flush=True)
                    submit_one()
                now = time.monotonic()
                # Rewriting a growing evidence index for every game is quadratic.
                if now - last_report >= 10 or not futures:
                    summary = report(output, results, args.games, config, now - started)
                    print(f"Progress: {summary['completed_games']}/{args.games} complete, "
                          f"{summary['failed_games']} errors, {summary['games_with_findings']} games with findings; "
                          f"{now - started:.0f}s elapsed", flush=True)
                    last_report = now
        except KeyboardInterrupt:
            interrupted = True
            print("Stopping workers; completed games are saved. Rerun the same command to resume.", flush=True)
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
            for future, seed in futures.items():
                if not future.cancelled() and future.done() and future.exception() is None:
                    results[seed] = compact_result(future.result())
            report(output, results, args.games, config, time.monotonic() - started)
        if interrupted:
            return 130
        return 1 if any(r["status"] != "complete" for r in results.values()) else 0
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--predators", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--stuck-seconds", type=float, default=60)
    parser.add_argument("--radius", type=float, default=15)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--diagnostics", action="store_true", help="C++ movement/collision traces, 30s approach, controls and recovery tracking")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "predator-stuck-scan")
    args = parser.parse_args()
    if args.diagnostics and args.backend != "cpp":
        parser.error("--diagnostics requires --backend cpp")
    if min(args.games, args.predators, args.workers) < 1 or args.start_seed < 0:
        parser.error("Game, predator and worker counts must be positive; start seed must be nonnegative")
    if not math.isfinite(args.radius) or args.radius <= 0:
        parser.error("--radius must be positive and finite")
    for name, value in (("seconds", args.seconds), ("stuck-seconds", args.stuck_seconds)):
        if not math.isfinite(value) or value <= 0 or not math.isclose(value / DT, round(value / DT), abs_tol=1e-8):
            parser.error(f"--{name} must be a positive multiple of {DT}")
    if args.seconds < args.stuck_seconds:
        parser.error("Game duration must be at least --stuck-seconds")
    config = dict(predators=args.predators, seconds=args.seconds, stuck_seconds=args.stuck_seconds,
                  backend=args.backend, diagnostics=args.diagnostics,
                  radius=args.radius, dt=DT, images=not args.no_images,
                  world=[1600, 1200], native_ambient_spawns=True,
                  detector="every rolling interval, anchored at its first sampled position")
    try:
        return run_batch(args, config)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
