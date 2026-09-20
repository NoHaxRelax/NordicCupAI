"""Native simulation and detection; Python only serializes finished games."""
from pathlib import Path
import gzip
import json
import math
import time
import traceback

import numpy as np
try:
    from . import _stuck
except ImportError as exc:
    raise ImportError("Build the scanner first: python survival-simulator/scripts/predator_stuck_cpp/build.py") from exc

_stuck.set_numpy_loops(np.sin, np.cos, np.arctan2, np.hypot)


def make_engine(seed, predators=100):
    words = []
    while seed:
        words.append(seed & 0xffffffff)
        seed >>= 32
    return _stuck.Engine(words or [0], 1600, 1200, 400, 0, predators, 32, 50, 0.1, True)


def image_environment(engine):
    """Flat biome colors for evidence crops; never consume simulation RNG."""
    import pygame
    from types import SimpleNamespace
    palette = np.array([(50, 110, 60), (80, 100, 60), (190, 165, 100),
                        (110, 155, 70), (50, 100, 165)], dtype=np.uint8)
    biomes = np.frombuffer(engine.biome_map(), dtype=np.uint8).reshape(1600, 1200)
    return SimpleNamespace(biome_surface=pygame.surfarray.make_surface(palette[biomes]),
                           obstacles=[SimpleNamespace(x=x, y=y, width=w, height=h)
                                      for x, y, w, h in engine.obstacles()])


def run_game(seed, config, output):
    # Import here to keep the native package independently usable for verification.
    import predator_stuck_scan as scan
    folder = Path(output) / "games" / f"seed-{seed:08d}"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(seed=seed, status="running", findings=[], simulated_seconds=0.0)
    try:
        if scan.stop_requested():
            result["status"] = "interrupted"
            return result
        engine = make_engine(seed, config["predators"])
        first_count = len(engine.predators())
        attempts = _stuck.ensure_predators(engine, config["predators"])
        result.update(initial_predators=len(engine.predators()), native_initial_successes=first_count,
                      additional_spawn_attempts=attempts, initial_overlaps=[])
        raw = _stuck.run_scan(engine, round(config["seconds"] / scan.DT),
                              round(config["stuck_seconds"] / scan.DT), config["radius"],
                              config.get("diagnostics", False), 300)
        from .diagnostics import FORMAT, POPULATION_COLUMNS, geometry, summarize
        detailed = config.get("diagnostics", False)
        obstacles = engine.obstacles()
        debug_by_id = {row[0]: row[1:] for row in raw["diagnostics"]}
        catalog = []
        for pid, (tick, x, y, heading, overlap) in enumerate(raw["births"]):
            catalog.append(dict(predator_id=pid, born_seconds=round(tick * scan.DT, 6),
                                initial_position=[x, y], initial_heading=heading,
                                spawned_overlapping_obstacle=overlap))
            if detailed:
                catalog[-1]["motion_totals"] = dict(zip(POPULATION_COLUMNS, raw["population"][pid]))
            if tick == 0 and overlap:
                result["initial_overlaps"].append(pid)
        if raw["events"] or detailed:
            (folder / "events").mkdir(exist_ok=True)
            scan.write_json(folder / "map.json", dict(width=1600, height=1200, obstacles=obstacles,
                biome_file="biomes.bin.gz" if detailed else None,
                biome_layout="uint8, x*1200+y; forest=0, swamp=1, desert=2, grassland=3, river=4"))
        if detailed:
            with gzip.open(folder / "biomes.bin.gz", "wb", compresslevel=1) as handle:
                handle.write(engine.biome_map())
        env = image_environment(engine) if raw["events"] and config["images"] else None
        for pid, biome, size, history in raw["events"]:
            anchor = history[0][1:3]
            event = dict(seed=seed, predator_id=pid, radius=config["radius"],
                         start_seconds=round(history[0][0] * scan.DT, 6),
                         detected_seconds=round(history[-1][0] * scan.DT, 6),
                         duration_seconds=round((history[-1][0] - history[0][0]) * scan.DT, 6),
                         anchor=list(anchor), end_position=list(history[-1][1:3]),
                         max_distance=max(math.hypot(s[1] - anchor[0], s[2] - anchor[1]) for s in history),
                         awake_samples=sum(not s[5] for s in history), sample_count=len(history),
                         predator_size=size, biome=biome,
                         **{k: v for k, v in catalog[pid].items() if k != "predator_id"})
            trace_path = folder / "events" / f"predator-{pid:04d}.json.gz"
            event["trace"] = trace_path.relative_to(Path(output)).as_posix()
            detail, exit_tick, max_after, final_x, final_y = debug_by_id[pid]
            event["follow_up"] = dict(escaped_after_detection=exit_tick >= 0,
                first_exit_seconds=round(exit_tick * scan.DT, 6) if exit_tick >= 0 else None,
                observation_end_seconds=config["seconds"], final_position=[final_x, final_y],
                max_distance_from_anchor_after_detection=max_after)
            if detailed:
                event["diagnostic_summary"] = summarize(detail, history[0][0])
                event["onset_geometry"] = geometry(anchor, size, obstacles)
                event["detection_geometry"] = geometry(history[-1][1:3], size, obstacles)
                event["approach_start_seconds"] = round(detail[0][0] * scan.DT, 6)
            trace = event | dict(dt=scan.DT, sample_columns=["tick", "x", "y", "heading", "energy", "resting"],
                                 samples=history)
            if detailed:
                trace.update(diagnostic_format=FORMAT, diagnostic_rows=detail)
            temporary = trace_path.with_suffix(".gz.tmp")
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
                json.dump(trace, handle, allow_nan=False, separators=(",", ":"))
            temporary.replace(trace_path)
            if env is not None:
                png = trace_path.with_suffix("").with_suffix(".png")
                scan.event_image(env, trace, png)
                event["image"] = png.relative_to(Path(output)).as_posix()
            result["findings"].append(event)
        if detailed and raw["control"] is not None:
            pid, detail = raw["control"]
            control_path = folder / "unflagged-control.json.gz"
            with gzip.open(control_path, "wt", encoding="utf-8", compresslevel=1) as handle:
                json.dump(dict(seed=seed, predator_id=pid, diagnostic_format=FORMAT, diagnostic_rows=detail,
                               selection="Lowest-index predator with no qualifying confinement interval in the full game"),
                          handle, separators=(",", ":"), allow_nan=False)
            result["control_trace"] = control_path.relative_to(Path(output)).as_posix()
        result.update(status="complete", simulated_seconds=config["seconds"],
                      final_predators=len(engine.predators()), total_tracked=len(catalog), predators=catalog)
    except Exception:
        result.update(status="error", error=traceback.format_exc())
    finally:
        result["wall_seconds"] = round(time.monotonic() - started, 3)
        scan.write_json(folder / "result.json", result)
    return result
