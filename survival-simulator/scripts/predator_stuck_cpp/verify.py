"""Compare native ticks and scan output to the current Python engine.

python survival-simulator/scripts/predator_stuck_cpp/verify.py --seconds 65 --seeds 0 1
Run on each target platform; cross-platform floating-point identity is not assumed.
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import predator_stuck_scan as scan
from predator_stuck_cpp import _stuck, make_engine


def detector_check():
    rng = random.Random(18)
    for radius in (1, 5, 15):
        for steps in (1, 7, 60, 600):
            x = y = 0.0
            points = []
            for tick in range(1300):
                if tick % 70 < 25:
                    x += rng.uniform(-3, 3)
                    y += rng.uniform(-3, 3)
                if tick % 101 == 0:
                    x += 40
                points.append((x, y))
            tracker = scan.ResidenceWindow(steps, radius)
            expected = [tracker.add((tick, x, y, 0, 0, False)) for tick, (x, y) in enumerate(points)]
            assert _stuck.detect_path(points, steps, radius) == expected, (radius, steps)
    for points, expected in [([(0., 0.), (12., 12.), (0., 0.)], False),
                             ([(0., 0.), (14., 0.), (0., 14.)], True),
                             ([(0., 0.), (40., 0.), (0., 0.)], False)]:
        assert _stuck.detect_path(points, 2, 15)[-1] is expected
    assert _stuck.detect_path([(0., 0.)] * 601, 600, 15) == [False] * 600 + [True]


def verify_seed(seed, seconds, predators):
    from src.core import SimulationCore
    started = time.monotonic()
    core = SimulationCore(seed=seed, starting_agents=0, starting_predators=predators)
    env = core.env
    native = make_engine(seed, predators)
    assert len(env.predators) == len(native.predators()), "Initial spawn count"
    while len(env.predators) < predators:
        env.spawn_predator()
    _stuck.ensure_predators(native, predators)
    assert native.obstacles() == [(o.x, o.y, o.width, o.height) for o in env.obstacles], "Map obstacles"
    names = {name: i for i, name in enumerate(("forest", "swamp", "desert", "grassland", "river"))}
    assert native.biome_map() == bytes(names[b.type] for b in env.biome_map.flat), "Biome map"
    windows, expected = [], {}
    ticks = round(seconds / scan.DT)
    last_print = started
    for tick in range(ticks + 1):
        if tick:
            env.non_agent_step(scan.DT)
            native.step([])
        py = [(p.x, p.y, p.direction, p.energy, p.resting) for p in env.predators]
        cpp = native.predators()
        if py != cpp:
            mismatch = next(((i, a, b) for i, (a, b) in enumerate(zip(py, cpp)) if a != b), None)
            raise AssertionError(f"seed={seed} tick={tick} predator mismatch: {mismatch}; counts={len(py)}/{len(cpp)}")
        assert env.rng.getstate() == native.rng_state(), f"seed={seed} tick={tick} RNG"
        assert not env.agents and not native.agents()
        while len(windows) < len(py):
            windows.append(scan.ResidenceWindow(600, 15))
        for pid, p in enumerate(py):
            if pid not in expected and windows[pid].add((tick, *p)):
                expected[pid] = list(windows[pid].history)
        if time.monotonic() - last_print > 10:
            print(f"Parity seed {seed}: {tick * scan.DT:.1f}/{seconds:g}s matched", flush=True)
            last_print = time.monotonic()
    bulk = make_engine(seed, predators)
    _stuck.ensure_predators(bulk, predators)
    raw = _stuck.run_scan(bulk, ticks, 600, 15, True, 300)
    actual = {pid: samples for pid, biome, size, samples in raw["events"]}
    assert actual == expected, "Bulk scanner differs from Python rolling windows"
    assert bulk.predators() == native.predators() and bulk.rng_state() == native.rng_state(), "Bulk final state"
    from predator_stuck_cpp.diagnostics import DETAIL_COLUMNS, INDEX
    import numpy as np
    for pid, rows, *_ in raw["diagnostics"]:
        for row in rows:
            assert len(row) == len(DETAIL_COLUMNS)
            ix = INDEX
            if row[ix["mode"]] <= 0:
                continue
            accepted = int(row[ix["accepted_candidate"]])
            attempts = int(row[ix["candidate_tests"]])
            expected_mask = (1 << (accepted if accepted >= 0 else attempts)) - 1
            assert int(row[ix["rejected_candidate_mask"]]) == expected_mask
            assert row[3] == row[ix["before_heading"]] + row[ix["turn"]]
            x, y = row[ix["before_x"]], row[ix["before_y"]]
            if accepted >= 0:
                angle = row[ix["absolute_move_direction"]]
                if accepted:
                    i = accepted - 1
                    angle += (math.pi / 18) * ((i + 1) // 2) * (-1 if i % 2 else 1)
                x += row[ix["scaled_distance"]] * np.cos(angle)
                y += row[ix["scaled_distance"]] * np.sin(angle)
            assert abs(row[1] - max(10, min(1590, x))) < 1e-12
            assert abs(row[2] - max(10, min(1190, y))) < 1e-12
    result = dict(seed=seed, seconds=seconds, starting_predators=predators,
                  final_predators=len(py), findings=len(expected), exact_ticks=ticks + 1,
                  wall_seconds=round(time.monotonic() - started, 3))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=65)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--predators", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seconds < 60:
        parser.error("At least 60 seconds required to verify confinement events")
    detector_check()
    print("Detector boundary, excursion and randomized path checks passed", flush=True)
    results = [verify_seed(seed, args.seconds, args.predators) for seed in args.seeds]
    if args.output:
        scan.write_json(args.output, dict(status="passed", tests=results, sources=scan.source_hashes("cpp")))


if __name__ == "__main__":
    main()
