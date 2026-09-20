"""How fast can agents gather enough terrain evidence to start a seed scan?

Time-to-seed decomposes as:

    T_seed = T_acquire + T_scan + T_refine

T_scan is 324 s on a 20-thread laptop and 19.2 s on V4's 6-shard cluster; T_refine is
milliseconds. V4's live recovery was 132 s end to end, so T_acquire was ~113 s of it --
by far the largest term, and the one nobody has optimised.

V4 waits for 128 anchored samples across 3 biomes (or 400 across 2 after 180 s). That
threshold is far more than a scan needs, because of the scan-early-then-refine result:
~12 diverse samples already cut the full domain to ~10^5 candidates, and every later
sample then narrows that list for milliseconds instead of a fresh sweep. So the real
question is: how soon can five agents see enough *distinct biomes over enough spread*?

This measures that floor. It drives the real engine with an outward-fanning exploration
policy, records each agent's public `biome` label every tick, and reports when the pooled
evidence crosses the thresholds a scan actually needs.

Positions here are the engine's true ones, so this is a LOWER BOUND: a live agent must
localise itself. Relative displacement is cheap (movement is deterministic given our
commands and the observed biome penalty), but the absolute offset needs wall anchoring.
Read the numbers as "acquisition cannot be faster than this", not as a live estimate.

    PYTHONPATH=survival-simulator python seed-search-fast/acquisition_curve.py
"""
import argparse
import json
import math
import pathlib
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fastsim import _mirror  # noqa: E402

_mirror.set_numpy_loops(np.sin, np.cos, np.arctan2, np.hypot)

W, H = 1600, 1200
RIVER = 4
HERE = pathlib.Path(__file__).resolve().parent
CHECKPOINTS = (2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180)


def fan_out_actions(state, headings):
    """Exploration policy: each agent walks a fixed outward heading at full speed.

    Movement and turning consume zero RNG (fastsim/_engine.cpp:1067-1098), so this cannot
    perturb the stream it is trying to observe. Agents bounce off the map edges.
    """
    acts = []
    for a in state["observations"]:
        aid = int(a["agent_id"])
        acts.append((aid, {
            "move_distance": a["speed"],
            "move_direction": headings.get(aid, 0.0),
            "turn_angle": 0.0,
            "spawn_agent": False,
        }))
    return acts


def collect(seed, seconds, n_agents=5):
    """Run one game, recording (x, y, label) per agent per tick from the true positions."""
    e = _mirror.Engine([seed], env_width=W, env_height=H, starting_agents=n_agents,
                       starting_fruits=40, starting_trees=40)
    st = e.state()
    ids = [int(a["agent_id"]) for a in st["observations"]]
    # fan out evenly so the pooled samples span the map instead of clustering
    headings = {aid: 2 * math.pi * k / max(1, len(ids)) for k, aid in enumerate(ids)}
    samples = []          # (tick, x, y, label)
    ticks = int(seconds * 10)
    for t in range(ticks):
        agents = {int(a[0]): a for a in e.agents()}
        for a in st["observations"]:
            aid = int(a["agent_id"])
            if aid not in agents:
                continue
            x, y = agents[aid][1], agents[aid][2]
            lab = e.biome_map()[int(x) * H + int(y)]
            if lab != RIVER:               # river overwrites the land map: no information
                samples.append((t, int(x), int(y), int(lab)))
        # bounce: re-aim any agent that has run into the boundary band
        for aid, a in agents.items():
            if a[1] < 60 or a[1] > W - 60 or a[2] < 60 or a[2] > H - 60:
                headings[aid] = headings.get(aid, 0.0) + math.pi / 2 + 0.3
        st = e.step(fan_out_actions(st, headings))
        if st["num_agents"] == 0:
            break
    return samples


def thin(samples, min_sep=60):
    """Consecutive ticks are 10 px apart and nearly redundant. Keep a spatially spread
    subset, which is what actually constrains the Voronoi map."""
    kept = []
    for _, x, y, l in samples:
        if all((x - kx) ** 2 + (y - ky) ** 2 >= min_sep * min_sep for kx, ky, _ in kept):
            kept.append((x, y, l))
    return kept


def evidence(kept):
    if not kept:
        return 0, 0, 0
    xs = [x for x, _, _ in kept]
    ys = [y for _, y, _ in kept]
    return len(kept), len({l for _, _, l in kept}), max(max(xs) - min(xs), max(ys) - min(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--probe-frac", type=float, default=0.01,
                    help="fraction of the domain to scan when projecting candidate counts")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--out", default="runs/acquisition")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filt = HERE / ("terrain_filter_fast.exe" if sys.platform == "win32" else "terrain_filter_fast")
    if not filt.exists():
        print("build terrain_filter_fast first (see README.md)")
        return 2

    chooser = random.Random(31337)
    targets = [chooser.randrange(0, 1 << 32) for _ in range(args.seeds)]
    probe_n = int((1 << 32) * args.probe_frac)

    print(f"{args.agents} agents fanning outward, {args.seeds} seeds, "
          f"projecting full-domain candidates from a {args.probe_frac*100:.0f}% probe\n")
    print(f"{'sim s':>6} {'samples':>8} {'biomes':>7} {'span px':>8} {'proj. candidates over 2^32':>28}")

    rows = {c: [] for c in CHECKPOINTS}
    for seed in targets:
        samples = collect(seed, max(CHECKPOINTS), args.agents)
        for c in CHECKPOINTS:
            upto = [s for s in samples if s[0] <= c * 10]
            kept = thin(upto)
            n, labels, span = evidence(kept)
            proj = None
            if labels >= 2 and n >= 4:
                f = out / f"acq_{seed}_{c}.txt"
                f.write_text("".join(f"{x} {y} {l}\n" for x, y, l in kept))
                p = subprocess.run([str(filt), str(f), "0", str(probe_n), str(args.threads),
                                    str(out / "acq_cand.txt")], capture_output=True, text=True)
                if p.returncode == 0:
                    hits = json.loads(p.stdout.strip().splitlines()[-1])["hits"]
                    proj = hits / args.probe_frac
                f.unlink(missing_ok=True)
            rows[c].append(dict(seed=seed, n=n, labels=labels, span=span, proj=proj))

    summary = []
    for c in CHECKPOINTS:
        rs = rows[c]
        med = lambda k: sorted(r[k] for r in rs)[len(rs) // 2]  # noqa: E731
        projs = [r["proj"] for r in rs if r["proj"] is not None]
        pstr = "-" if not projs else f"{sorted(projs)[len(projs)//2]:,.0f}"
        ready = sum(1 for r in rs if r["proj"] is not None and r["proj"] <= 5e6)
        print(f"{c:>6} {med('n'):>8} {med('labels'):>7} {med('span'):>8} {pstr:>20}"
              f"   scan-ready {ready}/{len(rs)}")
        summary.append(dict(sim_seconds=c, median_samples=med("n"), median_labels=med("labels"),
                            median_span=med("span"),
                            median_projected_candidates=(sorted(projs)[len(projs)//2] if projs else None),
                            scan_ready=ready, of=len(rs)))

    (out / "acquisition_summary.json").write_text(json.dumps(
        dict(agents=args.agents, seeds=targets, probe_frac=args.probe_frac,
             checkpoints=summary), indent=2) + "\n")
    print(f"\nwritten: {out/'acquisition_summary.json'}")
    print("\nNote: true positions are used, so this is the acquisition FLOOR. A live agent")
    print("must localise; relative displacement is cheap, the absolute offset is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
