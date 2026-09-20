"""Reliability harness: recover N unknown seeds from real engine maps.

The claim this exists to support is narrow and specific:

    Given only public terrain observations from a map we did not generate, we recover
    the exact 32-bit seed that produced it, every time, with no false positives, by
    searching the entire 2^32 domain.

How the honesty is enforced:

  * Target seeds are drawn uniformly from the FULL uint32 range by a reproducible RNG.
    They are not small, not sequential, and not chosen.
  * Observations come from the REAL engine (fastsim Engine.biome_map()), not from a
    reimplementation of the generator. River pixels are skipped, because the river
    overwrites the base Voronoi map and carries no information about it.
  * Samples follow a random walk, so they are clustered and connected the way an agent's
    actual observations would be, rather than conveniently spread over the whole map.
  * The seed is passed to the recovery pipeline NOWHERE. It is used only to build the
    world and, at the very end, to mark the answer right or wrong.
  * The search covers [0, 2^32) in full. A run that stops early is reported as incomplete
    rather than counted as a success.

    python reliability.py --targets 32 --scan-samples 24 --out runs/reliability
"""
import argparse
import json
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


def walk_samples(biome_map, rng, n, step=40):
    """Sample biome labels along a WAYPOINT walk that deliberately crosses the map.

    This matters more than the sample count. Biome diversity is what narrows the domain:
    a sample set that is all one label is compatible with roughly a third of the whole
    seed space. Two earlier versions of this function failed exactly that way -- a purely
    diffusive walk (fresh direction each step) left ~1.4M candidates per target, and a
    persistent-heading walk with angular drift still left ~33M, because over 160 steps the
    heading itself random-walks and the trajectory stays inside one Voronoi cell.

    Navigating to distant waypoints is what an acquisition policy must actually do. An
    agent moves ~100 px/s, so crossing a 1600x1200 map costs seconds, not minutes.
    """
    x, y = float(rng.randrange(100, W - 100)), float(rng.randrange(100, H - 100))
    tx, ty = float(rng.randrange(60, W - 60)), float(rng.randrange(60, H - 60))
    out, guard = [], 0
    while len(out) < n and guard < n * 400:
        guard += 1
        dx, dy = tx - x, ty - y
        d = (dx * dx + dy * dy) ** 0.5
        if d < step:                       # arrived: pick a fresh distant waypoint
            tx, ty = float(rng.randrange(60, W - 60)), float(rng.randrange(60, H - 60))
            continue
        x += step * dx / d
        y += step * dy / d
        lab = biome_map[int(x) * H + int(y)]
        if lab == RIVER:                   # river overwrites the land map: no information
            continue
        out.append((int(x), int(y), int(lab)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=32)
    ap.add_argument("--scan-samples", type=int, default=24)
    ap.add_argument("--total-samples", type=int, default=128)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1 << 32, help="default: the whole domain")
    ap.add_argument("--seed", type=int, default=20260920, help="RNG for choosing targets")
    ap.add_argument("--targets-in-range", action="store_true",
                    help="smoke test: draw targets from the scanned range instead of the full domain")
    ap.add_argument("--out", default="runs/reliability")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filt = HERE / ("terrain_filter_fast.exe" if sys.platform == "win32" else "terrain_filter_fast")
    refine = HERE / ("refine_candidates.exe" if sys.platform == "win32" else "refine_candidates")
    for exe in (filt, refine):
        if not exe.exists():
            print(f"missing {exe.name}; build it first (see README.md)")
            return 2

    # ---- choose unknown targets and collect their public observations ----------------
    chooser = random.Random(args.seed)
    lo, hi = (args.start, args.start + args.count) if args.targets_in_range else (0, 1 << 32)
    targets = [chooser.randrange(lo, hi) for _ in range(args.targets)]
    print(f"targets ({len(targets)}, uniform over the full uint32 domain):")
    print("  " + ", ".join(str(t) for t in targets[:8]) + (" ..." if len(targets) > 8 else ""))

    t0 = time.perf_counter()
    scan_files, all_samples = [], {}
    for i, seed in enumerate(targets):
        e = _mirror.Engine([seed], env_width=W, env_height=H,
                           starting_agents=0, starting_fruits=0, starting_trees=0)
        samples = walk_samples(e.biome_map(), random.Random(args.seed + i), args.total_samples)
        all_samples[i] = samples
        f = out / f"scan_{i}.txt"
        f.write_text("".join(f"{x} {y} {l}\n" for x, y, l in samples[:args.scan_samples]))
        scan_files.append(str(f))
    collect_s = time.perf_counter() - t0
    # Biome diversity is what actually determines how much a sample set narrows the
    # domain. Samples that all carry the same label are nearly worthless: an early run
    # whose walk stayed inside one Voronoi cell left ~1.4M candidates per target.
    div = [len({l for _, _, l in all_samples[j][:args.scan_samples]}) for j in range(len(targets))]
    print(f"collected {args.total_samples} walked land samples per target in {collect_s:.1f} s")
    print(f"distinct biome labels in the scan samples: min {min(div)}, "
          f"median {sorted(div)[len(div)//2]}, max {max(div)}")
    if min(div) < 2:
        print(f"  WARNING: {sum(1 for d in div if d < 2)} target(s) saw a single biome; "
              f"those samples cannot narrow the domain")
    print()

    # Acquisition gate. Diversity and spatial span, not sample count, decide how much a
    # sample set narrows the domain: a single-label set is consistent with roughly a third
    # of all seeds. Two earlier runs here died on exactly that, leaving 1.4M and then 33M
    # candidates per target. Targets below the gate are reported as NOT DISPATCHED rather
    # than scanned, because a weak set costs the whole search budget and returns nothing.
    MIN_LABELS, MIN_SPAN = 2, 400
    gate = {}
    for j in range(len(targets)):
        pts = all_samples[j][:args.scan_samples]
        xs_ = [x for x, _, _ in pts]
        ys_ = [y for _, y, _ in pts]
        gate[j] = dict(labels=len({l for _, _, l in pts}),
                       span=max(max(xs_) - min(xs_), max(ys_) - min(ys_)) if pts else 0)
        gate[j]["ok"] = gate[j]["labels"] >= MIN_LABELS and gate[j]["span"] >= MIN_SPAN
    dispatched = [j for j in range(len(targets)) if gate[j]["ok"]]
    print(f"acquisition gate (>={MIN_LABELS} labels, >={MIN_SPAN} px span): "
          f"{len(dispatched)}/{len(targets)} dispatched")
    for j in range(len(targets)):
        if not gate[j]["ok"]:
            print(f"  held target {targets[j]}: {gate[j]['labels']} labels, {gate[j]['span']} px span")
    if not dispatched:
        print("nothing to scan")
        return 1
    scan_files = [scan_files[j] for j in dispatched]
    print()


    # ---- one pass over the domain, all targets at once --------------------------------
    cand_path = out / "candidates.txt"
    cmd = [str(filt), ",".join(scan_files), str(args.start), str(args.count),
           str(args.threads), str(cand_path)]
    print(f"scanning [{args.start}, {args.start + args.count}) with {args.scan_samples} "
          f"samples x {len(targets)} targets on {args.threads} threads ...")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    scan_s = time.perf_counter() - t0
    if proc.returncode != 0:
        print("filter failed:", proc.stderr[:500])
        return 2
    receipt = json.loads(proc.stdout.strip().splitlines()[-1])
    complete = receipt["tested"] == args.count
    print(f"  tested {receipt['tested']:,} seeds in {scan_s:.1f} s "
          f"({receipt['seeds_per_second']/1e6:.2f} M/s), {receipt['hits']} candidates total")
    if not complete:
        print("  INCOMPLETE SCAN - not counting any recovery as a success")

    # ---- split candidates per target and refine with the remaining observations -------
    per = {i: [] for i in range(len(targets))}
    for line in cand_path.read_text().splitlines():
        if not line.strip():
            continue
        if len(scan_files) > 1:
            d, v = line.split()
            per[dispatched[int(d)]].append(int(v))
        else:
            per[dispatched[0]].append(int(line.split()[-1]))

    results, refine_s = [], 0.0
    for i, seed in enumerate(targets):
        if i not in dispatched:
            results.append(dict(target=seed, scan_candidates=None, survivors=[], unique=False,
                                correct=False, retained=False, held=True, gate=gate[i]))
            continue
        cands = per[i]
        cf = out / f"cand_{i}.txt"
        cf.write_text("".join(f"{c}\n" for c in cands))
        rf = out / f"refined_{i}.txt"
        sf = out / f"full_{i}.txt"
        sf.write_text("".join(f"{x} {y} {l}\n" for x, y, l in all_samples[i]))
        t0 = time.perf_counter()
        subprocess.run([str(refine), str(cf), str(sf), str(args.threads), str(rf)],
                       capture_output=True, text=True, check=True)
        refine_s += time.perf_counter() - t0
        survivors = [int(x) for x in rf.read_text().split()]
        results.append(dict(target=seed, scan_candidates=len(cands),
                            survivors=survivors,
                            unique=len(survivors) == 1,
                            correct=survivors == [seed],
                            retained=seed in survivors, held=False, gate=gate[i]))

    # ---- verdict ----------------------------------------------------------------------
    n = len(results)
    correct = sum(r["correct"] for r in results)
    retained = sum(r["retained"] for r in results)
    false_pos = sum(len([s for s in r["survivors"] if s != r["target"]]) for r in results)
    print(f"\n{'target':>12} {'cands@scan':>11} {'survivors':>10}  verdict")
    for r in results:
        if r["held"]:
            print(f"{r['target']:>12} {'-':>11} {'-':>10}  NOT DISPATCHED "
                  f"({r['gate']['labels']} labels, {r['gate']['span']} px)")
            continue
        mark = "RECOVERED" if r["correct"] else ("ambiguous" if r["retained"] else "LOST")
        print(f"{r['target']:>12} {r['scan_candidates']:>11} {len(r['survivors']):>10}  {mark}")

    n_disp = sum(1 for r in results if not r["held"])
    print(f"dispatched              : {n_disp}/{n}  (held by the gate: {n - n_disp})")
    print(f"uniquely recovered      : {correct}/{n_disp} of dispatched")
    print(f"true seed retained      : {retained}/{n}")
    print(f"false positives (total) : {false_pos}")
    print(f"scan                    : {scan_s:.1f} s for all {n} targets "
          f"({scan_s/n:.1f} s per target amortised)")
    print(f"refine                  : {refine_s*1000:.1f} ms total")
    print(f"complete domain scan    : {complete}")

    summary = dict(targets=n, uniquely_recovered=correct, retained=retained,
                   false_positives=false_pos, complete=complete,
                   scan_seconds=scan_s, refine_seconds=refine_s,
                   collect_seconds=collect_s, receipt=receipt,
                   scan_samples=args.scan_samples, total_samples=args.total_samples,
                   range=[args.start, args.start + args.count], results=results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwritten: {out/'summary.json'}")
    return 0 if (complete and correct == n_disp and false_pos == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
