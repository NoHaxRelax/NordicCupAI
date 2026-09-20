"""Verify and report the expanded-local-food policy speed optimization."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def ci(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (10_000, len(values)))].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def main():
    folder = ROOT / "docs/sharedfood-speed"
    before = json.loads((folder / "pc-before-100.json").read_text())
    after = json.loads((folder / "pc-optimized-100.json").read_text())
    optimized = json.loads((folder / "optimized-1000.json").read_text())

    assert [row["seed"] for row in before["rows"]] == [row["seed"] for row in after["rows"]]
    exact = [
        left["score"] == right["score"]
        and left["survival"] == right["survival"]
        and left["steps"] == right["steps"]
        for left, right in zip(before["rows"], after["rows"])
    ]
    assert all(exact)
    assert optimized["games"] == 1000

    original_rows = []
    for path in sorted((ROOT / "docs/sharedfood20/final").glob("shard-*/games.jsonl")):
        for line in path.open():
            row = json.loads(line)
            if row["model"] == "expanded_food_baseline":
                original_rows.append(row)
    original_rows.sort(key=lambda row: row["seed"])
    assert len(original_rows) == 1000

    old_scores = [row["score"] for row in original_rows]
    new_scores = [row["score"] for row in optimized["rows"]]
    result = {
        "pc_exact_games": sum(exact),
        "pc_games": len(exact),
        "pc_before": {key: value for key, value in before.items() if key != "rows"},
        "pc_after": {key: value for key, value in after.items() if key != "rows"},
        "policy_speedup_percent": 100 * (1 - after["policy_us_per_tick"] / before["policy_us_per_tick"]),
        "loop_cpu_speedup_percent": 100 * (1 - after["loop_cpu_us_per_tick"] / before["loop_cpu_us_per_tick"]),
        "wall_speedup_percent": 100 * (1 - after["mean_seconds_per_game"] / before["mean_seconds_per_game"]),
        "original_runpod_score": {
            "mean": float(np.mean(old_scores)),
            "ci95": ci(old_scores, 20260920),
        },
        "optimized_laptop_score": {
            "mean": float(np.mean(new_scores)),
            "ci95": ci(new_scores, 20260921),
        },
        "optimized_laptop_timing": {key: value for key, value in optimized.items() if key != "rows"},
        "cross_cpu_exact_scores": sum(a["score"] == b["score"] for a, b in zip(original_rows, optimized["rows"])),
    }
    (folder / "expanded-verification.json").write_text(json.dumps(result, indent=2) + "\n")

    old = result["pc_before"]
    new = result["pc_after"]
    score = result["optimized_laptop_score"]
    lines = [
        "# Expanded local food: policy speed optimization",
        "",
        "The game engine and policy parameters are unchanged. The optimized policy reuses temporary buffers, avoids rebuilding equivalent visibility sets, skips visibility work for objects already observed on the current tick, and reverses tree/fruit counting to visit each pair once.",
        "",
        "## Controlled speed comparison",
        "",
        "Same 100 seeds and SSH PC, with 12 workers for each frozen build. Every optimized game reproduced the original score, lifetime and number of simulation steps exactly.",
        "",
        "| Build | Policy µs/tick | Engine µs/tick | Whole-loop CPU µs/tick | Wall seconds/game |",
        "|---|---:|---:|---:|---:|",
        f'| Original | {old["policy_us_per_tick"]:.1f} | {old["engine_us_per_tick"]:.1f} | {old["loop_cpu_us_per_tick"]:.1f} | {old["mean_seconds_per_game"]:.2f} |',
        f'| Optimized | {new["policy_us_per_tick"]:.1f} | {new["engine_us_per_tick"]:.1f} | {new["loop_cpu_us_per_tick"]:.1f} | {new["mean_seconds_per_game"]:.2f} |',
        "",
        f'Policy time fell **{result["policy_speedup_percent"]:.1f}%** and whole-loop CPU time fell **{result["loop_cpu_speedup_percent"]:.1f}%**. Engine time is statistically noisy under concurrent load and was not optimized.',
        "",
        "## Score check",
        "",
        f'The optimized build scored **{score["mean"]:.1f}** over 1,000 full games on the laptop; its pointwise bootstrap 95% interval is **{score["ci95"][0]:.1f}–{score["ci95"][1]:.1f}**. The earlier unchanged-policy Runpod panel scored **{result["original_runpod_score"]["mean"]:.1f}**. CPU families can produce different floating-point trajectories, so the 100-game same-host exact comparison is the clean regression check.',
        "",
        "A tick is one update of the whole population. Timing includes concurrent-worker scheduling and should be compared within the same host/panel.",
    ]
    (folder / "EXPANDED_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
