#!/usr/bin/env python3
"""Compare two completed native evaluations seed by seed."""

import argparse
import json
import math
import pathlib
import random
import statistics


def rows(folder):
    values = [json.loads(line) for line in (folder / "games.jsonl").read_text().splitlines()
              if line.strip()]
    by_seed = {row["seed"]: row for row in values}
    if len(by_seed) != len(values):
        raise ValueError(f"duplicate seeds in {folder}")
    return by_seed


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values):
    return dict(mean=statistics.fmean(values), median=statistics.median(values),
                min=min(values), max=max(values),
                std=statistics.stdev(values) if len(values) > 1 else 0.0,
                p10=percentile(values, .1), p90=percentile(values, .9))


def paired_ci(values, seed=20260919, samples=20000):
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return [percentile(means, .025), percentile(means, .975)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=pathlib.Path)
    parser.add_argument("entrapment", type=pathlib.Path)
    args = parser.parse_args()
    baseline, entrapment = rows(args.baseline), rows(args.entrapment)
    if set(baseline) != set(entrapment):
        missing_left = sorted(set(entrapment) - set(baseline))
        missing_right = sorted(set(baseline) - set(entrapment))
        raise ValueError(f"seed sets differ; baseline missing {missing_left}, entrapment missing {missing_right}")
    seeds = sorted(baseline)
    score_base = [baseline[seed]["score"] for seed in seeds]
    score_trap = [entrapment[seed]["score"] for seed in seeds]
    score_delta = [right - left for left, right in zip(score_base, score_trap)]
    time_base = [baseline[seed]["simulated_seconds"] for seed in seeds]
    time_trap = [entrapment[seed]["simulated_seconds"] for seed in seeds]
    time_delta = [right - left for left, right in zip(time_base, time_trap)]
    output = {
        "games": len(seeds), "seed_min": seeds[0], "seed_max": seeds[-1],
        "baseline_score": describe(score_base), "entrapment_score": describe(score_trap),
        "paired_score_delta_entrapment_minus_baseline": {
            **describe(score_delta), "mean_95pct_bootstrap_ci": paired_ci(score_delta),
            "entrapment_wins": sum(value > 0 for value in score_delta),
            "ties": sum(value == 0 for value in score_delta),
            "baseline_wins": sum(value < 0 for value in score_delta),
        },
        "baseline_duration": describe(time_base), "entrapment_duration": describe(time_trap),
        "paired_duration_delta_entrapment_minus_baseline": {
            **describe(time_delta), "mean_95pct_bootstrap_ci": paired_ci(time_delta, 20260920),
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
