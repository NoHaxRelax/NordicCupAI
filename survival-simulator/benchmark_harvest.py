"""Matched-seed predator-free comparison. Truth is confined to diagnostics."""

import argparse
import json
from pathlib import Path
import statistics
import time

from src.core import SimulationCore
from src.utils.controllers.expert_policy import ExpertPolicy, load_config
from src.utils.controllers.population import TRAITS


ATTRIBUTES = {"vision_range": "vision_radius", "vision_angle": "cone_angle"}


def write_comparison(output):
    pairs = []
    for path in sorted(output.glob("baseline-seed-*.json")):
        baseline = json.loads(path.read_text(encoding="utf-8"))
        central_path = output / f"centralized-seed-{baseline['seed']}.json"
        if central_path.exists():
            central = json.loads(central_path.read_text(encoding="utf-8"))
            pairs.append((baseline, central))
    if not pairs:
        return
    lines = ["# Predator-free harvesting comparison", "",
             "Both policies receive public observations only. True fruit energies and traits are used for reporting.", "",
             "| Seed | Policy | Seconds | Score | Fruit bonus | Mean fruit energy | Population | Mean living trait score |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for pair in pairs:
        for row in pair:
            traits = "extinct" if row["living_trait_score"] is None else f"{row['living_trait_score']:.3f}"
            lines.append(f"| {row['seed']} | {row['policy']} | {row['seconds']:.0f} | {row['score']:.3f} | "
                         f"{row['fruit_score']:.3f} | {row['mean_fruit_energy']:.2f} | {row['population']} | {traits} |")
    baseline_score = statistics.mean(pair[0]["score"] for pair in pairs)
    central_score = statistics.mean(pair[1]["score"] for pair in pairs)
    baseline_food = sum(pair[0]["fruit_score"] for pair in pairs)
    central_food = sum(pair[1]["fruit_score"] for pair in pairs)
    lines += ["", f"Mean total score: {baseline_score:.3f} → {central_score:.3f} "
              f"({100 * (central_score / baseline_score - 1):+.2f}%).",
              f"Combined fruit bonus: {baseline_food:.3f} → {central_food:.3f} "
              f"({100 * (central_food / baseline_food - 1):+.2f}%)." if baseline_food else "No baseline fruit was collected.",
              "", "These are short seeded checks, not a guarantee of a higher score on every map or over the 3000-second horizon. "
              "Births and other policy choices change subsequent random draws. A trait score of 1.0 is the founder average.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(seed, seconds, centralized, output, coverage=None):
    config = load_config()
    config = config.model_copy(update={"harvest": config.harvest.model_copy(update={"enabled": centralized})})
    if coverage is not None:
        config = config.model_copy(update={"harvest": config.harvest.model_copy(update={
            "coverage": config.harvest.coverage.model_copy(update={"enabled": coverage})})})
    policy = ExpertPolicy(config)
    sim = SimulationCore(seed=seed, predators_enabled=False)
    baseline = {trait: statistics.mean(getattr(a, ATTRIBUTES.get(trait, trait)) for a in sim.env.agents) for trait in TRAITS}
    energies, birth_scores, parent_scores, death_causes = [], [], [], {}
    births = 0
    rotted = 0
    visited = set()
    from benchmark_mapping import free_cell_areas, coverage_percent
    free_areas = free_cell_areas(sim.env, 120)

    def score_traits(agent):
        return statistics.mean(getattr(agent, ATTRIBUTES.get(trait, trait)) / baseline[trait] for trait in TRAITS)

    def event(kind, **data):
        nonlocal births, rotted
        if kind == "fruit_eaten":
            energies.append(data["fruit"].energy)
        elif kind == "birth":
            births += 1
            birth_scores.append(score_traits(data["agent"]))
            if data.get("parent") is not None:
                parent_scores.append(score_traits(data["parent"]))
        elif kind == "death":
            cause = data["cause"]
            death_causes[cause] = death_causes.get(cause, 0) + 1
        elif kind == "fruit_rot":
            rotted += 1

    sim.env.event_sink = event
    actions, trace, costs = [], [], []
    next_sample = 0.
    started = time.perf_counter()
    while sim.env.time < seconds - 1e-9 and sim.env.agents:
        state = sim.step(actions)
        before = time.perf_counter()
        decisions = policy.actions_for_step(state["observations"], state["sim_time"])
        costs.append(time.perf_counter() - before)
        actions = [(a.agent_id, a) for a in decisions]
        visited.update((int(a.x // 120), int(a.y // 120)) for a in sim.env.agents)
        if sim.env.time >= next_sample - 1e-9:
            trace.append(dict(time=sim.env.time, score=sim.env.score, population=len(sim.env.agents),
                              tracks=len(policy.harvest.tracks), assignments=len(policy.harvest.assignments),
                              population_target=policy.harvest.population_target))
            trace[-1]["visited_area_percent"] = coverage_percent(visited, free_areas)
            next_sample += 30
            print(f"seed={seed} {'central' if centralized else 'baseline'} t={sim.env.time:.0f}s "
                  f"population={len(sim.env.agents)} score={sim.env.score:.2f}", flush=True)
    result = dict(seed=seed, policy="centralized" if centralized else "baseline", seconds=sim.env.time,
                  score=sim.env.score, fruit_score=sum(energies) / 1000, fruit_eaten=len(energies),
                  mean_fruit_energy=statistics.mean(energies) if energies else 0,
                  ripe_fruit_percent=100 * sum(e >= 56 for e in energies) / max(1, len(energies)),
                  population=len(sim.env.agents), births=births, deaths=death_causes,
                  fruit_rotted=rotted, visited_area_percent=coverage_percent(visited, free_areas),
                  coverage_enabled=config.harvest.coverage.enabled,
                  survey=policy.harvest.coverage.snapshot().get("survey_percent"),
                  living_trait_score=statistics.mean(score_traits(a) for a in sim.env.agents) if sim.env.agents else None,
                  mean_child_trait_score=statistics.mean(birth_scores) if birth_scores else None,
                  mean_parent_trait_score=statistics.mean(parent_scores) if parent_scores else None,
                  controller_mean_ms=1000 * statistics.mean(costs), wall_seconds=time.perf_counter() - started,
                  trace=trace)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{result['policy']}-seed-{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if centralized:
        (output / f"planner-seed-{seed}.json").write_text(json.dumps(policy.planner.snapshot(), indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42])
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--policy", choices=["both", "baseline", "centralized"], default="both")
    parser.add_argument("--output", type=Path, default=Path("runs/harvest-benchmark"))
    parser.add_argument("--coverage", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3000:
        parser.error("--seconds must be in (0, 3000]")
    for seed in args.seeds:
        for central in (False, True):
            if args.policy in ("both", "centralized" if central else "baseline"):
                run(seed, args.seconds, central, args.output, coverage=args.coverage)
    write_comparison(args.output)
