"""Local, observational run telemetry. Never feeds hidden state to the policy."""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import uuid

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[2]
HORIZON_SECONDS = 3000.0
TRAITS = ("speed", "sprint_speed", "max_energy", "hearing_radius", "vision_range", "vision_angle")
TRAIT_ATTRIBUTES = {"vision_range": "vision_radius", "vision_angle": "cone_angle"}


class DiagnosticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    output_directory: str
    trait_sample_interval_seconds: float = Field(gt=0)
    comparison_age_seconds: float = Field(gt=0)
    minimum_cohort_size: int = Field(ge=2, strict=True)
    meaningful_change_fraction: float = Field(gt=0, le=1)
    progress_interval_seconds: float = Field(gt=0)
    plot_dpi: int = Field(ge=72, le=600)
    open_report: bool


def load_diagnostics_config(path=None):
    path = Path(path) if path else ROOT / "config" / "diagnostics.json"
    return DiagnosticsConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _correlation(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RunDiagnostics:
    def __init__(self, sim, config=None, output_directory=None, policy_config=None, planner_config=None):
        self.sim = sim
        self.env = sim.env
        if self.env.event_sink is not None:
            raise ValueError("This environment already has an event recorder")
        self.config = config if config is not None else load_diagnostics_config()
        base = Path(output_directory or self.config.output_directory)
        if not base.is_absolute():
            base = ROOT / base
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = base / f"{stamp}_seed-{sim.seed}_{uuid.uuid4().hex[:8]}"
        self.directory.mkdir(parents=True)
        self.events = (self.directory / "events.jsonl").open("w", encoding="utf-8")
        self.agents = {}
        self.samples = []
        self.bins = {}
        self.food_energy = 0.0
        self.predation_loss = 0.0
        self.predation_bonus = 0.0
        self.fruit_spawned = 0
        self.fruit_eaten = 0
        self.fruit_rotted = 0
        self.living = 0
        self.peak_population = 0
        self.step_start = self.env.time
        self.step_end = self.env.time
        self.last_sample_time = self.env.time
        self.last_decisions = {}
        self._finished = False
        self.metadata = {
            "seed": sim.seed, "python": platform.python_version(),
            "platform": platform.platform(), "diagnostics": self.config.model_dump(),
            "policy": policy_config,
            "planner": planner_config,
            "simulation": {key: getattr(sim, key) for key in (
                "env_width", "env_height", "chunk_size", "starting_agents",
                "starting_predators", "starting_fruits", "starting_trees", "dt",
            )},
        }
        self.metadata["simulation"]["predators_enabled"] = getattr(sim, "predators_enabled", True)
        for args, key in ((["rev-parse", "HEAD"], "git_revision"),
                          (["status", "--porcelain"], "git_status")):
            try:
                result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                        text=True, timeout=5, check=True)
                self.metadata[key] = result.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                self.metadata[key] = None
        for agent in self.env.agents:
            self.on_event("birth", agent=agent, parent=None)
        for fruit in self.env.fruits:
            self.on_event("fruit_spawn", fruit=fruit)
        self.env.event_sink = self.on_event
        self.sample()
        (self.directory / "run_config.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False), encoding="utf-8"
        )

    def on_event(self, event_type, **data):
        event = {"event": event_type, "time": self.step_end}
        if event_type == "birth":
            agent, parent = data["agent"], data["parent"]
            parent_id = parent.agent_id if parent is not None else None
            generation = self.agents[parent_id]["generation"] + 1 if parent_id is not None else 0
            row = {
                "agent_id": agent.agent_id, "parent_id": parent_id, "generation": generation,
                "birth_time": self.step_start, "birth_energy": float(agent.energy),
                "death_time": None, "death_cause": None, "death_energy": None,
                "children": 0, "fruit_count": 0, "fruit_energy": 0.0,
                "children_by_comparison_age": 0, "food_by_comparison_age": 0.0,
                "could_sprint_at_last_decision": None, "saw_predator_at_last_decision": None,
            }
            row.update({trait: float(getattr(agent, TRAIT_ATTRIBUTES.get(trait, trait)))
                        for trait in TRAITS})
            self.agents[agent.agent_id] = row
            if parent_id is not None:
                parent_row = self.agents[parent_id]
                parent_row["children"] += 1
                if self.step_start - parent_row["birth_time"] <= self.config.comparison_age_seconds + 1e-9:
                    parent_row["children_by_comparison_age"] += 1
            self.living += 1
            self.peak_population = max(self.peak_population, self.living)
            event.update(row)
            event["time"] = self.step_start
        elif event_type == "death":
            agent = data["agent"]
            row = self.agents[agent.agent_id]
            row.update(death_time=self.step_end, death_cause=data["cause"],
                       death_energy=float(agent.energy))
            decision = self.last_decisions.pop(agent.agent_id, {})
            row["could_sprint_at_last_decision"] = decision.get("could_sprint")
            row["saw_predator_at_last_decision"] = decision.get("saw_predator")
            if data["cause"] == "predation":
                loss = float(agent.energy) / 100
                self.predation_loss += loss
                # The engine can kill an agent after aging made its energy
                # negative. Preserve that signed score adjustment exactly.
                self.predation_bonus += max(0.0, -loss)
            self.living -= 1
            event.update(agent_id=agent.agent_id, cause=data["cause"],
                         energy=float(agent.energy), last_decision=decision)
        elif event_type == "fruit_spawn":
            self.fruit_spawned += 1
            event.update(fruit_id=data["fruit"].fruit_id)
        elif event_type == "fruit_eaten":
            agent, fruit = data["agent"], data["fruit"]
            energy = float(fruit.energy)
            self.fruit_eaten += 1
            self.food_energy += energy
            row = self.agents[agent.agent_id]
            row["fruit_count"] += 1
            row["fruit_energy"] += energy
            if self.step_end - row["birth_time"] <= self.config.comparison_age_seconds + 1e-9:
                row["food_by_comparison_age"] += energy
            event.update(agent_id=agent.agent_id, fruit_id=fruit.fruit_id, energy=energy)
        elif event_type == "fruit_rot":
            self.fruit_rotted += 1
            event.update(fruit_id=data["fruit"].fruit_id)
        self.events.write(json.dumps(event, allow_nan=False) + "\n")

    def before_step(self, actions):
        self.step_start = self.env.time
        self.step_end = self.env.time + self.sim.dt
        self._food_before = self.food_energy
        self._loss_before = self.predation_loss
        for agent_id, action in actions:
            agent = self.env.agents_dict.get(agent_id)
            if agent is None:
                continue
            observed = self.env.agent_observations.get(agent_id, [])
            self.last_decisions[agent_id] = {
                "energy": float(agent.energy),
                "could_sprint": bool(agent.energy >= agent.max_energy / 5),
                "saw_predator": any(obj["type"] == "Predator" for obj in observed),
                "requested_distance": float(action.move_distance),
            }

    def _bin(self, second):
        return self.bins.setdefault(second, {
            "second_start": second, "observed_seconds": 0.0,
            "survival_gain": 0.0, "food_gain": 0.0, "predation_loss": 0.0,
        })

    def after_step(self):
        # Rounding only aligns floating-point tick times with second boundaries.
        start, end = round(self.step_start, 9), round(self.env.time, 9)
        cursor = start
        while cursor < end:
            second = math.floor(cursor)
            duration = min(end, second + 1) - cursor
            row = self._bin(second)
            row["observed_seconds"] += duration
            row["survival_gain"] += duration
            cursor += duration
        if end > start:
            row = self._bin(math.ceil(end) - 1)
            row["food_gain"] += (self.food_energy - self._food_before) / 1000
            row["predation_loss"] += self.predation_loss - self._loss_before
        if self.env.time - self.last_sample_time + 1e-9 >= self.config.trait_sample_interval_seconds:
            self.sample()

    def sample(self):
        living = self.env.agents
        row = {
            "time": float(self.env.time), "score": float(self.env.score),
            "population": len(living), "predators": len(self.env.predators),
            "fruit_spawned": self.fruit_spawned,
            "food_score": self.food_energy / 1000, "predation_loss": self.predation_loss,
            "energy_mean": _mean([agent.energy for agent in living]),
        }
        for trait in TRAITS:
            values = [float(getattr(agent, TRAIT_ATTRIBUTES.get(trait, trait))) for agent in living]
            row[f"{trait}_mean"] = _mean(values)
            row[f"{trait}_p10"] = float(np.percentile(values, 10)) if values else None
            row[f"{trait}_p90"] = float(np.percentile(values, 90)) if values else None
        self.samples.append(row)
        self.last_sample_time = self.env.time

    def generation_analysis(self):
        age = self.config.comparison_age_seconds
        rows = list(self.agents.values())
        groups = []
        eligible_all = []
        for generation in sorted({row["generation"] for row in rows}):
            cohort = [row for row in rows if row["generation"] == generation]
            # Require the same potential follow-up even for recent early deaths.
            eligible = [row for row in cohort if row["birth_time"] + age <= self.env.time + 1e-9]
            eligible_all.extend(eligible)
            summary = {
                "generation": generation, "born": len(cohort),
                "alive_at_end": sum(row["death_time"] is None for row in cohort),
                "eligible_at_comparison_age": len(eligible),
                "survival_at_comparison_age": _mean([
                    row["death_time"] is None or row["death_time"] >= row["birth_time"] + age - 1e-9
                    for row in eligible
                ]),
                "offspring_by_comparison_age": _mean([row["children_by_comparison_age"] for row in eligible]),
                "food_energy_by_comparison_age": _mean([row["food_by_comparison_age"] for row in eligible]),
            }
            summary.update({f"{trait}_mean": _mean([row[trait] for row in cohort]) for trait in TRAITS})
            groups.append(summary)

        minimum = self.config.minimum_cohort_size
        def compare(first, last):
            if first is None or last is None or first is last:
                return "No later generation is available for comparison."
            if min(first["eligible_at_comparison_age"], last["eligible_at_comparison_age"]) < minimum:
                return f"Insufficient follow-up: need at least {minimum} agents per generation with a full {age:g}-second opportunity."
            survival_delta = last["survival_at_comparison_age"] - first["survival_at_comparison_age"]
            offspring_delta = last["offspring_by_comparison_age"] - first["offspring_by_comparison_age"]
            margin = self.config.meaningful_change_fraction
            offspring_margin = margin * max(1.0, first["offspring_by_comparison_age"])
            if survival_delta >= 0 and offspring_delta >= 0 and (
                survival_delta >= margin or offspring_delta >= offspring_margin
            ):
                return "Later agents performed better on the measured survival/reproduction criteria in this run; this is suggestive, not proof of adaptation."
            if survival_delta <= 0 and offspring_delta <= 0 and (
                survival_delta <= -margin or offspring_delta <= -offspring_margin
            ):
                return "Later agents performed worse on the measured survival/reproduction criteria in this run."
            if survival_delta * offspring_delta < 0:
                return "Mixed result: survival and reproduction changed in opposite directions."
            return "No clear improvement under the configured practical-change thresholds."

        first = groups[0] if groups else None
        last = groups[-1] if groups else None
        assessable = [group for group in groups if group["generation"] > 0
                      and group["eligible_at_comparison_age"] >= minimum]
        latest = assessable[-1] if assessable else None
        fitness = [row["children_by_comparison_age"] for row in eligible_all]
        correlations = {
            trait: _correlation([row[trait] for row in eligible_all], fitness)
            if len(eligible_all) >= minimum else None for trait in TRAITS
        }
        return {
            "comparison_age_seconds": age, "generations": groups,
            "first_generation": first, "last_generation": last,
            "last_generation_verdict": compare(first, last),
            "latest_assessable_generation": latest,
            "latest_assessable_verdict": compare(first, latest),
            "trait_offspring_correlations": correlations,
            "eligible_agents": len(eligible_all),
            "interpretation": (
                "Generation is parent depth; generations overlap. Correlations and cohort changes are descriptive, "
                "not causal tests of natural selection. Food, predators, birth location, relatedness, and starting "
                "energy differ. Founders start at 150 energy; ordinary newborns at 75. Living agents have censored "
                "lifetimes, so raw mean lifespans are not used to rank generations. Trait increases alone do not establish fitness."
            ),
        }

    def finish(self, reason, make_plots=True):
        if self._finished:
            return self.directory
        self._finished = True
        self.env.event_sink = None
        self.events.close()
        if self.samples[-1]["time"] != self.env.time:
            self.sample()
        second_rows = []
        cumulative = 0.0
        for _, row in sorted(self.bins.items()):
            gain = row["survival_gain"] + row["food_gain"] - row["predation_loss"]
            cumulative += gain
            second_rows.append({**row, "score_gain": gain,
                                "score_gain_per_second": gain / row["observed_seconds"],
                                "cumulative_score": cumulative})
        agent_rows = []
        for row in self.agents.values():
            end = self.env.time if row["death_time"] is None else row["death_time"]
            agent_rows.append({**row, "observed_lifetime_seconds": end - row["birth_time"],
                               "lifetime_censored": row["death_time"] is None})
        evolution = self.generation_analysis()
        # Fruit.grow may overshoot 60 by at most one 2*dt growth increment.
        fruit_energy_bound = self.fruit_spawned * (60.0 + 2 * self.sim.dt)
        elapsed_bound = self.env.time + fruit_energy_bound / 1000 + self.predation_bonus
        benchmark = max(HORIZON_SECONDS, self.env.time) + fruit_energy_bound / 1000 + self.predation_bonus
        deaths = [row for row in agent_rows if row["death_time"] is not None]
        predation = [row for row in deaths if row["death_cause"] == "predation"]
        score = float(self.env.score)
        summary = {
            "seed": self.sim.seed, "end_reason": reason, "simulated_seconds": float(self.env.time),
            "final_score": score, "survival_score": float(self.env.time),
            "food_score": self.food_energy / 1000, "predation_loss": self.predation_loss,
            "score_reconstruction_error": score - (self.env.time + self.food_energy / 1000 - self.predation_loss),
            "horizon_seconds": HORIZON_SECONDS,
            "survival_target_percent": min(100.0, 100 * self.env.time / HORIZON_SECONDS),
            "true_theoretical_maximum": None,
            "optimistic_score_benchmark": benchmark,
            "benchmark_percent": 100 * score / benchmark,
            "elapsed_run_score_upper_bound": elapsed_bound,
            "elapsed_bound_percent": 100 * score / elapsed_bound if elapsed_bound else None,
            "fruit_energy_upper_bound_per_spawn": 60 + 2 * self.sim.dt,
            "benchmark_definition": (
                "3000 survival points (including any terminal tick overrun) + a full-ripeness energy allowance "
                "for every fruit spawned in the observed run / 1000 + any observed negative-energy predation bonus. "
                "This is an optimistic reference, not the theoretical optimum: future spawns after an early stop "
                "are unknown, alternate policies can change the random trajectory, and not all food is reachable. "
                "The elapsed-run bound uses actual elapsed time instead of 3000. Percentages are not clipped at zero."
            ),
            "founders": sum(row["parent_id"] is None for row in agent_rows),
            "births": sum(row["parent_id"] is not None for row in agent_rows),
            "alive_at_end": len(self.env.agents), "peak_population": self.peak_population,
            "deaths": len(deaths), "predation_deaths": len(predation),
            "energy_depletion_deaths": sum(row["death_cause"] == "energy_depletion" for row in deaths),
            "other_deaths": sum(row["death_cause"] not in ("predation", "energy_depletion") for row in deaths),
            "predation_without_sprint_energy": sum(row["could_sprint_at_last_decision"] is False for row in predation),
            "predation_without_observed_predator": sum(row["saw_predator_at_last_decision"] is False for row in predation),
            "predation_without_prior_decision": sum(row["could_sprint_at_last_decision"] is None for row in predation),
            "fruit_spawned": self.fruit_spawned, "fruit_eaten": self.fruit_eaten,
            "fruit_rotted": self.fruit_rotted, "evolution": evolution,
        }
        self.summary = summary
        write_csv(self.directory / "per_second.csv", second_rows)
        write_csv(self.directory / "traits_over_time.csv", self.samples)
        write_csv(self.directory / "agents.csv", agent_rows)
        write_csv(self.directory / "generations.csv", evolution["generations"])
        (self.directory / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        from src.utils.run_report import write_report
        write_report(self.directory, summary, self.samples, second_rows, agent_rows,
                     self.config, make_plots=make_plots)
        return self.directory
