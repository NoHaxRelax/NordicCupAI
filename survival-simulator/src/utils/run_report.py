"""Export static plots and a self-contained local HTML run recap."""

from html import escape
import os

import numpy as np

from src.utils.run_diagnostics import TRAITS


LABELS = {
    "speed": "Walking speed", "sprint_speed": "Sprint speed",
    "max_energy": "Energy capacity", "hearing_radius": "Hearing radius",
    "vision_range": "Vision range", "vision_angle": "Vision angle (rad)",
}


def _number(value, digits=2):
    return "—" if value is None else f"{value:,.{digits}f}"


def normalized_traits(samples):
    """Living means relative to the fixed initial population mean (100%).

    Empty populations and zero/missing starting values have no defined ratio.
    Keep those gaps rather than plotting them as zero or changing the baseline.
    """
    result = {}
    for trait in TRAITS:
        key = f"{trait}_mean"
        initial = samples[0].get(key) if samples else None
        result[trait] = np.array([
            100 * row[key] / initial
            if initial is not None and initial > 0 and row.get(key) is not None
            else np.nan for row in samples
        ])
    return result


def _make_plots(directory, summary, samples, seconds, agents, config):
    # Keep font cache writes inside the run directory; never open GUI figures.
    os.environ.setdefault("MPLCONFIGDIR", str(directory / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "figure.facecolor": "white",
        "axes.titleweight": "bold",
    })
    teal, orange, red = "#087f8c", "#e8a33d", "#c64c4c"

    def save(fig, name):
        fig.savefig(directory / name, dpi=config.plot_dpi, facecolor="white")
        plt.close(fig)

    times = [row["time"] for row in samples]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), layout="constrained", sharex=True)
    fig.suptitle(f"Run {summary['seed']} · score and population", fontsize=17)
    axes[0].plot(times, [row["score"] for row in samples], color=teal, label="Actual score")
    axes[0].plot(times, times, color="#777777", linestyle="--", label="Survival contribution")
    axes[0].set(ylabel="Cumulative points")
    axes[0].legend(loc="upper left")
    if seconds:
        bin_times = [row["second_start"] + row["observed_seconds"] for row in seconds]
        axes[1].plot(bin_times, [row["score_gain_per_second"] for row in seconds],
                     color=teal, label="Net gain / second")
        axes[1].plot(bin_times, [row["food_gain"] / row["observed_seconds"] for row in seconds],
                     color=orange, label="Fruit / second", alpha=0.85)
        axes[1].plot(bin_times, [-row["predation_loss"] / row["observed_seconds"] for row in seconds],
                     color=red, label="Predation adjustment / second", alpha=0.85)
        axes[1].axhline(1, color="#777777", linestyle=":", label="Survival: +1 / second")
    axes[1].set(ylabel="Points / simulated second")
    if seconds:
        axes[1].legend(loc="best", ncols=2, fontsize=9)
    axes[2].plot(times, [row["population"] for row in samples], color=teal, label="Agents")
    axes[2].plot(times, [row["predators"] for row in samples], color=red, label="Predators")
    axes[2].set(xlabel="Simulated seconds", ylabel="Population")
    axes[2].legend()
    axes[2].set_xlim(0, max(1, summary["simulated_seconds"]))
    save(fig, "score.png")

    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    percentages = normalized_traits(samples)
    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#9467bd", "#9b7600")
    styles = ("-", "--", "-.", ":", "--", "-.")
    for trait, color, style in zip(TRAITS, colors, styles):
        ax.plot(times, percentages[trait], color=color, linestyle=style,
                linewidth=2, label=LABELS[trait].replace(" (rad)", ""))
    ax.axhline(100, color="#777777", linestyle=":", linewidth=1, zorder=0)
    ax.set(title="Evolution of living agents", xlabel="Simulated seconds",
           ylabel="Average trait / starting average (%)",
           xlim=(0, max(1, summary["simulated_seconds"])))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=3, frameon=False)
    if not any(np.isfinite(values).any() for values in percentages.values()):
        ax.text(0.5, 0.5, "No living-population trait data", ha="center", va="center", transform=ax.transAxes)
    save(fig, "traits.png")

    evolution = summary["evolution"]
    groups = evolution["generations"]
    age = evolution["comparison_age_seconds"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    fig.suptitle(f"Generation outcomes · equal first {age:g} seconds of opportunity", fontsize=15)
    gens = np.array([row["generation"] for row in groups])
    if groups:
        axes[0, 0].bar(gens - 0.2, [row["born"] for row in groups], width=0.4,
                       color=orange, label="Born")
        axes[0, 0].bar(gens + 0.2, [row["eligible_at_comparison_age"] for row in groups],
                       width=0.4, color=teal, label="Full follow-up opportunity")
        axes[0, 0].legend(fontsize=9)
    axes[0, 0].set(title="Cohort sizes", ylabel="Agents")
    for ax, key, title, scale in (
        (axes[0, 1], "survival_at_comparison_age", f"Survived to {age:g} seconds (%)", 100),
        (axes[1, 0], "offspring_by_comparison_age", f"Mean offspring in first {age:g} seconds", 1),
        (axes[1, 1], "food_energy_by_comparison_age", f"Mean fruit energy in first {age:g} seconds", 1),
    ):
        values = [np.nan if row[key] is None else row[key] * scale for row in groups]
        bars = ax.bar(gens, values, color=teal)
        for bar, row in zip(bars, groups):
            if row["eligible_at_comparison_age"] < config.minimum_cohort_size:
                bar.set_alpha(0.4)
                bar.set_hatch("//")
        ax.set(title=title)
        if key == "survival_at_comparison_age":
            ax.set_ylim(0, 105)
    for ax in axes.flat:
        ax.set_xlabel("Generation (parent depth; 0 = founders)")
        ax.set_xticks(gens)
    fig.supxlabel("Pale / hatched bars: small sample. Missing bars: no full follow-up opportunity.", fontsize=9)
    save(fig, "generations.png")

    eligible = [row for row in agents if row["birth_time"] + age <= summary["simulated_seconds"] + 1e-9]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    fig.suptitle(f"Trait–reproduction associations · offspring in first {age:g} seconds", fontsize=15)
    for ax, trait in zip(axes.flat, TRAITS):
        if eligible:
            points = ax.scatter([row[trait] for row in eligible],
                                [row["children_by_comparison_age"] for row in eligible],
                                c=[row["birth_time"] for row in eligible], cmap="viridis",
                                alpha=0.6, s=22, edgecolors="none")
        else:
            ax.text(0.5, 0.5, "Insufficient follow-up", ha="center", va="center", transform=ax.transAxes)
        correlation = evolution["trait_offspring_correlations"][trait]
        ax.set(title=f"{LABELS[trait]} · r = {_number(correlation)}", xlabel=LABELS[trait], ylabel="Offspring")
    if eligible:
        fig.colorbar(points, ax=axes.ravel().tolist(), label="Birth time (simulated seconds)", shrink=0.8)
    save(fig, "selection.png")


def write_report(directory, summary, samples, seconds, agents, config, make_plots=True):
    if make_plots:
        _make_plots(directory, summary, samples, seconds, agents, config)
    evolution = summary["evolution"]
    first, last = evolution["first_generation"], evolution["last_generation"]
    alive = [row for row in agents if row["death_time"] is None]
    trait_rows = []
    for trait in TRAITS:
        initial = first[f"{trait}_mean"] if first else None
        final = last[f"{trait}_mean"] if last else None
        living = float(np.mean([row[trait] for row in alive])) if alive else None
        change = 100 * (final / initial - 1) if initial and final is not None else None
        trait_rows.append(f"<tr><td>{LABELS[trait]}</td><td>{_number(initial)}</td>"
                          f"<td>{_number(final)}</td><td>{_number(change)}%</td><td>{_number(living)}</td></tr>")
    generation_rows = []
    for row in evolution["generations"]:
        survival = row["survival_at_comparison_age"]
        generation_rows.append(
            f"<tr><td>{row['generation']}</td><td>{row['born']}</td><td>{row['alive_at_end']}</td>"
            f"<td>{row['eligible_at_comparison_age']}</td><td>{_number(None if survival is None else 100 * survival)}%</td>"
            f"<td>{_number(row['offspring_by_comparison_age'])}</td><td>{_number(row['food_energy_by_comparison_age'])}</td></tr>"
        )
    latest = evolution["latest_assessable_generation"]
    latest_text = ""
    if latest and last and latest["generation"] != last["generation"]:
        latest_text = (f"<p>Latest generation with sufficient follow-up: <b>{latest['generation']}</b>. "
                       f"{escape(evolution['latest_assessable_verdict'])}</p>")
    image = lambda name, alt: f'<img src="{name}" alt="{alt}" loading="lazy">' if make_plots else ""
    partial = summary["end_reason"] not in ("extinction", "horizon")
    text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Survival recap &middot; seed {summary['seed']}</title>
<style>
body{{margin:0;background:#f4f6f7;color:#21343a;font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1140px;margin:auto;padding:32px 24px}}h1{{font-size:32px;margin:0}}
h2{{margin:0 0 10px;font-size:23px}}p{{margin:8px 0}}.muted{{color:#576c74}}
section{{background:white;border:1px solid #dfe7e9;border-radius:12px;padding:22px;margin:20px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:14px;margin-top:20px}}
.card{{background:#e8f2f2;padding:18px;border-radius:10px}}.card strong{{display:block;font-size:28px}}
img{{display:block;width:100%;height:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:right;padding:9px;border-bottom:1px solid #dde5e7}}th:first-child,td:first-child{{text-align:left}}
th{{background:#eef4f4}}.table{{overflow-x:auto}}a{{color:#087f8c}}
details{{margin-top:16px}}summary{{cursor:pointer;color:#087f8c;font-weight:600}}
</style></head><body><main>
<h1>Run recap &middot; seed {summary['seed']}</h1>
<p class="muted">{_number(summary['simulated_seconds'])} simulated seconds &middot; {escape(summary['end_reason'].replace('_', ' '))}{' &middot; partial run' if partial else ''}</p>
<div class="cards">
<div class="card">Score<strong>{_number(summary['final_score'], 3)}</strong></div>
<div class="card">Survival target<strong>{_number(summary['survival_target_percent'])}%</strong>of 3,000 seconds</div>
<div class="card">Optimistic benchmark<strong>{_number(summary['benchmark_percent'])}%</strong>not the theoretical maximum</div>
<div class="card">Population<strong>{summary['alive_at_end']} alive</strong>{summary['births']} births &middot; peak {summary['peak_population']}</div>
</div>
<section><h2>Score and population</h2>
{image('score.png', 'Cumulative score, score per second, and population')}
<p class="muted">{summary['fruit_eaten']} fruit eaten &middot; {summary['predation_deaths']} eaten by predators &middot; {summary['energy_depletion_deaths']} energy-depletion deaths</p>
</section>
<section><h2>Evolutionary traits</h2>
<p class="muted">Living-agent averages. Starting average = 100%; 120% means 20% higher.</p>
{image('traits.png', 'All six living-population trait averages as percentages of their initial averages')}
<details><summary>Trait values</summary>
<div class="table"><table><thead><tr><th>Trait</th><th>Founder mean</th><th>Last generation mean</th><th>Change</th><th>Final living mean</th></tr></thead>
<tbody>{''.join(trait_rows)}</tbody></table></div>
</details></section>
<section><h2>Generation outcomes</h2>
<p><b>{escape(evolution['last_generation_verdict'])}</b></p>
{image('generations.png', 'Generation sizes, survival, reproduction and food at a common age')}
<details><summary>Comparison details</summary>
{latest_text}
<p>Compared over the first {config.comparison_age_seconds:g} seconds of life; at least {config.minimum_cohort_size} agents per cohort. Last generation = deepest lineage, including deaths. Larger traits alone do not imply better fitness.</p>
<div class="table"><table><thead><tr><th>Generation</th><th>Born</th><th>Alive</th><th>Eligible</th><th>Survival</th><th>Mean offspring</th><th>Mean fruit energy</th></tr></thead>
<tbody>{''.join(generation_rows)}</tbody></table></div>
<p class="muted">{escape(evolution['interpretation'])}</p>
</details>
<details><summary>Trait-offspring associations</summary>
{image('selection.png', 'Relationships between inherited traits and offspring at a common age')}
<p class="muted">{evolution['eligible_agents']} eligible agents. Correlation is not proof of selection.</p>
</details></section>
<section><details><summary>Score benchmark and death details</summary>
<p>{escape(summary['benchmark_definition'])}</p>
<p>Elapsed-run upper bound: {_number(summary['elapsed_run_score_upper_bound'], 3)} points ({_number(summary['elapsed_bound_percent'])}% achieved). True maximum unknown.</p>
<p>Predation: {summary['predation_without_sprint_energy']} lacked sprint energy; {summary['predation_without_observed_predator']} had no observed predator; {summary['predation_without_prior_decision']} had no prior decision. Categories overlap and do not establish causes.</p>
<p>{summary['other_deaths']} other deaths &middot; {summary['fruit_spawned']} fruit spawned &middot; {summary['fruit_rotted']} rotted.</p>
</details>
<details><summary>Download data</summary><p>
<a href="summary.json">Summary</a> &middot; <a href="per_second.csv">Score</a> ?
<a href="traits_over_time.csv">Traits</a> &middot; <a href="agents.csv">Agents</a> ?
<a href="generations.csv">Generations</a> &middot; <a href="events.jsonl">Events</a> ?
<a href="run_config.json">Configuration</a></p>
</details></section>
</main></body></html>"""
    (directory / "report.html").write_text(text, encoding="utf-8")
