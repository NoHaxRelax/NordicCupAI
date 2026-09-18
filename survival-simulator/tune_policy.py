"""Resumable, single-node Bayesian policy tuning with parallel seed evaluations.

One coordinator owns the Optuna journal. Workers only simulate episodes and
write distinct result files, so no shared SQLite database or service is needed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import copy
import hashlib
from importlib.metadata import version
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import statistics
import time

from tuning_evaluation import atomic_json, configure_process, evaluate_seed, initialize_worker

configure_process()

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from src.utils.controllers.expert_policy import ExpertConfig, load_config
from src.utils.controllers.global_planner import PlannerConfig, load_planner_config


ROOT = Path(__file__).resolve().parent
SPACE = {
    "harvest.coverage.orchard_revisit_seconds": FloatDistribution(8., 30.),
    "harvest.coverage.cell_size": CategoricalDistribution([75., 100., 125.]),
    "harvest.coverage.food_imbalance_fraction": FloatDistribution(.15, .5),
    "harvest.food_budget_per_agent_second": FloatDistribution(4., 10.),
    "harvest.ripe_energy": FloatDistribution(48., 60.),
    "harvest.emergency_energy": FloatDistribution(70., 140.),
    "harvest.parent_reserve": FloatDistribution(80., 160.),
    "reproduction.selection.post_alignment.normal_cooldown_seconds": FloatDistribution(6., 16.),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def get_nested(value, path):
    for key in path.split("."):
        value = value[key]
    return value


def configured(base, params):
    value = copy.deepcopy(base)
    for path, setting in params.items():
        keys = path.split(".")
        section = value
        for key in keys[:-1]:
            section = section[key]
        section[keys[-1]] = setting
    return ExpertConfig.model_validate(value).model_dump(mode="json")


def summarize(results):
    scores = [row["score"] for row in results]
    return dict(mean_score=statistics.mean(scores),
                score_std=statistics.stdev(scores) if len(scores) > 1 else 0.,
                minimum_score=min(scores), extinction_rate=statistics.mean(row["extinct"] for row in results),
                mean_population=statistics.mean(row["population"] for row in results),
                seeds=[row["seed"] for row in results], scores=scores)


def source_hash():
    files = [ROOT / "tune_policy.py", ROOT / "tuning_evaluation.py", *sorted((ROOT / "src").rglob("*.py"))]
    return digest({str(path.relative_to(ROOT)).replace("\\", "/"):
                   hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files})


def check_manifest(path, expected):
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != expected:
            changed = [key for key in expected.keys() | previous.keys() if expected.get(key) != previous.get(key)]
            raise ValueError(f"Study settings changed ({', '.join(sorted(changed))}); use a new --output directory.")
    else:
        atomic_json(path, expected)


@contextmanager
def coordinator_lock(output):
    """OS lock releases on exit/crash; never mistake another live job for a crash."""
    with (output / "coordinator.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise RuntimeError("Another coordinator holds this output directory. Use one job per study.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            release()


def baseline_params(base):
    params = {key: get_nested(base, key) for key in SPACE}
    for key, distribution in SPACE.items():
        setting = params[key]
        if isinstance(distribution, CategoricalDistribution):
            valid = setting in distribution.choices
        else:
            valid = distribution.low <= setting <= distribution.high
        if not valid:
            raise ValueError(f"Baseline {key}={setting} is outside the search space. Adjust SPACE before starting.")
    return params


def retry_interrupted(study, frozen):
    attrs = {"baseline": frozen.user_attrs.get("baseline", False), "retry_of": frozen.number}
    # Completed seed files are keyed by configuration and will be reused.
    if set(frozen.params) == set(SPACE):
        study.enqueue_trial(frozen.params, user_attrs=attrs)
    elif attrs["baseline"]:
        study.enqueue_trial(frozen.system_attrs.get("fixed_params", {}), user_attrs=attrs)


def repair_journal_tail(journal):
    """Recover a torn final append, with the coordinator lock already held.

    Optuna ignores an incomplete final line when reading, but appending after
    it makes the journal unreadable. Preserve the fragment before truncating.
    Corruption anywhere except the last record is an error, never discarded.
    """
    if not journal.exists():
        return
    with journal.open("r+b") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                return
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("Incomplete final record")
                json.loads(line)
            except (ValueError, UnicodeError) as exc:
                if handle.read(1):
                    raise ValueError(f"Invalid journal record at byte {offset}; history was left unchanged") from exc
                backup = journal.with_name(f"{journal.name}.torn-tail-{time.time_ns()}.bin")
                with backup.open("xb") as fragment:
                    fragment.write(line)
                    fragment.flush()
                    os.fsync(fragment.fileno())
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
                print(f"Recovered interrupted journal write; preserved fragment in {backup.name}", flush=True)
                return


def open_study(output, optimizer_seed, startup_trials, base):
    journal = output / "optuna.journal"
    repair_journal_tail(journal)
    storage = JournalStorage(JournalFileBackend(str(journal), lock_obj=JournalFileOpenLock(str(journal))))
    sampler = optuna.samplers.TPESampler(seed=optimizer_seed, n_startup_trials=startup_trials,
                                        multivariate=True, constant_liar=True)
    study = optuna.create_study(storage=storage, study_name="central-harvest", direction="maximize",
                                sampler=sampler, load_if_exists=True)
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL)
        retry_interrupted(study, trial)
    params = baseline_params(base)
    if not any(t.user_attrs.get("baseline") and t.state != TrialState.FAIL for t in study.trials):
        study.enqueue_trial(params, user_attrs={"baseline": True})
    # A resumed search should not restart the sampler's random sequence.
    study.sampler = optuna.samplers.TPESampler(seed=optimizer_seed + len(study.trials),
        n_startup_trials=startup_trials, multivariate=True, constant_liar=True)
    return study


def candidate_directory(output, expert):
    path = output / "candidates" / digest(expert)[:20]
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "expert_policy.json"
    if not config_path.exists():
        atomic_json(config_path, expert)
    return path


def submit_seeds(pool, output, expert, planner, seeds, seconds, phase, deadline, checkpoint_seconds=60.):
    directory = candidate_directory(output, expert) / phase
    return {pool.submit(evaluate_seed, expert, planner, seed, seconds,
                        str(directory / f"seed-{seed}.json"), deadline, checkpoint_seconds): seed for seed in seeds}


def export_search(study, output, base, planner):
    trials = sorted(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)),
                    key=lambda t: (-t.value, t.number))
    rows = [dict(trial=t.number, mean_score=t.value, params=t.params, baseline=t.user_attrs.get("baseline", False),
                 metrics=t.user_attrs.get("metrics")) for t in trials]
    atomic_json(output / "search_results.json", rows)
    if trials:
        atomic_json(output / "best_search_expert_policy.json", configured(base, trials[0].params))
        atomic_json(output / "best_search_global_planner.json", planner)
    lines = ["# Policy search progress", "", f"Completed trials: {len(trials)}.", "",
             "Scores use the shorter training episodes. Full-length validation is reported in `report.md`.", "",
             "| Trial | Mean score | Baseline |", "| --- | ---: | --- |"]
    lines += [f"| {row['trial']} | {row['mean_score']:.4f} | {'yes' if row['baseline'] else ''} |" for row in rows]
    lines += ["", ("Best training settings: `best_search_expert_policy.json` and `best_search_global_planner.json`."
                   if trials else "First trial pending; checkpoints and completed seeds are retained when stopped."), ""]
    (output / "search_report.md").write_text("\n".join(lines), encoding="utf-8")
    return trials


def search(pool, study, args, base, planner, deadline, stop):
    # Parallelize seeds as well as trials. With 24 cores and 3 seeds this keeps
    # 8 candidates in flight, giving TPE feedback sooner than 24 serial batches.
    width = max(1, args.workers // len(args.train_seeds))
    active, futures = {}, {}
    completed = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    heartbeat = time.monotonic()
    while True:
        while (not stop.is_set() and time.monotonic() < deadline and len(active) < width
               and completed + len(active) < args.trials):
            trial = study.ask(fixed_distributions=SPACE)
            expert = configured(base, trial.params)
            jobs = submit_seeds(pool, args.output, expert, planner, args.train_seeds,
                                args.train_seconds, "train", deadline, args.checkpoint_seconds)
            active[trial.number] = dict(trial=trial, results={}, remaining=len(jobs), interrupted=False)
            futures.update({future: (trial.number, seed) for future, seed in jobs.items()})
            print(f"Started trial {trial.number}; {len(active)} configurations in flight", flush=True)
        if not futures:
            break
        done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
        for future in done:
            number, seed = futures.pop(future)
            pending = active[number]
            try:
                result = future.result()
            except Exception as exc:
                study.tell(pending["trial"], state=TrialState.FAIL)
                retry_interrupted(study, study.trials[number])
                atomic_json(args.output / "worker_error.json", dict(trial=number, seed=seed, error=repr(exc)))
                stop.set()
                raise RuntimeError(f"Trial {number}, seed {seed} failed; see worker_error.json") from exc
            pending["remaining"] -= 1
            if result is None:
                pending["interrupted"] = True
            else:
                pending["results"][seed] = result
            if pending["remaining"]:
                continue
            trial = pending["trial"]
            if pending["interrupted"]:
                trial.set_user_attr("interrupted", True)
                study.tell(trial, state=TrialState.FAIL)
                retry_interrupted(study, study.trials[number])
            else:
                metrics = summarize([pending["results"][seed] for seed in args.train_seeds])
                trial.set_user_attr("metrics", metrics)
                study.tell(trial, metrics["mean_score"])
                completed += 1
                print(f"Trial {number}: mean score={metrics['mean_score']:.4f}, "
                      f"extinction={metrics['extinction_rate']:.0%}, completed={completed}", flush=True)
            del active[number]
            export_search(study, args.output, base, planner)
        if time.monotonic() - heartbeat >= 60:
            print(f"Search: {completed} complete, {len(active)} active, "
                  f"{max(0, deadline - time.monotonic()) / 3600:.2f} hours remaining", flush=True)
            heartbeat = time.monotonic()


def select_finalists(trials, base, count):
    baseline = baseline_params(base)
    selected = [dict(label="baseline", params=baseline, expert=base)]
    seen = {digest(base)}
    for trial in trials:
        expert = configured(base, trial.params)
        key = digest(expert)
        if key in seen:
            continue
        selected.append(dict(label=f"trial-{trial.number}", params=trial.params, expert=expert))
        seen.add(key)
        if len(selected) >= count + 1:
            break
    return selected


def validation_report(output, candidates, results, seeds, planner):
    rows = []
    for candidate in candidates:
        label = candidate["label"]
        available = results[label]
        complete = all(seed in available for seed in seeds)
        rows.append(dict(label=label, complete=complete, completed_seeds=sorted(available),
                         config_directory=str(candidate_directory(output, candidate["expert"]).relative_to(output)),
                         metrics=summarize([available[seed] for seed in seeds]) if complete else None))
    ranked = sorted((row for row in rows if row["complete"]),
                    key=lambda row: (-row["metrics"]["mean_score"], row["label"]))
    baseline = next((row for row in ranked if row["label"] == "baseline"), None)
    # Do not promote a candidate without a complete, comparable baseline.
    winner = ranked[0] if ranked and baseline else None
    if winner:
        candidate = next(c for c in candidates if c["label"] == winner["label"])
        atomic_json(output / "best_expert_policy.json", candidate["expert"])
        atomic_json(output / "best_global_planner.json", planner)
    report = dict(winner=winner["label"] if winner else None, baseline_complete=baseline is not None,
                  all_candidates_complete=all(row["complete"] for row in rows),
                  compared_seeds=seeds, candidates=rows,
                  note="Validation seeds select the winner; use additional untouched seeds for an unbiased final assessment.")
    atomic_json(output / "validation_results.json", report)
    lines = ["# Policy tuning results", "", "Predator-free; higher mean game score is better.", "",
             "Only candidates with every validation seed completed are ranked. Early extinction counts as a completed episode; wall-clock timeouts do not.", "",
             "| Candidate | Completed seeds | Mean score | Extinction rate | Mean final population |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in sorted(rows, key=lambda r: (not r["complete"], -(r["metrics"] or {}).get("mean_score", 0))):
        metrics = row["metrics"]
        values = (f"{metrics['mean_score']:.4f} | {metrics['extinction_rate']:.1%} | {metrics['mean_population']:.1f}"
                  if metrics else "pending | pending | pending")
        lines.append(f"| {row['label']} | {len(row['completed_seeds'])}/{len(seeds)} | {values} |")
    if winner:
        deltas = [a - b for a, b in zip(winner["metrics"]["scores"], baseline["metrics"]["scores"])]
        lines += ["", f"Selected: **{winner['label']}**. Mean paired improvement over baseline: {statistics.mean(deltas):+.4f} score.",
                  "", "Load both `best_expert_policy.json` and `best_global_planner.json` to reproduce the tuned settings."]
    else:
        lines += ["", "No validated winner yet. Resume this directory to finish validation."]
    if not report["all_candidates_complete"]:
        lines += ["", "Validation is incomplete. The current selection, if present, is provisional."]
    lines += ["", report["note"], ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def validate(pool, study, args, base, planner, deadline, stop):
    trials = export_search(study, args.output, base, planner)
    run_state = read_run_state(args.output)
    if run_state.get("phase") == "validation":
        candidates = [{**row, "expert": configured(base, row["params"])} for row in run_state["finalists"]]
    else:
        candidates = select_finalists(trials, base, args.finalists)
    results = {candidate["label"]: {} for candidate in candidates}
    futures = {}
    finalists = [{k: v for k, v in c.items() if k != "expert"} for c in candidates]
    atomic_json(args.output / "run_state.json", dict(phase="validation", finalists=finalists))
    atomic_json(args.output / "finalists.json", finalists)
    for candidate in candidates:
        jobs = submit_seeds(pool, args.output, candidate["expert"], planner, args.validation_seeds,
                            args.validation_seconds, "validation", deadline, args.checkpoint_seconds)
        futures.update({future: (candidate["label"], seed) for future, seed in jobs.items()})
    validation_report(args.output, candidates, results, args.validation_seeds, planner)
    heartbeat = time.monotonic()
    while futures:
        done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
        for future in done:
            label, seed = futures.pop(future)
            try:
                result = future.result()
            except Exception:
                stop.set()
                raise
            if result is not None:
                results[label][seed] = result
                print(f"Validation {label}, seed {seed}: score={result['score']:.4f}, "
                      f"population={result['population']}", flush=True)
            validation_report(args.output, candidates, results, args.validation_seeds, planner)
        if time.monotonic() - heartbeat >= 60:
            print(f"Validation: {len(futures)} episodes pending, "
                  f"{max(0, deadline - time.monotonic()) / 3600:.2f} hours remaining", flush=True)
            heartbeat = time.monotonic()
    if all(all(seed in results[c["label"]] for seed in args.validation_seeds) for c in candidates):
        atomic_json(args.output / "run_state.json", dict(phase="complete"))


def read_run_state(output):
    path = output / "run_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def resolve_options(args):
    """Keep cluster defaults while allowing one short laptop command."""
    if args.output is None:
        if args.laptop and os.name == "nt" and os.environ.get("LOCALAPPDATA"):
            # Frequent checkpoints and journal appends should stay out of OneDrive.
            args.output = Path(os.environ["LOCALAPPDATA"]) / "NordicCupAI" / "bo-laptop"
        else:
            args.output = ROOT / "runs" / ("bo-laptop" if args.laptop else "bo-dtu")
    if args.workers is None:
        args.workers = min(6, max(1, (os.cpu_count() or 2) // 2)) if args.laptop else 1
    if args.total_hours is None:
        args.total_hours = 8. if args.laptop else 11.5
    if args.search_hours is None:
        args.search_hours = .75 * args.total_hours if args.laptop else min(8., .75 * args.total_hours)
    if args.startup_trials is None:
        args.startup_trials = 16 if args.laptop else 24
    if args.finalists is None:
        args.finalists = 2 if args.laptop else 7
    if args.keep_awake is None:
        args.keep_awake = args.laptop and os.name == "nt"
    return args


@contextmanager
def keep_awake(enabled):
    """Temporarily prevent Windows idle sleep; allow the display to turn off."""
    active = False
    if enabled and os.name == "nt":
        import ctypes
        set_state = ctypes.windll.kernel32.SetThreadExecutionState
        active = bool(set_state(0x80000001))  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if not active:
            print("Could not prevent idle sleep; check Windows power settings.", flush=True)
    try:
        yield
    finally:
        if active:
            set_state(0x80000000)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--laptop", action="store_true", help="Laptop defaults: up to 6 CPU workers, 8 hours, 2 finalists plus baseline")
    cli.add_argument("--output", type=Path, help="Resume directory; laptop Windows default is LOCALAPPDATA/NordicCupAI/bo-laptop")
    cli.add_argument("--expert-config", type=Path)
    cli.add_argument("--planner-config", type=Path)
    cli.add_argument("--workers", type=int)
    cli.add_argument("--trials", type=int, default=300, help="Total completed search trials, including previous submissions")
    cli.add_argument("--startup-trials", type=int)
    cli.add_argument("--search-hours", type=float, help="Search time per invocation; 0 runs validation only")
    cli.add_argument("--total-hours", "--hours", type=float, help="Whole invocation budget, including validation")
    cli.add_argument("--checkpoint-seconds", type=float, default=60., help="Wall-clock interval for saving in-progress episodes")
    cli.add_argument("--keep-awake", action=argparse.BooleanOptionalAction, help="Prevent Windows idle sleep while running; default on with --laptop")
    cli.add_argument("--train-seconds", type=float, default=600.)
    cli.add_argument("--validation-seconds", type=float, default=3000.)
    cli.add_argument("--train-seeds", type=int, nargs="+", default=[1, 7, 42])
    cli.add_argument("--validation-seeds", type=int, nargs="+", default=[1001, 1007, 1042])
    cli.add_argument("--finalists", type=int, help="Top distinct candidates, in addition to baseline")
    cli.add_argument("--optimizer-seed", type=int, default=20260917)
    return cli


def run(args):
    args = resolve_options(args)
    started = time.monotonic()
    if not (math.isfinite(args.search_hours) and math.isfinite(args.total_hours)
            and 0 <= args.search_hours < args.total_hours):
        raise ValueError("Require 0 <= --search-hours < --total-hours")
    if min(args.workers, args.trials, args.startup_trials, args.finalists) < 1:
        raise ValueError("Workers, trials, startup-trials and finalists must be positive")
    if not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds <= 0:
        raise ValueError("--checkpoint-seconds must be finite and positive")
    if not 0 < args.train_seconds <= args.validation_seconds <= 3000:
        raise ValueError("Require 0 < train-seconds <= validation-seconds <= 3000")
    if set(args.train_seeds) & set(args.validation_seeds):
        raise ValueError("Training and validation seeds must be disjoint")
    if any(len(seeds) != len(set(seeds)) for seeds in (args.train_seeds, args.validation_seeds)):
        raise ValueError("Seed lists must not contain duplicates")
    allocation = os.environ.get("LSB_DJOB_NUMPROC")
    if allocation and args.workers > int(allocation):
        raise ValueError("--workers exceeds the allocated LSF CPU slots")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    base = load_config(args.expert_config).model_dump(mode="json")
    base["harvest"]["enabled"] = base["harvest"]["coverage"]["enabled"] = True
    planner = load_planner_config(args.planner_config).model_dump(mode="json")
    planner["biome_inference"]["adapt_to_wall_clock"] = False
    planner["draw_overlay"] = False
    base = ExpertConfig.model_validate(base).model_dump(mode="json")
    planner = PlannerConfig.model_validate(planner).model_dump(mode="json")
    baseline_params(base)
    if not (planner["mapping_enabled"] and planner["population_after_alignment"]
            and base["reproduction"]["selection"]["enabled"]
            and base["reproduction"]["selection"]["post_alignment"]["enabled"]):
        raise ValueError("Tuning requires shared mapping and post-alignment population selection to be enabled")
    manifest = dict(format_version=2, source_hash=source_hash(), expert=base, planner=planner,
                    python=platform.python_version(), system=platform.system(),
                    packages={name: version(name) for name in ("optuna", "numpy", "scipy", "pydantic", "pygame", "shapely")},
                    search_space={name: optuna.distributions.distribution_to_json(dist) for name, dist in SPACE.items()},
                    train_seeds=args.train_seeds, validation_seeds=args.validation_seeds,
                    train_seconds=args.train_seconds, validation_seconds=args.validation_seconds,
                    optimizer_seed=args.optimizer_seed, startup_trials=args.startup_trials,
                    predators_enabled=False, objective="mean game score over completed fixed-horizon episodes")
    context = multiprocessing.get_context("spawn")
    stop = context.Event()

    def request_stop(signum, frame):
        print("Stopping: saving active episodes and trial history. Please wait for workers to finish checkpointing.", flush=True)
        stop.set()

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with coordinator_lock(args.output), keep_awake(args.keep_awake):
            check_manifest(args.output / "manifest.json", manifest)
            atomic_json(args.output / "base_expert_policy.json", base)
            atomic_json(args.output / "base_global_planner.json", planner)
            study = open_study(args.output, args.optimizer_seed, args.startup_trials, base)
            export_search(study, args.output, base, planner)
            print(f"Output: {args.output}\nCPU workers: {args.workers}; predators: off; "
                  f"checkpoint interval: {args.checkpoint_seconds:g}s\n"
                  f"Budget: {args.total_hours:g} hours. Ctrl+C saves and stops; repeat the command to resume.", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                     initializer=initialize_worker, initargs=(stop,)) as pool:
                try:
                    if read_run_state(args.output).get("phase") == "validation":
                        print("Resuming unfinished validation before any new search.", flush=True)
                    else:
                        atomic_json(args.output / "run_state.json", dict(phase="search"))
                        search(pool, study, args, base, planner, started + args.search_hours * 3600, stop)
                    if not stop.is_set():
                        validate(pool, study, args, base, planner, started + args.total_hours * 3600, stop)
                finally:
                    stop.set()
            print(f"Saved results in {args.output}", flush=True)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    command = parser()
    try:
        run(command.parse_args())
    except (ValueError, RuntimeError) as error:
        command.exit(1, f"Error: {error}\n")
