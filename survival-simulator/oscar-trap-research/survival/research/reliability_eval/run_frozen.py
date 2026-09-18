"""Bounded-RAM executor and fixed-denominator summarizer for the frozen batch."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "research"), str(ROOT)]
from protocol import PROTOCOL, PROTOCOL_SHA256
from score_receipt import score

OUT = ROOT / "results" / "reliability_eval"
STOP = Path("/tmp/predator-intake-stop")


def policy_dependency_manifest(module_name: str) -> dict:
    """Hash the recursive local-Python import closure under research/."""
    research_root = (ROOT / "research").resolve()
    pending = [module_name]
    visited_modules = set()
    files = {}
    while pending:
        name = pending.pop()
        if not name or name in visited_modules:
            continue
        visited_modules.add(name)
        parent = name.rpartition(".")[0]
        if parent:
            pending.append(parent)
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, AttributeError, ValueError):
            continue
        if spec is None or spec.origin is None or spec.origin in {"built-in", "frozen"}:
            continue
        path = Path(spec.origin).resolve()
        try:
            relative = path.relative_to(research_root)
        except ValueError:
            continue
        if path.suffix != ".py":
            continue
        files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"cannot inspect policy dependency {path}: {exc}") from exc
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    try:
                        base = importlib.util.resolve_name(
                            "." * node.level + (node.module or ""), package)
                    except (ImportError, ValueError):
                        continue
                else:
                    base = node.module or ""
                pending.append(base)
                pending.extend(f"{base}.{alias.name}" for alias in node.names
                               if base and alias.name != "*")
    if not files:
        raise RuntimeError(f"no local research dependencies found for {module_name!r}")
    digest = hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "files": dict(sorted(files.items()))}


def lower_bound(successes: int, trials: int, confidence: float = .95) -> float:
    """Exact one-sided Clopper-Pearson lower bound, using stdlib only."""
    if successes <= 0:
        return 0.0
    alpha = 1.0 - confidence
    if successes == trials:
        return alpha ** (1.0 / trials)
    def upper_tail(p):
        return sum(math.comb(trials, i) * p**i * (1-p)**(trials-i)
                   for i in range(successes, trials + 1))
    low, high = 0.0, successes / trials
    for _ in range(100):
        middle = (low + high) / 2
        if upper_tail(middle) < alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def run_case(case: dict, policy: str, policy_sha256: str,
             dependency_manifest: dict) -> dict:
    row = dict(case)
    if STOP.exists():
        return row | {"status": "not_started_stop_sentinel", "pass": False}
    log = OUT / "logs" / f"case-{case['case']:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run", "--user", "--scope", "-p", "MemoryMax=3G",
        "-p", "MemoryHigh=2G", "-p", "MemorySwapMax=0", "--",
        sys.executable, str(ROOT / "research/simple_chase/run_streaming_v2.py"),
        "--policy", policy, "--seconds", str(PROTOCOL["seconds"]),
        "--map-seed", str(case["map_seed"]), "--fixture-seed",
        str(case["fixture_seed"]), "--predators", "1", "--station-bait",
        "--native-width", str(PROTOCOL["native_width"]),
    ]
    with log.open("w") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT)
    row["exit_code"] = completed.returncode
    candidates = list((ROOT / "results/simple_chase").glob(
        f"real-map-*-m{case['map_seed']}-f{case['fixture_seed']}-*.json"))
    if completed.returncode or len(candidates) != 1:
        return row | {"status": "process_or_receipt_failure", "pass": False,
                      "candidate_receipts": [str(p.relative_to(ROOT)) for p in candidates]}
    try:
        scored = score(candidates[0])
    except Exception as exc:
        return row | {"status": "scoring_error", "pass": False,
                      "error": f"{type(exc).__name__}: {exc}"}
    current_dependencies = policy_dependency_manifest(policy.partition(":")[0])
    if (current_dependencies != dependency_manifest
            or scored.get("policy") != policy
            or scored.get("policy_sha256") != policy_sha256
            or json.loads(candidates[0].read_text()).get("map_seed") != case["map_seed"]
            or json.loads(candidates[0].read_text()).get("fixture_seed") != case["fixture_seed"]):
        return row | {"status": "policy_or_seed_freeze_mismatch", "pass": False,
                      "receipt": str(candidates[0].relative_to(ROOT))}
    sidecar = OUT / "scores" / f"case-{case['case']:02d}.score.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(scored, indent=2) + "\n")
    return row | {"status": "scored", "pass": scored["pass"],
                  "receipt": scored["receipt"], "score": str(sidecar.relative_to(ROOT)),
                  "policy_sha256": scored["policy_sha256"]}


def summarize(rows: list[dict], policy: str) -> dict:
    by_case = {row["case"]: row for row in rows}
    complete = [by_case.get(case["case"], case | {
        "status": "missing", "pass": False}) for case in PROTOCOL["cases"]]
    successes = sum(row.get("pass") is True for row in complete)
    bound = lower_bound(successes, len(complete), PROTOCOL["confidence"])
    return {
        "schema": "guide-delivery-reliability-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "policy": policy,
        "successes": successes,
        "trials": len(complete),
        "one_sided_confidence": PROTOCOL["confidence"],
        "clopper_pearson_lower_bound": bound,
        "establishes_over_95_percent": bound > .95,
        "rows": complete,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="module:Class frozen candidate")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    module_name, separator, _ = args.policy.partition(":")
    if not separator:
        raise SystemExit("policy must be module:Class")
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"cannot resolve policy module {module_name!r}")
    policy_sha256 = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    dependency_manifest = policy_dependency_manifest(module_name)
    OUT.mkdir(parents=True, exist_ok=True)
    execution = OUT / "execution.json"
    rows = []
    if execution.exists():
        old = json.loads(execution.read_text())
        if (old.get("protocol_sha256") != PROTOCOL_SHA256
                or old.get("policy") != args.policy
                or old.get("policy_sha256") != policy_sha256
                or old.get("policy_dependencies") != dependency_manifest):
            raise SystemExit("existing execution belongs to a different protocol or policy")
        rows = old.get("rows", [])
    if not args.summarize_only:
        done = {row["case"] for row in rows}
        pending = [case for case in PROTOCOL["cases"] if case["case"] not in done]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, case, args.policy, policy_sha256,
                                   dependency_manifest)
                       for case in pending]
            for future in as_completed(futures):
                rows.append(future.result())
                execution.write_text(json.dumps({"protocol_sha256": PROTOCOL_SHA256,
                    "policy": args.policy, "policy_sha256": policy_sha256,
                    "policy_dependencies": dependency_manifest,
                    "rows": rows}, indent=2) + "\n")
    summary = summarize(rows, args.policy)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
