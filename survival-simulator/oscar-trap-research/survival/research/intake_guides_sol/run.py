"""Recorded fixture runner for observation-only rest-synchronised intake.

The wall, two holders, predator arrivals and one later guide per arrival are
fixture supplied.  This privilege is setup only.  The policy receives JSON
round-tripped native observation DTOs and public time, exactly as the preserved
wall-funnel harness enforces.  Food is replenished by that disclosed fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parents[0] / "wall_funneling" / "observed_run.py"

sys.path.insert(0, str(SOURCE.parent))
_spec = importlib.util.spec_from_file_location("wall_funneling_observed_runner", SOURCE)
_runner = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_runner)

_policy_spec = importlib.util.spec_from_file_location("intake_guides_policy", HERE / "observed_policy.py")
_policy = importlib.util.module_from_spec(_policy_spec)
assert _policy_spec.loader is not None
_policy_spec.loader.exec_module(_policy)
ObservedFunnel = _policy.ObservedFunnel

OUT = ROOT / "results" / "intake_guides_sol"
POLICY_HASH = hashlib.sha256((HERE / "observed_policy.py").read_bytes()).hexdigest()
_BaseRecorder = _runner.ReplayRecorder


class IntakeRecorder(_BaseRecorder):
    def __init__(self, *args, **kwargs):
        kwargs["policy"] = "observation-only-rest-synchronised-intake-v1"
        kwargs["notes"] = (__doc__ + " Later guide/predator arrivals are supplied at prepared approach positions; "
                           "recruitment and routing to those positions are not demonstrated. "
                           "Predator stationarity is inferred only from successive native observations.")
        kwargs["policy_sha256"] = POLICY_HASH
        super().__init__(*args, **kwargs)


def run(**kwargs):
    _runner.OUT = OUT
    _runner.StagedGuideFunnel = ObservedFunnel
    _runner.EXPERIMENTAL_HASH = POLICY_HASH
    _runner.ReplayRecorder = IntakeRecorder
    kwargs["staged_guides"] = True
    kwargs.setdefault("width", 30)
    kwargs.setdefault("length", 100)
    kwargs.setdefault("baits", 2)
    kwargs.setdefault("awake", True)
    return _runner.run(**kwargs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predators", type=int, default=4)
    p.add_argument("--waves", type=int)
    p.add_argument("--interval", type=float, default=30.)
    p.add_argument("--seconds", type=float, default=150.)
    p.add_argument("--seed", type=int, default=4101)
    p.add_argument("--spread", type=float, default=15.)
    p.add_argument("--depth-spread", type=float, default=40.)
    p.add_argument("--horizontal", action="store_true")
    p.add_argument("--native", action="store_true")
    a = vars(p.parse_args())
    if a["waves"] is None:
        a["waves"] = a["predators"]
    result = run(**a)
    print(json.dumps({k:v for k,v in result.items() if k not in ("trace","events")}, indent=2))
