"""Locate the short-overlap survival boundary before any longer validation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

from run import run, ROOT, BASE_SOURCE_SHA256

STOP = Path("/tmp/predator-intake-stop")
OUT = ROOT / "results" / "short_overlap_sol"
CASES = [
    dict(length=overlap, right_length=overlap, face_offset=0, gap=gap, seed=741)
    for overlap in (35., 40., 45., 50., 54.)
    for gap in (11., 15., 19.)
]


def job(case):
    if STOP.exists():
        return {**case, "stopped": True}
    result = run(depth=5, predators=1, seconds=30, interval=1, native=True,
                 every=1, lateral_spread=0, heading_jitter=0,
                 horizontal=True, **case)
    print({k: result.get(k) for k in
           ("overlap", "gap", "seconds", "bait_alive", "joint_success")},
          flush=True)
    return result


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(job, CASES))
    (OUT / "frontier-summary.json").write_text(json.dumps({
        "schema": "short-overlap-frontier-v1",
        "base_fixture_sha256": BASE_SOURCE_SHA256,
        "selection": "predeclared overlaps 35/40/45/50/54 x gaps 11/15/19",
        "rows": rows,
    }, indent=2) + "\n")
