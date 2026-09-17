"""Predeclared deployment and single-predator retention screen for policy v2."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

from run_v2 import run, ROOT, SOURCE_HASHES

STOP = Path("/tmp/predator-intake-stop")
OUT = ROOT / "results" / "short_overlap_sol" / "v2"
CASES = [dict(length=o, right_length=o, face_offset=0, gap=g, seed=761)
         for o in (10.3, 20., 30., 40., 50., 54., 55.)
         for g in (11., 15., 19.)]


def job(case):
    if STOP.exists(): return {**case, "stopped": True}
    row = run(depth=5, predators=1, seconds=30, interval=1, native=True,
              every=1, lateral_spread=0, heading_jitter=0,
              horizontal=True, **case)
    print({k: row.get(k) for k in ("overlap", "gap", "seconds", "mapped",
                                   "bait_alive", "acquired", "final_joint")},
          flush=True)
    return row


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(job, CASES))
    (OUT / "screen-summary.json").write_text(json.dumps({
        "schema": "short-overlap-v2-screen-v1", "source_hashes": SOURCE_HASHES,
        "selection": "overlaps 10.3/20/30/40/50/54/55 x gaps 11/15/19",
        "rows": rows,
    }, indent=2) + "\n")
