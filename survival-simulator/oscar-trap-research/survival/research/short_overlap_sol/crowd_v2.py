"""Predeclared 33-predator crowd screen for accepting policy v2."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from run_v2 import run, ROOT, SOURCE_HASHES

STOP = Path("/tmp/predator-intake-stop")
OUT = ROOT / "results" / "short_overlap_sol" / "v2"
CASES = [dict(length=o, right_length=o, face_offset=0, gap=g, seed=s)
         for o in (10.3, 20., 30.) for g in (11., 15., 19.) for s in (771, 772)]


def job(case):
    if STOP.exists(): return {**case, "stopped": True}
    row = run(depth=5, predators=33, seconds=120, interval=.1, native=True,
              every=1, lateral_spread=15, heading_jitter=.3,
              horizontal=True, **case)
    print({k: row.get(k) for k in ("overlap", "gap", "seed", "seconds",
          "bait_alive", "acquired", "physical_losses", "final_joint")}, flush=True)
    return row


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(job, CASES))
    (OUT / "crowd-summary.json").write_text(json.dumps({
        "schema": "short-overlap-v2-crowd-v1", "source_hashes": SOURCE_HASHES,
        "selection": "overlaps 10.3/20/30 x gaps 11/15/19 x seeds771/772",
        "rows": rows,
    }, indent=2) + "\n")
