"""Representative short-overlap depth-5 crowd tests with native recordings."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

from run import run, ROOT, BASE_SOURCE_SHA256

STOP = Path("/tmp/predator-intake-stop")
OUT = ROOT / "results" / "short_overlap_sol"

# Fixed in advance: lower edge of the 10-20 bin, and representatives of the
# 20-30 and 30-40 census bins. Test all previously established native gap
# extremes/center with two simulator seeds. No result-dependent selection.
CASES = [
    dict(length=overlap, right_length=overlap, face_offset=0, gap=gap, seed=seed)
    for overlap in (10.3, 20.0, 30.0)
    for gap in (11.0, 15.0, 19.0)
    for seed in (731, 732)
]


def job(case):
    if STOP.exists():
        return {**case, "stopped": True}
    result = run(
        depth=5, predators=33, seconds=120, interval=.1, native=True,
        every=1, lateral_spread=15, heading_jitter=.3, horizontal=True,
        **case,
    )
    print({k: result.get(k) for k in
           ("overlap", "gap", "seed", "bait_alive", "acquired",
            "final_joint", "joint_success")}, flush=True)
    return result


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(job, CASES))
    summary = {
        "schema": "short-overlap-crowd-v1",
        "base_fixture_sha256": BASE_SOURCE_SHA256,
        "selection": "predeclared 3 overlaps x 3 gap widths x 2 seeds",
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

