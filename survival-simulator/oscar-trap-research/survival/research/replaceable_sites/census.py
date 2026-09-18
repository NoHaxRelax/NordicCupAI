"""Compare frozen v5 geometry with replaceable-site geometry; no simulation."""
from __future__ import annotations

import json
from pathlib import Path

from replaceable_sites.selector import enumerate_sites

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "real_map_census"
OUT = ROOT / "results" / "replaceable_sites"


def payload(row):
    return {"width": 1600, "height": 1200, "obstacles": row["obstacles"]}


def old_eligible(static_map):
    # Same geometry and ordering as policy_v5_short_site, without importing or
    # constructing a runtime controller. Disabling second access gives its
    # exact short-overlap predicate and clear-runup checks.
    return enumerate_sites(static_map, min_overlap=20.0,
                           require_second_access=False,
                           allow_offset_approach=False)


def main():
    rows = []
    for path in sorted(SOURCE.glob("map-*.json")):
        raw = json.loads(path.read_text())
        static_map = payload(raw)
        old = old_eligible(static_map)
        new = enumerate_sites(static_map)
        rows.append({
            "seed": raw["seed"], "old_short_site": bool(old),
            "replaceable_site": bool(new),
            "old_site_count": len(old), "replaceable_site_count": len(new),
            "old_boundary_site": any(s["boundary_indices"] for s in old),
            "replaceable_boundary_site": any(s["boundary_indices"] for s in new),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "replaceable-static-site-census-v1",
        "seed_range": [min(r["seed"] for r in rows), max(r["seed"] for r in rows)],
        "maps": len(rows), "bait_depth": 5.0, "gap": [10.9, 19.1],
        "minimum_overlap": 20.0,
        "old_short_site_maps": sum(r["old_short_site"] for r in rows),
        "replaceable_site_maps": sum(r["replaceable_site"] for r in rows),
        "old_boundary_site_maps": sum(r["old_boundary_site"] for r in rows),
        "replaceable_boundary_site_maps": sum(r["replaceable_boundary_site"] for r in rows),
        "rows": rows,
        "notes": [
            "Exact preserved native generated-map rectangles; no actors and no simulation steps.",
            "Native arena walls are obstacle indices 0-3 and are included in both predicates.",
            "Second access is a static radius-5.01 straight path through the opposite channel mouth.",
            "It is not evidence of safe replacement while a predator occupies the site.",
        ],
    }
    (OUT / "census.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
