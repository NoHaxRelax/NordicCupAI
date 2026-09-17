"""Small deterministic contract audit; does not create or step an environment."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

from replaceable_sites.selector import enumerate_sites, select_site
from simple_chase.policy_v5_short_site import SimpleChase as FrozenV5

ROOT = Path(__file__).resolve().parents[2]
MAPS = ROOT / "results" / "real_map_census"
OUT = ROOT / "results" / "replaceable_sites"


def main():
    mismatches = []
    deterministic = True
    immutable = True
    compatible = True
    required = {"mouth", "inward", "cross", "goal", "far", "hold", "gap",
                "overlap", "obstacle_indices", "axis"}
    for path in sorted(MAPS.glob("map-*.json")):
        raw = json.loads(path.read_text())
        static_map = {"width": 1600, "height": 1200,
                      "obstacles": copy.deepcopy(raw["obstacles"])}
        before = copy.deepcopy(static_map)
        legacy = enumerate_sites(static_map, min_overlap=20,
                                 require_second_access=False,
                                 allow_offset_approach=False)
        old = object.__new__(FrozenV5)
        old.width, old.height, old.rects = 1600, 1200, raw["obstacles"]
        try:
            old._select_site()
            frozen_accepts = True
        except ValueError:
            frozen_accepts = False
        if frozen_accepts != bool(legacy):
            mismatches.append(raw["seed"])
        current = enumerate_sites(static_map)
        deterministic &= current == enumerate_sites(static_map)
        immutable &= static_map == before
        compatible &= all(required <= site.keys() for site in current)
        if current:
            compatible &= select_site(static_map) == current[0]
    signature = inspect.signature(select_site)
    report = {
        "schema": "replaceable-site-contract-v1",
        "maps_checked": 128,
        "frozen_v5_acceptance_mismatch_seeds": mismatches,
        "deterministic": deterministic,
        "input_immutable": immutable,
        "legacy_keys_preserved": compatible,
        "selector_parameters": list(signature.parameters),
        "static_only_interface": list(signature.parameters) == ["static_map", "kwargs"],
        "selector_sha256": hashlib.sha256(
            (Path(__file__).parent / "selector.py").read_bytes()).hexdigest(),
        "simulation_steps": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contract.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert not mismatches and deterministic and immutable and compatible


if __name__ == "__main__":
    main()
