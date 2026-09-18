"""Native every-frame multi-bait harness layered on streaming v2.

The underlying loop, arranged guide/predator pair, action validation, native
dynamics, evaluator, and recorder remain streaming v2. Setup adds one native
stationary agent at each selected static site. Bait IDs use deterministic
static-map order; only the runtime policy's first localized DTO selects one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from multi_site.policy import MultiSiteGuide, dispersed_sites
from replaceable_sites.selector_compact import enumerate_sites
from simple_chase import run_streaming_v2 as base


HERE = Path(__file__).resolve()
_policy_instance = None
_original_spawn = base._spawn_fixture
_original_hashes = base._harness_hashes


class NativeMultiSite(MultiSiteGuide):
    def __init__(self, static_map, bait_id=0, guide_id=1):
        global _policy_instance
        sites = dispersed_sites(enumerate_sites(static_map), limit=8)
        if not sites:
            raise ValueError("no eligible multi-bait static sites")
        ids = [0, *range(2, len(sites) + 1)]
        super().__init__(static_map, bait_id=bait_id, guide_id=guide_id,
                         bait_sites=dict(zip(ids, sites)))
        self.deployment_sites = sites
        self.site = sites[0]
        _policy_instance = self

    def configure_deployment(self, ordered_sites):
        ids = [0, *range(2, len(ordered_sites) + 1)]
        self.bait_sites = dict(zip(ids, ordered_sites))
        self.selected_bait_id = None
        # Preserve a truthful compatible site before the first DTO chooses.
        self.site = ordered_sites[0]


def _spawn_multi(env, fixture, *, site, station_bait):
    if not station_bait:
        raise ValueError("multi-site harness requires --station-bait")
    agents, predators, rows = _original_spawn(
        env, fixture, site=site, station_bait=True)
    policy = _policy_instance
    if policy is None:
        raise RuntimeError("multi-site policy was not constructed")
    guide = agents[1]
    # IDs follow the selector's deterministic static-map order. Runtime DTO
    # localization, never fixture truth, decides which ID/site the guide uses.
    ordered = policy.deployment_sites
    policy.configure_deployment(ordered)
    bait = agents[0]
    bait.x, bait.y = ordered[0]["goal"]
    rows["agents"][0].update(x=bait.x, y=bait.y,
                             setup_repositioned=True,
                             multi_site_bait=True, site_index=0)
    for index, extra_site in enumerate(ordered[1:], start=1):
        goal = tuple(extra_site["goal"])
        if env._in_obstacle(goal, radius=5., obstacles=env.obstacles):
            raise ValueError("multi-site bait goal intersects native agent radius")
        # Native spawn uses a 20x20 top-left box and rejects the deliberately
        # narrow trap goal. Follow the disclosed bait-0 setup pattern: create
        # the native agent at the already validated guide point, then arrange
        # its centre at the radius-5-safe static goal before the first frame.
        extra = env.spawn_agent(x=guide.x, y=guide.y)
        if extra is None:
            raise ValueError("native spawn rejected multi-site setup staging")
        extra.x, extra.y = goal
        rows["agents"].append({"slot": index + 1, "x": extra.x, "y": extra.y,
                               "heading": extra.direction,
                               "agent_id": extra.agent_id,
                               "arranged_at_site": True,
                               "setup_repositioned": True,
                               "multi_site_bait": True,
                               "site_index": index})
    env._update_spatial_grid()
    # The base loop intentionally binds only its scored bait and guide here;
    # every extra bait remains in env and therefore in every DTO/action/frame.
    return agents, predators, rows


def _load_policy(_spec):
    return NativeMultiSite, HERE


def _hashes():
    rows = _original_hashes()
    rows[str(HERE.relative_to(base.ROOT))] = hashlib.sha256(HERE.read_bytes()).hexdigest()
    policy_file = HERE.parent / "policy.py"
    rows[str(policy_file.relative_to(base.ROOT))] = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-seed", type=int, required=True)
    parser.add_argument("--fixture-seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=300.)
    parser.add_argument("--native-width", type=int, default=320)
    args = parser.parse_args()
    base._load_policy = _load_policy
    base._spawn_fixture = _spawn_multi
    base._harness_hashes = _hashes
    path, result = base.run(
        policy_spec="multi_site.run_multi_site:NativeMultiSite",
        map_seed=args.map_seed, fixture_seed=args.fixture_seed,
        seconds=args.seconds, predators=1, station_bait=True,
        native_render=True, native_width=args.native_width,
    )
    result["multi_site_baits"] = [
        {"bait_id": aid, "site": base.json_clone(site)}
        for aid, site in sorted(_policy_instance.bait_sites.items())
    ]
    result["multi_site_bait_count"] = len(_policy_instance.bait_sites)
    result["base_evaluator_scope"] = "bait_id_0_only; use multi_site/score_any_bait.py"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)
    return result


if __name__ == "__main__":
    main()
