"""Strict any-static-bait causal audit over immutable native replay frames.

This protocol is separate from frozen single-bait v4, but uses the same native
chase-qualified guide -> bait target-chain semantics for each declared site.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research"),
                str(ROOT / "research" / "reliability_eval")]

from debugger.replay_stream import inspect_stream, iter_frames
from score_receipt_v4 import TargetChain, _eligible_guides, _guide_chase_gate

PROTOCOL = {
    "schema": "multi-site-any-bait-causal-protocol-v1",
    "seconds": 300.0,
    "latest_acquisition_seconds": 270.0,
    "guide_to_bait_handoff_max_active_seconds": 3.0,
    "physical_confirmation_seconds": 2.0,
    "frame_seconds": 0.1,
    "physical_radius": 75.0,
    "maximum_inward_projection": 10.0,
    "rule": (
        "For any statically declared bait/site pair, a native bait target may "
        "start a chain within three predator-active seconds of a replayed native "
        "chase-gate-qualified eligible-guide target. Until physical confirmation "
        "and for every remaining frame through 300 seconds, that site's target "
        "must be its bait or native rest and the predator must remain physically "
        "inside that same site. Rest may continue but cannot initiate a chain. "
        "Switching to another bait, guide, None, or other target breaks the site chain."
    ),
}
PROTOCOL_SHA256 = hashlib.sha256(json.dumps(
    PROTOCOL, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _physical(predator, site):
    if predator is None:
        return False
    dx = predator["x"] - site["mouth"][0]
    dy = predator["y"] - site["mouth"][1]
    return (math.hypot(dx, dy) <= PROTOCOL["physical_radius"] + .001
            and dx * site["inward"][0] + dy * site["inward"][1]
            <= PROTOCOL["maximum_inward_projection"] + .001)


def score(receipt_path):
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text())
    base = {
        "schema": "multi-site-any-bait-causal-score-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "single_bait_semantics": "equivalent to frozen v4 target-chain semantics",
        "receipt": str(receipt_path.relative_to(ROOT)),
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "policy": receipt.get("policy"),
        "policy_sha256": receipt.get("policy_hash"),
        "pass": False,
    }
    declared = receipt.get("multi_site_baits")
    if (not isinstance(declared, list) or not declared
            or receipt.get("multi_site_bait_count") != len(declared)
            or len({row.get("bait_id") for row in declared}) != len(declared)):
        return base | {"reason": "missing_or_invalid_static_bait_mapping"}
    if receipt.get("schema") != "real-map-intake-result-v2":
        return base | {"reason": "unsupported_or_setup_receipt"}
    if (receipt.get("mode") != "direct_guide"
            or receipt.get("station_bait") is not True
            or receipt.get("native_render") is not True
            or receipt.get("arranged_adjacent_awake_start") is not True
            or receipt.get("agent_energy_refilled") is not True
            or receipt.get("initial_predators") != 1):
        return base | {"reason": "receipt_not_protocol_configuration"}
    replay_path = ROOT / receipt["replay"]
    replay_check = inspect_stream(replay_path)
    if (receipt.get("record_every_ticks") != 1
            or replay_check["frames"] != receipt.get("replay_frames")
            or replay_check["native_frames"] != replay_check["frames"]
            or replay_check.get("valid") is not True):
        return base | {"reason": "replay_not_complete_every-frame_native",
                       "replay_check": replay_check}
    if (receipt.get("reason") != "horizon"
            or receipt.get("seconds", 0) < PROTOCOL["seconds"] - .051
            or replay_check.get("reason") != "horizon"
            or replay_check.get("duration", 0)
            < PROTOCOL["seconds"] - .051):
        return base | {"reason": "full_300_second_horizon_not_completed",
                       "replay_check": replay_check}

    sites = {row["bait_id"]: row["site"] for row in declared}
    eligible_guides, attribution_scope = _eligible_guides(receipt)
    switches = [row for row in receipt.get("target_switches", [])
                if row.get("tracked_predator_slot") == 0]
    switch_index = 0
    target = None
    tracked_id = None
    trackers = {}
    for bait_id, site in sites.items():
        trackers[bait_id] = {
            "site": site,
            "chain": TargetChain(
                bait_id,
                max_active_seconds=PROTOCOL[
                    "guide_to_bait_handoff_max_active_seconds"]),
            "acquisition": None,
            "handoff": None,
            "loss_after_acquisition": None,
            "bait_left_static_goal": None,
            "final_inside": False,
            "final_target": None,
        }

    for frame in iter_frames(replay_path):
        now = float(frame["t"])
        if tracked_id is None and frame["predators"]:
            tracked_id = frame["predators"][0]["id"]
        predator = next((row for row in frame["predators"]
                         if row["id"] == tracked_id), None)
        while switch_index < len(switches) and switches[switch_index]["time"] <= now + .001:
            target = switches[switch_index]["target"]
            switch_index += 1
        chase_qualified = _guide_chase_gate(
            predator, frame["agents"], target, eligible_guides)
        agents = {row["id"]: row for row in frame["agents"]}
        for bait_id, state in trackers.items():
            site = state["site"]
            bait = agents.get(bait_id)
            if (bait is None or math.dist((bait["x"], bait["y"]),
                                         site["goal"]) > .01):
                if state["bait_left_static_goal"] is None:
                    state["bait_left_static_goal"] = now
            inside = _physical(predator, site)
            if state["acquisition"] is None:
                confirmed = state["chain"].update(
                    time=now, target=target,
                    eligible_guide_ids=eligible_guides,
                    physical=inside,
                    guide_chase_qualified=chase_qualified)
                if confirmed is not None:
                    state["acquisition"] = confirmed["physical_start"]
                    state["handoff"] = confirmed
            elif (not inside or target not in (bait_id, "resting")):
                if state["loss_after_acquisition"] is None:
                    state["loss_after_acquisition"] = now
            state["final_inside"] = inside
            state["final_target"] = target

    site_results = []
    for bait_id, state in sorted(trackers.items()):
        handoff = state["handoff"]
        acquisition = state["acquisition"]
        passed = bool(
            acquisition is not None
            and acquisition <= PROTOCOL["latest_acquisition_seconds"] + .001
            and state["loss_after_acquisition"] is None
            and state["bait_left_static_goal"] is None
            and state["final_inside"]
            and state["final_target"] in (bait_id, "resting")
        )
        if acquisition is None:
            reason = "no_unbroken_guide_to_this_bait_physical_chain"
        elif acquisition > PROTOCOL["latest_acquisition_seconds"] + .001:
            reason = "site_acquisition_after_270_seconds"
        elif state["bait_left_static_goal"] is not None:
            reason = "declared_bait_not_static_at_site"
        elif state["loss_after_acquisition"] is not None:
            reason = "same_site_physical_or_target_containment_not_continuous"
        else:
            reason = "pass"
        site_results.append({
            "bait_id": bait_id,
            "pass": passed,
            "reason": reason,
            "physical_acquisition": acquisition,
            "bait_target_handoff": None if handoff is None else handoff["handoff_time"],
            "handoff_guide_id": None if handoff is None else handoff["handoff_guide_id"],
            "guide_to_bait_active_seconds": None if handoff is None else handoff["active_gap"],
            "guide_to_bait_wall_seconds": None if handoff is None else handoff["wall_gap"],
            "loss_after_acquisition": state["loss_after_acquisition"],
            "bait_left_static_goal": state["bait_left_static_goal"],
            "final_inside_same_site": state["final_inside"],
            "final_target": state["final_target"],
        })
    winners = [row for row in site_results if row["pass"]]
    return base | {
        "pass": bool(winners),
        "reason": "pass" if winners else "no_static_bait_site_satisfied_protocol",
        "bait_count": len(sites),
        "passing_bait_ids": [row["bait_id"] for row in winners],
        "eligible_guide_ids": sorted(eligible_guides),
        "attribution_scope": attribution_scope,
        "site_results": site_results,
        "replay_check": replay_check,
        "caveat": ("Any declared static bait may pass, but each candidate is "
                   "evaluated independently. Targeting another bait breaks the "
                   "current site's chain; physical retention without a qualifying "
                   "guide handoff remains unsupported rather than causal success."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.receipt)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
