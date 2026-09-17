"""Candidate v4 scorer: native target handoff precedes physical acquisition."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from debugger.replay_stream import inspect_stream, iter_frames
from protocol_v4 import PROTOCOL, PROTOCOL_SHA256


def _eligible_guides(receipt: dict):
    """Return causal scope without treating nearest-guide telemetry as truth.

    A single-guide receipt remains scoped to its assigned guide.  Once the
    policy has created or identified reserve guides, v4's colony-lineage scope
    accepts the native predator's actual target among those policy-controlled
    agents.  Birth records cover short-lived children absent from the next DTO
    and therefore absent from ``guide_role_ids``.
    """
    assigned = receipt.get("guide_id")
    role_ids = set(receipt.get("guide_role_ids") or [])
    born_ids = {row.get("child_id") for row in receipt.get("child_births", [])}
    eligible = ({assigned} | role_ids | born_ids) - {None, receipt.get("bait_id")}
    colony = bool((eligible - {assigned}) or receipt.get("child_births"))
    return eligible, ("native_lineage_any_policy_controlled_guide"
                      if colony else "single_assigned_guide")


class TargetChain:
    """Pure target/handoff state machine shared by replay scoring and tests."""

    def __init__(self, bait_id, *, max_active_seconds=3.0):
        self.bait_id = bait_id
        self.max_active_seconds = max_active_seconds
        self.previous_time = None
        self.previous_target = object()
        self.last_guide_time = None
        self.active_since_guide = None
        self.chain = None

    def update(self, *, time, target, eligible_guide_ids, physical):
        elapsed = 0.0 if self.previous_time is None else time - self.previous_time
        changed = target != self.previous_target
        if target in eligible_guide_ids:
            self.last_guide_time = time
            self.active_since_guide = 0.0
            self.last_guide_id = target
        elif target != "resting" and self.active_since_guide is not None:
            self.active_since_guide += elapsed

        if self.chain is not None and target not in (self.bait_id, "resting"):
            self.chain = None

        if (self.chain is None and changed and target == self.bait_id
                and self.last_guide_time is not None
                and self.active_since_guide is not None
                and self.active_since_guide <= self.max_active_seconds + .001):
            self.chain = {
                "handoff_time": time,
                "last_guide_target": self.last_guide_time,
                "handoff_guide_id": self.last_guide_id,
                "active_gap": self.active_since_guide,
                "wall_gap": time - self.last_guide_time,
                "physical_start": None,
                "physical_ticks": 0,
            }

        confirmed = None
        if self.chain is not None:
            if physical and target in (self.bait_id, "resting"):
                if self.chain["physical_ticks"] == 0:
                    self.chain["physical_start"] = time
                self.chain["physical_ticks"] += 1
            else:
                self.chain["physical_start"] = None
                self.chain["physical_ticks"] = 0
            if self.chain["physical_ticks"] >= round(
                    PROTOCOL["active_acquisition_seconds"] * 10):
                confirmed = dict(self.chain)

        self.previous_time = time
        self.previous_target = target
        return confirmed


def score(receipt_path: Path) -> dict:
    receipt_path = receipt_path.resolve()
    receipt = json.loads(receipt_path.read_text())
    base = {
        "schema": "guide-delivery-causal-score-v4",
        "protocol_sha256": PROTOCOL_SHA256,
        "receipt": str(receipt_path.relative_to(ROOT)),
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "policy": receipt.get("policy"),
        "policy_sha256": receipt.get("policy_hash"),
        "pass": False,
    }
    if receipt.get("schema") != "real-map-intake-result-v2":
        return base | {"reason": "unsupported_or_setup_receipt"}
    if (receipt.get("mode") != PROTOCOL["mode"]
            or receipt.get("station_bait") is not True
            or receipt.get("native_render") is not True
            or receipt.get("arranged_adjacent_awake_start") is not True
            or receipt.get("agent_energy_refilled") is not True):
        return base | {"reason": "receipt_not_protocol_configuration"}
    replay_path = ROOT / receipt["replay"]
    replay_check = inspect_stream(replay_path)
    if (receipt.get("record_every_ticks") != 1
            or replay_check["frames"] != receipt.get("replay_frames")
            or replay_check["native_frames"] != replay_check["frames"]):
        return base | {"reason": "replay_not_complete_every-frame_native",
                       "replay_check": replay_check}
    if (receipt.get("reason") != "horizon"
            or receipt.get("seconds", 0) < PROTOCOL["seconds"] - .051):
        return base | {"reason": "horizon_not_completed",
                       "replay_check": replay_check}
    if not receipt.get("bait_alive") or receipt.get("bait_left_after_deployment") is not None:
        return base | {"reason": "bait_not_stably_alive_at_site",
                       "replay_check": replay_check}
    if receipt.get("initial_predators") != 1:
        return base | {"reason": "protocol_requires_one_tracked_predator",
                       "replay_check": replay_check}

    switches = [row for row in receipt.get("target_switches", [])
                if row.get("tracked_predator_slot") == 0]
    switch_index = 0
    target = None
    tracked_id = None
    chain = TargetChain(receipt["bait_id"], max_active_seconds=
                        PROTOCOL["guide_to_bait_handoff_max_active_seconds"])
    eligible_guides, attribution_scope = _eligible_guides(receipt)
    acquisition = None
    handoff = None
    hold_ticks = 0
    loss_after_acquisition = None
    guide_alive_at_acquisition = None

    for frame in iter_frames(replay_path):
        now = float(frame["t"])
        if tracked_id is None and frame["predators"]:
            tracked_id = frame["predators"][0]["id"]
        predator = next((p for p in frame["predators"] if p["id"] == tracked_id), None)
        while switch_index < len(switches) and switches[switch_index]["time"] <= now + .001:
            target = switches[switch_index]["target"]
            switch_index += 1
        if predator is None:
            physical = False
        else:
            delta = (predator["x"] - receipt["site"]["mouth"][0],
                     predator["y"] - receipt["site"]["mouth"][1])
            physical = (math.hypot(*delta) <= 75.001
                        and sum(delta[i] * receipt["site"]["inward"][i]
                                for i in range(2)) <= 10.001)

        if acquisition is None:
            confirmed = chain.update(time=now, target=target,
                                     eligible_guide_ids=eligible_guides,
                                     physical=physical)
            if confirmed is not None:
                acquisition = confirmed["physical_start"]
                handoff = confirmed
                hold_ticks = confirmed["physical_ticks"]
                guide_alive_at_acquisition = any(
                    a["id"] == confirmed["handoff_guide_id"]
                    for a in frame["agents"])
        else:
            contained = physical and target in (receipt["bait_id"], "resting")
            if contained:
                hold_ticks += 1
            elif loss_after_acquisition is None:
                loss_after_acquisition = now

    pass_value = bool(
        acquisition is not None
        and acquisition <= PROTOCOL["latest_active_acquisition_seconds"] + .001
        and receipt.get("tracked_followed_active_guide", 0) >= 1
        and hold_ticks >= round(PROTOCOL["hold_seconds"] * 10)
        and loss_after_acquisition is None
    )
    if acquisition is None:
        reason = "no_unbroken_guide_to_bait_chain"
    elif acquisition > PROTOCOL["latest_active_acquisition_seconds"] + .001:
        reason = "active_bait_acquisition_after_270_seconds"
    elif receipt.get("tracked_followed_active_guide", 0) < 1:
        reason = "no_native_chase-gate_guide_follow_evidence"
    elif hold_ticks < round(PROTOCOL["hold_seconds"] * 10) or loss_after_acquisition is not None:
        reason = "containment_not_continuous_through_300_second_horizon"
    else:
        reason = "pass"
    return base | {
        "pass": pass_value,
        "reason": reason,
        "active_bait_acquisition": acquisition,
        "bait_target_handoff": None if handoff is None else handoff["handoff_time"],
        "last_guide_target_before_handoff": None if handoff is None else handoff["last_guide_target"],
        "handoff_guide_id": None if handoff is None else handoff["handoff_guide_id"],
        "eligible_guide_ids": sorted(eligible_guides),
        "attribution_scope": attribution_scope,
        "guide_to_bait_active_seconds": None if handoff is None else handoff["active_gap"],
        "guide_to_bait_wall_seconds": None if handoff is None else handoff["wall_gap"],
        "guide_alive_at_acquisition": guide_alive_at_acquisition,
        "guide_death": next((d["time"] for d in receipt.get("deaths", [])
                             if d["agent_id"] == receipt.get("guide_id")), None),
        "stable_containment_tail_seconds": round(hold_ticks / 10, 1),
        "loss_after_acquisition": loss_after_acquisition,
        "guide_release_observed": next((event for event in receipt.get("policy_events", [])
                                        if event.get("kind") == "guide_release_observed"), None),
        "guide_returned_to_staging": next((event for event in receipt.get("policy_events", [])
                                           if event.get("kind") == "guide_returned_to_staging"), None),
        "replay_check": replay_check,
        "caveat": ("V4 attributes a temporally continuous native guide-target to "
                   "bait-target chain before physical intake; it does not establish "
                   "counterfactual causation. Guide survival remains neutral."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.receipt)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
