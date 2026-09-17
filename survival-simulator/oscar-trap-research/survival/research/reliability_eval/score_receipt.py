"""Strict post-hoc delivery scorer over immutable receipt and replay files."""
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
from protocol import PROTOCOL, PROTOCOL_SHA256


def _active_guide_at(receipt: dict, when: float):
    active = receipt.get("guide_id")
    for event in receipt.get("guide_role_trace", []):
        if event["time"] <= when + 1e-6:
            active = event["active_guide_id"]
    return active


def score(receipt_path: Path) -> dict:
    receipt_path = receipt_path.resolve()
    receipt = json.loads(receipt_path.read_text())
    base = {
        "schema": "guide-delivery-causal-score-v2",
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
    confirmation_start = None
    confirmation_ticks = 0
    confirmation_handoff = None
    confirmation_active_gap = None
    confirmation_wall_gap = None
    acquisition = None
    handoff = None
    last_guide_target_time = None
    active_seconds_since_guide_target = None
    hold_ticks = 0
    loss_after_acquisition = None
    guide_alive_at_acquisition = None
    required_acquisition_ticks = round(PROTOCOL["active_acquisition_seconds"] * 10)
    required_hold_ticks = round(PROTOCOL["hold_seconds"] * 10)
    previous_time = None

    for frame in iter_frames(replay_path):
        now = float(frame["t"])
        elapsed = 0.0 if previous_time is None else now - previous_time
        previous_time = now
        if tracked_id is None and frame["predators"]:
            tracked_id = frame["predators"][0]["id"]
        predator = next((p for p in frame["predators"] if p["id"] == tracked_id), None)
        while switch_index < len(switches) and switches[switch_index]["time"] <= now + .001:
            target = switches[switch_index]["target"]
            switch_index += 1
        active_guide = _active_guide_at(receipt, now)
        if target == active_guide:
            last_guide_target_time = now
            active_seconds_since_guide_target = 0.0
        elif target != "resting" and active_seconds_since_guide_target is not None:
            active_seconds_since_guide_target += elapsed
        if predator is None:
            physical = False
        else:
            delta = (predator["x"] - receipt["site"]["mouth"][0],
                     predator["y"] - receipt["site"]["mouth"][1])
            physical = (math.hypot(*delta) <= 75.001
                        and sum(delta[i] * receipt["site"]["inward"][i]
                                for i in range(2)) <= 10.001)

        # Only an active bait target can start confirmation. Native rest is valid
        # after that observed handoff, provided physical containment never breaks.
        if acquisition is None:
            if confirmation_start is None:
                if physical and target == receipt["bait_id"]:
                    confirmation_start = now
                    confirmation_ticks = 1
                    if (last_guide_target_time is not None
                            and active_seconds_since_guide_target is not None
                            and active_seconds_since_guide_target
                            <= PROTOCOL["guide_to_bait_handoff_max_active_seconds"] + .001):
                        confirmation_handoff = last_guide_target_time
                        confirmation_active_gap = active_seconds_since_guide_target
                        confirmation_wall_gap = now - last_guide_target_time
            elif physical and target in (receipt["bait_id"], "resting"):
                confirmation_ticks += 1
            else:
                confirmation_start = None
                confirmation_ticks = 0
                confirmation_handoff = None
                confirmation_active_gap = None
                confirmation_wall_gap = None
            if confirmation_ticks >= required_acquisition_ticks:
                    acquisition = confirmation_start
                    hold_ticks = required_acquisition_ticks
                    guide_alive_at_acquisition = any(
                        a["id"] == active_guide for a in frame["agents"])
                    handoff = confirmation_handoff
        else:
            # After active acquisition, native rest is acceptable containment.
            contained = physical and target in (receipt["bait_id"], "resting")
            if contained:
                hold_ticks += 1
            else:
                if loss_after_acquisition is None:
                    loss_after_acquisition = now

    pass_value = bool(
        acquisition is not None
        and acquisition <= PROTOCOL["latest_active_acquisition_seconds"] + .001
        and handoff is not None
        and receipt.get("tracked_followed_active_guide", 0) >= 1
        and hold_ticks >= required_hold_ticks
        and loss_after_acquisition is None
    )
    if acquisition is None:
        reason = "no_active_bait_acquisition"
    elif acquisition > PROTOCOL["latest_active_acquisition_seconds"] + .001:
        reason = "active_bait_acquisition_after_270_seconds"
    elif handoff is None:
        reason = "no_recent_guide_to_bait_handoff"
    elif receipt.get("tracked_followed_active_guide", 0) < 1:
        reason = "no_native_chase-gate_guide_follow_evidence"
    elif hold_ticks < required_hold_ticks or loss_after_acquisition is not None:
        reason = "containment_not_continuous_through_300_second_horizon"
    else:
        reason = "pass"
    return base | {
        "pass": pass_value,
        "reason": reason,
        "active_bait_acquisition": acquisition,
        "last_guide_target_before_acquisition": handoff,
        "guide_to_bait_active_seconds": confirmation_active_gap,
        "guide_to_bait_wall_seconds": confirmation_wall_gap,
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
        "caveat": ("This establishes a temporally continuous native guide-target to "
                   "active-bait handoff, not counterfactual causation. Guide survival "
                   "is explicitly neutral. Release/return events are ordinary-observation "
                   "policy evidence reported separately from delivery success."),
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
