"""Frozen reliability protocol. Change means a new experiment, not an edit."""
from __future__ import annotations

import hashlib
import json

PROTOCOL = {
    "schema": "guide-delivery-reliability-protocol-v3",
    "cases": [
        {"case": index, "map_seed": 11000 + index,
         "fixture_seed": 21000 + index}
        for index in range(1, 60)
    ],
    "seconds": 300.0,
    "predators": 1,
    "station_bait": True,
    "mode": "direct_guide",
    "native_render": True,
    "native_width": 320,
    "record_every_ticks": 1,
    "hold_seconds": 30.0,
    "latest_active_acquisition_seconds": 270.0,
    "active_acquisition_seconds": 2.0,
    "acquisition_rule": (
        "first physical active bait target starts confirmation and fixes the "
        "guide-handoff timestamp; no more than three predator-active seconds may "
        "elapse since its last guide target, while native rest is excluded and "
        "the wall-clock gap is retained; the next two seconds must remain physical "
        "and may target bait or enter native rest; rest cannot start confirmation"
    ),
    "guide_to_bait_handoff_max_active_seconds": 3.0,
    "confidence": 0.95,
    "required_lower_bound_exclusive": 0.95,
    "denominator_rule": (
        "all 59 frozen cases; unsupported maps, setup errors, process failures, "
        "missing/invalid receipts, incomplete replays, and scoring errors fail"
    ),
    "guide_outcome_rule": (
        "guide death and guide survival are both permitted; neither is itself "
        "success or failure"
    ),
    "scope_rule": (
        "single already-engaged predator only; 33-predator sequential throughput "
        "and whole-game retention require separate evaluations"
    ),
}


def canonical_bytes() -> bytes:
    return json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()


PROTOCOL_SHA256 = hashlib.sha256(canonical_bytes()).hexdigest()
