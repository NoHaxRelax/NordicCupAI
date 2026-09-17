"""Candidate v4 protocol; separate from the immutable v3 scorer and sidecars."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from protocol import PROTOCOL as V3_PROTOCOL

PROTOCOL = deepcopy(V3_PROTOCOL)
PROTOCOL.update({
    "schema": "guide-delivery-reliability-protocol-v4",
    "acquisition_rule": (
        "a native active-bait target switch within three predator-active seconds "
        "of the last native chase-gate-qualified eligible-guide target starts a "
        "handoff chain anywhere; closest-visible guide labels from the native "
        "pivot branch do not count as guide pursuit; every "
        "subsequent target through physical entry and two-second confirmation "
        "must be bait or native rest; rest may continue but cannot initiate the "
        "chain; awake None or another target breaks it"
    ),
    "attribution_scopes": {
        "single_assigned_guide": (
            "where no reserve lineage exists, the native target must be the "
            "original assigned guide"
        ),
        "native_lineage_any_policy_controlled_guide": (
            "where policy-controlled reserve lineage exists, any native target "
            "identified by original guide_id, guide_role_ids, or child_births "
            "may initiate the handoff; active_guide_id is nearest-distance public "
            "telemetry and is not treated as the predator's target"
        ),
    },
    "legacy_active_guide_counter_rule": (
        "tracked_followed_active_guide is not a pass requirement in v4 because "
        "it compares pursuit with fluctuating nearest-guide telemetry; the "
        "per-frame replayed native chase gate for the actual target supplies "
        "the causal pursuit evidence instead"
    ),
})


def canonical_bytes() -> bytes:
    return json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()


PROTOCOL_SHA256 = hashlib.sha256(canonical_bytes()).hexdigest()
