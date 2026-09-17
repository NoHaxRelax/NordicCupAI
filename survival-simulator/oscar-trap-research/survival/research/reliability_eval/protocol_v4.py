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
        "of the last active-guide target starts a handoff chain anywhere; every "
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
})


def canonical_bytes() -> bytes:
    return json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()


PROTOCOL_SHA256 = hashlib.sha256(canonical_bytes()).hexdigest()
