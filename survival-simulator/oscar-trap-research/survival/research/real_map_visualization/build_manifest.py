"""Publish only complete native-image real-map replays in 100-frame chunks."""
from pathlib import Path
import gzip
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "real_map_intake_sol"


def classify(receipt):
    if receipt.get("outcome") == "unsupported_map_no_eligible_site":
        return "unsupported"
    if (str(receipt.get("policy", "")).endswith("NoBirthDiagnostic") or
            str(receipt.get("failure_stage", "")).startswith("no_birth_diagnostic")):
        return "no-birth diagnostic"
    if receipt.get("conditional_delivery_status") == "no_tracked_encounter_excluded":
        return "excluded"
    if receipt.get("conditional_delivery_status") == "encounter_without_native_child":
        return "no-birth diagnostic"
    seconds = float(receipt.get("seconds", 0))
    requested = float(receipt.get("requested_seconds", seconds))
    if receipt.get("success") and requested >= 3000 and seconds >= requested - .1:
        return "full game"
    if receipt.get("success") is False and seconds >= 30:
        return "failure"
    return "short pilot"


def main():
    rows = []
    frame_root = HERE / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    corrections = {}
    for correction_path in RESULTS.glob("*.correction.json"):
        correction = json.loads(correction_path.read_text())
        if correction.get("schema") != "real-map-intake-correction-v1":
            raise ValueError(f"unknown correction schema: {correction_path.name}")
        corrections[correction["receipt"]] = (correction_path, correction)
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.endswith(".correction.json"):
            continue
        receipt = json.loads(path.read_text())
        correction_path = None
        if path.name in corrections:
            correction_path, correction = corrections[path.name]
            receipt.update(correction["overrides"])
        # SmokePolicy receipts validate plumbing only and are explicitly not
        # predator-intake evidence.
        if str(receipt.get("policy", "")).endswith("SmokePolicy"):
            continue
        replay_rel = receipt.get("replay")
        if not replay_rel:
            if receipt.get("outcome") == "unsupported_map_no_eligible_site":
                rows.append({
                    "kind": "unsupported", "frames": 0,
                    "title": f"Unsupported map · seed {receipt.get('map_seed')}",
                    "map_seed": receipt.get("map_seed"),
                    "fixture_seed": receipt.get("fixture_seed"),
                    "receipt": "/" + str(path.relative_to(ROOT)),
                    "note": receipt.get("error", "No eligible refuge on this map."),
                })
            continue
        replay_path = ROOT / replay_rel
        if not replay_path.exists():
            continue
        data = json.loads(gzip.decompress(replay_path.read_bytes()))
        frames = data.get("frames", [])
        # State-only recordings are deliberately excluded. The viewer never
        # substitutes reconstructed or keyframed imagery for native pixels.
        if not frames or not all("native_image" in frame for frame in frames):
            continue
        expected = round(float(receipt["seconds"]) * 10) + 1
        if len(frames) != expected:
            raise AssertionError((path.name, len(frames), expected))
        if not all(abs(float(frame["t"]) - index / 10) < .001
                   for index, frame in enumerate(frames)):
            raise AssertionError(f"non-contiguous frame times in {path.name}")
        chunk_dir = frame_root / path.stem
        chunk_dir.mkdir(exist_ok=True)
        for start in range(0, len(frames), 100):
            payload = [{"t": frame["t"], "native_image": frame["native_image"]}
                       for frame in frames[start:start + 100]]
            chunk = chunk_dir / f"{start // 100}.json.gz"
            if not chunk.exists():
                chunk.write_bytes(gzip.compress(
                    json.dumps(payload, separators=(",", ":")).encode(),
                    compresslevel=1))
        kind = classify(receipt)
        reproduced = bool(receipt.get("reproduction_of") or
                          receipt.get("run_label") == "reproduction" or
                          receipt.get("reproduction"))
        child_guide_event = any(event.get("kind") == "lineage_guide_role"
                                and event.get("agent_id") not in (None, 1)
                                for event in receipt.get("policy_events", []))
        inferred_guide = "child" if child_guide_event else "parent"
        guide_role = receipt.get("guide_role") or inferred_guide
        parent_role = receipt.get("parent_role") or (
            "gather_evade" if guide_role == "child" else "guide")
        child_role = receipt.get("child_role") or (
            "guide" if guide_role == "child" else "gather_evade")
        label = "reproduction · " if reproduced else ""
        rows.append({
            "kind": kind, "reproduction": reproduced,
            "title": (f"{label}{kind} · map {receipt.get('map_seed')} · "
                      f"fixture {receipt.get('fixture_seed')} · {receipt['seconds']:g}s · "
                      f"{path.stem[-8:]}"),
            "chunks": f"frames/{path.stem}/", "frames": len(frames),
            "seconds": receipt["seconds"], "requested": receipt.get("requested_seconds"),
            "reason": receipt.get("reason"), "success": receipt.get("success", False),
            "map_seed": receipt.get("map_seed"), "fixture_seed": receipt.get("fixture_seed"),
            "bait_deployed_at": receipt.get("bait_deployed_at"),
            "guided_delivered": receipt.get("guided_delivered", 0),
            "first_encounter": receipt.get("first_guide_predator_observation"),
            "failure_stage": receipt.get("failure_stage"),
            "gaze_route_exercised": receipt.get("gaze_route_exercised"),
            "guide_role": guide_role, "parent_role": parent_role,
            "child_role": child_role,
            "predators": receipt.get("initial_predators"),
            "random_all_starts": not receipt.get("station_bait", False),
            "receipt": "/" + str(path.relative_to(ROOT)),
            "correction": (None if correction_path is None else
                           "/" + str(correction_path.relative_to(ROOT))),
            "replay": "/" + replay_rel,
        })
    order = {"full game": 0, "failure": 1, "no-birth diagnostic": 2,
             "excluded": 3, "short pilot": 4, "unsupported": 5}
    rows.sort(key=lambda row: (order[row["kind"]], -row.get("seconds", 0), row["title"]))
    (HERE / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"cases": len(rows),
                      "native_frames": sum(row["frames"] for row in rows),
                      "native_only": True, "chunk_size": 100}))


if __name__ == "__main__":
    main()
