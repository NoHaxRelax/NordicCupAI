"""Publish only complete native-image real-map replays in 100-frame chunks."""
from pathlib import Path
import gzip
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "debugger"))
from replay_stream import iter_frames
RESULT_DIRS = [ROOT / "results" / name for name in ("real_map_intake_sol", "simple_chase", "release_validation", "replaceable_sites", "terrain_guide", "sequential_real_map")]
RESULT_DIRS.extend(ROOT / "results/short_overlap_sol" / name for name in ("approach_probe", "reacquire_probe", "v13_fresh", "v15_regression", "v16_emergency_probe", "v19_early_safety", "v20_contact_safe", "v20_fresh16"))
RESULT_DIRS.extend(p for p in (ROOT / "results/short_overlap_sol").iterdir()
                   if p.is_dir() and p.name.startswith(("v26_", "v28_", "v29_", "v30_", "v31_", "v33_")) and p not in RESULT_DIRS)
RESULT_DIRS.extend(p for p in (ROOT / "results/reacquisition_sol").iterdir()
                   if p.is_dir() and p.name != "replays")


def classify(receipt):
    if str(receipt.get("conditional_delivery_status", "")).startswith("invalid_"):
        return "invalid setup"
    if str(receipt.get("schema", "")).startswith("sequential-real-map-encounter-v"):
        return "sequential pilot" if receipt.get("success") else "failure"
    if receipt.get("schema") in ("replaceable-site-native-handoff-audit-v1", "replaceable-site-native-handoff-v1", "replaceable-site-native-crowd-handoff-v1"):
        return "prepared handoff"
    if receipt.get("outcome") == "unsupported_map_no_eligible_site":
        return "unsupported"
    if (str(receipt.get("policy", "")).endswith("NoBirthDiagnostic") or
            str(receipt.get("failure_stage", "")).startswith("no_birth_diagnostic")):
        return "no-birth diagnostic"
    if receipt.get("conditional_delivery_status") == "no_tracked_encounter_excluded":
        return "excluded"
    if receipt.get("conditional_delivery_status") == "encounter_without_native_child":
        return "no-birth diagnostic"
    if receipt.get("reason") == "stop sentinel":
        return "partial pilot"
    seconds = float(receipt.get("seconds", 0))
    requested = float(receipt.get("requested_seconds", seconds))
    if receipt.get("success") and requested >= 3000 and seconds >= requested - .1:
        return "full game"
    if receipt.get("success"):
        return "capture pilot"
    if receipt.get("success") is False and seconds >= 30:
        return "failure"
    return "short pilot"


def main():
    stop = Path("/tmp/predator-intake-stop")
    if stop.exists() and "IDE crash" in stop.read_text():
        print("Viewer processing paused after IDE crash; saved recordings remain intact.")
        return
    rows = []
    manifest_path = HERE / "manifest.json"
    cached_rows = {}
    cached_at = 0
    if manifest_path.exists():
        cached_at = manifest_path.stat().st_mtime_ns
        cached_rows = {row.get("receipt"): row for row in json.loads(manifest_path.read_text())}
    frame_root = HERE / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    corrections = {}
    for correction_path in (p for directory in RESULT_DIRS for p in directory.glob("*.correction.json")):
        correction = json.loads(correction_path.read_text())
        if correction.get("schema") != "real-map-intake-correction-v1":
            raise ValueError(f"unknown correction schema: {correction_path.name}")
        corrections[correction["receipt"]] = (correction_path, correction)
    for path in sorted(p for directory in RESULT_DIRS for p in directory.glob("*.json")):
        if path.name.endswith(".correction.json"):
            continue
        receipt = json.loads(path.read_text())
        if not isinstance(receipt,dict):
            continue
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
        cache_key = "/" + str(path.relative_to(ROOT))
        cached = cached_rows.get(cache_key)
        source_paths = [path, replay_path] + ([correction_path] if correction_path else [])
        # Original recordings are immutable; reuse already validated native chunks
        # only when all sources predate the last manifest and every chunk exists.
        if (cached and cached.get("frames") and (path.parent.name == "real_map_intake_sol" or cached.get("setup_label"))
                and all(p.stat().st_mtime_ns <= cached_at for p in source_paths)
                and all((HERE / cached["chunks"] / f"{i}.json.gz").exists()
                        for i in range((cached["frames"] + 99) // 100))):
            rows.append(cached)
            continue
        expected = round(float(receipt["seconds"]) * 10) + 1
        chunk_dir = frame_root / path.stem
        chunk_dir.mkdir(exist_ok=True)
        frame_count = 0
        payload = []
        native = True
        def publish_chunk(items, index):
            chunk = chunk_dir / f"{index}.json.gz"
            if not chunk.exists():
                temporary = chunk.with_suffix(f".{__import__('os').getpid()}.tmp")
                temporary.write_bytes(gzip.compress(json.dumps(items,separators=(",",":")).encode(),compresslevel=1))
                temporary.replace(chunk)
        # Chunks are atomically written from this immutable replay. An interrupted
        # library rebuild can validate its existing native-only chunks without
        # decompressing the much larger entity-state stream again.
        chunk_paths = [chunk_dir / f"{i}.json.gz" for i in range((expected+99)//100)]
        reuse_chunks = all(p.exists() and p.stat().st_mtime_ns >= replay_path.stat().st_mtime_ns
                           for p in chunk_paths)
        def chunk_frames():
            for chunk in chunk_paths:
                with gzip.open(chunk, "rt") as stream:
                    yield from json.load(stream)
        source_frames = chunk_frames() if reuse_chunks else iter_frames(replay_path)
        for frame in source_frames:
            if "native_image" not in frame:
                native = False
                break
            if abs(float(frame["t"]) - frame_count / 10) >= .001:
                raise AssertionError(f"non-contiguous frame times in {path.name}")
            payload.append({"t":frame["t"],"native_image":frame["native_image"]})
            frame_count += 1
            if len(payload)==100:
                publish_chunk(payload,frame_count//100-1)
                payload=[]
        if not native or not frame_count:
            continue
        if frame_count != expected:
            raise AssertionError((path.name,frame_count,expected))
        if payload:
            publish_chunk(payload,frame_count//100)
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
        setup_label = "random all starts" if not receipt.get("station_bait") else "arranged bait diagnostic"
        if receipt.get("arranged_adjacent_awake_start"):
            guide_role, parent_role, child_role = "single guide", "not_applicable", "not_spawned"
            setup_label = "adjacent awake start; predeployed bait"
        if receipt.get("arranged_aligned_final_approach"):
            setup_label = "aligned final-approach control; predeployed bait"
        if kind == "prepared handoff":
            setup_label = receipt.get('setup_label', 'prepared native-map bait replacement')
            guide_role, parent_role, child_role = 'replacement bait', 'old bait', 'not_spawned'
        if kind == "invalid setup":
            setup_label = receipt.get('setup_label', 'invalid privileged setup; excluded from reliability')
        if str(receipt.get("schema", "")).startswith("sequential-real-map-encounter-v"):
            setup_label = "repeated random predator + adjacent guide; predeployed bait"
            guide_role, parent_role, child_role = "independent guides", "not_applicable", "not_spawned"
            receipt["guided_delivered"] = receipt["successful_encounters"]
            receipt["initial_predators"] = receipt["requested_encounters"]
        label = ("reproduction · " if reproduced else "")
        if receipt.get("arranged_adjacent_awake_start"):
            label += ("aligned control · " if receipt.get("arranged_aligned_final_approach") else "adjacent start · ")
        rows.append({
            "kind": kind, "reproduction": reproduced, "setup_label": setup_label,
            "title": (f"{label}{kind} · map {receipt.get('map_seed')} · "
                      f"fixture {receipt.get('fixture_seed')} · {receipt['seconds']:g}s · "
                      f"{path.stem[-8:]}"),
            "chunks": f"frames/{path.stem}/", "frames": frame_count,
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
    strict_scores = {}
    for score_path in sorted((ROOT / "results/reliability_eval").rglob("*.score.json"), key=lambda p:p.stat().st_mtime_ns):
        score = json.loads(score_path.read_text())
        if score.get("receipt") and score.get("schema") == "guide-delivery-causal-score-v2":
            strict_scores["/" + score["receipt"].lstrip("/")] = (score_path, score)
    v4_scores = {}
    sys.path.insert(0, str(ROOT / "research/reliability_eval"))
    from protocol_v4 import PROTOCOL_SHA256 as current_v4_hash
    for score_path in sorted((ROOT / "results").rglob("*.score-v4.json"), key=lambda p:p.stat().st_mtime_ns):
        score = json.loads(score_path.read_text())
        if (score.get("receipt") and score.get("schema") == "guide-delivery-causal-score-v4"
                and score.get("protocol_sha256") == current_v4_hash):
            v4_scores["/" + score["receipt"].lstrip("/")] = (score_path, score)
    any_bait_scores = {}
    for score_path in sorted((ROOT / "results/multi_site").rglob("*.strict-any-bait.score.json"), key=lambda p:p.stat().st_mtime_ns):
        score = json.loads(score_path.read_text())
        if score.get('schema') == 'multi-site-any-bait-causal-score-v1':
            any_bait_scores['/' + score['receipt'].lstrip('/')] = (score_path, score)
    # Apply evaluator sidecars to presentation only, including cached rows.
    # Immutable original success stays visible separately.
    for row in rows:
        original = json.loads((ROOT / row['receipt'].lstrip('/')).read_text())
        if row.get('kind') == 'invalid setup':
            row['success'] = False
            row['guided_delivered'] = 0
            continue
        if (original.get('schema') == 'real-map-intake-result-v2'
                and row['receipt'] not in strict_scores and row['receipt'] not in v4_scores):
            row['kind'] = 'unscored native run'
            row['success'] = None
            row['guided_delivered'] = original.get('guided_delivered', 0)
            row['failure_stage'] = 'causal scoring pending; raw harness count only'
            row.pop('strict_v4_score', None)
            row.pop('capture_audit', None)
            policy_label = original.get('policy', 'policy').split(':')[0].split('.')[-1]
            row['title'] = (f"[{policy_label}] unscored · map {row.get('map_seed')} · "
                            f"{row.get('seconds',0):g}s · {Path(row['receipt']).stem[-8:]}")
        if original.get('multi_site_baits'):
            row['setup_label'] = (f"adjacent awake start; {len(original['multi_site_baits'])} static bait sites; "
                                  "site chosen from guide observations")
        if original.get('child_births'):
            row['guide_role'], row['parent_role'], row['child_role'] = 'guide lineage', 'parent guide', 'native reserve guides'
            row['guide_births'] = len(original['child_births'])
            row['guide_deaths'] = sum(d.get('role') != 'bait' for d in original.get('deaths', []))
            row['setup_label'] = f"adjacent awake start; {row['guide_births']} native reserve births; predeployed bait"
        policy = original.get('policy', '')
        if policy and not row.get('policy_label'):
            row['policy_label'] = policy.split(':')[0].split('.')[-1].removeprefix('policy_')
            row['title'] = '[' + row['policy_label'] + '] ' + row['title']
        audit_path = ROOT / "results/simple_chase/active_bait_audits" / (Path(row["receipt"]).stem + ".audit.json")
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            row["capture_audit"] = "/" + str(audit_path.relative_to(ROOT))
            row["original_harness_success"] = audit["original_harness_success"]
            if audit["distant_autonomous_capture"]:
                row["kind"] = "failure"
                row["success"] = False
                row["guided_delivered"] = 0
                row["failure_stage"] = "autonomous capture after distant guide death"
                if not row["title"].startswith("guiding failure · autonomous capture · "):
                    row["title"] = "guiding failure · autonomous capture · " + row["title"]
        if row["receipt"] in strict_scores:
            score_path, score = strict_scores[row["receipt"]]
            row["capture_audit"] = "/" + str(score_path.relative_to(ROOT))
            row["original_harness_success"] = original.get("success")
            row["strict_score"] = score["pass"]
            row["strict_reason"] = score["reason"]
            row["success"] = score["pass"]
            row["guided_delivered"] = int(score["pass"])
            if row["kind"] != "unsupported":
                row["kind"] = "capture pilot" if score["pass"] else "failure"
            row["failure_stage"] = None if score["pass"] else score["reason"]
            row["title"] = (f"[{row.get('policy_label', 'policy')}] strict {'PASS' if score['pass'] else 'FAIL'} · "
                f"map {row.get('map_seed')} · fixture {row.get('fixture_seed')} · {row.get('seconds', 0):g}s · "
                f"{Path(row['receipt']).stem[-8:]}")
        if row["receipt"] in v4_scores:
            score_path, score = v4_scores[row["receipt"]]
            row["strict_v3_score"] = row.get("strict_score")
            row["strict_v4_score"] = score["pass"]
            row["capture_audit"] = "/" + str(score_path.relative_to(ROOT))
            row["strict_reason"] = score["reason"]
            row["success"] = score["pass"]
            row["guided_delivered"] = int(score["pass"])
            if row["kind"] != "unsupported":
                row["kind"] = "capture pilot" if score["pass"] else "failure"
            row["failure_stage"] = None if score["pass"] else score["reason"]
            row["title"] = (f"[{row.get('policy_label', 'policy')}] v4 {'PASS' if score['pass'] else 'FAIL'} · "
                f"map {row.get('map_seed')} · fixture {row.get('fixture_seed')} · {row.get('seconds', 0):g}s · "
                f"{Path(row['receipt']).stem[-8:]}")
        if row['receipt'] in any_bait_scores:
            score_path, score = any_bait_scores[row['receipt']]
            row['capture_audit'] = '/' + str(score_path.relative_to(ROOT))
            row['strict_any_bait_score'] = score['pass']
            row['success'] = score['pass']
            row['guided_delivered'] = int(score['pass'])
            row['failure_stage'] = None if score['pass'] else score['reason']
            row['kind'] = 'capture pilot' if score['pass'] else 'failure'
            row['title'] = (f"[multi-site] any-bait {'PASS' if score['pass'] else 'FAIL'} · "
                            f"map {row.get('map_seed')} · {row.get('seconds', 0):g}s · "
                            f"{Path(row['receipt']).stem[-8:]}")
        retention_path = (ROOT / row['receipt'].lstrip('/')).with_suffix('.retention-audit.json')
        if row.get('kind') == 'prepared handoff' and retention_path.exists():
            audit = json.loads(retention_path.read_text())
            row['capture_audit'] = '/' + str(retention_path.relative_to(ROOT))
            row['success'] = audit['pass_value']
            row['prepared_retention_score'] = audit['pass_value']
            row['setup_label'] = ('prepared boundary refuge; rear bait replacement' if original.get('boundary_fixture')
                                  else 'prepared interior refuge; rear bait replacement')
            row['title'] = (f"prepared33 retention {'PASS' if audit['pass_value'] else 'FAIL'} · "
                            f"map {row.get('map_seed')} · {row.get('seconds', 0):g}s · "
                            f"{Path(row['receipt']).stem[-8:]}")
    order = {"full game": 0, "capture pilot": .5, "sequential pilot": .6, "prepared handoff": .75, "unscored native run": .9, "partial pilot": 3.5, "failure": 1, "no-birth diagnostic": 2,
             "excluded": 3, "short pilot": 4, "unsupported": 5, "invalid setup": 6}
    rows.sort(key=lambda row: (order[row["kind"]], -row.get("seconds", 0), row["title"]))
    temporary = HERE / f"manifest.{__import__('os').getpid()}.tmp"
    temporary.write_text(json.dumps(rows, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps({"cases": len(rows),
                      "native_frames": sum(row["frames"] for row in rows),
                      "native_only": True, "chunk_size": 100}))


if __name__ == "__main__":
    main()
