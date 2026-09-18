#!/usr/bin/env python3
"""Run the prepared score shards against validation only, then verify receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path("/private/tmp/nordic-ai-cup-secrets")
ACTIVE_PLAN = ROOT / "artifacts/drone-api-tests/active-plan.json"


def run_portal(action: str, output: Path, url_file: Path | None = None) -> dict:
    command = [
        sys.executable,
        str(ROOT / "drone/portal.py"),
        action,
        "--output",
        str(output),
    ]
    if url_file is not None:
        command.extend(["--url-file", str(url_file)])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Portal {action} failed; inspect {output.parent}")
    return json.loads(output.read_text())


def free_port() -> int:
    for port in range(9054, 9071):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free callback port in the reserved range")


def wait_for_port(port: int, process: subprocess.Popen, deadline_seconds: int = 15) -> None:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Callback recorder exited during startup")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError("Callback recorder did not start listening")


def stop_child(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_complete_capture(
    plan: dict, capture_root: Path, deadline_seconds: int = 90, max_missing: int = 0
) -> list[int]:
    """Do not switch plans while any callback from the completed shard may still be writing."""
    expected_frames = set(range(1, 250))
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        by_frame = {}
        for path in sorted((capture_root / plan["name"]).glob("*/*.json")):
            row = json.loads(path.read_text())
            frame = int(row["frame"])
            if row["response"]["annotations"] != plan["predictions_by_frame"].get(str(frame), []):
                raise RuntimeError(f"Captured response differs from fixed plan at frame {frame}")
            by_frame[frame] = row
        if set(by_frame) == expected_frames:
            return []
        time.sleep(1)
    missing = sorted(expected_frames - set(by_frame))
    if len(missing) <= max_missing:
        return missing
    raise RuntimeError(f"Capture incomplete after completed validation; missing frames: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "artifacts/drone-api-tests/score-assessment-20260917",
    )
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--max-missing-capture", type=int, default=0)
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=ROOT / "data/drone/capture/score-assessment-20260917",
    )
    args = parser.parse_args()
    bundle = args.bundle
    capture_root = args.capture_root
    manifest = json.loads((bundle / "manifest.json").read_text())
    plans = [bundle / item["file"] for item in manifest["plans"]]
    plans_to_run = []
    for plan_path in plans:
        plan = json.loads(plan_path.read_text())
        folder = ROOT / "artifacts/drone-api-tests/score-probes" / plan["name"]
        receipt = folder / "queue.json"
        if receipt.exists():
            if not args.resume_completed:
                raise RuntimeError(f"Refusing to resubmit existing probe receipt: {receipt}")
            saved_plan = json.loads((folder / "plan.json").read_text())
            result = json.loads((folder / "result.json").read_text())
            if saved_plan != plan or result.get("errors"):
                raise RuntimeError(f"Existing probe is not a clean completion: {folder}")
            continue
        plans_to_run.append(plan_path)

    before = run_portal("status", bundle / "status-before-live-run.json")
    if before.get("n_evaluations") != 0:
        raise RuntimeError("Unexpected evaluation count; refusing to continue")
    latest = before.get("validations", [{}])[0]
    if latest and not latest.get("finished_at"):
        raise RuntimeError("A validation appears to be active; refusing to collide with it")

    port = free_port()
    route_file = PRIVATE / "route"
    endpoint_file = PRIVATE / "score-shards-endpoint-url"
    tunnel_log_path = PRIVATE / "score-shards-tunnel.log"
    capture_log_path = bundle / "live-capture.log"
    capture_log = capture_log_path.open("ab", buffering=0)
    tunnel_fd = os.open(tunnel_log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(tunnel_log_path, 0o600)
    capture = None
    tunnel = None
    try:
        capture = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "drone/capture.py"),
                "--port",
                str(port),
                "--output",
                str(capture_root),
                "--route-file",
                str(route_file),
                "--plan-file",
                str(ACTIVE_PLAN),
            ],
            stdout=capture_log,
            stderr=capture_log,
        )
        wait_for_port(port, capture)

        tunnel = subprocess.Popen(
            [
                str(PRIVATE / "cloudflared"),
                "tunnel",
                "--url",
                f"http://127.0.0.1:{port}",
                "--no-autoupdate",
                "--protocol",
                "http2",
            ],
            stdout=tunnel_fd,
            stderr=tunnel_fd,
        )
        os.close(tunnel_fd)
        tunnel_fd = -1
        deadline = time.monotonic() + 75
        public_url = None
        while time.monotonic() < deadline:
            if tunnel.poll() is not None:
                raise RuntimeError("Temporary relay exited during startup")
            match = re.search(
                r"https://[a-z0-9-]+\.trycloudflare\.com",
                tunnel_log_path.read_text(errors="ignore"),
            )
            if match:
                public_url = match.group(0)
                break
            time.sleep(1)
        if public_url is None:
            raise RuntimeError("Temporary relay did not publish an endpoint")

        route = route_file.read_text().strip()
        endpoint_fd = os.open(endpoint_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.chmod(endpoint_file, 0o600)
        with os.fdopen(endpoint_fd, "w") as stream:
            stream.write(f"{public_url}/{route}/predict\n")

        # A newly registered quick tunnel can be visible to cloudflared before its
        # public hostname is reachable from the official verifier. Retry only the
        # non-mutating verification step; no validation is submitted in this loop.
        verified = None
        verification_errors = []
        for attempt in range(1, 10):
            if tunnel.poll() is not None or capture.poll() is not None:
                raise RuntimeError("Callback service exited during verification")
            try:
                candidate = run_portal("verify", bundle / "live-verify.json", endpoint_file)
            except RuntimeError as exc:
                verification_errors.append({"attempt": attempt, "error": str(exc)})
            else:
                if (
                    not candidate.get("errors")
                    and candidate.get("can_connect")
                    and candidate.get("can_predict")
                    and candidate.get("response_status_code") == 200
                ):
                    verified = candidate
                    break
                verification_errors.append({"attempt": attempt, "response": candidate})
            time.sleep(5)
        if verified is None:
            (bundle / "live-verify-retries.json").write_text(
                json.dumps(verification_errors, indent=2) + "\n"
            )
            raise RuntimeError("Official verification did not confirm the callback")
        print(json.dumps({"stage": "verified", "port": port}), flush=True)

        capture_audit = []
        for index, plan_path in enumerate(plans_to_run, start=1):
            plan = json.loads(plan_path.read_text())
            print(json.dumps({"stage": "submitting_validation", "shard": index}), flush=True)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "drone/run_probe.py"),
                    str(plan_path),
                    "--url-file",
                    str(endpoint_file),
                ]
            )
            if result.returncode:
                raise RuntimeError(f"Validation shard {index} failed; no later shard was submitted")
            missing = wait_for_complete_capture(
                plan, capture_root, max_missing=args.max_missing_capture
            )
            capture_audit.append({"name": plan["name"], "missing_frames": missing})
            print(json.dumps({"stage": "capture_checked", "name": plan["name"], "missing_frames": missing}), flush=True)

        summary_path = bundle / "live-score-summary.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "drone/analyze_validation_score_shards.py"),
                "--bundle",
                str(bundle),
                "--capture-root",
                str(capture_root),
                "--max-missing-capture",
                str(args.max_missing_capture),
                "--output",
                str(summary_path),
            ]
        )
        if result.returncode:
            raise RuntimeError("Receipt verification failed")

        after = run_portal("status", bundle / "status-after-live-run.json")
        if after.get("n_evaluations") != 0:
            raise RuntimeError("Evaluation count changed unexpectedly")
        print(
            json.dumps(
                {
                    "stage": "complete",
                    "validations_before": before.get("n_validations"),
                    "validations_after": after.get("n_validations"),
                    "evaluations_after": after.get("n_evaluations"),
                    "summary": str(summary_path),
                    "capture_audit": capture_audit,
                }
            ),
            flush=True,
        )
    finally:
        if tunnel_fd >= 0:
            os.close(tunnel_fd)
        stop_child(tunnel)
        stop_child(capture)
        capture_log.close()


if __name__ == "__main__":
    main()
