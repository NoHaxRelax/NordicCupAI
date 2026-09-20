#!/usr/bin/env python3
"""Run one V4 search shard against an HTTPS-exposed server."""
import argparse, json, subprocess, time
from pathlib import Path

import requests


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Token-bearing base URL, without /predict")
    p.add_argument("--shard", required=True, type=int, choices=range(1, 6))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--filter", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    task = None
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        r = session.get(f"{args.url}/worker/{args.shard}", timeout=20)
        r.raise_for_status()
        body = r.json()
        if body.get("stop"):
            return
        if not body.get("waiting"):
            task = body
            break
        time.sleep(1)
    if task is None:
        raise RuntimeError("server never supplied a shard")
    atomic_json(args.out / "task.json", task)
    atomic_json(args.out / "initial-public.json", task["initial_public"])
    terrain = args.out / "terrain.txt"
    terrain.write_text("".join(f"{x} {y} {b}\n" for x, y, b in task["samples"]))
    coordinator = args.filter.parent / "streaming_verification" / "coordinator"
    stream = args.out / "stream"
    command = [str(coordinator), "stream", str(args.filter), str(terrain),
               str(args.out / "initial-public.json"), str(args.config),
               str(task["start"]), str(task["count"]), str(stream)]
    with (args.out / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    sent = set()
    verified = stream / "verified.txt"
    while process.poll() is None:
        if verified.exists():
            for word in verified.read_text().split():
                seed = int(word)
                if seed not in sent:
                    r = session.post(f"{args.url}/candidate", json={
                        "game_id": task["game_id"], "seed": seed,
                        "worker": f"https-shard-{args.shard}"}, timeout=20)
                    r.raise_for_status()
                    sent.add(seed)
        status = session.get(f"{args.url}/worker/{args.shard}", timeout=20).json()
        if status.get("stop"):
            process.terminate()
            break
        time.sleep(.2)
    process.wait(timeout=30)
    if verified.exists():
        for word in verified.read_text().split():
            seed = int(word)
            if seed not in sent:
                session.post(f"{args.url}/candidate", json={
                    "game_id": task["game_id"], "seed": seed,
                    "worker": f"https-shard-{args.shard}"}, timeout=20).raise_for_status()
    atomic_json(args.out / "worker-summary.json", {
        "shard": args.shard, "game_id": task["game_id"],
        "returncode": process.returncode, "candidates_sent": sorted(sent)})


if __name__ == "__main__":
    main()
