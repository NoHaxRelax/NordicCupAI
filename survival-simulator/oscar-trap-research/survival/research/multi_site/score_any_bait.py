"""Posthoc any-bait causal audit over immutable native replay frames."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path


FRAME = re.compile(
    r'"t":([0-9.]+),"score":.*?"agents":(\[.*?\]),"predators":(\[.*?\]),"fruits":',
    re.DOTALL)


def native_track(path):
    """Yield compact creature rows without materializing PNG frame payloads."""
    buffer = ""
    with gzip.open(path, "rt") as stream:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            buffer += chunk
            while True:
                match = FRAME.search(buffer)
                if match is None:
                    # A frame prefix is small; discard completed image payloads
                    # while retaining enough for a split frame header.
                    if len(buffer) > 2_000_000:
                        marker = buffer.rfind('{"t":')
                        buffer = buffer[marker:] if marker >= 0 else buffer[-100_000:]
                    break
                agents = json.loads(match[2])
                predators = json.loads(match[3])
                tracked = next(row for row in predators if row["id"] == 0)
                yield float(match[1]), agents, tracked
                buffer = buffer[match.end():]


def score(receipt_path):
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    sites = {row["bait_id"]: row["site"] for row in receipt["multi_site_baits"]}
    replay = receipt_path.parents[2] / receipt["replay"]
    guide_ids = set(receipt.get("guide_role_ids", [receipt["guide_id"]]))
    switches = receipt["target_switches"]
    chains = []
    for index, row in enumerate(switches):
        if row["target"] not in sites:
            continue
        prior = next((old for old in reversed(switches[:index])
                      if old["target"] in guide_ids), None)
        if prior is not None and 0 <= row["time"] - prior["time"] <= 3.:
            chains.append({"guide_time": prior["time"], "guide_id": prior["target"],
                           "bait_time": row["time"], "bait_id": row["target"]})

    frames = list(native_track(replay))
    tracks = [(t, predator["x"], predator["y"])
              for t, _, predator in frames]
    by_time = {round(t, 1): (agents, predator) for t, agents, predator in frames}
    results = []
    for chain in chains:
        agents, predator = by_time.get(round(chain["guide_time"], 1), ([], None))
        guide = next((row for row in agents if row["id"] == chain["guide_id"]), None)
        chase_gate = False
        chase_distance = None
        chase_rel_dir = None
        if guide is not None and predator is not None:
            chase_distance = math.dist((predator["x"], predator["y"]),
                                       (guide["x"], guide["y"]))
            chase_rel_dir = ((math.atan2(guide["y"] - predator["y"],
                                         guide["x"] - predator["x"])
                              - predator["direction"] + math.pi)
                             % (2 * math.pi) - math.pi)
            chase_gate = (abs(chase_rel_dir) > math.pi / 2
                          or chase_distance < predator["hearing_radius"] * 1.5)
        site = sites[chain["bait_id"]]
        mouth, inward = site["mouth"], site["inward"]
        physical = []
        for t, x, y in tracks:
            dx, dy = x - mouth[0], y - mouth[1]
            along = dx * inward[0] + dy * inward[1]
            physical.append((t, math.hypot(dx, dy) <= 75 and along <= 10))
        streak = 0
        entered = None
        for t, inside in physical:
            if t < chain["bait_time"]:
                continue
            streak = streak + 1 if inside else 0
            if streak >= 20:
                entered = round(t - 1.9, 1)
                break
        retention = (entered is not None and all(inside for t, inside in physical
                                                 if t >= physical[-1][0] - 30.))
        results.append({**chain, "physical_entry_time": entered,
                        "native_target_selected_guide": True,
                        "guide_chase_gate": chase_gate,
                        "guide_distance": chase_distance,
                        "guide_relative_direction": chase_rel_dir,
                        "retained_final_30s": retention,
                        "success": chase_gate and entered is not None and retention})
    return {"receipt": str(receipt_path), "criteria": {
                "guide_to_bait_target_seconds": 3., "physical_streak_seconds": 2.,
                "retention_tail_seconds": 30.},
            "bait_count": len(sites), "causal_chains": results,
            "success": any(row["success"] for row in results)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = score(args.receipt)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
