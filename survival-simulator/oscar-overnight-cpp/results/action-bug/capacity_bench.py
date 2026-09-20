#!/usr/bin/env python3
"""Measure the duplicate-action HTTP boundary locally, without a simulation.

This mirrors the expensive parts of the reference validation path:
our JSON response construction, their JSON decode, and one Pydantic model per
action.  It intentionally uses a zero-cost no-op action and never opens a
network connection.
"""
import argparse
import gc
import json
import resource
import time

from pydantic import BaseModel


class ActionRequest(BaseModel):
    agent_id: int
    move_distance: float
    move_direction: float
    turn_angle: float
    spawn_agent: bool = False


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def parse_sizes(raw: str) -> list[int]:
    result = []
    for part in raw.split(","):
        value = int(part.strip().replace("_", ""))
        if value <= 0:
            raise ValueError("sizes must be positive")
        result.append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="20000,50000,100000,200000,400000,600000,800000,1000000")
    args = parser.parse_args()
    print("actions,payload_mb,serialize_s,loads_s,pydantic_s,total_s,peak_rss_mb", flush=True)
    for count in parse_sizes(args.sizes):
        action = {
            "agent_id": 0,
            "move_distance": 0.0,
            "move_direction": 0.0,
            "turn_angle": 0.0,
            "spawn_agent": False,
        }
        started = time.perf_counter()
        actions = [action] * count
        body = json.dumps({"actions": actions}, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":"))
        serialized = time.perf_counter() - started

        started = time.perf_counter()
        decoded = json.loads(body)["actions"]
        loaded = time.perf_counter() - started

        started = time.perf_counter()
        models = [ActionRequest(**item) for item in decoded]
        validated = time.perf_counter() - started
        total = serialized + loaded + validated
        print(
            f"{count},{len(body) / 1e6:.3f},{serialized:.3f},{loaded:.3f},"
            f"{validated:.3f},{total:.3f},{rss_mb():.1f}",
            flush=True,
        )
        del actions, body, decoded, models
        gc.collect()


if __name__ == "__main__":
    main()
