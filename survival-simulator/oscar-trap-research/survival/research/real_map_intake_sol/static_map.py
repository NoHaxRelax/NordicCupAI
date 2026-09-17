"""Lossless, JSON-compatible snapshots of generated static world geometry.

The policy is allowed this payload once at construction.  It contains no live
engine references and no creature, fruit, tree, energy, rest, or RNG state.
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib

import numpy as np


TERRAIN_LABELS = ("forest", "grassland", "swamp", "desert", "river")


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf8")


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def snapshot_environment(env) -> dict:
    """Return an exact static map without retaining references to ``env``."""
    label_to_code = {label: index for index, label in enumerate(TERRAIN_LABELS)}
    terrain = np.fromiter(
        (label_to_code[biome.type] for biome in env.biome_map.flat),
        dtype=np.uint8, count=env.biome_map.size,
    ).reshape(env.biome_map.shape)
    compressed = zlib.compress(terrain.tobytes(order="C"), level=9)
    obstacles = [
        {
            "index": index,
            "x": float(obstacle.x),
            "y": float(obstacle.y),
            "width": float(obstacle.width),
            "height": float(obstacle.height),
        }
        for index, obstacle in enumerate(env.obstacles)
    ]
    return {
        "schema": "survival-static-map-v1",
        "width": int(env.width),
        "height": int(env.height),
        "obstacles": obstacles,
        "terrain": {
            "encoding": "zlib-base64-u8-c-order",
            "shape": [int(terrain.shape[0]), int(terrain.shape[1])],
            "labels": list(TERRAIN_LABELS),
            "move_penalty": {
                label: float(next(
                    biome.move_penalty for biome in env.biome_map.flat
                    if biome.type == label
                ))
                for label in sorted(set(biome.type for biome in env.biome_map.flat))
            },
            "data": base64.b64encode(compressed).decode("ascii"),
        },
    }


def decode_terrain(payload: dict) -> np.ndarray:
    """Decode a payload terrain grid for routing or contract tests."""
    terrain = payload["terrain"]
    if terrain["encoding"] != "zlib-base64-u8-c-order":
        raise ValueError("unsupported terrain encoding")
    raw = zlib.decompress(base64.b64decode(terrain["data"], validate=True))
    shape = tuple(terrain["shape"])
    expected = int(np.prod(shape))
    if len(raw) != expected:
        raise ValueError(f"terrain byte count {len(raw)} != {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


def json_clone(payload):
    """Break references and reject values outside the JSON policy boundary."""
    return json.loads(json.dumps(payload, allow_nan=False))

