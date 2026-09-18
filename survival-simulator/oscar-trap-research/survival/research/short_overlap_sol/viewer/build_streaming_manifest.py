"""Build bounded-memory every-frame chunks from completed short-wall replays."""
from __future__ import annotations
import gzip
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "short_overlap_sol"
CHUNK = 20


def frames(path):
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with gzip.open(path, "rt", encoding="utf8") as stream:
        while True:
            data = stream.read(1024 * 1024)
            if not data and not buffer:
                break
            buffer += data
            if not started:
                marker = buffer.find('"frames":[')
                if marker < 0:
                    if len(buffer) > 32:
                        buffer = buffer[-32:]
                    if data:
                        continue
                    raise ValueError(f"frames marker missing: {path}")
                buffer = buffer[marker + len('"frames":['):]
                started = True
            while True:
                buffer = buffer.lstrip(" \r\n\t,")
                if buffer.startswith("]"):
                    return
                try:
                    frame, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                yield frame
                buffer = buffer[end:]
            if not data:
                raise ValueError(f"truncated frames: {path}")


def selected_receipts():
    rows = []
    for seed in range(10000, 10012):
        candidates = list((RESULTS / "native_v4_stream").glob(f"map-{seed}-s120-summary.json"))
        if candidates:
            payload = json.loads(candidates[0].read_text())
            result = payload["result"]
            receipt = ROOT / result["replay"].replace("replays/", "").replace(".json.gz", ".json")
        else:
            pool = ((RESULTS / "native_v3_stream") if seed >= 10006 else
                    (RESULTS / "native_v2"))
            matches = list(pool.glob(f"native-shortv2-m{seed}-*.json"))
            matches = [p for p in matches if not p.name.endswith("correction.json")]
            if not matches:
                raise FileNotFoundError(f"no completed receipt for map {seed}")
            receipt = max(matches, key=lambda p: json.loads(p.read_text()).get("seconds", 0))
            result = json.loads(receipt.read_text())
        if result.get("seconds") != 120.0:
            raise ValueError(f"not a completed 120s case: {receipt}")
        replay = ROOT / result["replay"]
        rows.append((seed, receipt, replay, result))
    return rows


def main():
    manifest = []
    for seed, receipt, replay, result in selected_receipts():
        chunk_dir = HERE / "chunks" / f"map-{seed}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        bucket = []
        count = 0
        for frame in frames(replay):
            if "native_image" not in frame:
                raise ValueError(f"missing native image at frame {count}: {replay}")
            bucket.append(frame)
            count += 1
            if len(bucket) == CHUNK:
                with gzip.open(chunk_dir / f"{(count-1)//CHUNK}.json.gz", "wt") as out:
                    json.dump(bucket, out, separators=(",", ":"))
                bucket = []
        if bucket:
            with gzip.open(chunk_dir / f"{(count-1)//CHUNK}.json.gz", "wt") as out:
                json.dump(bucket, out, separators=(",", ":"))
        expected = round(result["seconds"] * 10) + 1
        if count != expected:
            raise ValueError(f"frame count {count} != {expected}: {replay}")
        manifest.append({
            "map_seed": seed, "seconds": result["seconds"], "frames": count,
            "overlap": result["overlap"], "gap": result["gap"],
            "acquired": result["acquired"], "losses": result["physical_losses"],
            "final_joint": result["final_joint"], "chunk_size": CHUNK,
            "chunks": f"chunks/map-{seed}/",
            "receipt": os.path.relpath(receipt, HERE),
            "replay": os.path.relpath(replay, HERE),
        })
        print(json.dumps({"map_seed": seed, "frames": count}), flush=True)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
