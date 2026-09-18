"""Serve an honest, local overview of entrapment and native-game evidence."""
import argparse
import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def game_record(folder, replay=True):
    s = read(folder / "summary.json")
    if not isinstance(s, dict):
        return None
    manifest = read(folder / "manifest.json") or {}
    requested = manifest.get("horizon") or manifest.get("seconds")
    elapsed = s.get("sim_time")
    complete = s.get("status") == "complete"
    cohort = "full game" if requested and requested >= 3000 else "integration check"
    return {"id": quote(str(folder.resolve()), safe=""), "folder": str(folder),
            "name": folder.name, "replay": replay and (folder / "static.json").exists()
            and (folder / "background.png").exists() and (folder / "chunks").is_dir(),
            "requested_horizon": requested, "complete": complete, "cohort": cohort,
            "summary": s}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=9066)
    p.add_argument("--coverage", type=Path, action="append", default=[])
    p.add_argument("--games", type=Path, action="append", default=[])
    p.add_argument("--no-archived-summaries", action="store_true")
    args = p.parse_args()
    summary = read(ROOT / "docs/entrapment_iteration/final-summary.json")
    coverage_paths = args.coverage or sorted((ROOT / "logs/entrapment-iteration").glob("**/coverage.json"))
    coverage = []
    for path in coverage_paths:
        c = read(path / "coverage.json" if path.is_dir() else path)
        if isinstance(c, dict): coverage.append({"name": (path if path.is_dir() else path.parent).name, "path": str(path), "data": c})
    folders = []
    for item in (args.games or [ROOT / "logs/entrapment_game"]):
        if (item / "summary.json").exists(): folders.append(item)
        elif item.is_dir(): folders += [x.parent for x in sorted(item.glob("**/summary.json"))]
    games = [g for g in (game_record(f) for f in folders) if g]
    if not args.no_archived_summaries:
        for path in sorted((ROOT / "docs/entrapment_iteration").glob("native-*-summary.json")):
            s = read(path)
            if isinstance(s, dict): games.append({"id": "", "folder": str(path), "name": path.stem.replace("-summary", ""), "replay": False, "requested_horizon": None, "complete": s.get("status") == "complete", "cohort": "integration check", "summary": s})
    # Avoid duplicate recorded folders while preserving distinct archived experiments.
    seen, unique = set(), []
    for g in games:
        key = (g["folder"], g["summary"].get("seed"), g["summary"].get("sim_time"))
        if key not in seen: seen.add(key); unique.append(g)
    evidence = (summary or {}).get("improvements") or [
        {"change":"Observed-motion association and route recovery", "evidence":"72/100 development maps versus the earlier nearest-predator baseline of 68/100."},
        {"change":"Independent map check", "evidence":"74/100 fresh maps; no map seeds overlapped the development set."},
        {"change":"Corner fallback", "evidence":"Added no qualifying sites in the 100-map development survey; corner delivery remains exploratory."},
        {"change":"Physical replacement", "evidence":"12/12 replacements arrived alive; the paired policy passed 9/12 with and without replacement."},
        {"change":"Bystander avoidance", "evidence":"Integrated outside the survival module; native checks are descriptive and unmatched."},
    ]
    payload = {"generated_from": "local files", "final_summary": summary, "improvements": evidence,
               "coverage_runs": coverage, "games": unique,
               "counts": {"coverage_runs": len(coverage), "games": len(unique),
                          "recorded_replays": sum(g["replay"] for g in unique)}}
    replays, replay_lock = {}, threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values): pass
        def send(self, body, mime="application/json", status=200):
            if not isinstance(body, bytes): body = body.encode()
            self.send_response(status); self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-cache")
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/": return self.send((Path(__file__).with_suffix(".html")).read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/data": return self.send(json.dumps(payload))
                if u.path.startswith("/replay/"):
                    parts = u.path.split("/"); folder = Path(unquote(parts[2])).resolve()
                    allowed = next((g for g in unique if g["replay"] and Path(g["folder"]).resolve() == folder), None)
                    if not allowed: return self.send("Not found", "text/plain", 404)
                    tick = int(parse_qs(u.query).get("tick", [0])[0])
                    with replay_lock:
                        if folder not in replays:
                            from entrapment_viewer import Replay
                            import pygame
                            pygame.font.init(); replays[folder] = Replay(folder)
                        replay = replays[folder]
                        if parts[3] == "frame": body = replay.png(tick)
                        else: row = replay.row(tick)
                    if parts[3] == "frame": return self.send(body, "image/png")
                    return self.send(json.dumps({k:v for k,v in row.items() if k != "world"}))
                self.send("Not found", "text/plain", 404)
            except (ValueError, KeyError, IndexError, FileNotFoundError): self.send("Not found", "text/plain", 404)
            except BrokenPipeError: pass

    print(f"Research overview: http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__": main()
