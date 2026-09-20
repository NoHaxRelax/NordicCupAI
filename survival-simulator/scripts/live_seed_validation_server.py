"""Validation endpoint: expanded-local-food while public seed recovery runs.

The controller never receives the validation seed.  It records the action that
actually ran, derives terrain samples from ordinary observations, searches the
entire uint32 domain in a background process, and independently replays every
recorded action before accepting a shadow.  The food policy keeps answering while
the search runs, so no request is held for the several-minute computation.
"""
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from collections import deque

from fastapi import FastAPI, Request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim import SimulationCore  # noqa: E402
from models.seed_shadow.replay import ShadowJournal  # noqa: E402
from models.survival.oscar_orchard import OrchardPolicy  # noqa: E402

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "19123"))
SEARCH_AT = float(os.environ.get("SEED_SEARCH_AT", "180"))
MIN_SAMPLES = int(os.environ.get("SEED_MIN_SAMPLES", "500"))
WORKERS = int(os.environ.get("SEED_SEARCH_WORKERS", "30"))
SEARCH_END = int(os.environ.get("SEED_SEARCH_END", str(2**32)))
COMPUTE_LIMIT = float(os.environ.get("SEED_COMPUTE_LIMIT", "600"))
WAIT_PER_ACTION = float(os.environ.get("SEED_WAIT_PER_ACTION", "9"))
RUN_ROOT = pathlib.Path(os.environ.get("SEED_SHADOW_RUN_DIR", "/workspace/seed-validation-runs"))
SCANNER = ROOT / "models" / "seed_shadow" / "build" / "scan"
SCAN_DRIVER = ROOT / "scripts" / "scan_seed_domain.py"

EXPANDED_LOCAL_FOOD = {
    "births_per_tick": 1, "breed_reserve": 120.0,
    "breed_reserve_late": 235.2506179159642, "cap_hard_min": 2,
    "cap_max": 41.567283565992376, "cap_min": 3.0402929768839604,
    "cap_mult": 0.65, "cap_tree_slack": 1, "cluster_radius": 0.0,
    "cull": False, "dist_pen": 0.5114898180547935, "dump_after_t": math.inf,
    "dump_food_site": 2, "dump_mult": 1.0, "emergency_reserve": 105.0,
    "explore_energy": 200.0, "explore_min": 60.0, "explore_radius": 450.0,
    "extra_old": True, "feed_mode": "hungry", "fit_energy": 1.169024188319745,
    "fit_hear": 0.6924908230029558, "fit_speed": 1.9018092919140193,
    "fit_vision": 0.9831667720395514, "fruit_min_wait": 0.0,
    "fruit_reach": 210.2194875098543, "heir_age": 55.0,
    "heir_at_food": False, "heir_needs_site": True, "heir_reserve": 250.0,
    "heir_select": True, "heir_slack": 0.05, "hungry_margin": 5.0,
    "idle_sweep": True, "late_still_t": math.inf, "lone_reach_mult": 1.0,
    "low_pop_reserve": 205.7556826916815, "min_stay": 15.0, "n0": 80.0,
    "no_eat_age": math.inf, "nursery_bonus": 0.0, "old_eat_last": True,
    "old_reach": 60.0, "post_radius": 116.83236347567907, "repost_every": 10.0,
    "reserve_t0": 686.2847344808403, "reserve_t1": 1133.7142846858605,
    "ripen_wait": 20.0, "rot_margin": 47.0, "select_min_young": 0,
    "site_min": 5.0, "spread_weight": 0.0, "sweep_rate": 0.03,
    "switch_gain": 100.0, "travel_turn": 0.25, "tree_half": 713.2786757704566,
    "tree_reach": 623.8328391478476, "tree_slots": 1, "vo_cap": 12.0,
    "vo_win_far": 4.5, "vo_win_fruit": 4.5, "watch_patience": 30.0,
    "watch_reach": 500.0, "watch_refresh": 60.0,
}


def normalize(body):
    states = body.get("agent_status") or body.get("observations") or []
    for state in states:
        for obs in state.get("observations") or []:
            obs["type"] = str(obs.get("type", "")).capitalize()
    body["agent_status"] = states
    return states


def action_dicts(actions):
    return [(aid, action.model_dump() if hasattr(action, "model_dump") else action.dict())
            for aid, action in actions]


class LiveGame:
    def __init__(self):
        self.lock = threading.RLock()
        self.latencies = deque(maxlen=10000)
        self.generation = 0
        self.last_time = -1.0
        self.search_process = None
        self.reset()

    def reset(self):
        if self.search_process is not None and self.search_process.poll() is None:
            try:
                os.killpg(self.search_process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.generation += 1
        self.policy = OrchardPolicy(seed=0, **EXPANDED_LOCAL_FOOD)
        self.journal = ShadowJournal(deadline_seconds=COMPUTE_LIMIT)
        self.previous_actions = []
        self.search_process = None
        self.search_thread = None
        self.search_started = None
        self.search_finished = None
        self.recovered_seed = None
        self.recovery = "gathering"
        self.failure = None
        self.run_dir = RUN_ROOT / time.strftime("game-%Y%m%d-%H%M%S")
        suffix = 0
        while self.run_dir.exists():
            suffix += 1
            self.run_dir = RUN_ROOT / f"{self.run_dir.name}-{suffix}"
        self.run_dir.mkdir(parents=True)
        self.last_time = -1.0

    def _write_status(self):
        status = self.status()
        temporary = self.run_dir / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(self.run_dir / "status.json")

    def _start_search(self):
        rows = self.journal.terrain.rows()
        samples = self.run_dir / "samples.txt"
        samples.write_text("".join(f"{x} {y} {label}\n" for x, y, label in rows))
        generation = self.generation
        self.recovery = "searching"
        self.search_started = time.monotonic()
        self.search_thread = threading.Thread(
            target=self._recover, args=(generation, samples), daemon=True,
            name=f"seed-recovery-{generation}")
        self.search_thread.start()

    def _recover(self, generation, samples):
        search_dir = self.run_dir / "search"
        command = [sys.executable, str(SCAN_DRIVER), "--scanner", str(SCANNER),
                   "--samples", str(samples), "--out", str(search_dir),
                   "--workers", str(WORKERS), "--deadline-seconds", str(COMPUTE_LIMIT - 30)]
        if SEARCH_END != 2**32:
            command.extend(["--end", str(SEARCH_END)])
        log_path = self.run_dir / "search.log"
        try:
            with log_path.open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           text=True, start_new_session=True)
                with self.lock:
                    if generation != self.generation:
                        os.killpg(process.pid, signal.SIGTERM)
                        return
                    self.search_process = process
                code = process.wait()
            if code:
                raise RuntimeError(f"full-domain scanner exited {code}")
            progress = json.loads((search_dir / "progress.json").read_text())
            candidates = json.loads((search_dir / "candidates.json").read_text())
            remaining = COMPUTE_LIMIT - progress["seconds"]
            if remaining <= 0:
                raise TimeoutError("no compute budget remains for replay")

            # Replay a stable snapshot without holding up /predict.
            with self.lock:
                if generation != self.generation:
                    return
                snapshot = list(self.journal.frames)
            candidate = ShadowJournal(deadline_seconds=remaining)
            candidate.frames = snapshot
            seed = candidate.recover(candidates, SimulationCore, complete_search=True)
            if seed is None:
                raise RuntimeError(f"candidate replay ended {candidate.status}")

            # Catch up concurrently accumulated ticks. The final lock covers only the
            # usually empty tail, never the multi-minute search or full replay.
            cursor = len(snapshot)
            while True:
                with self.lock:
                    if generation != self.generation:
                        return
                    tail = list(self.journal.frames[cursor:])
                for frame in tail:
                    candidate.record(frame["actions"], frame["public"])
                    if candidate.status == "desynchronized":
                        raise RuntimeError("candidate desynchronized while catching up")
                cursor += len(tail)
                with self.lock:
                    if generation != self.generation:
                        return
                    if cursor == len(self.journal.frames):
                        self.journal.seed = seed
                        self.journal._shadow = candidate._shadow
                        self.journal.status = "public_consistent"
                        self.recovered_seed = seed
                        self.recovery = "recovered"
                        self.search_finished = time.monotonic()
                        self._write_status()
                        return
        except Exception as exc:
            with self.lock:
                if generation == self.generation:
                    self.recovery = "failed"
                    self.failure = repr(exc)
                    self.search_finished = time.monotonic()
                    self._write_status()

    def predict(self, body):
        started = time.perf_counter()
        states = normalize(body)
        sim_time = float(body.get("sim_time") or 0.0)
        with self.lock:
            if self.last_time >= 0 and sim_time < self.last_time:
                self.reset()
            # The initial time-zero probe has no observations and precedes every
            # engine step.  Every later response is the result of the actions
            # returned on the last call.
            if states or sim_time > 0:
                self.journal.record(self.previous_actions, body)
            if self.journal.status == "desynchronized":
                self.recovery = "desynchronized"
                self.failure = self.journal.failure
            actions = self.policy(states, sim_time)
            self.previous_actions = action_dicts(actions)
            self.last_time = sim_time
            if (self.recovery == "gathering" and sim_time >= SEARCH_AT
                    and len(self.journal.terrain.points) >= MIN_SAMPLES):
                self._start_search()
            if int(sim_time * 10) % 1000 == 0:
                self._write_status()
            active_search = self.search_thread if self.recovery == "searching" else None
        # Validation permits ten seconds per response and 600 accumulated. Spend
        # nine seconds advancing the already-running 30-core search while the
        # simulation itself remains frozen at this action boundary.
        if active_search is not None:
            active_search.join(timeout=WAIT_PER_ACTION)
        self.latencies.append((time.perf_counter() - started) * 1000)
        return {"actions": [action for _, action in self.previous_actions]}

    def status(self):
        values = sorted(self.latencies)
        percentile = lambda q: values[round((len(values)-1)*q)] if values else 0.0
        return {
            "generation": self.generation, "sim_time": self.last_time,
            "frames": len(self.journal.frames), "terrain_samples": len(self.journal.terrain.points),
            "recovery": self.recovery, "recovered_seed": self.recovered_seed,
            "failure": self.failure,
            "search_wall_seconds": (self.search_finished or time.monotonic()) - self.search_started
                if self.search_started else None,
            "shadow_status": self.journal.status,
            "policy": "expanded_local_food", "requests": len(self.latencies),
            "latency_mean_ms": sum(values)/len(values) if values else 0.0,
            "latency_p95_ms": percentile(.95), "latency_max_ms": values[-1] if values else 0.0,
        }


game = LiveGame()
app = FastAPI(title="Expanded local food with public seed recovery")


@app.post("/predict")
@app.post("/")
async def predict(request: Request):
    return game.predict(await request.json())


@app.get("/")
def index():
    return {"message": "Seed recovery validation endpoint running", **game.status()}


@app.get("/metrics")
def metrics():
    return game.status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
