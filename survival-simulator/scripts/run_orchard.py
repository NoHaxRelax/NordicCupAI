"""Evaluate Orchard with only public observations crossing a process boundary.

Run with the main project's Python environment. Simulator imports stay inside
run() so Windows spawn never imports the engine into the policy worker.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if not args.render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import sys
    sys.path.insert(0, str(ROOT))
    from models.observation_only import ObservationOnlyOrchard
    from src.core import SimulationCore
    from src.elements.environment import Environment
    from src.utils.DTOs import ActionRequest

    # Evaluation scenario only; this object is never sent to the policy worker.
    path = args.output / f"seed-{args.seed}.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite a completed evaluation: {path}")
    Environment.spawn_predator = lambda self, *a, **kw: None
    started = time.perf_counter()
    sim = SimulationCore(seed=args.seed)
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ("models/survival/oscar_orchard.py", "models/observed_bounds.py",
                           "models/observation_only.py", "scripts/run_orchard.py")}
    state = sim.step([])
    steps, next_sample, samples = 1, 0., []
    peak = state["num_agents"]
    screen = None
    if args.render:
        import pygame
        from scripts.playback import PlaybackControls
        pygame.init()
        screen = pygame.display.set_mode((1100, 877), pygame.RESIZABLE)
        playback, clock = PlaybackControls(args.speed), pygame.time.Clock()
        font = pygame.font.SysFont(None, 22)
    reason = "horizon"
    with ObservationOnlyOrchard(policy_seed=args.policy_seed, policy_kwargs=args.policy_kwargs) as policy:
        try:
            running = True
            while running and state["num_agents"] and steps * sim.dt < args.seconds - 1e-9:
                if screen is not None:
                    playback.advance(clock.tick(60) / 1000.)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running, reason = False, "window_closed"
                        playback.handle_event(event, screen.get_size())
                deadline = time.perf_counter() + .04
                while running and state["num_agents"] and steps * sim.dt < args.seconds - 1e-9:
                    if screen is not None and not playback.consume_step(sim.dt):
                        break
                    # Only public observations and time cross the process boundary.
                    decisions = policy(state["observations"], state["sim_time"])
                    actions = [(aid, ActionRequest.model_validate(action)) for aid, action in decisions]
                    state = sim.step(actions)
                    steps += 1
                    peak = max(peak, state["num_agents"])
                    if state["sim_time"] >= next_sample:
                        sample = dict(seconds=round(state["sim_time"], 2), score=state["score"],
                                      population=state["num_agents"], wall_seconds=round(time.perf_counter() - started, 2))
                        samples.append(sample)
                        print(f"seed {args.seed}: {sample['seconds']:.0f}/{args.seconds:g}s, "
                              f"alive {sample['population']}, score {sample['score']:.2f}", flush=True)
                        write_json(path.with_suffix(".progress.json"), dict(seed=args.seed, **sample))
                        next_sample = state["sim_time"] + 100.
                    if screen is not None and time.perf_counter() >= deadline:
                        break
                if screen is not None:
                    viewport = screen.subsurface((0, 0, screen.get_width(), max(1, screen.get_height() - playback.HEIGHT)))
                    sim.env.draw(viewport)
                    playback.draw(screen, font)
                    pygame.display.set_caption(f"Orchard | {state['sim_time']:.1f}s | alive {state['num_agents']} | score {state['score']:.2f}")
                    pygame.display.flip()
        except KeyboardInterrupt:
            reason = "interrupted"
        finally:
            if screen is not None:
                pygame.quit()
        if state["num_agents"] == 0:
            reason = "extinct"
        result = dict(seed=args.seed, policy_seed=args.policy_seed, horizon_seconds=args.seconds,
                      reason=reason, policy_kwargs=args.policy_kwargs,
                      seconds=state["sim_time"], score=state["score"], population=state["num_agents"],
                      extinct=state["num_agents"] == 0, peak_population=peak, predators_enabled=False,
                      predators_remaining=len(sim.env.predators), steps=steps,
                      wall_seconds=time.perf_counter() - started, platform=platform.platform(),
                      source_sha256=hashes, audit=dict(policy.audit), samples=samples)
        if hasattr(policy, "last_audit"):
            result["final_audit"] = policy.last_audit
        write_json(path, result)
        path.with_suffix(".progress.json").unlink(missing_ok=True)
        print(json.dumps({k: result[k] for k in ("seed", "seconds", "score", "population", "wall_seconds")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=0,
                        help="Independent controller RNG seed; never derived from the world seed")
    parser.add_argument("--seconds", type=float, default=3000.)
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "orchard")
    parser.add_argument("--render", action="store_true", help="Show the native world with speed buttons")
    parser.add_argument("--speed", type=int, choices=(1, 2, 5, 10, 20), default=1)
    parser.add_argument("--policy-kwargs", type=json.loads, default={}, help="JSON Orchard parameter overrides")
    args = parser.parse_args()
    if not .1 <= args.seconds <= 3000:
        parser.error("Require 0.1 <= seconds <= 3000")
    if not isinstance(args.policy_kwargs, dict):
        parser.error("--policy-kwargs must be a JSON object")
    run(args)


if __name__ == "__main__":
    main()
