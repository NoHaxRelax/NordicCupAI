"""Watch the agents' shared world model being built, beside the real world.

Run it with the project venv, from anywhere; the working directory does not
matter, but an interpreter without pygame and shapely does:

    python survival-simulator/scripts/view_world_model.py --seed 1007
    python survival-simulator/scripts/view_world_model.py --seed 1007 \
        --screenshot runs/model.png --headless

The left half is simulator truth, drawn only here. The right half is
`GlobalPlanner.snapshot()` and nothing else: estimated poses, mapped rock
faces, remembered trees, observed biome samples and the predator belief -- every one of them derived from agent observations. Groups
that have not aligned keep separate panels, because until they share a frame
they genuinely do not share a map.

`--grade` is the single exception, and it is off by default. It draws real
predator positions onto the estimated map so the belief can be scored against
them, which means the right half then contains privileged data and is no
longer an honest picture of what the agents know. It says so on the panel
while it is on. Without it, nothing on the right comes from the simulator.

This exists alongside local_playground.py, which does the same comparison with
diagnostics attached. That entry point cannot run in this worktree: it calls
SimulationCore(predators_enabled=...) and RunDiagnostics expects
Environment.event_sink, neither of which this vendored simulator copy has.
This script drives SimulationCore directly and skips diagnostics instead.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_MISSING = [name for name in ("pygame", "shapely", "scipy", "numpy", "pydantic")
            if importlib.util.find_spec(name) is None]
if _MISSING:
    # Otherwise this surfaces as a traceback from inside src.elements.creature,
    # which does not point at the real cause: the wrong interpreter.
    raise SystemExit(
        f"Missing {', '.join(_MISSING)} for {sys.executable}.\n"
        "Run this with the project virtualenv rather than a base conda install, e.g.\n"
        "  <repo>/survival-simulator/.venv/Scripts/python.exe "
        "survival-simulator/scripts/view_world_model.py --seed 1007")

import pygame

from src.core import SimulationCore
from src.utils.controllers.expert_policy import ExpertConfig, ExpertPolicy, load_config
from src.utils.controllers.global_planner import PlannerConfig, load_planner_config
from src.utils.map_renderer import MapRenderer, draw_comparison


def configure(expert_path, planner_path, belief, particles, escape, hold):
    """Load the tuned files, then apply only the switches this viewer offers."""
    expert = load_config(expert_path).model_dump()
    expert["harvest"]["enabled"] = True
    expert["memory"]["belief_escape_enabled"] = escape
    expert["memory"]["belief_flee_seconds"] = hold
    planner = load_planner_config(planner_path).model_dump()
    planner["draw_overlay"] = False
    planner["biome_inference"]["adapt_to_wall_clock"] = False
    planner["predator_belief"]["enabled"] = belief
    planner["predator_belief"]["particles"] = particles
    return ExpertConfig.model_validate(expert), PlannerConfig.model_validate(planner)


TRUTH = (236, 244, 252)


def grade_truth(sim, planner, tally=None):
    """Pair each belief with the nearest real predator and score the spread.

    Only anchored groups are graded. An unanchored group's frame is a private
    gauge, so comparing a world coordinate against it would measure the wrong
    thing entirely. Grading is kept separate from drawing so a headless run
    accumulates the whole episode rather than only the frame it saves.
    """
    truths = [(predator.x, predator.y) for predator in sim.env.predators]
    graded = []
    if not truths:
        return graded
    for track in planner.predator_tracker.tracks.values():
        group = planner.estimator.groups.get(track.group_id)
        if group is None or not group.anchored:
            continue
        # MapView.pixel builds a Vector2, which rejects a numpy array.
        mean = (float(track.mean[0]), float(track.mean[1]))
        nearest = min(truths, key=lambda point: math.dist(point, mean))
        error = math.dist(nearest, mean)
        # A predator no track is claiming is unknown, not mispredicted.
        if error > 300.:
            continue
        spread = max(track.spread, 1e-6)
        graded.append((track.group_id, mean, nearest, error, spread))
        if tally is not None:
            tally["graded"] += 1
            tally["covered"] += int(error <= spread)
    return graded


def overlay_truth(screen, sim, planner, renderer, graded):
    """Mark real predator positions on the estimated map and label the error.

    This is the only place truth touches the estimated view, and it lives here
    rather than in MapRenderer so that renderer keeps its snapshot-only
    contract: nothing the policy consumes can reach simulator state through it.

    A connector runs from each belief to the nearest real predator. Solid means
    truth fell inside that track's own reported spread, dashed means it did
    not, so a filter claiming more precision than it has looks wrong on sight
    rather than only in a metric.
    """
    truths = [(predator.x, predator.y) for predator in sim.env.predators]
    for group_id, group in planner.estimator.groups.items():
        view = renderer.views.get(group_id)
        if view is None or not group.anchored:
            continue
        # Say so on the panel. A screenshot of this must never be mistaken for
        # the observation-only view it otherwise is.
        notice = "GRADING: white = real predators (privileged)"
        width = renderer.small_font.size(notice)[0]
        corner = (view.rect.right - width - 12, view.rect.y + 8)
        pygame.draw.rect(screen, (46, 26, 30),
                         pygame.Rect(corner[0] - 5, corner[1] - 2, width + 10, 17),
                         border_radius=3)
        renderer._text(screen, notice, corner, TRUTH, renderer.small_font)
        for point in truths:
            x, y = view.pixel(point)
            if view.rect.collidepoint((x, y)):
                pygame.draw.line(screen, TRUTH, (x - 7, y - 7), (x + 7, y + 7), 2)
                pygame.draw.line(screen, TRUTH, (x - 7, y + 7), (x + 7, y - 7), 2)
    for group_id, mean, nearest, error, spread in graded:
        view = renderer.views.get(group_id)
        if view is None:
            continue
        start, end = view.pixel(mean), view.pixel(nearest)
        if not view.rect.collidepoint(start):
            continue
        if error <= spread:
            pygame.draw.line(screen, TRUTH, start, end, 1)
        else:
            renderer._dashed(screen, TRUTH, start, end)
        renderer._text(screen, f"err {error:.0f} / spread {spread:.0f}",
                       (start[0] + 12, start[1] + 6), TRUTH, renderer.small_font)


def compose(screen, sim, planner, renderer, actual, label, graded=None):
    """Ground truth is confined to the left surface; the right gets a snapshot."""
    sim.env.draw(actual)
    for agent in sim.env.agents:
        color = renderer._agent_color(agent.agent_id)
        position = (round(agent.x), round(agent.y))
        pygame.draw.circle(actual, color, position, 9, 2)
        actual.blit(renderer.marker_font.render(f"#{agent.agent_id}", True, color),
                    (position[0] + 12, position[1] - 28))
    for predator in sim.env.predators:
        # Truth for the eye only, so the belief can be judged against it.
        position = (round(predator.x), round(predator.y))
        pygame.draw.circle(actual, (240, 86, 98), position, 13, 3)
        facing = (position[0] + 26 * math.cos(predator.direction),
                  position[1] + 26 * math.sin(predator.direction))
        pygame.draw.line(actual, (240, 86, 98), position, facing, 3)
    draw_comparison(screen, actual, planner.snapshot(), renderer, label)
    if graded is not None:
        overlay_truth(screen, sim, planner, renderer, graded)


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--seed", type=int, help="World seed; random when omitted.")
    cli.add_argument("--seconds", type=float, default=600., help="Stop after this much sim time.")
    cli.add_argument("--predators", type=int, default=3, help="Predators present from the start.")
    cli.add_argument("--speed", type=int, default=4, help="Sim ticks attempted per frame.")
    cli.add_argument("--headless", action="store_true", help="No window; pair with --screenshot.")
    cli.add_argument("--screenshot", metavar="PATH", help="Save the final frame as a PNG.")
    cli.add_argument("--snapshot-json", metavar="PATH", help="Save the final snapshot for inspection.")
    cli.add_argument("--no-belief", action="store_true", help="Draw the map without the predator belief.")
    cli.add_argument("--grade", action="store_true",
                     help="Draw real predator positions on the estimated map to score the "
                          "belief. Privileged data: the right panel stops being "
                          "observation-only while this is on.")
    cli.add_argument("--particles", type=int, default=96, help="Particles per predator track.")
    cli.add_argument("--belief-escape", action="store_true",
                     help="Also let the belief steer the flee heading.")
    cli.add_argument("--belief-hold", type=float, default=0.,
                     help="Seconds a confident belief may hold the danger state open.")
    cli.add_argument("--config", help="Expert policy JSON path.")
    cli.add_argument("--planner-config", help="Global planner JSON path.")
    args = cli.parse_args()
    if args.headless and not (args.screenshot or args.snapshot_json):
        cli.error("--headless produces no output without --screenshot or --snapshot-json")
    if args.seconds <= 0 or args.speed < 1 or args.predators < 0:
        cli.error("Require positive seconds, speed >= 1 and predators >= 0")

    seed = random.randint(0, 2**32 - 1) if args.seed is None else args.seed
    expert, planner = configure(args.config, args.planner_config, not args.no_belief,
                                args.particles, args.belief_escape, args.belief_hold)
    policy = ExpertPolicy(expert, planner)
    sim = SimulationCore(seed=seed, starting_predators=args.predators)
    print(f"Seed: {seed}")
    print(f"Predator belief: {'on' if planner.predator_belief.enabled else 'off'}"
          f" | steering: {expert.memory.belief_escape_enabled}"
          f" | hold: {expert.memory.belief_flee_seconds:g}s")
    if not args.headless:
        print("Keys: B biomes   P predator belief   F fit   drag pan   wheel zoom")
        if args.grade:
            print("Grading overlay ON: white crosses on the estimated map are real predator")
            print("  positions, which the agents do not know. Connector solid = truth inside")
            print("  the track's spread, dashed = outside. Omit --grade for an honest view.")

    pygame.init()
    renderer = MapRenderer()
    actual = pygame.Surface((sim.env_width, sim.env_height))
    screen = None
    if not args.headless:
        info = pygame.display.Info()
        height = int(info.current_h * .9)
        screen = pygame.display.set_mode((min(int(info.current_w * .95), int(height * 1.9)), height),
                                         pygame.RESIZABLE)
        pygame.display.set_caption(f"Shared world model · seed {seed}")
    clock = pygame.time.Clock()
    reason, actions = "horizon", []
    # Running calibration tally, so over-confidence is readable live rather
    # than only afterwards in a metric.
    tally = dict(graded=0, covered=0) if args.grade else None
    graded = [] if args.grade else None
    state = sim.step([])
    try:
        while state["sim_time"] < args.seconds and state["num_agents"]:
            if not args.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        reason = "window_closed"
                        state = dict(state, num_agents=0)
                        break
                    renderer.handle_event(event)
                if reason == "window_closed":
                    break
            deadline = time.perf_counter() + .05
            for _ in range(args.speed):
                actions = [(action.agent_id, action) for action in policy.actions_for_step(
                    state["observations"], sim_time=state["sim_time"])]
                state = sim.step(actions)
                if tally is not None:
                    graded = grade_truth(sim, policy.planner, tally)
                if not state["num_agents"] or state["sim_time"] >= args.seconds:
                    break
                if not args.headless and time.perf_counter() >= deadline:
                    break
            if not args.headless:
                tracks = len(policy.planner.predator_tracker.tracks)
                coverage = ('' if not tally or not tally["graded"] else
                            f' | inside spread {tally["covered"] / tally["graded"]:.0%}')
                compose(screen, sim, policy.planner, renderer, actual,
                        f'Score {state["score"]:.1f} | Agents {state["num_agents"]} | '
                        f'Predators {len(sim.env.predators)} | Beliefs {tracks}'
                        f'{coverage} | {state["sim_time"]:.1f}s', graded)
                pygame.display.flip()
                clock.tick(60)
        else:
            reason = "extinction" if not state["num_agents"] else "horizon"
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        snapshot = policy.planner.snapshot()
        if args.screenshot:
            image = pygame.Surface((1900, 1000))
            compose(image, sim, policy.planner, renderer, actual,
                    f'Seed {seed} | Score {sim.env.score:.1f} | Agents {len(sim.env.agents)} | '
                    f'Predators {len(sim.env.predators)} | {sim.env.time:.1f}s', graded)
            destination = Path(args.screenshot).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(image, str(destination))
            print(f"Screenshot: {destination}")
        if args.snapshot_json:
            destination = Path(args.snapshot_json).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8")
            print(f"Snapshot: {destination}")
        pygame.quit()
        groups = snapshot["groups"]
        anchored = sum(bool(group.get("anchored")) for group in groups)
        beliefs = sum(len((group.get("predator_belief") or {}).get("tracks", [])) for group in groups)
        print(f"Ended ({reason}) at {sim.env.time:.1f}s | score {sim.env.score:.2f}")
        print(f"Frames: {len(groups)} ({anchored} anchored) | predator beliefs held: {beliefs}")
        if tally and tally["graded"]:
            # A well-calibrated cloud puts truth inside one RMS radius about
            # 63% of the time; materially less means the spread is understated.
            print(f"Calibration: truth inside the reported spread "
                  f"{tally['covered'] / tally['graded']:.1%} of {tally['graded']} graded "
                  f"track-ticks (well-calibrated is near 63%)")
        policy.reset()


if __name__ == "__main__":
    main()
