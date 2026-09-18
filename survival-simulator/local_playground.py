import argparse
import json
import math
from pathlib import Path
import random
import time
import webbrowser

import pygame

from src.core import SimulationCore
from src.utils.controllers.expert_policy import ExpertPolicy, load_config
from src.utils.controllers.global_planner import load_planner_config
from src.utils.map_renderer import MapRenderer, draw_comparison
from src.utils.planner_overlay import draw_planner_overlay
from src.utils.playback import PlaybackControls
from src.utils.run_diagnostics import HORIZON_SECONDS, RunDiagnostics, load_diagnostics_config


def _draw_map_comparison(screen, sim, planner, renderer, actual_surface, label):
    """Ground-truth drawing is confined to the actual-world side of the viewer."""
    sim.env.draw(actual_surface)
    for agent in sim.env.agents:
        color = renderer._agent_color(agent.agent_id)
        position = (round(agent.x), round(agent.y))
        pygame.draw.circle(actual_surface, color, position, 9, 2)
        text = renderer.marker_font.render(f"#{agent.agent_id}", True, color)
        actual_surface.blit(text, (position[0] + 12, position[1] - 28))
    draw_comparison(screen, actual_surface, planner.snapshot(), renderer, label)


def local_simulation(verbose=True, seed=None, config_path=None,
                     diagnostics_config_path=None, output_dir=None,
                     max_seconds=HORIZON_SECONDS, open_report=None, diagnostics=True,
                     planner_config_path=None, map_view=True, map_screenshot=None,
                     predators_enabled=True, central_harvest=None, speed=1, policy_mode=None):
    """Run the expert, optionally render, and save a recap even on window close."""
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= HORIZON_SECONDS:
        raise ValueError(f"max_seconds must be in (0, {HORIZON_SECONDS:g}]")
    playback = PlaybackControls(speed)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    policy_config = load_config(config_path)
    if policy_mode is not None:
        policy_config = policy_config.model_copy(update={"policy_mode": policy_mode})
    if central_harvest is not None:
        policy_config = policy_config.model_copy(update={"harvest": policy_config.harvest.model_copy(
            update={"enabled": central_harvest})})
    policy = ExpertPolicy(policy_config, load_planner_config(planner_config_path))
    report_config = load_diagnostics_config(diagnostics_config_path)
    sim = SimulationCore(seed=seed, predators_enabled=predators_enabled)
    recorder = RunDiagnostics(sim, report_config, output_dir, policy.config.model_dump(),
                              policy.planner.config.model_dump()) if diagnostics else None
    screen, clock, font = None, None, None
    map_renderer = MapRenderer() if (verbose and map_view) or map_screenshot else None
    actual_surface = pygame.Surface((sim.env_width, sim.env_height)) if map_renderer else None
    reason = "error"
    next_progress = 0.0
    print(f"Seed: {seed}")
    print(f"Policy: {policy_config.policy_mode}")
    coordinated = policy_config.policy_mode == "simple" or policy_config.harvest.enabled
    print(f"Predators: {'enabled' if predators_enabled else 'disabled'} | Central harvest: {coordinated}")
    if recorder:
        print(f"Run data: {recorder.directory}")

    try:
        if verbose:
            pygame.init()
            info = pygame.display.Info()
            screen_height = int(info.current_h * 0.9)
            aspect = 1.9 if map_view else sim.env_width / sim.env_height
            screen_width = min(int(info.current_w * 0.95), int(screen_height * aspect))
            screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
            pygame.display.set_caption(f"Survival simulator · expert · seed {seed}")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont(None, 24)

        actions = []
        while True:
            events = pygame.event.get() if verbose else []
            if any(event.type == pygame.QUIT for event in events):
                reason = "window_closed"
                break
            for event in events:
                if playback.handle_event(event, screen.get_size()):
                    continue
                if map_view and map_renderer:
                    map_renderer.handle_event(event)
            if verbose:
                playback.advance(clock.tick(60) / 1000.)
            frame_deadline = time.perf_counter() + playback.MAX_FRAME_WORK_SECONDS
            finished = False
            while not verbose or playback.consume_step(sim.dt):
                if recorder:
                    recorder.before_step(actions)
                state = sim.step(actions)
                if recorder:
                    recorder.after_step()
                actions = [
                    (action.agent_id, action) for action in policy.actions_for_step(
                        state["observations"], sim_time=state["sim_time"]
                    )
                ]
                if sim.env.time >= next_progress:
                    detail = ''
                    if policy_config.policy_mode == 'simple':
                        detail = (f' | Under {policy_config.simple.renewal_age:g}s: '
                                  f'{policy.harvest.population_plan.get("young", 0)}/{policy.harvest.population_target}'
                                  f' | Aging scouts: {len(policy.harvest.retired)}')
                    print(f'Score: {state["score"]:.2f} | Agents alive: {state["num_agents"]} | Time: {sim.env.time:.2f}{detail}')
                    next_progress += report_config.progress_interval_seconds
                if state["num_agents"] == 0:
                    reason, finished = "extinction", True
                elif sim.env.time + 1e-9 >= max_seconds:
                    reason = "horizon" if max_seconds == HORIZON_SECONDS else "time_limit"
                    finished = True
                # Keep event handling/rendering responsive even when the policy
                # cannot compute the requested speed. Always decide every tick.
                if finished or not verbose or time.perf_counter() >= frame_deadline:
                    break

            if verbose:
                target = f' (target {policy.harvest.population_target})' if policy.harvest.active else ''
                if policy.harvest.active and policy_config.policy_mode == 'simple':
                    target = (f' (under {policy_config.simple.renewal_age:g}s: '
                              f'{policy.harvest.population_plan.get("young", 0)}/{policy.harvest.population_target}; '
                              f'scouts: {len(policy.harvest.retired)})')
                label = f'Score: {sim.env.score:.2f} | Agents: {len(sim.env.agents)}{target} | Time: {sim.env.time:.1f}s'
                viewport = screen.subsurface((0, 0, screen.get_width(),
                                              max(1, screen.get_height() - playback.HEIGHT)))
                if map_view:
                    _draw_map_comparison(viewport, sim, policy.planner, map_renderer, actual_surface, label)
                else:
                    sim.env.draw(viewport)
                    draw_planner_overlay(viewport, sim.env, policy.planner, font)
                    viewport.blit(font.render(label, True, (255, 255, 255)), (20, 20))
                playback.draw(screen, font)
                pygame.display.flip()
            if finished:
                break
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        if map_screenshot:
            image_surface = pygame.Surface((1600, 900))
            label = f"Seed {seed} | Score {sim.env.score:.2f} | Agents {len(sim.env.agents)} | Time {sim.env.time:.1f}s"
            _draw_map_comparison(image_surface, sim, policy.planner, map_renderer, actual_surface, label)
            screenshot_path = Path(map_screenshot).expanduser().resolve()
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(image_surface, str(screenshot_path))
            print(f"Map comparison: {screenshot_path}")
        pygame.quit()
        print(f"Run ended ({reason}). Score: {sim.env.score:.3f} | Seed: {seed}")
        if recorder:
            (recorder.directory / "planner_final.json").write_text(
                json.dumps(policy.planner.snapshot(), indent=2, allow_nan=False), encoding="utf-8"
            )
            directory = recorder.finish(reason)
            summary = recorder.summary
            print(f"Survival target: {summary['survival_target_percent']:.2f}% | "
                  f"Score / optimistic benchmark: {summary['benchmark_percent']:.2f}%")
            print(summary["evolution"]["last_generation_verdict"])
            report_path = directory / "report.html"
            print(f"Recap: {report_path}")
            should_open = report_config.open_report if open_report is None else open_report
            if should_open:
                webbrowser.open(report_path.as_uri())
        policy.reset()
    return recorder.directory if recorder else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the expert and save score/evolution diagnostics.")
    parser.add_argument("--headless", action="store_true", help="Run without a Pygame window.")
    parser.add_argument("--policy", choices=("standard", "simple"), default=None,
                        help="Select the existing policy or the simple population/territory/food policy.")
    parser.add_argument("--no-predators", action="store_true", help="Disable initial and later predator spawns.")
    parser.add_argument("--central-harvest", action=argparse.BooleanOptionalAction, default=None,
                        help="Coordinate survival, feeding and generation renewal after shared absolute mapping.")
    parser.add_argument("--seed", type=int, help="World seed for reproducible runs on the same OS.")
    parser.add_argument("--speed", type=int, choices=PlaybackControls.SPEEDS, default=1,
                        help="Initial graphical playback speed; buttons or keys 1-5 change it during a run. Headless runs are unthrottled.")
    parser.add_argument("--config", help="Expert policy JSON path.")
    parser.add_argument("--planner-config", help="Global planner JSON path.")
    parser.add_argument("--map-view", action=argparse.BooleanOptionalAction, default=True,
                        help="Show actual simulation beside the agents' independent internal map (default: on).")
    parser.add_argument("--map-screenshot", metavar="PATH",
                        help="Save the final actual/internal map comparison as PNG, including in headless mode.")
    parser.add_argument("--diagnostics-config", help="Diagnostics JSON path.")
    parser.add_argument("--output-dir", help="Report directory (relative to survival-simulator, or absolute).")
    parser.add_argument("--max-seconds", type=float, default=HORIZON_SECONDS,
                        help="Stop a short run early; the survival benchmark remains 3000 seconds.")
    parser.add_argument("--open-report", action="store_true", default=None,
                        help="Open the saved HTML recap in your default browser.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Disable event recording and reports.")
    args = parser.parse_args()
    local_simulation(verbose=not args.headless, seed=args.seed, config_path=args.config,
                     diagnostics_config_path=args.diagnostics_config, output_dir=args.output_dir,
                     max_seconds=args.max_seconds, open_report=args.open_report,
                     diagnostics=not args.no_diagnostics, planner_config_path=args.planner_config,
                     map_view=args.map_view, map_screenshot=args.map_screenshot,
                     predators_enabled=not args.no_predators, central_harvest=args.central_harvest,
                     speed=args.speed, policy_mode=args.policy)
