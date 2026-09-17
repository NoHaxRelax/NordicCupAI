"""Local rendering only: align an estimated map to the picture for inspection.

The controller never receives this environment or the display transform.
"""

import colorsys

import numpy as np
import pygame

from src.utils.controllers.world_estimator import rotate


def draw_planner_overlay(screen, env, planner, font):
    if not planner.config.enabled or not planner.config.draw_overlay:
        return
    zoom = min(screen.get_width() / env.width, screen.get_height() / env.height)
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    actual_agents = {agent.agent_id: agent for agent in env.agents}
    for group_id, plan in planner.plans.items():
        poses = [p for p in planner.estimator.poses.values()
                 if p.group_id == group_id and p.agent_id in actual_agents]
        if not poses:
            continue
        # This is only a display gauge for otherwise arbitrary local coordinates.
        anchor = min(poses, key=lambda pose: pose.agent_id)
        actual = actual_agents[anchor.agent_id]
        angle = actual.direction - anchor.heading
        offset = np.array((actual.x, actual.y)) - rotate(anchor.position, angle)

        def pixel(point):
            return tuple(int(round(value * zoom)) for value in rotate(point, angle) + offset)

        colors = {}
        for section in plan.sections:
            hue = (group_id * 0.61803398875 + section.section_id * 0.38196601125) % 1
            color = tuple(round(value * 255) for value in colorsys.hsv_to_rgb(hue, 0.6, 1))
            colors[section.section_id] = color
            x0, y0, x1, y1 = section.bounds
            corners = [pixel(p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            pygame.draw.polygon(overlay, (*color, 22), corners)
            pygame.draw.polygon(overlay, (*color, 145), corners, 1)
            center = pixel(section.center)
            pygame.draw.circle(overlay, (*color, 240), center, 5, 2)
            label = font.render(f'{group_id}:{section.section_id} food~{section.food_weight:.1f}', True, color)
            overlay.blit(label, (center[0] + 6, center[1] + 6))
        for pose in poses:
            index = plan.assignments.get(pose.agent_id)
            if index is None:
                continue
            active = pose.agent_id in planner.hints
            uncertain = pose.uncertainty > planner.config.estimator.max_position_uncertainty
            color = (*colors[index], 50 if uncertain else (230 if active else 100))
            start, end = pixel(pose.position), pixel(plan.sections[index].center)
            pygame.draw.line(overlay, color, start, end, 2 if active else 1)
            overlay.blit(font.render(str(pose.agent_id), True, colors[index]), start)
    screen.blit(overlay, (0, 0))
    label = f'Estimated planner: {len(planner.plans)} map groups | {sum(len(p.sections) for p in planner.plans.values())} sections'
    screen.blit(font.render(label, True, (255, 255, 255)), (20, 44))
