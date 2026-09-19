"""Offline spectator diagrams from actual recorded frames, never rerun a game."""
import json
import math
from pathlib import Path


def render_frames(folder, frames):
    import pygame
    folder = Path(folder)
    static = json.loads((folder/'static.json').read_text())
    scale = min(1100/static['width'], 800/static['height'])
    pygame.font.init()
    font = pygame.font.Font(None, 18)
    selected = sorted(set([0, max(0, len(frames)-101), len(frames)-1]))
    files = []
    for index in selected:
        frame = json.loads(frames[index][2])
        canvas = pygame.Surface((round(static['width']*scale), round(static['height']*scale)+40))
        canvas.fill((242, 244, 236))
        def point(x, y):
            return round(x*scale), round(y*scale)+40
        for o in static['obstacles']:
            pygame.draw.rect(canvas, (90, 95, 100), (*point(o['x'], o['y']), round(o['width']*scale), round(o['height']*scale)))
        world = frame['world']
        for tree in world['trees']:
            pygame.draw.circle(canvas, (175, 206, 165), point(tree['x'], tree['y']), max(2, round(tree['radius']*scale)), 1)
        for fruit in world['fruits']:
            pygame.draw.circle(canvas, (219, 135, 36), point(fruit['x'], fruit['y']), max(2, round(fruit['radius']*scale)))
        roles = frame.get('policy', {}).get('roles', {})
        for predator in world['predators']:
            pygame.draw.circle(canvas, (195, 35, 45), point(predator['x'], predator['y']), max(4, round(predator['size']*scale)))
        for agent in world['agents']:
            p = point(agent['x'], agent['y'])
            role = roles.get(str(agent['agent_id']), '?')
            colour = (230, 177, 0) if 'bait' in role else (20, 156, 182) if role == 'guide' else (35, 80, 170)
            pygame.draw.circle(canvas, colour, p, max(3, round(agent['size']*scale)))
            pygame.draw.line(canvas, colour, p, point(agent['x']+20*math.cos(agent['direction']), agent['y']+20*math.sin(agent['direction'])), 2)
            label = font.render(f"{agent['agent_id']} {role} E{agent['energy']:.0f}", True, (20, 25, 35))
            canvas.blit(label, (p[0]+5, p[1]-14))
        caption = font.render(f"Spectator truth | t={frame['sim_time']:.1f}s | tick {frame['tick']} | pre-action | red: predators", True, (20, 25, 35))
        canvas.blit(caption, (10, 10))
        name = f"frame-{frame['tick']:06d}.png"
        pygame.image.save(canvas, folder/name)
        files.append(name)
    return files
