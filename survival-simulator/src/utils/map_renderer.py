"""Render the controller's snapshot without access to simulator state.

Each disconnected group owns its view transform. Boundary anchoring changes that
group's coordinate frame, so an old user pan/zoom is discarded on frame revision.
"""

from dataclasses import dataclass, field
import colorsys
import math

import numpy as np
import pygame


BACKGROUND = (12, 19, 28)
PANEL = (18, 28, 39)
UNKNOWN = (8, 14, 22)
TEXT = (224, 233, 241)
MUTED = (140, 160, 177)
TRAP_COLORS = {"wall": (255, 190, 65), "slot": (55, 231, 163), "shelter": (188, 150, 247)}
BIOME_COLORS = {
    "forest": (41, 118, 61), "grassland": (96, 174, 75),
    "swamp": (85, 112, 95), "desert": (210, 167, 95),
    "river": (61, 162, 217),
}


def panel_layout(rect, count, gap=10):
    """Non-overlapping panels, with enough width for IDs and anchoring status."""
    rect = pygame.Rect(rect)
    if count < 1:
        return []
    columns = 1 if count <= 2 else 2
    rows = math.ceil(count / columns)
    width = (rect.width - gap * (columns - 1)) // columns
    height = (rect.height - gap * (rows - 1)) // rows
    return [pygame.Rect(rect.x + (i % columns) * (width + gap),
                        rect.y + (i // columns) * (height + gap), width, height)
            for i in range(count)]


@dataclass
class MapView:
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 1, 1))
    center: pygame.Vector2 = field(default_factory=pygame.Vector2)
    scale: float = 1.0
    zoom: float = 1.0
    pan: pygame.Vector2 = field(default_factory=pygame.Vector2)
    revision: object = None

    def pixel(self, point):
        result = pygame.Vector2(self.rect.center) + self.pan
        result += (pygame.Vector2(point) - self.center) * self.scale * self.zoom
        return round(result.x), round(result.y)


class MapRenderer:
    """Snapshot-only internal map, with wheel zoom, drag pan and F to fit."""

    def __init__(self):
        pygame.font.init()
        self.title_font = pygame.font.SysFont("Segoe UI", 19, bold=True)
        self.font = pygame.font.SysFont("Segoe UI", 14)
        self.small_font = pygame.font.SysFont("Segoe UI", 12)
        self.marker_font = pygame.font.SysFont("Segoe UI", 26, bold=True)
        self.views = {}
        self.dragging = None
        self.last_mouse = (0, 0)
        self.show_biome_estimates = True
        self.show_trap_estimates = True
        self.biome_surfaces = {}

    def _text(self, surface, text, position, color=TEXT, font=None, max_width=None):
        font = font or self.font
        text = str(text)
        if max_width is not None:
            while text and font.size(text)[0] > max_width:
                text = text[:-2].rstrip(".") + "." if len(text) > 1 else ""
        surface.blit(font.render(text, True, color), position)

    def _view_at(self, position):
        return next((key for key, view in self.views.items()
                     if view.rect.collidepoint(position)), None)

    def handle_event(self, event):
        """Only adjust the panel under the pointer; never change controller data."""
        if hasattr(event, "pos"):
            self.last_mouse = event.pos
        if event.type == pygame.MOUSEBUTTONDOWN:
            key = self._view_at(event.pos)
            if event.button == 1:
                self.dragging = key
            elif event.button == 2 and key in self.views:
                self.views[key].zoom = 1.0
                self.views[key].pan.update(0, 0)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = None
        elif event.type == pygame.MOUSEMOTION and self.dragging in self.views:
            self.views[self.dragging].pan += pygame.Vector2(event.rel)
        elif event.type == pygame.MOUSEWHEEL:
            key = self._view_at(self.last_mouse)
            if key in self.views:
                view = self.views[key]
                previous = view.zoom
                view.zoom = min(20, max(0.2, view.zoom * 1.2 ** event.y))
                pointer = pygame.Vector2(self.last_mouse) - pygame.Vector2(view.rect.center)
                view.pan = pointer - (pointer - view.pan) * (view.zoom / previous)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_b:
            self.show_biome_estimates = not self.show_biome_estimates
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
            self.show_trap_estimates = not self.show_trap_estimates
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            key = self._view_at(self.last_mouse)
            targets = [self.views[key]] if key in self.views else self.views.values()
            for view in targets:
                view.zoom = 1.0
                view.pan.update(0, 0)

    def _fit(self, group, agents, rect):
        key = group["group_id"]
        view = self.views.setdefault(key, MapView())
        revision = (group.get("frame_revision", 0), group.get("anchored", False))
        if revision != view.revision:
            view.zoom, view.pan, view.revision = 1.0, pygame.Vector2(), revision
        view.rect = rect
        size = group.get("world_size")
        if group.get("anchored") and size:
            points = [(0, 0), tuple(size)]
        else:
            points = [agent["position"] for agent in agents]
            points += [tree["position"] for tree in group.get("trees", [])]
            points += [sample["position"] for sample in group.get("biomes", [])]
            points += [edge[name] for edge in group.get("edges", []) for name in ("start", "end")]
            points += [site["bait"] for site in (group.get("trap_estimate") or {}).get("sites", [])]
        if not points:
            points = [(0, 0)]
        low = pygame.Vector2(min(p[0] for p in points), min(p[1] for p in points))
        high = pygame.Vector2(max(p[0] for p in points), max(p[1] for p in points))
        extent = high - low
        view.center = (high + low) / 2
        view.scale = min(max(1, rect.width - 34) / max(160, extent.x),
                         max(1, rect.height - 34) / max(160, extent.y))
        return view

    @staticmethod
    def _agent_color(agent_id):
        return tuple(round(channel * 255) for channel in
                     colorsys.hsv_to_rgb((agent_id * 0.61803398875) % 1, 0.5, 1))

    @staticmethod
    def _dashed(surface, color, start, end):
        vector = pygame.Vector2(end) - pygame.Vector2(start)
        distance = vector.length()
        if distance < 1:
            return
        direction = vector / distance
        # Clipping also bounds the number of dashes when a panel is zoomed in.
        clipped = surface.get_clip().clipline(start, end)
        if not clipped:
            return
        start, end = map(pygame.Vector2, clipped)
        distance = (end - start).length()
        for offset in range(0, int(distance), 10):
            pygame.draw.line(surface, color, start + direction * offset,
                             start + direction * min(offset + 4, distance), 1)

    def _draw_group(self, surface, rect, group, agents, snapshot):
        pygame.draw.rect(surface, PANEL, rect, border_radius=7)
        group_id = group["group_id"]
        anchored = bool(group.get("anchored"))
        status = "absolute frame" if anchored else "local frame"
        if anchored and not group.get("world_size"):
            status += " | bounds incomplete"
        self._text(surface, f"Cluster {group_id}  |  {status}", (rect.x + 10, rect.y + 7),
                   (120, 219, 179) if anchored else TEXT, max_width=rect.width - 20)
        counts = f'{len(agents)} agents   {len(group.get("edges", []))} edges   {len(group.get("biomes", []))} biome samples'
        traps = group.get("trap_estimate")
        if traps is not None:
            counts += f'   {len(traps["sites"])} potential traps' + (' (stale)' if traps.get("stale") else '')
        self._text(surface, counts, (rect.x + 10, rect.y + 28), MUTED,
                   self.small_font, rect.width - 20)
        plot = pygame.Rect(rect.x + 7, rect.y + 49, max(1, rect.width - 14), max(1, rect.height - 73))
        pygame.draw.rect(surface, UNKNOWN, plot)
        view = self._fit(group, agents, plot)
        original_clip = surface.get_clip()
        surface.set_clip(plot)
        estimate = group.get("biome_estimate")
        if estimate and self.show_biome_estimates:
            self._draw_biome_estimate(surface, view, group["group_id"], estimate)
        size = group.get("world_size")
        if anchored and size:
            corners = [view.pixel(p) for p in [(0, 0), (size[0], 0), tuple(size), (0, size[1])]]
            pygame.draw.lines(surface, (64, 107, 99), True, corners, 1)
        for sample in group.get("biomes", []):
            color = BIOME_COLORS.get(sample.get("biome", ""), MUTED)
            pygame.draw.circle(surface, color, view.pixel(sample["position"]), 3)
        for edge in group.get("edges", []):
            pygame.draw.line(surface, (180, 188, 199), view.pixel(edge["start"]), view.pixel(edge["end"]), 2)
        for tree in group.get("trees", []):
            pygame.draw.circle(surface, (124, 179, 111), view.pixel(tree["position"]), 4, 1)
        by_id = {agent["agent_id"]: agent for agent in agents}
        harvest = snapshot.get("harvest", {})
        if harvest.get("active") and harvest.get("group_id") == group["group_id"]:
            coverage = harvest.get("coverage", {})
            pitch = coverage.get("grid_pitch")
            samples = coverage.get("points", [])
            if pitch and samples:
                origin = [min(p["position"][axis] for p in samples) for axis in (0, 1)]
                cells = {tuple(round((p["position"][axis] - origin[axis]) / pitch[axis]) for axis in (0, 1)): p
                         for p in samples}
                for (x, y), sample in cells.items():
                    owner = sample.get("owner")
                    if owner is None:
                        continue
                    px, py = sample["position"]
                    color = tuple(round(v * .6) for v in self._agent_color(owner))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if cells.get((x + dx, y + dy), {}).get("owner") == owner:
                            continue
                        if dx:
                            a, b = (px + dx * pitch[0] / 2, py - pitch[1] / 2), (px + dx * pitch[0] / 2, py + pitch[1] / 2)
                        else:
                            a, b = (px - pitch[0] / 2, py + dy * pitch[1] / 2), (px + pitch[0] / 2, py + dy * pitch[1] / 2)
                        pygame.draw.line(surface, color, view.pixel(a), view.pixel(b), 1)
            for position in coverage.get("homes", {}).values():
                x, y = view.pixel(position)
                pygame.draw.line(surface, (88, 208, 216), (x - 4, y), (x + 4, y), 1)
                pygame.draw.line(surface, (88, 208, 216), (x, y - 4), (x, y + 4), 1)
            patrols = set(coverage.get("patrol_ids", []))
            targets = dict(coverage.get("targets", {}))
            targets.update(coverage.get("gap_targets", {}))
            for agent_id, position in targets.items():
                agent = by_id.get(int(agent_id))
                if agent is not None and int(agent_id) in patrols and not harvest.get("navigation"):
                    self._dashed(surface, (88, 208, 216), view.pixel(agent["position"]), view.pixel(position))
            for patch in coverage.get("unknown_patches", []):
                pygame.draw.circle(surface, (88, 208, 216), view.pixel(patch["position"]), 7, 1)
            fruits = {fruit["track_id"]: fruit for fruit in harvest.get("fruits", [])}
            for fruit in fruits.values():
                color = (246, 204, 87) if fruit["ripe_in_seconds"] <= 0 else (129, 195, 88)
                x, y = view.pixel(fruit["position"])
                pygame.draw.rect(surface, color, (x - 3, y - 3, 7, 7), 1)
            for agent_id, track_id in harvest.get("assignments", {}).items():
                agent, fruit = by_id.get(int(agent_id)), fruits.get(track_id)
                if agent is not None and fruit is not None and not harvest.get("navigation"):
                    self._dashed(surface, (181, 151, 68), view.pixel(agent["position"]), view.pixel(fruit["position"]))
            tasks = harvest.get("tasks", {})
            for agent_id, route in harvest.get("navigation", {}).items():
                agent = by_id.get(int(agent_id))
                if agent is None or route.get("status") in ("waiting", "blocked", "arrived"):
                    continue
                task = tasks.get(agent_id, tasks.get(str(agent_id), {}))
                color = (181, 151, 68) if task.get("kind") == "collect fruit" else (88, 208, 216)
                path = [agent["position"]] + route.get("waypoints", [])
                for start, end in zip(path, path[1:]):
                    self._dashed(surface, color, view.pixel(start), view.pixel(end))
        now = snapshot.get("sim_time") or 0
        subdued_history = harvest.get("active") and harvest.get("coverage", {}).get("homes")
        history_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA) if subdued_history else surface
        if subdued_history:
            history_surface.set_clip(surface.get_clip())
        for link in snapshot.get("links", []):
            observer, target = by_id.get(link["observer_id"]), by_id.get(link["target_id"])
            if observer is None or target is None:
                continue
            age = max(0, now - link.get("last_seen", now))
            brightness = 0.3 + 0.7 * math.exp(-age / 30)
            color = tuple(round(value * brightness) for value in (93, 139, 177))
            if subdued_history:
                color += (30,)
            self._dashed(history_surface, color, view.pixel(observer["position"]), view.pixel(target["position"]))
        if subdued_history:
            surface.blit(history_surface, (0, 0))
        if traps is not None and self.show_trap_estimates:
            self._draw_trap_estimate(surface, view, traps)
        roles = snapshot.get("roles", {})
        labels = []
        markers = [pygame.Rect(view.pixel(agent["position"]), (0, 0)).inflate(14, 14)
                   for agent in agents]
        for agent in agents:
            position = view.pixel(agent["position"])
            if not plot.collidepoint(position):
                continue
            color = self._agent_color(agent["agent_id"])
            uncertainty = max(0, float(agent.get("uncertainty", 0)))
            radius = max(7, min(24, round(uncertainty * view.scale * view.zoom)))
            if uncertainty:
                pygame.draw.circle(surface, tuple(int(value * 0.4) for value in color), position, radius, 1)
            heading = agent.get("heading", 0)
            facing = (position[0] + 13 * math.cos(heading), position[1] + 13 * math.sin(heading))
            pygame.draw.line(surface, color, position, facing, 2)
            pygame.draw.circle(surface, color, position, 5)
            role = agent.get("role", roles.get(agent["agent_id"], roles.get(str(agent["agent_id"]), "")))
            if isinstance(role, dict):
                role = role.get("role", "")
            task = harvest.get("tasks", {}).get(agent["agent_id"],
                                               harvest.get("tasks", {}).get(str(agent["agent_id"])))
            if task is not None:
                role = {"collect fruit": "collect", "patrol territory": "patrol", "scout patch": "scout",
                        "return to territory": "return",
                        "wait for ripening": "ripening", "blocked; choose another target": "blocked"}.get(task["kind"], task["kind"])
            label = f'#{agent["agent_id"]} {role}'.strip()
            if "trait_score" in agent:
                label += f' {agent["trait_score"]:.2f}x' + ("*" if agent.get("elite") else "")
            width, height = self.small_font.size(label)
            candidates = [(9, -15), (9, 8), (-width - 9, -15),
                          (-width - 9, 8), (9, -32), (9, 25)]
            for dx, dy in candidates:
                label_rect = pygame.Rect(position[0] + dx, position[1] + dy, width, height)
                label_rect.clamp_ip(plot.inflate(-4, -4))
                if not any(label_rect.colliderect(other) for other in labels + markers):
                    break
            labels.append(label_rect.inflate(4, 3))
            pygame.draw.rect(surface, UNKNOWN, label_rect.inflate(3, 2))
            self._text(surface, label, label_rect.topleft, color, self.small_font)
        # A scale bar makes independently fitted cluster maps comparable.
        units = 10 ** math.floor(math.log10(max(1e-6, 60 / (view.scale * view.zoom))))
        pixels = max(1, round(units * view.scale * view.zoom))
        left, bottom = plot.x + 10, plot.bottom - 9
        pygame.draw.line(surface, MUTED, (left, bottom), (left + pixels, bottom), 2)
        self._text(surface, f"{units:g} units", (left, bottom - 17), MUTED, self.small_font)
        surface.set_clip(original_clip)
        footer = f'Center {view.center.x:.0f}, {view.center.y:.0f}  |  zoom {view.zoom:.1f}x'
        if estimate and self.show_biome_estimates:
            agreement = estimate.get("land_sample_agreement")
            footer += f'  |  Estimated biomes: {len(estimate["sites"])} sites'
            if agreement is not None:
                footer += f' / {agreement:.0%} sample fit'
            schedule = estimate.get("refit_schedule")
            if schedule is not None:
                remaining = max(0, schedule["next_check_at"] - (snapshot.get("sim_time") or 0))
                footer += f' | check in {remaining:.0f}s'
        self._text(surface, footer, (rect.x + 10, rect.bottom - 20), MUTED,
                   self.small_font, rect.width - 20)

    def _draw_trap_estimate(self, surface, view, estimate):
        for site in estimate.get("sites", []):
            kind = site["kind"]
            color = MUTED if estimate.get("stale") else TRAP_COLORS[kind]
            x, y = view.pixel(site["bait"])
            if kind != "shelter":
                px, py = view.pixel(site["predator_side"])
                self._dashed(surface, color, (x, y), (px, py))
                pygame.draw.line(surface, color, (px - 5, py - 5), (px + 5, py + 5), 2)
                pygame.draw.line(surface, color, (px - 5, py + 5), (px + 5, py - 5), 2)
            if not surface.get_clip().inflate(28, 28).collidepoint(x, y):
                continue
            if kind == "shelter":
                pygame.draw.circle(surface, UNKNOWN, (x, y), 10)
                pygame.draw.circle(surface, color, (x, y), 10, 3)
            else:
                points = ([(x, y - 12), (x + 11, y + 9), (x - 11, y + 9)] if kind == "wall" else
                          [(x, y - 12), (x + 12, y), (x, y + 12), (x - 12, y)])
                pygame.draw.polygon(surface, UNKNOWN, points)
                pygame.draw.polygon(surface, color, points, 3)

    def _draw_biome_estimate(self, surface, view, group_id, estimate):
        cached = self.biome_surfaces.get(group_id)
        if cached is None or cached[0] != estimate["fingerprint"]:
            labels = np.array(estimate["labels"], dtype=int)
            confidence = np.array(estimate["confidence"])
            colors = np.full((*labels.shape, 3), UNKNOWN, dtype=np.uint8)
            for index, label in enumerate(estimate["palette"]):
                mask = labels == index
                base = np.array(BIOME_COLORS.get(label, MUTED))
                colors[mask] = np.clip(base * (.18 + .47 * confidence[mask, None]), 0, 255).astype(np.uint8)
            raster = pygame.surfarray.make_surface(np.transpose(colors, (1, 0, 2)))
            self.biome_surfaces[group_id] = estimate["fingerprint"], raster
        else:
            raster = cached[1]
        x0, y0, x1, y1 = estimate["bounds"]
        left, top = view.pixel((x0, y0))
        right, bottom = view.pixel((x1, y1))
        destination = pygame.Rect(left, top, max(1, right - left), max(1, bottom - top))
        visible = destination.clip(surface.get_clip())
        if visible.width and visible.height:
            # Crop BEFORE scaling: zooming a whole map by 20x must not allocate
            # a giant off-screen surface. Only visible grid cells are enlarged.
            nx, ny = raster.get_size()
            sx = max(0, math.floor((visible.left - left) * nx / destination.width))
            sy = max(0, math.floor((visible.top - top) * ny / destination.height))
            ex = min(nx, math.ceil((visible.right - left) * nx / destination.width))
            ey = min(ny, math.ceil((visible.bottom - top) * ny / destination.height))
            if destination.width / nx > 128 or destination.height / ny > 128:
                # A very coarse grid at extreme zoom needs only a handful of
                # rectangles, with no enlarged image allocation at all.
                for y in range(sy, ey):
                    for x in range(sx, ex):
                        a = round(left + x * destination.width / nx), round(top + y * destination.height / ny)
                        b = round(left + (x + 1) * destination.width / nx), round(top + (y + 1) * destination.height / ny)
                        pygame.draw.rect(surface, raster.get_at((x, y)), (a[0], a[1], b[0] - a[0], b[1] - a[1]))
            else:
                target = pygame.Rect(round(left + sx * destination.width / nx), round(top + sy * destination.height / ny),
                                     max(1, round((ex - sx) * destination.width / nx)),
                                     max(1, round((ey - sy) * destination.height / ny)))
                surface.blit(pygame.transform.scale(raster.subsurface((sx, sy, ex - sx, ey - sy)), target.size), target)
        for border in estimate["borders"]:
            base = (216, 195, 153) if border["kind"] == "land" else (82, 178, 219)
            color = tuple(round(channel * (.25 + .65 * border["confidence"])) for channel in base)
            start, end = view.pixel(border["start"]), view.pixel(border["end"])
            if border["kind"] == "land":
                self._dashed(surface, color, start, end)
            else:
                pygame.draw.line(surface, color, start, end, 1)

    def draw(self, surface, snapshot, rect=None):
        """Draw a JSON-compatible planner snapshot. No environment is accepted."""
        rect = pygame.Rect(rect or surface.get_rect())
        pygame.draw.rect(surface, BACKGROUND, rect)
        groups = {group["group_id"]: group for group in snapshot.get("groups", [])}
        agents = snapshot.get("agents", [])
        for agent in agents:
            groups.setdefault(agent["group_id"], {"group_id": agent["group_id"]})
        anchored = sum(bool(groups[agent["group_id"]].get("anchored")) for agent in agents)
        if len(groups) == 1 and agents:
            status = ("Shared absolute coordinates" if anchored == len(agents)
                      else "Shared local coordinates | seeking a boundary anchor")
        else:
            status = f"{len(groups)} separate frames | {anchored}/{len(agents)} agents anchored"
        if snapshot.get("phase") == "population":
            status += " | Growing population"
        coverage = snapshot.get("harvest", {}).get("coverage", {})
        if snapshot.get("harvest", {}).get("active") and coverage.get("survey_percent") is not None:
            status += f' | Survey samples: {coverage["survey_percent"]:.0f}%'
        self._text(surface, "Agents' internal map", (rect.x + 12, rect.y + 10), font=self.title_font)
        self._text(surface, status,
                   (rect.x + 12, rect.y + 37), MUTED, self.small_font, rect.width - 24)
        self._text(surface, "Drag: pan   Wheel: zoom   F: fit   B: biomes   T: trap sites",
                   (rect.x + 12, rect.y + 56), MUTED, self.small_font, rect.width - 24)
        ordered = [groups[key] for key in sorted(groups)]
        self.views = {key: view for key, view in self.views.items() if key in groups}
        self.biome_surfaces = {key: value for key, value in self.biome_surfaces.items() if key in groups}
        harvest_legend_height = 36 if snapshot.get("harvest", {}).get("active") else 0
        trap_legend_height = 36 if any(group.get("trap_estimate") is not None for group in ordered) else 0
        content = pygame.Rect(rect.x + 10, rect.y + 83, rect.width - 20,
                              max(1, rect.height - 163 - harvest_legend_height - trap_legend_height))
        if not ordered:
            self._text(surface, "Waiting for agent observations", content.move(12, 20).topleft, MUTED)
        for group, panel in zip(ordered, panel_layout(content, len(ordered))):
            agents = [agent for agent in snapshot.get("agents", []) if agent["group_id"] == group["group_id"]]
            self._draw_group(surface, panel, group, agents, snapshot)
        legend_y = rect.bottom - 69 - harvest_legend_height - trap_legend_height
        self._text(surface, "Solid: observed edges   Rings: trees   Dashes: sighting history",
                   (rect.x + 12, legend_y), MUTED, self.small_font, rect.width - 24)
        self._text(surface, "Dots: observed biome   Shading / tan borders: estimates   *: elite",
                   (rect.x + 12, legend_y + 18), MUTED, self.small_font, rect.width - 24)
        x = rect.x + 12
        for name, color in BIOME_COLORS.items():
            width = self.small_font.size(name)[0] + 24
            if x + width > rect.right:
                break
            pygame.draw.circle(surface, color, (x + 3, legend_y + 45), 3)
            self._text(surface, name, (x + 11, legend_y + 36), MUTED, self.small_font)
            x += width
        if snapshot.get("harvest", {}).get("active"):
            simple = snapshot["harvest"].get("mode") == "simple"
            self._text(surface, ("Gold squares: known fruit   Gold dashes: collection targets" if simple else
                                "Fruit squares: green growing / gold ripe   Gold dashes: harvest targets"),
                       (rect.x + 12, legend_y + 54), MUTED, self.small_font, rect.width - 24)
            self._text(surface, ("Cyan crosses: assigned homes   Cyan lines: patrols   Labels: task" if simple else
                                "Colored outlines: territories   Cyan: patrols / gaps   Labels: task"),
                       (rect.x + 12, legend_y + 72), MUTED, self.small_font, rect.width - 24)
        if trap_legend_height:
            self._text(surface, "Potential traps: amber triangles = walls   green diamonds = slots",
                       (rect.x + 12, legend_y + 54 + harvest_legend_height), MUTED, self.small_font, rect.width - 24)
            self._text(surface, "Purple rings = shelters   Grey = refresh pending   Cross = predator side",
                       (rect.x + 12, legend_y + 72 + harvest_legend_height), MUTED, self.small_font, rect.width - 24)


def draw_comparison(surface, actual_surface, snapshot, renderer, label=""):
    """Compose a pre-rendered actual world and the independent estimated view."""
    surface.fill(BACKGROUND)
    divider = surface.get_width() // 2
    renderer._text(surface, "Actual simulation", (14, 10), font=renderer.title_font)
    renderer._text(surface, label, (14, 39), MUTED, renderer.small_font, divider - 28)
    rect = pygame.Rect(12, 70, divider - 24, surface.get_height() - 84)
    zoom = min(rect.width / actual_surface.get_width(), rect.height / actual_surface.get_height())
    scaled = pygame.transform.smoothscale(actual_surface,
              (max(1, round(actual_surface.get_width() * zoom)), max(1, round(actual_surface.get_height() * zoom))))
    surface.blit(scaled, scaled.get_rect(center=rect.center))
    pygame.draw.line(surface, (53, 68, 82), (divider, 0), (divider, surface.get_height()))
    renderer.draw(surface, snapshot, (divider + 1, 0, surface.get_width() - divider - 1, surface.get_height()))
