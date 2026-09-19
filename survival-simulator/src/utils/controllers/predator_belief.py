"""Probabilistic predator positions from sightings, facing and the native AI.

No simulator objects, predator identities, hidden energy, rest state or biome
truth are accepted. A sighting is an ordinary (distance, angle, rel_dir)
observation anchored in the observer's estimated frame, so the belief inherits
that frame's error rather than pretending to world coordinates.

Between sightings a weighted particle set is advanced through a replica of
`Predator.step`. The chase/pivot branch is decided against OUR OWN estimated
poses and headings, which the policy knows far better than it knows the
predator, so the spread reflects what the native AI could have done instead of
an isotropic guess. Everything genuinely hidden -- energy, terrain multiplier,
the exact wander draw -- stays as per-particle nuisance state that is sampled,
never assumed.

Two measurements shrink the belief. A sighting reweights particles by position
and by the heading recovered from `rel_dir`. Absence within an agent's hearing
radius removes particles outright: the engine appends every creature inside
that radius before any visibility test, so "silent" there is exact rather than
merely suggestive. Absence inside the vision cone is not used by default,
because a predator behind a rock is unobserved but present.

Rock geometry conditions the prediction rather than only correcting it after
the fact. A particle sees an agent only along a clear line: the engine's ray
caster stops vision at any obstacle face, so a predator cannot chase what a
boulder hides, and the branch it takes -- not merely where it lands -- depends
on the mapped edges. Hearing is the exception the engine itself makes, being
reported before any visibility test, so it is never occluded. Where a predator
could not physically stand is taken from rectangles recovered from adjoining
faces, matching Environment._in_obstacle's filled-rectangle test, with a
clearance test carrying the faces that have not resolved into a rectangle yet.

Known approximations, all conservative in the sense that they widen the belief
or leave it unchanged rather than inventing confidence:
  * Line of sight is tested against mapped edges only. A rock nobody has
    observed yet cannot hide anything, so the belief is over-confident about
    unexplored ground rather than under-confident about mapped ground.
  * The vision test uses the cone and the segment, not the engine's exact
    visibility polygon, so a particle grazing the very edge of a face may
    disagree with the engine by one tick.
  * Rest and wake use the engine thresholds but start from a sampled energy,
    so a freshly seen predator carries the full 11-vs-15 speed ambiguity.
"""

from dataclasses import dataclass, field
import itertools
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# Engine constants. Sources: src/elements/predator.py (speeds, size, hearing,
# vision, chase gate), src/elements/creature.py (cone_angle default pi/3),
# src/elements/environment.py (energy costs, rest thresholds, collision sweep).
PREDATOR_WALK = 11.
PREDATOR_SPRINT = 15.
PREDATOR_SIZE = 10.
PREDATOR_MAX_ENERGY = 200.
PREDATOR_HEARING = 60.
PREDATOR_VISION = 250.
PREDATOR_HALF_CONE = math.pi / 6.
CHASE_HEARING_FACTOR = 1.5
TURN_CLAMP = .3
TURN_DEADZONE = .05
PIVOT_OFFSET = math.pi / 4.
EDGE_MINIMUM_DISTANCE = 2.
WANDER_TURN = .1
WALK_COST = .05
SPRINT_COST = .5
LOW_ENERGY_FRACTION = .2
WAKE_FRACTION = .5
REST_RECOVERY_PER_SECOND = 30.
COLLISION_STEP = math.pi / 18.
# Environment.non_agent_step eats an agent when the gap is under the summed
# radii. Ten for the predator plus five for an agent.
CATCH_DISTANCE = 15.
TERRAIN_MULTIPLIERS = (1., .8, .5, .3)


class PredatorBeliefConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(default=False, strict=True)
    particles: int = Field(default=96, ge=8, le=1024)
    max_tracks: int = Field(default=8, ge=1, le=64)
    # Deliberately past the reactive flee radius. A predator sighted at 180
    # units is exactly the one whose next position has to be predicted rather
    # than observed, and the old 100-unit gate discarded it before any memory.
    sighting_radius: float = Field(default=260., gt=0)
    # One engine tick moves a predator by a distance, not by a speed, so the
    # propagation counts ticks. Core's default dt is 1/10.
    tick_seconds: float = Field(default=.1, gt=0)
    max_ticks_per_update: int = Field(default=32, ge=1, le=256)
    # Sightings of one predator by several agents in a group merge first; the
    # cluster then localises it better than any single bearing.
    merge_radius: float = Field(default=18., gt=0)
    association_tolerance: float = Field(default=24., ge=0)
    measurement_uncertainty: float = Field(default=6., gt=0)
    heading_uncertainty: float = Field(default=.25, gt=0)
    max_unseen_seconds: float = Field(default=12., gt=0)
    # Past this the belief no longer says anything a uniform prior would not.
    max_spread: float = Field(default=170., gt=0)
    negative_evidence: bool = Field(default=True, strict=True)
    # Absence inside the vision cone, which is only sound because `occlusion`
    # below checks the line of sight first; a predator behind mapped rock is
    # unobserved but present, and is not removed.
    vision_negative_evidence: bool = Field(default=True, strict=True)
    negative_evidence_margin: float = Field(default=10., ge=0)
    # Vision blocked by mapped rock. Switching this off makes a particle able
    # to chase an agent through a boulder, and makes cone-based absence unsafe.
    occlusion: bool = Field(default=True, strict=True)
    # Recover filled rectangles from adjoining faces so a rock's interior is
    # forbidden, not just the neighbourhood of an observed face.
    infer_rectangles: bool = Field(default=True, strict=True)
    # Environment's own recovery sweep is 36 steps of 10 degrees.
    collision_attempts: int = Field(default=36, ge=0, le=36)
    edge_avoidance: bool = Field(default=True, strict=True)
    max_edges_considered: int = Field(default=96, ge=0, le=1024)
    position_noise: float = Field(default=.5, ge=0)
    heading_noise: float = Field(default=.03, ge=0)
    # Radius used to turn the particle cloud into "a predator is near me".
    threat_radius: float = Field(default=130., gt=0)
    threat_min_probability: float = Field(default=.1, ge=0, le=1)
    seed: int = Field(default=20260919, ge=0)


def wrap(angle):
    return (angle + math.pi) % math.tau - math.pi


def rotate(vector, angle):
    cos, sin = math.cos(angle), math.sin(angle)
    x, y = np.asarray(vector, dtype=float).reshape(-1, 2).T
    return np.stack((x * cos - y * sin, x * sin + y * cos), axis=1)


def predator_sightings(observations, radius):
    """Read predator bearings, keeping the heading when the engine supplied it.

    `rel_dir` is the bearing from the predator to the observer minus the
    predator's own heading, so it recovers that heading exactly once the
    observer's pose is known. It is optional here: a caller's fixture, or a
    future payload change, may omit it and the track then starts heading-blind.
    """
    sightings = []
    for observation in observations:
        if observation.get("type") != "Predator":
            continue
        try:
            distance = float(observation["distance"])
            angle = float(observation["angle"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(distance) and math.isfinite(angle)) or distance > radius:
            continue
        relative = observation.get("rel_dir")
        try:
            relative = None if relative is None else float(relative)
        except (TypeError, ValueError):
            relative = None
        if relative is not None and not math.isfinite(relative):
            relative = None
        sightings.append((distance, angle, relative))
    return sorted(sightings, key=lambda row: (row[0], row[1]))


def nearest_edge_offsets(points, edges):
    """Offset from each point to its closest point on the closest segment.

    Mirrors the per-edge closest-point search in Predator.step, which works in
    the predator frame and then keeps the globally nearest result.
    """
    starts, vectors = edges[:, 0], edges[:, 1] - edges[:, 0]
    lengths = np.maximum(np.sum(vectors * vectors, axis=1), 1e-12)
    delta = points[:, None, :] - starts[None, :, :]
    projection = np.clip(np.sum(delta * vectors[None, :, :], axis=2) / lengths, 0., 1.)
    closest = starts[None, :, :] + projection[..., None] * vectors[None, :, :]
    offsets = closest - points[:, None, :]
    index = np.argmin(np.hypot(offsets[..., 0], offsets[..., 1]), axis=1)
    return np.take_along_axis(offsets, index[:, None, None], 1)[:, 0, :]


def blocked(points, edges, radius):
    """True where a circle of `radius` about the point touches a mapped edge."""
    if not len(edges) or not len(points):
        return np.zeros(len(points), dtype=bool)
    starts, vectors = edges[:, 0], edges[:, 1] - edges[:, 0]
    lengths = np.maximum(np.sum(vectors * vectors, axis=1), 1e-12)
    delta = points[:, None, :] - starts[None, :, :]
    projection = np.clip(np.sum(delta * vectors[None, :, :], axis=2) / lengths, 0., 1.)
    closest = starts[None, :, :] + projection[..., None] * vectors[None, :, :]
    return np.any(np.hypot(*(closest - points[:, None, :]).transpose(2, 0, 1)) < radius - 1e-6, axis=1)


def observed_rectangles(edges, tolerance=.5, maximum_side=200.):
    """Recover rectangles only from adjoining complete horizontal/vertical faces.

    Parallel faces alone are ambiguous: they might belong to different rocks.
    A rectangle gives the filled region Environment._in_obstacle tests, so the
    belief can forbid a rock's interior and not merely the strip beside a face.
    No assumption about an unseen opposite face comes from world truth.
    """
    horizontal, vertical = [], []
    for pair in np.asarray(edges, dtype=float).reshape((-1, 2, 2)):
        if not np.isfinite(pair).all():
            continue
        low, high = pair.min(axis=0), pair.max(axis=0)
        dx, dy = high - low
        if 2 * tolerance < dx < maximum_side and dy <= tolerance:
            horizontal.append((float(low[0]), float(high[0]), float(pair[:, 1].mean())))
        elif 2 * tolerance < dy < maximum_side and dx <= tolerance:
            vertical.append((float(low[1]), float(high[1]), float(pair[:, 0].mean())))
    if not horizontal or not vertical:
        return []
    horizontal = np.array(sorted(set(horizontal)))
    vertical = np.array(sorted(set(vertical)))
    candidates = []
    for x0, x1, y in horizontal:
        adjacent = ((np.minimum(abs(vertical[:, 2] - x0), abs(vertical[:, 2] - x1)) <= tolerance)
                    & (np.minimum(abs(vertical[:, 0] - y), abs(vertical[:, 1] - y)) <= tolerance))
        for y0, y1, x in vertical[adjacent]:
            h_spans = np.maximum(abs(horizontal[:, 0] - x0), abs(horizontal[:, 1] - x1)) <= tolerance
            v_spans = np.maximum(abs(vertical[:, 0] - y0), abs(vertical[:, 1] - y1)) <= tolerance
            faces = [bool(np.any(h_spans & (abs(horizontal[:, 2] - side) <= tolerance))) for side in (y0, y1)]
            faces += [bool(np.any(v_spans & (abs(vertical[:, 2] - side) <= tolerance))) for side in (x0, x1)]
            bounds = tuple(round(float(value), 5) for value in (x0, y0, x1, y1))
            candidates.append((sum(faces), bounds))
    # A rectangle seen from several corners is one obstacle, even when small
    # independent pose errors shift the reconstructed copies slightly.
    result, retained = [], []
    for faces, bounds in sorted(set(candidates), key=lambda row: (-row[0], row[1])):
        if any(max(abs(a - b) for a, b in zip(bounds, other)) <= tolerance for other in retained):
            continue
        retained.append(bounds)
        x0, y0, x1, y1 = bounds
        result.append(dict(bounds=list(bounds), observed_faces=faces,
                           rectangle=[x0, y0, x1 - x0, y1 - y0]))
    return sorted(result, key=lambda item: item["bounds"])


def segments_clear(starts, ends, edges):
    """True where the straight segment start->end crosses no mapped edge.

    The engine's ray caster stops vision at an obstacle face, so this is the
    test that decides whether a particle can see an agent at all. Endpoints are
    excluded from the crossing, because a segment that merely terminates on a
    face is not blocked by it.
    """
    if not len(edges) or not len(starts):
        return np.ones(len(starts), dtype=bool)
    origin = np.asarray(starts, dtype=float)[:, None, :]
    ray = (np.asarray(ends, dtype=float) - np.asarray(starts, dtype=float))[:, None, :]
    face = edges[None, :, 0, :]
    span = (edges[:, 1] - edges[:, 0])[None, :, :]
    denominator = ray[..., 0] * span[..., 1] - ray[..., 1] * span[..., 0]
    offset = face - origin
    parallel = np.abs(denominator) < 1e-12
    safe = np.where(parallel, 1., denominator)
    along = (offset[..., 0] * span[..., 1] - offset[..., 1] * span[..., 0]) / safe
    across = (offset[..., 0] * ray[..., 1] - offset[..., 1] * ray[..., 0]) / safe
    crossing = (~parallel & (along > 1e-9) & (along < 1. - 1e-9)
                & (across > 1e-9) & (across < 1. - 1e-9))
    return ~np.any(crossing, axis=1)


@dataclass
class Obstacles:
    """Mapped rock in one group's frame: faces plus any recovered rectangles."""

    edges: np.ndarray
    rectangles: np.ndarray

    @property
    def known(self):
        return bool(len(self.edges)) or bool(len(self.rectangles))


EMPTY_OBSTACLES = Obstacles(np.empty((0, 2, 2)), np.empty((0, 4)))


def occupied(points, obstacles, radius):
    """True where a creature of `radius` could not stand.

    Environment._in_obstacle tests a *filled* rectangle inflated by the radius,
    so a rock's interior is forbidden and not merely the strip beside a face.
    Recovered rectangles give that exactly; the clearance test covers faces
    that have not resolved into a rectangle yet.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points) or obstacles is None or not obstacles.known:
        return np.zeros(len(points), dtype=bool)
    result = blocked(points, obstacles.edges, radius)
    if len(obstacles.rectangles):
        low = obstacles.rectangles[None, :, :2] - radius
        high = obstacles.rectangles[None, :, 2:] + radius
        inside = np.all((points[:, None, :] > low + 1e-6) & (points[:, None, :] < high - 1e-6), axis=2)
        result = result | np.any(inside, axis=1)
    return result


@dataclass
class PredatorTrack:
    """A weighted particle set for one predator, in one group's frame."""

    track_id: int
    group_id: int
    positions: np.ndarray
    headings: np.ndarray
    energies: np.ndarray
    terrain: np.ndarray
    resting: np.ndarray
    weights: np.ndarray
    first_seen: float
    last_seen: float
    sightings: int = 1
    heading_known: bool = True
    observers: set = field(default_factory=set)

    @property
    def mean(self):
        return np.sum(self.positions * self.weights[:, None], axis=0)

    @property
    def spread(self):
        """Weighted RMS distance from the mean: one number for "how vague"."""
        offsets = self.positions - self.mean
        return float(math.sqrt(max(0., np.sum(self.weights * np.sum(offsets * offsets, axis=1)))))

    def probability_within(self, point, radius):
        offsets = self.positions - np.asarray(point, dtype=float)
        return float(np.sum(self.weights[np.hypot(offsets[:, 0], offsets[:, 1]) <= radius]))

    def snapshot(self, now=None, particle_limit=128):
        """JSON-compatible summary. `particles` is subsampled for drawing.

        `weights` is kept alongside, normalised over the subsample, so a
        density rendering can weight each particle instead of assuming the
        cloud is uniform between resamples.
        """
        mean = self.mean
        step = max(1, math.ceil(len(self.positions) / max(1, particle_limit)))
        kept = self.weights[::step]
        total = float(kept.sum())
        return dict(track_id=self.track_id, group_id=self.group_id,
                    mean=[float(mean[0]), float(mean[1])], spread=round(self.spread, 3),
                    particles=[[round(float(x), 2), round(float(y), 2)]
                               for x, y in self.positions[::step]],
                    weights=[round(float(w) / total, 6) for w in kept] if total > 0 else
                            [round(1. / max(1, len(kept)), 6)] * len(kept),
                    seconds_unseen=(None if now is None else
                                    round(max(0., float(now) - self.last_seen), 3)),
                    heading=(None if not self.heading_known else
                             round(float(math.atan2(np.sum(self.weights * np.sin(self.headings)),
                                                    np.sum(self.weights * np.cos(self.headings)))), 4)),
                    resting_probability=round(float(np.sum(self.weights[self.resting])), 4),
                    first_seen=self.first_seen, last_seen=self.last_seen,
                    sightings=self.sightings, observers=sorted(self.observers))


@dataclass(frozen=True)
class PredatorThreat:
    """What one agent should do about the belief, in its own facing frame."""

    probability: float
    distance: float
    angle: float
    escape_direction: float
    spread: float
    seconds_unseen: float
    track_id: int


class PredatorTracker:
    """Maintain one particle set per inferred predator across a population."""

    def __init__(self, config: PredatorBeliefConfig | None = None,
                 movement_factors: dict | None = None, unknown_movement_factor: float = 1.,
                 cell_size: float = 40.):
        self.config = config if config is not None else PredatorBeliefConfig()
        self.movement_factors = dict(movement_factors or {})
        self.unknown_movement_factor = float(unknown_movement_factor)
        self.cell_size = float(cell_size)
        self.reset()

    def reset(self):
        self.tracks: dict[int, PredatorTrack] = {}
        self.threats: dict[int, PredatorThreat] = {}
        self.last_time: float | None = None
        self._rng = np.random.default_rng(self.config.seed)
        self._ids = itertools.count(1)
        self._obstacles: dict[int, tuple] = {}

    def _rock(self, group_id, group, edges):
        """Cache the group's rock geometry until its observed faces change.

        Recovering rectangles walks the face list, so it must not run per tick.
        """
        key = (len(edges), getattr(group, "_edge_revision", 0),
               None if group is None else len(group.edges))
        cached = self._obstacles.get(group_id)
        if cached is not None and cached[0] == key:
            return cached[1]
        rectangles = np.empty((0, 4))
        if self.config.infer_rectangles and len(edges) >= 2:
            bounds = [inferred["rectangle"] for inferred in observed_rectangles(edges)]
            if bounds:
                boxes = np.asarray(bounds, dtype=float)
                rectangles = np.column_stack((boxes[:, :2], boxes[:, :2] + boxes[:, 2:]))
        obstacles = Obstacles(np.asarray(edges, dtype=float).reshape(-1, 2, 2), rectangles)
        self._obstacles[group_id] = (key, obstacles)
        return obstacles

    # -- particle initialisation -------------------------------------------

    def _sample_energy(self, count):
        """A predator wakes above half its maximum and then only spends.

        Nothing in an observation reveals the level, so the prior spans the
        active range. The 40-unit threshold below which the engine caps a
        sprint at walking speed therefore carries real probability mass.
        """
        return self._rng.uniform(1., PREDATOR_MAX_ENERGY, size=count)

    def _sample_terrain(self, positions, group):
        """Prefer a mapped biome under the particle; sample when unmapped."""
        factors = np.full(len(positions), np.nan)
        if group is not None and group.biomes and self.movement_factors:
            cells = np.floor(positions / self.cell_size).astype(int)
            for index, cell in enumerate(map(tuple, cells)):
                sample = group.biomes.get(cell)
                if sample is not None:
                    factors[index] = self.movement_factors.get(
                        sample.biome, self.unknown_movement_factor)
        unknown = np.isnan(factors)
        if unknown.any():
            choices = (sorted(set(self.movement_factors.values())) if self.movement_factors
                       else list(TERRAIN_MULTIPLIERS))
            factors[unknown] = self._rng.choice(np.asarray(choices, dtype=float),
                                                size=int(unknown.sum()))
        return factors

    def _new_track(self, group_id, position, heading, now, group, observers, obstacles=None):
        count = self.config.particles
        spread = self.config.measurement_uncertainty
        centre = np.asarray(position, dtype=float)
        positions = centre + self._rng.normal(0., spread, size=(count, 2))
        if obstacles is not None and obstacles.known:
            # Measurement error can scatter a particle into rock, where no
            # predator can stand. Redraw those rather than start the belief
            # somewhere the engine would never allow.
            for _ in range(4):
                trapped = occupied(positions, obstacles, PREDATOR_SIZE)
                if not trapped.any():
                    break
                positions[trapped] = centre + self._rng.normal(
                    0., spread, size=(int(trapped.sum()), 2))
        if heading is None:
            headings = self._rng.uniform(-math.pi, math.pi, size=count)
        else:
            headings = wrap(heading + self._rng.normal(0., self.config.heading_uncertainty, size=count))
        return PredatorTrack(
            track_id=next(self._ids), group_id=group_id, positions=positions, headings=headings,
            energies=self._sample_energy(count), terrain=self._sample_terrain(positions, group),
            resting=np.zeros(count, dtype=bool), weights=np.full(count, 1. / count),
            first_seen=now, last_seen=now, heading_known=heading is not None,
            observers=set(observers),
        )

    # -- prediction ---------------------------------------------------------

    def _advance(self, track, agents, headings, obstacles, group):
        """One engine tick of Predator.step applied to every particle."""
        config = self.config
        obstacles = obstacles or EMPTY_OBSTACLES
        edges = obstacles.edges
        positions, pose_headings = track.positions, track.headings
        count = len(positions)
        awake = ~track.resting

        request = np.zeros(count)
        direction = np.zeros(count)
        turn = np.zeros(count)
        acting = np.zeros(count, dtype=bool)

        target_distance = np.full(count, np.inf)
        sensed_any = np.zeros(count, dtype=bool)
        if len(agents):
            delta = agents[None, :, :] - positions[:, None, :]
            distances = np.hypot(delta[..., 0], delta[..., 1])
            bearings = np.arctan2(delta[..., 1], delta[..., 0])
            relative = wrap(bearings - pose_headings[:, None])
            # Creature.observe reports everything inside the hearing radius
            # before any visibility test, so hearing is never occluded. Sight
            # is the cone out to the vision range, and it stops at rock.
            heard = distances <= PREDATOR_HEARING
            seen = (~heard & (distances <= PREDATOR_VISION)
                    & (np.abs(relative) <= PREDATOR_HALF_CONE))
            if config.occlusion and len(edges) and seen.any():
                # Only the candidates that already passed range and cone need
                # the line test, which keeps it a small array rather than one
                # entry per particle-agent pair.
                rows, columns = np.nonzero(seen)
                seen[rows, columns] = segments_clear(positions[rows], agents[columns], edges)
            sensed = heard | seen
            gated = np.where(sensed, distances, np.inf)
            nearest = np.argmin(gated, axis=1)
            rows = np.arange(count)
            target_distance = gated[rows, nearest]
            sensed_any = np.isfinite(target_distance)
            target_angle = relative[rows, nearest]
            # The gate is the agent's gaze, which we know exactly: rel_dir of
            # the predator as the *agent* would report it.
            looking = wrap(np.arctan2(-delta[rows, nearest, 1], -delta[rows, nearest, 0])
                           - headings[nearest])

            chase = sensed_any & ((np.abs(looking) > math.pi / 2)
                                  | (target_distance < PREDATOR_HEARING * CHASE_HEARING_FACTOR))
            pivot = sensed_any & ~chase

            strength = np.clip(target_angle * .5, -TURN_CLAMP, TURN_CLAMP)
            outside = np.abs(target_angle) > TURN_DEADZONE
            # Inside the dead zone the engine emits no turn signal at all and
            # steers by the raw bearing instead of the clamped strength.
            request = np.where(chase, np.minimum(PREDATOR_SPRINT, target_distance), request)
            direction = np.where(chase, np.where(outside, strength, target_angle), direction)
            turn = np.where(chase & outside, strength, turn)

            sign = -np.sign(looking)
            pivot_direction = target_angle + sign * PIVOT_OFFSET
            # The engine predicts its own pivot with a full 15 units even when
            # energy or terrain later shortens the actual translation.
            x = target_distance * np.cos(target_angle) - PREDATOR_SPRINT * np.cos(pivot_direction)
            y = target_distance * np.sin(target_angle) - PREDATOR_SPRINT * np.sin(pivot_direction)
            request = np.where(pivot, PREDATOR_SPRINT, request)
            direction = np.where(pivot, pivot_direction, direction)
            turn = np.where(pivot, np.arctan2(y, x), turn)
            acting = sensed_any.copy()

        idle = awake & ~sensed_any
        if idle.any():
            avoided = np.zeros(count, dtype=bool)
            if config.edge_avoidance and len(edges):
                offsets = nearest_edge_offsets(positions[idle], edges)
                gaps = np.hypot(offsets[:, 0], offsets[:, 1])
                local = wrap(np.arctan2(offsets[:, 1], offsets[:, 0]) - pose_headings[idle])
                # The ray caster only returns faces inside the cone, so a wall
                # behind the predator does not steer it, and a face hidden
                # behind a nearer rock is never returned at all.
                visible = (gaps <= PREDATOR_VISION) & (np.abs(local) <= PREDATOR_HALF_CONE)
                if config.occlusion and visible.any():
                    candidates = np.flatnonzero(visible)
                    targets = positions[idle][candidates] + offsets[candidates]
                    visible[candidates] = segments_clear(
                        positions[idle][candidates], targets, edges)
                effective = np.maximum(gaps - PREDATOR_SIZE, EDGE_MINIMUM_DISTANCE)
                steer = np.where(local > 0, -math.pi / effective, math.pi / effective)
                rows = np.flatnonzero(idle)[visible]
                request[rows] = PREDATOR_WALK
                direction[rows] = steer[visible]
                turn[rows] = steer[visible]
                acting[rows] = True
                avoided[rows] = True
            wander = idle & ~avoided
            if wander.any():
                # turn(uniform(-.1, .1)) then move(speed, None): a null
                # direction travels along the heading the tick started with.
                request[wander] = PREDATOR_WALK
                direction[wander] = 0.
                turn[wander] = self._rng.uniform(-WANDER_TURN, WANDER_TURN, size=int(wander.sum()))
                acting[wander] = True

        acting &= awake
        # update_entity_position: the sprint cap collapses to walking speed
        # below a fifth of maximum energy, and energy is charged before the
        # biome multiplier shortens the step.
        capped = np.where(track.energies < PREDATOR_MAX_ENERGY * LOW_ENERGY_FRACTION,
                          np.minimum(request, PREDATOR_WALK), np.minimum(request, PREDATOR_SPRINT))
        capped = np.where(acting, np.maximum(capped, 0.), 0.)
        cost = np.where(capped <= PREDATOR_WALK, capped * WALK_COST,
                        PREDATOR_WALK * WALK_COST + (capped - PREDATOR_WALK) * SPRINT_COST)
        cost += np.where(acting, np.minimum(math.pi, np.abs(turn)) / math.tau, 0.)

        travel = capped * track.terrain
        absolute = pose_headings + direction
        moved = positions + np.stack((travel * np.cos(absolute), travel * np.sin(absolute)), axis=1)
        if obstacles.known and config.collision_attempts:
            stuck = np.flatnonzero(occupied(moved, obstacles, PREDATOR_SIZE) & (travel > 0))
            if len(stuck):
                # Environment tests 0, -10, +10, -20, ... degrees and keeps the
                # original point when every candidate is blocked. All the
                # candidates are tested in one pass rather than one at a time.
                sweep = np.array([COLLISION_STEP * ((attempt + 1) // 2) * (-1.) ** attempt
                                  for attempt in range(config.collision_attempts)])
                adjusted = absolute[stuck][:, None] + sweep[None, :]
                candidates = (positions[stuck][:, None, :] + travel[stuck][:, None, None]
                              * np.stack((np.cos(adjusted), np.sin(adjusted)), axis=2))
                free = ~occupied(candidates.reshape(-1, 2), obstacles, PREDATOR_SIZE)
                free = free.reshape(len(stuck), -1)
                first = np.argmax(free, axis=1)
                rows = np.arange(len(stuck))
                resolved = candidates[rows, first]
                moved[stuck] = np.where(free[rows, first][:, None], resolved, positions[stuck])

        track.positions = moved + self._rng.normal(0., config.position_noise, size=moved.shape)
        track.headings = wrap(np.where(acting, pose_headings + turn, pose_headings)
                              + self._rng.normal(0., config.heading_noise, size=count))
        energies = np.where(acting, track.energies - cost, track.energies)
        # Resting regenerates 30 a second and wakes above half of maximum.
        resting = track.resting | (energies <= 0.)
        energies = np.where(resting, energies + REST_RECOVERY_PER_SECOND * config.tick_seconds,
                            energies)
        track.resting = resting & (energies <= PREDATOR_MAX_ENERGY * WAKE_FRACTION)
        track.energies = np.clip(energies, -PREDATOR_MAX_ENERGY, PREDATOR_MAX_ENERGY)
        track.terrain = self._sample_terrain(track.positions, group)

    # -- measurement --------------------------------------------------------

    def _reweight(self, track, position, heading, sightline=None, obstacles=None):
        """Gaussian position likelihood, plus heading when rel_dir supplied.

        Rock rules candidates out rather than merely shading them. A predator
        cannot be standing inside a mapped rock, and one reported by sight
        rather than by hearing must have had a clear line to that observer.
        """
        config = self.config
        offsets = track.positions - np.asarray(position, dtype=float)
        squared = np.sum(offsets * offsets, axis=1)
        likelihood = np.exp(-.5 * squared / config.measurement_uncertainty ** 2)
        if heading is not None:
            error = wrap(track.headings - heading)
            likelihood = likelihood * np.exp(-.5 * (error / config.heading_uncertainty) ** 2)
        if obstacles is not None and obstacles.known:
            likelihood = np.where(occupied(track.positions, obstacles, PREDATOR_SIZE),
                                  0., likelihood)
            if config.occlusion and sightline is not None and len(obstacles.edges):
                observer = np.repeat(np.asarray(sightline, dtype=float)[None, :],
                                     len(track.positions), axis=0)
                likelihood = np.where(
                    segments_clear(observer, track.positions, obstacles.edges), likelihood, 0.)
        weights = track.weights * likelihood
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 1e-12:
            # The prediction missed entirely. Trust the measurement, not a
            # rescued cloud that no longer covers the predator.
            return False
        track.weights = weights / total
        self._resample(track)
        return True

    def _resample(self, track):
        """Systematic resampling once the effective sample size halves."""
        count = len(track.weights)
        effective = 1. / float(np.sum(track.weights ** 2))
        if effective >= count / 2.:
            return
        offset = float(self._rng.uniform(0., 1. / count))
        cursor = (offset + np.arange(count) / count)
        picks = np.searchsorted(np.cumsum(track.weights), cursor, side="right")
        picks = np.clip(picks, 0, count - 1)
        track.positions = track.positions[picks] + self._rng.normal(
            0., self.config.position_noise, size=(count, 2))
        track.headings = track.headings[picks]
        track.energies = track.energies[picks]
        track.terrain = track.terrain[picks]
        track.resting = track.resting[picks]
        track.weights = np.full(count, 1. / count)

    def _apply_negative_evidence(self, group_id, envelopes, matched_positions, obstacles=None):
        """Remove particles an agent would have sensed but did not report.

        Hearing is not occluded in the engine, so silence inside that radius is
        a hard measurement. Silence inside the vision cone is only a
        measurement where the line of sight is clear: a predator behind mapped
        rock is unobserved but present, and has to survive. A particle near a
        position that *was* reported is spared, otherwise a live sighting would
        erase its own track.
        """
        config = self.config
        for track in list(self.tracks.values()):
            if track.group_id != group_id:
                continue
            survives = np.ones(len(track.positions), dtype=bool)
            for centre, heading, hearing, cone, span in envelopes:
                offsets = track.positions - centre
                distances = np.hypot(offsets[:, 0], offsets[:, 1])
                silent = distances <= max(0., hearing - config.negative_evidence_margin)
                if config.vision_negative_evidence and cone > 0 and span > 0:
                    bearing = wrap(np.arctan2(offsets[:, 1], offsets[:, 0]) - heading)
                    watched = ((distances <= max(0., cone - config.negative_evidence_margin))
                               & (np.abs(bearing) <= span / 2.))
                    if config.occlusion and obstacles is not None and len(obstacles.edges) and watched.any():
                        rows = np.flatnonzero(watched)
                        eye = np.repeat(np.asarray(centre, dtype=float)[None, :], len(rows), axis=0)
                        watched[rows] = segments_clear(eye, track.positions[rows], obstacles.edges)
                    silent |= watched
                survives &= ~silent
            for position in matched_positions:
                offsets = track.positions - position
                survives |= np.hypot(offsets[:, 0], offsets[:, 1]) <= config.merge_radius
            if survives.all():
                continue
            weights = np.where(survives, track.weights, 0.)
            total = float(weights.sum())
            if total <= 1e-9:
                # Every hypothesis was ruled out; the predator is simply gone.
                self.tracks.pop(track.track_id, None)
                continue
            track.weights = weights / total
            self._resample(track)

    # -- association --------------------------------------------------------

    def _cluster(self, measurements):
        """Merge sightings of one predator seen by several agents at once.

        A member's sightline is the observer's position when that sighting came
        from vision rather than hearing, which is the case where a clear line
        must have existed. Hearing carries no such constraint, so it stays None.
        """
        clusters = []
        for measurement in measurements:
            position = measurement[0]
            for cluster in clusters:
                if math.dist(cluster["position"], position) <= self.config.merge_radius:
                    cluster["members"].append(measurement)
                    stack = np.array([row[0] for row in cluster["members"]], dtype=float)
                    cluster["position"] = tuple(stack.mean(axis=0))
                    break
            else:
                clusters.append(dict(position=tuple(position), members=[measurement]))
        merged = []
        for cluster in clusters:
            headings = [row[1] for row in cluster["members"] if row[1] is not None]
            heading = (None if not headings else
                       math.atan2(float(np.mean(np.sin(headings))), float(np.mean(np.cos(headings)))))
            sightline = next((row[3] for row in cluster["members"] if row[3] is not None), None)
            merged.append((np.asarray(cluster["position"], dtype=float), heading,
                           {row[2] for row in cluster["members"]}, sightline))
        return merged

    def _associate(self, group_id, clusters, ticks, now, group, obstacles=None):
        """Greedy nearest-mean matching inside a reachability gate.

        Predator observations carry no identity, so association is geometric.
        The gate is what one predator could physically have covered since the
        track was last corrected, widened by the measurement error.
        """
        config = self.config
        candidates = [track for track in self.tracks.values() if track.group_id == group_id]
        reach = PREDATOR_SPRINT * max(1, ticks) + config.association_tolerance
        pairs = sorted(
            ((math.dist(track.mean, cluster[0]), index, track.track_id)
             for index, cluster in enumerate(clusters) for track in candidates),
            key=lambda row: (row[0], row[1], row[2]))
        taken_clusters, taken_tracks, matched = set(), set(), []
        for distance, index, track_id in pairs:
            if index in taken_clusters or track_id in taken_tracks or distance > reach:
                continue
            taken_clusters.add(index)
            taken_tracks.add(track_id)
            matched.append((track_id, index))
        for track_id, index in matched:
            position, heading, observers, sightline = clusters[index]
            track = self.tracks[track_id]
            if not self._reweight(track, position, heading, sightline, obstacles):
                replacement = self._new_track(group_id, position, heading, now, group,
                                              observers, obstacles)
                replacement.track_id = track.track_id
                replacement.first_seen = track.first_seen
                replacement.sightings = track.sightings
                replacement.observers = track.observers | observers
                self.tracks[track_id] = replacement
                track = replacement
            track.last_seen = now
            track.sightings += 1
            track.observers |= observers
            if heading is not None:
                track.heading_known = True
        for index, cluster in enumerate(clusters):
            if index in taken_clusters:
                continue
            position, heading, observers, _ = cluster
            track = self._new_track(group_id, position, heading, now, group, observers, obstacles)
            self.tracks[track.track_id] = track

    # -- public API ---------------------------------------------------------

    def update(self, groups, poses, agent_states, now):
        """Advance, measure and prune the belief for one simulation tick.

        `groups` and `poses` are the WorldEstimator's own containers; nothing
        is read from them except mapped edges, biome samples and agent poses.
        """
        config = self.config
        if not config.enabled:
            return
        if not agent_states or not poses:
            self.tracks.clear()
            self.threats.clear()
            self.last_time = None if not agent_states else now
            return
        if self.last_time is not None and now <= self.last_time:
            # Retries must not integrate motion or count a sighting twice.
            if now < self.last_time:
                self.reset()
                self.last_time = now
            return
        elapsed = config.tick_seconds if self.last_time is None else now - self.last_time
        ticks = min(config.max_ticks_per_update,
                    max(1, int(round(elapsed / config.tick_seconds))))
        states = {state["agent_id"]: state for state in agent_states}

        members: dict[int, list] = {}
        for pose in poses.values():
            if pose.agent_id in states:
                members.setdefault(pose.group_id, []).append(pose)

        rock, sensors = {}, {}
        for group_id, group_poses in members.items():
            group = groups.get(group_id) if groups else None
            edges = np.empty((0, 2, 2))
            if group is not None and group.edges and config.max_edges_considered:
                edges = np.asarray(group.edge_positions(), dtype=float)
                if len(edges) > config.max_edges_considered:
                    # Keep the faces nearest whatever the prediction will touch:
                    # the agents and the beliefs themselves, since a track that
                    # has drifted away from the group must still be predicted
                    # against the rocks around *it*.
                    anchors = [pose.position for pose in group_poses]
                    anchors += [track.mean for track in self.tracks.values()
                                if track.group_id == group_id]
                    anchors = np.asarray(anchors, dtype=float).reshape(-1, 2)
                    midpoints = edges.mean(axis=1)
                    gaps = np.min(np.hypot(*(midpoints[:, None, :] - anchors[None, :, :]
                                             ).transpose(2, 0, 1)), axis=1)
                    edges = edges[np.argsort(gaps)[:config.max_edges_considered]]
            rock[group_id] = self._rock(group_id, group, edges)
            sensors[group_id] = (
                np.array([pose.position for pose in group_poses], dtype=float).reshape(-1, 2),
                np.array([pose.heading for pose in group_poses], dtype=float).reshape(-1),
            )
        self._obstacles = {key: value for key, value in self._obstacles.items() if key in rock}

        for _ in range(ticks):
            for track in self.tracks.values():
                positions, headings = sensors.get(track.group_id, (np.empty((0, 2)), np.empty(0)))
                self._advance(track, positions, headings, rock.get(track.group_id),
                              groups.get(track.group_id) if groups else None)

        for group_id, group_poses in members.items():
            group = groups.get(group_id) if groups else None
            measurements, envelopes = [], []
            for pose in group_poses:
                state = states[pose.agent_id]
                observations = state.get("observations") or ()
                hearing = float(state.get("hearing_radius") or 0.)
                sightings = predator_sightings(observations, config.sighting_radius)
                for distance, angle, relative in sightings:
                    offset = rotate((distance * math.cos(angle), distance * math.sin(angle)),
                                    pose.heading)[0]
                    position = pose.position + offset
                    heading = (None if relative is None else
                               wrap(math.atan2(-offset[1], -offset[0]) - relative))
                    # Beyond the hearing radius the engine only reports what the
                    # ray caster could see, so a clear line existed. Within it,
                    # the report says nothing about rock in between.
                    sightline = pose.position.copy() if distance > hearing else None
                    measurements.append((position, heading, pose.agent_id, sightline))
                if config.negative_evidence and observations:
                    envelopes.append((
                        pose.position, pose.heading,
                        float(state.get("hearing_radius") or 0.),
                        float(state.get("vision_range") or 0.),
                        float(state.get("vision_angle") or 0.),
                    ))
            clusters = self._cluster(measurements)
            obstacles = rock.get(group_id)
            if envelopes:
                self._apply_negative_evidence(
                    group_id, envelopes, [cluster[0] for cluster in clusters], obstacles)
            if clusters:
                self._associate(group_id, clusters, ticks, now, group, obstacles)

        self._prune(now, set(members))
        self._refresh_threats(members, states, now)
        self.last_time = now

    def _prune(self, now, live_groups):
        config = self.config
        for track_id, track in list(self.tracks.items()):
            if (track.group_id not in live_groups
                    or now - track.last_seen > config.max_unseen_seconds
                    or track.spread > config.max_spread
                    or not np.isfinite(track.positions).all()):
                self.tracks.pop(track_id)
        if len(self.tracks) <= config.max_tracks:
            return
        # Keep the freshest, then the best supported: a stale diffuse cloud is
        # the first thing worth forgetting.
        ordered = sorted(self.tracks.values(),
                         key=lambda track: (-track.last_seen, track.spread, -track.sightings))
        self.tracks = {track.track_id: track for track in ordered[:config.max_tracks]}

    def _refresh_threats(self, members, states, now):
        config = self.config
        self.threats = {}
        for group_id, group_poses in members.items():
            tracks = [track for track in self.tracks.values() if track.group_id == group_id]
            if not tracks:
                continue
            for pose in group_poses:
                best = None
                for track in tracks:
                    probability = track.probability_within(pose.position, config.threat_radius)
                    if probability < config.threat_min_probability:
                        continue
                    if best is None or probability > best[0]:
                        best = (probability, track)
                if best is None:
                    continue
                probability, track = best
                offset = track.mean - pose.position
                bearing = math.atan2(offset[1], offset[0])
                self.threats[pose.agent_id] = PredatorThreat(
                    probability=round(float(probability), 6),
                    distance=float(math.hypot(*offset)),
                    angle=wrap(bearing - pose.heading),
                    escape_direction=wrap(bearing + math.pi - pose.heading),
                    spread=track.spread,
                    seconds_unseen=float(max(0., now - track.last_seen)),
                    track_id=track.track_id,
                )

    def danger(self, group_id, points, radius=None):
        """Probability that some predator sits within `radius` of each point."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        radius = self.config.threat_radius if radius is None else float(radius)
        miss = np.ones(len(points))
        for track in self.tracks.values():
            if track.group_id != group_id:
                continue
            offsets = points[:, None, :] - track.positions[None, :, :]
            near = np.hypot(offsets[..., 0], offsets[..., 1]) <= radius
            miss *= 1. - np.clip(near @ track.weights, 0., 1.)
        return 1. - miss

    def snapshot(self, group_id=None):
        """JSON-compatible estimated state for diagnostics; never ground truth."""
        return dict(
            enabled=self.config.enabled,
            sim_time=self.last_time,
            tracks=[track.snapshot(self.last_time) for track in sorted(
                self.tracks.values(), key=lambda track: track.track_id)
                if group_id is None or track.group_id == group_id],
            threats={agent_id: dict(
                probability=threat.probability, distance=round(threat.distance, 3),
                angle=round(threat.angle, 4), escape_direction=round(threat.escape_direction, 4),
                spread=round(threat.spread, 3), seconds_unseen=round(threat.seconds_unseen, 3),
                track_id=threat.track_id)
                for agent_id, threat in sorted(self.threats.items())},
        )
