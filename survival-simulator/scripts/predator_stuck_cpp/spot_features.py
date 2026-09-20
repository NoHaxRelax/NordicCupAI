"""Geometry and trajectory measurements for archived predator confinement cases.

This module reads data only: it does not import the simulation extension.
"""
import gzip
import json
import math

import numpy as np

from diagnostics import DETAIL_COLUMNS, INDEX as IX

FEATURE_VERSION = 1
BIOMES = ('forest', 'swamp', 'desert', 'grassland', 'river')
PENALTIES = (1., .5, .8, 1., .3)


def expanded_rectangles(obstacles, radius):
    o = np.asarray(obstacles, dtype=float)
    return np.column_stack((o[:, 0]-radius, o[:, 1]-radius,
                            o[:, 0]+o[:, 2]+radius, o[:, 1]+o[:, 3]+radius))


def point_clearances(x, y, rects):
    """Negative means strictly inside a square-expanded collision rectangle."""
    return np.maximum.reduce((rects[:, 0]-x, x-rects[:, 2], rects[:, 1]-y, y-rects[:, 3]))


def endpoint_clearances(x, y, step, angles, rects):
    """Minimum signed clearance over rectangles, for each attempted endpoint."""
    px = x + step*np.cos(angles)
    py = y + step*np.sin(angles)
    if not len(rects):
        return np.full(len(angles), np.inf)
    return np.minimum.reduce([
        np.maximum.reduce((r[0]-px, px-r[2], r[1]-py, py-r[3])) for r in rects])


def geometry_features(x, y, heading, step, requested_angle, obstacles, radius=10.):
    rects = expanded_rectangles(obstacles, radius)
    clearance = point_clearances(x, y, rects)
    order = np.argsort(clearance, kind='stable')
    nearest = int(order[0])
    near = clearance <= max(15., step)+1e-8
    local = rects[near]
    # Distances to faces whose perpendicular span contains the center. These
    # labels describe nearby faces, not a topological claim about a "corner".
    faces = []
    for distance, span in ((x-rects[:, 2], (rects[:, 1] <= y)&(y <= rects[:, 3])),
                           (rects[:, 0]-x, (rects[:, 1] <= y)&(y <= rects[:, 3])),
                           (y-rects[:, 3], (rects[:, 0] <= x)&(x <= rects[:, 2])),
                           (rects[:, 1]-y, (rects[:, 0] <= x)&(x <= rects[:, 2]))):
        valid = distance[(distance >= 0)&span]
        faces.append(float(valid.min()) if len(valid) else 1e6)
    left, right, above, below = faces
    corners = np.concatenate((rects[:, [0, 1]], rects[:, [0, 3]],
                              rects[:, [2, 1]], rects[:, [2, 3]]))
    corner_distance = float(np.hypot(corners[:, 0]-x, corners[:, 1]-y).min())
    angles = requested_angle + np.arange(36)*math.pi/18
    grid = endpoint_clearances(x, y, step, angles, local)
    sampled = endpoint_clearances(x, y, step, np.arange(360)*math.pi/180, local)
    # A circle contained strictly inside one rectangle cannot have any legal
    # endpoint, for ANY heading. This is stronger than sampling directions.
    contained = bool(np.any(clearance < -step-1e-8))
    result = dict(x=float(x), y=float(y), heading=float(heading % (2*math.pi)),
        step=float(step), overlap_count=int((clearance < 0).sum()),
        nearest_clearance=float(clearance[order[0]]),
        second_clearance=float(clearance[order[1]]),
        third_clearance=float(clearance[order[2]]), nearest_obstacle=nearest,
        nearby_obstacles_2=int((clearance <= 2).sum()),
        nearby_obstacles_15=int((clearance <= 15).sum()),
        nearest_boundary_clearance=float(clearance[:4].min()),
        nearest_internal_clearance=float(clearance[4:].min()) if len(clearance)>4 else 1e6,
        expanded_corner_distance=corner_distance,
        perpendicular_faces_2=int(min(left, right)<=2 and min(above, below)<=2),
        perpendicular_faces_15=int(min(left, right)<=15 and min(above, below)<=15),
        opposing_faces_15=int((left<=15 and right<=15) or (above<=15 and below<=15)),
        boundary_corner_15=int((clearance[:4] <= 15).sum() >= 2),
        free_step_directions_36=int((grid >= 0).sum()),
        borderline_step_directions_36=int((np.abs(grid)<1e-8).sum()),
        sampled_free_direction_fraction=float((sampled >= 0).mean()),
        step_circle_inside_one_obstacle=int(contained))
    r = rects[nearest]
    # Angle from heading to the nearest point of the nearest collision box.
    dx = float(np.clip(x, r[0], r[2])-x)
    dy = float(np.clip(y, r[1], r[3])-y)
    result['heading_to_nearest_box_degrees'] = math.degrees(math.atan2(
        math.sin(math.atan2(dy, dx)-heading), math.cos(math.atan2(dy, dx)-heading))) if dx or dy else 0.
    return result


def period_features(active):
    """Empirical recurrences, not an exact or infinite periodicity proof."""
    tail = active[-128:]
    result = dict(position_period=0, pose_period=0, best_pose_return_fraction=0.)
    if len(tail) < 32:
        return result
    for lag in range(1, 17):
        pos = np.hypot(tail[lag:, 1]-tail[:-lag, 1], tail[lag:, 2]-tail[:-lag, 2]) <= 1e-6
        delta = tail[lag:, 3]-tail[:-lag, 3]
        heading = np.abs(np.arctan2(np.sin(delta), np.cos(delta))) <= 1e-6
        pose_rate = float((pos & heading).mean())
        if not result['position_period'] and pos.mean() >= .99:
            result['position_period'] = lag
        if not result['pose_period'] and pose_rate >= .99:
            result['pose_period'] = lag
        result['best_pose_return_fraction'] = max(result['best_pose_return_fraction'], pose_rate)
    return result


def window_features(trace, world, population, is_case, shard):
    if trace['diagnostic_format']['columns'] != DETAIL_COLUMNS:
        raise ValueError('Unsupported diagnostic columns')
    rows = np.asarray(trace['diagnostic_rows'], dtype=float)
    if rows.ndim != 2 or rows.shape[1] != len(DETAIL_COLUMNS):
        raise ValueError('Invalid diagnostic array')
    end_tick = int(rows[-1, 0])
    start_tick = round(trace['start_seconds']*10) if is_case else end_tick-600
    interval = rows[rows[:, 0] > start_tick]
    if len(interval) != 600 or not np.array_equal(interval[:, 0], np.arange(start_tick+1, end_tick+1)):
        raise ValueError('Window must contain 600 consecutive transitions')
    if is_case:
        samples = np.asarray(trace['samples'], dtype=float)
        if len(samples) != 601 or np.hypot(samples[:, 1]-samples[0, 1], samples[:, 2]-samples[0, 2]).max() > 15+1e-8:
            raise ValueError('Invalid confinement interval')
    active = interval[interval[:, IX['mode']] > 0]
    if not len(active):
        raise ValueError('A 60-second window unexpectedly contains no active ticks')
    a, n = active, len(active)
    blocked = a[:, IX['accepted_candidate']] < 0
    displacement = np.hypot(a[:, 1]-a[:, IX['before_x']], a[:, 2]-a[:, IX['before_y']])
    kind = ('all_active_moves_blocked' if blocked.all() else 'mixed_blocking' if blocked.any()
            else 'moving_confinement') if is_case else 'unflagged_control'
    turns = a[:, IX['turn']]
    edge = a[:, IX['mode']] == 1
    quarter = np.abs(np.abs(turns)-math.pi/2) < 1e-12
    last = a[-1]
    before_biome = int(last[IX['before_biome']])
    pid = trace['predator_id']
    birth = population[pid]
    result = dict(seed=trace['seed'], predator_id=pid, shard=shard, is_case=int(is_case),
        classification=kind, biome=BIOMES[before_biome], start_seconds=start_tick/10,
        detected_seconds=end_tick/10, spawned_overlapping=int(birth['spawned_overlapping_obstacle']),
        active_ticks=n, blocked_fraction=float(blocked.mean()),
        fallback_fraction=float((a[:, IX['accepted_candidate']] > 0).mean()),
        edge_fraction=float(edge.mean()), wander_fraction=float((a[:, IX['mode']]==2).mean()),
        quarter_turn_fraction=float(quarter.mean()),
        stationary_fraction=float((displacement < 1e-9).mean()),
        selected_edge_close_fraction=float((edge & (a[:, IX['edge_clearance']] <= 2+1e-10)).mean()),
        selected_edge_clearance_min=float(a[edge, IX['edge_clearance']].min()) if edge.any() else 1e6,
        selected_edge_clearance_max=float(a[edge, IX['edge_clearance']].max()) if edge.any() else 1e6,
        turn_sign_changes=int((turns[1:]*turns[:-1] < 0).sum()),
        bounds_clamped_ticks=int(a[:, IX['bounds_clamped']].sum()),
        travelled_distance=float(displacement.sum()),
        bbox_width=float(np.ptp(interval[:, 1])), bbox_height=float(np.ptp(interval[:, 2])),
        center_x=float(a[-128:, 1].mean()), center_y=float(a[-128:, 2].mean()),
        biome_count=int(len(np.unique(a[:, IX['before_biome']]))),
        escaped_after_detection=int(trace['follow_up']['escaped_after_detection']) if is_case else 0,
        followup_seconds=600-end_tick/10 if is_case else 0.,
        trace=trace.get('trace', f"games/seed-{trace['seed']:08d}/unflagged-control.json.gz"))
    result.update(period_features(a))
    result['four_step_position_fraction'] = float((np.hypot(a[4:, 1]-a[:-4, 1], a[4:, 2]-a[:-4, 2]) < 1e-6).mean()) if n>4 else 0.
    # Both an entry pose and a late pose: the first detected interval can
    # include an approach before the repeating part of the path is established.
    for prefix, row in (('entry', a[0]), ('late', last)):
        geo = geometry_features(row[IX['before_x']], row[IX['before_y']], row[IX['before_heading']],
            row[IX['scaled_distance']], row[IX['absolute_move_direction']], world['obstacles'],
            trace.get('predator_size', 10.))
        result.update({prefix+'_'+key: value for key, value in geo.items()})
    result['late_logged_all_blocked'] = int(blocked[-1])
    # Reconstruct the 36 distinct native candidate directions at the late pose.
    # Near-boundary rounding is reported separately rather than called a mismatch.
    result['candidate_reconstruction_mismatch'] = int(
        not result['late_borderline_step_directions_36'] and
        (result['late_free_step_directions_36']==0) != bool(blocked[-1]))
    return result


def cluster_spots(cases, radius=5.):
    """Connected components of late path centers within radius, within one map."""
    parent = list(range(len(cases)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    cells = {}
    for i, row in enumerate(cases):
        x, y = row['center_x'], row['center_y']
        cell = (math.floor(x/radius), math.floor(y/radius))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((cell[0]+dx, cell[1]+dy), ()):
                    if math.hypot(x-cases[j]['center_x'], y-cases[j]['center_y']) <= radius:
                        parent[find(i)] = find(j)
        cells.setdefault(cell, []).append(i)
    groups = {}
    for i, row in enumerate(cases):
        groups.setdefault(find(i), []).append(row)
    spots = []
    for number, members in enumerate(groups.values()):
        spot_id = f"{members[0]['seed']:08d}-{number:03d}"
        for row in members:
            row['spot_id'] = spot_id
        xs = [r['center_x'] for r in members]; ys = [r['center_y'] for r in members]
        spots.append(dict(spot_id=spot_id, seed=members[0]['seed'], predators=len(members),
            center_x=sum(xs)/len(xs), center_y=sum(ys)/len(ys),
            center_span_x=max(xs)-min(xs), center_span_y=max(ys)-min(ys),
            classifications='|'.join(sorted({r['classification'] for r in members})),
            predator_ids='|'.join(str(r['predator_id']) for r in members),
            representative_trace=members[0]['trace'], shard=members[0]['shard']))
    return spots


def process_game(bundle):
    seed, shard, files, cluster_radius = bundle
    game = json.loads(files['result.json'])
    world = json.loads(files['map.json'])
    if game['status'] != 'complete' or game['seed'] != seed:
        raise ValueError(f'Incomplete or mismatched game {seed}')
    population = {p['predator_id']: p for p in game['predators']}
    cases = []
    for event in game['findings']:
        name = 'events/'+event['trace'].rsplit('/', 1)[1]
        trace = json.loads(gzip.decompress(files[name]))
        if trace['seed'] != seed or trace['predator_id'] != event['predator_id']:
            raise ValueError('Mismatched event file')
        cases.append(window_features(trace, world, population, True, shard))
    controls = []
    if game.get('control_trace'):
        trace = json.loads(gzip.decompress(files['unflagged-control.json.gz']))
        if trace['predator_id'] in {r['predator_id'] for r in cases}:
            raise ValueError('Control is a flagged predator')
        controls.append(window_features(trace, world, population, False, shard))
        controls[0]['spot_id'] = ''
    spots = cluster_spots(cases, cluster_radius)
    return dict(version=FEATURE_VERSION, seed=seed, shard=shard, rows=cases+controls,
                spots=spots, initial_predators=game['initial_predators'],
                total_predators=game['total_tracked'])
