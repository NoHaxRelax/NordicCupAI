"""Observed entrances and constructive exit tests for recorded confinement.

Exit witnesses allow arbitrary headings but retain native biome-scaled, fixed
walking steps and endpoint collisions. They do not claim the predator policy
will choose those headings. An unsuccessful bounded search is inconclusive.
"""
import gzip
import heapq
import json
import math
from pathlib import Path

import numpy as np

from diagnostics import INDEX as IX
from spot_features import PENALTIES, expanded_rectangles

ENTRANCE_VERSION = 1
EPS = 1e-8


def clearances(points, rects):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(rects):
        return np.full(len(points), np.inf)
    x, y = points[:, 0], points[:, 1]
    return np.minimum.reduce([np.maximum.reduce((r[0]-x, x-r[2], r[1]-y, y-r[3])) for r in rects])


def segment_is_clear(start, end, rects):
    """Test the segment against OPEN rectangle interiors, as a separate metric.

    The native engine itself checks only step endpoints.
    """
    a, b = np.asarray(start), np.asarray(end)
    delta = b-a
    for r in rects:
        lo, hi = 0., 1.
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-15:
                if not r[axis] < a[axis] < r[axis+2]:
                    lo, hi = 1., 0.
                    break
            else:
                t1 = (r[axis]-a[axis])/delta[axis]
                t2 = (r[axis+2]-a[axis])/delta[axis]
                lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
        if lo < hi-1e-12:
            return False
    return True


def observe_entry(trace, rects):
    rows = np.asarray(trace['diagnostic_rows'], dtype=float)
    if len(rows)>1 and not np.all(np.diff(rows[:, 0]) == 1):
        raise ValueError('Approach trace is not consecutive')
    anchor = np.asarray(trace['anchor']); radius = trace['radius']
    before = rows[:, [IX['before_x'], IX['before_y']]]
    after = rows[:, [IX['x'], IX['y']]]
    source_distance = np.linalg.norm(before-anchor, axis=1)
    target_distance = np.linalg.norm(after-anchor, axis=1)
    crossings = np.flatnonzero((source_distance > radius+EPS) & (target_distance <= radius) & (rows[:, IX['mode']] > 0))
    result = dict(entry_evidence='not_observed_in_saved_approach', observed_legal_entry=0,
        entry_seconds=None, crossing_from_x=None, crossing_from_y=None,
        crossing_to_x=None, crossing_to_y=None, entry_segment_clear=None,
        entered_no_observed_exit=0, entered_then_escaped=0,
        entered_no_exit_with_60s_followup=0, observed_residence_seconds=None,
        initial_position_outside_circle=int(np.linalg.norm(np.asarray(trace['initial_position'])-anchor)>radius+EPS))
    if not len(crossings):
        if trace['start_seconds'] == trace['born_seconds']:
            result['entry_evidence'] = 'confinement_since_birth_without_observed_entry'
        elif result['initial_position_outside_circle']:
            result['entry_evidence'] = 'born_outside_but_crossing_not_in_saved_approach'
        return result, rows, None
    i = int(crossings[-1]); row = rows[i]
    result.update(entry_seconds=float(row[0]*.1),
        crossing_from_x=float(before[i, 0]), crossing_from_y=float(before[i, 1]),
        crossing_to_x=float(after[i, 0]), crossing_to_y=float(after[i, 1]),
        entry_segment_clear=int(segment_is_clear(before[i], after[i], rects)))
    if row[IX['accepted_candidate']] < 0 or row[IX['bounds_clamped']] != 0:
        result['entry_evidence'] = 'crossing_not_an_accepted_unclamped_move'
        return result, rows, None
    if np.any(clearances(np.array([before[i], after[i]]), rects) < 0):
        result['entry_evidence'] = 'crossing_has_an_overlapping_endpoint'
        return result, rows, None
    if np.any(target_distance[i:] > radius+EPS):
        raise ValueError('An observed departure follows the final entry')
    if rows[-1, 0]-row[0] < 600:
        result['entry_evidence'] = 'observed_entry_but_less_than_60s_residence'
        return result, rows, None
    follow = trace['follow_up']
    escaped = bool(follow['escaped_after_detection'])
    observed_end = follow['first_exit_seconds'] if escaped else follow['observation_end_seconds']
    result.update(entry_evidence='observed_legal_entry', observed_legal_entry=1,
        entered_no_observed_exit=int(not escaped), entered_then_escaped=int(escaped),
        entered_no_exit_with_60s_followup=int(not escaped and follow['observation_end_seconds']-trace['detected_seconds']>=60),
        observed_residence_seconds=float(observed_end-result['entry_seconds']))
    return result, rows, i


def steps_at(points, biomes):
    points = np.asarray(points).reshape(-1, 2)
    width, height = biomes.shape
    x = np.clip(points[:, 0].astype(int), 0, width-1)
    y = np.clip(points[:, 1].astype(int), 0, height-1)
    return 11*np.asarray(PENALTIES)[biomes[x, y]]


def circle_partition_angles(position, step, rects):
    """All angular sign-change boundaries for circle/rectangle intersections."""
    x, y = position
    angles = [0.]
    for r in rects:
        for bound in (r[0], r[2]):
            ratio = (bound-x)/step
            if -1 <= ratio <= 1:
                a = math.acos(ratio); angles.extend((a, -a))
        for bound in (r[1], r[3]):
            ratio = (bound-y)/step
            if -1 <= ratio <= 1:
                a = math.asin(ratio); angles.extend((a, math.pi-a))
    boundaries = np.unique(np.remainder(angles, 2*math.pi))
    ends = np.r_[boundaries[1:], boundaries[0]+2*math.pi]
    return np.r_[boundaries, (boundaries+ends)/2]


def prove_no_first_step(position, step, rects):
    # Test all angular partitions AND their boundaries. A positive margin is
    # required, so an isolated legal tangency is never counted as blocked.
    angles = circle_partition_angles(position, step, rects)
    points = np.asarray(position) + step*np.column_stack((np.cos(angles), np.sin(angles)))
    return bool(np.all(clearances(points, rects) < -EPS))


def validate_route(path, anchor, radius, biomes, rects):
    path = np.asarray(path, dtype=float)
    if len(path)<2 or np.linalg.norm(path[-1]-anchor) <= radius+EPS:
        return False
    if np.any(clearances(path, rects) < EPS):
        return False
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return bool(np.all(np.abs(lengths-steps_at(path[:-1], biomes)) <= 1e-7))


def reverse_recorded_route(rows, entry_index, anchor, radius, biomes, rects):
    forward = np.vstack((rows[entry_index, [IX['before_x'], IX['before_y']]], rows[entry_index:, 1:3]))
    path = forward[::-1]
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1)>1e-9]]
    return path if validate_route(path, anchor, radius, biomes, rects) else None


def find_exit(start, anchor, radius, biomes, rects, max_nodes=256, angle_step=5., cell_size=.5):
    """Bounded greedy search; a found path is revalidated without quantization.

    Actual coordinates are never snapped. Grid keys only limit exploration.
    Thus positive witnesses are meaningful; search failures are not proofs.
    """
    start = np.asarray(start, dtype=float); anchor = np.asarray(anchor, dtype=float)
    positions = [start]; parents = [-1]
    queue = [(-float(np.linalg.norm(start-anchor)), 0)]
    visited = {tuple(np.floor(start/cell_size).astype(int))}
    base_angles = np.arange(0, 360, angle_step)*math.pi/180
    expanded = 0
    def route(index, endpoint):
        points = [endpoint]
        while index >= 0:
            points.append(positions[index]); index = parents[index]
        return np.asarray(points[::-1])
    while queue and expanded < max_nodes:
        _, index = heapq.heappop(queue)
        p = positions[index]; expanded += 1
        step = float(steps_at([p], biomes)[0])
        outward = math.atan2(p[1]-anchor[1], p[0]-anchor[0])
        angles = np.r_[base_angles, outward]
        points = p + step*np.column_stack((np.cos(angles), np.sin(angles)))
        points = points[clearances(points, rects) >= EPS]
        if not len(points):
            # Rescue openings narrower than the regular angular grid.
            angles = circle_partition_angles(p, step, rects)
            points = p + step*np.column_stack((np.cos(angles), np.sin(angles)))
            points = points[clearances(points, rects) >= EPS]
        distances = np.linalg.norm(points-anchor, axis=1)
        outside = np.flatnonzero(distances > radius+EPS)
        if len(outside):
            path = route(index, points[outside[0]])
            if not validate_route(path, anchor, radius, biomes, rects):
                raise ValueError('Search produced an invalid exit witness')
            return path, expanded
        for point, distance in zip(points, distances):
            key = tuple(np.floor(point/cell_size).astype(int))
            if key in visited: continue
            visited.add(key)
            parents.append(index); positions.append(point)
            heapq.heappush(queue, (-float(distance), len(positions)-1))
    return None, expanded


def empty_exit_fields():
    return dict(exit_test='not_tested_without_observed_entry', exit_route_steps=0,
        exit_search_expanded=0, no_legal_first_step_proven=0,
        entered_behavioral_trap=0, entered_geometric_sink=0, entered_exit_unresolved=0)


def process_entrance_game(bundle, base_cache, max_nodes=256, angle_step=5.):
    seed, shard, files, _ = bundle
    with gzip.open(Path(base_cache)/f'seed-{seed:08d}.json.gz', 'rt', encoding='utf-8') as f:
        result = json.load(f)
    if result['seed'] != seed:
        raise ValueError('Base cache seed mismatch')
    world = json.loads(files['map.json'])
    game = json.loads(files['result.json'])
    if game['status'] != 'complete':
        raise ValueError('Incomplete source game')
    case_rows = {r['predator_id']:r for r in result['rows'] if r['is_case']}
    if set(case_rows) != {e['predator_id'] for e in game['findings']}:
        raise ValueError('Base cache findings differ from source game')
    biomes = np.frombuffer(gzip.decompress(files['biomes.bin.gz']), dtype=np.uint8).reshape(world['width'], world['height'])
    witnesses = {}
    for event in game['findings']:
        pid = event['predator_id']; row = case_rows[pid]
        trace = json.loads(gzip.decompress(files['events/'+event['trace'].rsplit('/', 1)[1]]))
        rects = expanded_rectangles(world['obstacles'], trace['predator_size'])
        entry, rows, index = observe_entry(trace, rects)
        row.update(entry); row.update(empty_exit_fields())
        if index is None:
            continue
        anchor = np.asarray(trace['anchor']); radius = float(trace['radius'])
        # Before leaving the circle no endpoint can be further than radius+11.
        near = clearances_by_rectangle(anchor, rects) <= radius+11+EPS
        local = rects[near]
        path = reverse_recorded_route(rows, index, anchor, radius, biomes, local)
        if path is not None:
            row['exit_test'] = 'verified_reverse_entry_route'
            witnesses[str(pid)] = dict(kind='reverse_recorded_entry', trace=event['trace'],
                entry_tick=int(rows[index, 0]), detection_tick=int(rows[-1, 0]),
                route_steps=len(path)-1, anchor=anchor.tolist(), radius=radius)
        else:
            start = np.asarray(trace['end_position'])
            step = float(steps_at([start], biomes)[0])
            if prove_no_first_step(start, step, local):
                row['exit_test'] = 'no_legal_first_step_proven'
                row['no_legal_first_step_proven'] = 1
            else:
                path, expanded = find_exit(start, anchor, radius, biomes, local, max_nodes, angle_step)
                row['exit_search_expanded'] = expanded
                if path is None:
                    row['exit_test'] = 'unresolved_search_limit_or_resolution'
                else:
                    row['exit_test'] = 'verified_searched_exit_route'
                    witnesses[str(pid)] = dict(kind='searched_exit', trace=event['trace'],
                        anchor=anchor.tolist(), radius=radius, path=path.tolist())
        if path is not None:
            row['exit_route_steps'] = len(path)-1
        if row['entered_no_observed_exit']:
            row['entered_behavioral_trap'] = int(path is not None)
            row['entered_geometric_sink'] = row['no_legal_first_step_proven']
            row['entered_exit_unresolved'] = int(row['exit_test']=='unresolved_search_limit_or_resolution')
    for row in result['rows']:
        if not row['is_case']:
            row.update(dict(entry_evidence='not_applicable_unflagged_control', observed_legal_entry=0,
                entry_seconds=None, crossing_from_x=None, crossing_from_y=None,
                crossing_to_x=None, crossing_to_y=None, entry_segment_clear=None,
                entered_no_observed_exit=0, entered_then_escaped=0,
                entered_no_exit_with_60s_followup=0, observed_residence_seconds=None,
                initial_position_outside_circle=0))
            row.update(empty_exit_fields())
    for spot in result['spots']:
        members = [r for r in case_rows.values() if r['spot_id']==spot['spot_id']]
        for field in ('observed_legal_entry', 'entered_no_observed_exit', 'entered_then_escaped',
                      'entered_behavioral_trap', 'entered_geometric_sink', 'entered_exit_unresolved'):
            spot[field] = sum(r[field] for r in members)
    result.update(entrance_version=ENTRANCE_VERSION, exit_witnesses=witnesses)
    return result


def clearances_by_rectangle(position, rects):
    x, y = position
    return np.maximum.reduce((rects[:, 0]-x, x-rects[:, 2], rects[:, 1]-y, y-rects[:, 3]))
