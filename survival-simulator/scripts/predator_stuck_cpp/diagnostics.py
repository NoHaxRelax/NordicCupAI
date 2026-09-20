"""Interpret passive C++ telemetry without changing simulation decisions."""
import math
from collections import Counter

DETAIL_COLUMNS = ["tick", "x", "y", "heading", "energy", "resting",
    "before_x", "before_y", "before_heading", "before_energy", "before_resting", "before_biome",
    "mode", "visible_edges", "local_obstacles", "requested_distance", "relative_move_direction",
    "has_move_direction", "turn", "scaled_distance", "absolute_move_direction", "candidate_tests",
    "accepted_candidate", "rejected_candidate_mask", "first_blocking_obstacle",
    "selected_edge_x1", "selected_edge_y1", "selected_edge_x2", "selected_edge_y2",
    "closest_edge_x", "closest_edge_y", "edge_clearance", "edge_angle", "bounds_clamped"]
INDEX = {name: i for i, name in enumerate(DETAIL_COLUMNS)}
POPULATION_COLUMNS = ["rest_ticks", "active_ticks", "edge_avoidance_ticks", "wander_ticks",
    "all_candidates_blocked_ticks", "successful_fallback_ticks", "longest_blocked_active_run", "distance_travelled"]


def geometry(position, size, obstacles):
    x, y = position
    nearby = []
    overlaps = []
    for oid, (ox, oy, w, h) in enumerate(obstacles):
        # Signed clearance from the exact square-expanded collision rectangle.
        clearance = max(ox-size-x, x-(ox+w+size), oy-size-y, y-(oy+h+size))
        if clearance < 0:
            overlaps.append(oid)
        nearby.append((clearance, oid))
    return dict(overlapping_obstacles=overlaps,
                nearest_obstacles=[dict(obstacle_id=i, signed_collision_clearance=d)
                                   for d, i in sorted(nearby)[:8]],
                distance_to_world_edge=min(x, y, 1600-x, 1200-y))


def summarize(rows, start_tick):
    ix = INDEX
    interval = [r for r in rows if r[0] > start_tick]
    active = [r for r in interval if r[ix["mode"]] > 0]
    failed = [r for r in active if r[ix["accepted_candidate"]] < 0]
    fallback = [r for r in active if r[ix["accepted_candidate"]] > 0]
    zero = [r for r in active if math.hypot(r[1]-r[ix["before_x"]], r[2]-r[ix["before_y"]]) < 1e-9]
    turns = [r[ix["turn"]] for r in active]
    blockers = Counter(int(r[ix["first_blocking_obstacle"]]) for r in active
                       if r[ix["first_blocking_obstacle"]] >= 0)
    return dict(interval_ticks=len(interval), active_ticks=len(active), resting_ticks=len(interval)-len(active),
                edge_avoidance_ticks=sum(r[ix["mode"]] == 1 for r in active),
                wander_ticks=sum(r[ix["mode"]] == 2 for r in active),
                all_candidates_blocked_ticks=len(failed), successful_fallback_ticks=len(fallback),
                zero_displacement_active_ticks=len(zero),
                turn_sign_changes=sum(a*b < 0 for a, b in zip(turns, turns[1:])),
                total_absolute_turn=sum(abs(t) for t in turns),
                quarter_turn_ticks=sum(abs(abs(t)-math.pi/2) < 1e-12 for t in turns),
                four_active_step_returns=sum(math.dist(active[i][1:3], active[i-4][1:3]) < 1e-6
                                            for i in range(4, len(active))),
                biome_active_ticks=dict(Counter(int(r[ix["before_biome"]]) for r in active)),
                travelled_distance=sum(math.hypot(r[1]-r[ix["before_x"]], r[2]-r[ix["before_y"]]) for r in active),
                blocker_counts=dict(blockers),
                boundary_blocked_ticks=sum(n for oid, n in blockers.items() if oid < 4))


FORMAT = dict(version=1, columns=DETAIL_COLUMNS,
    modes={"-1": "birth/no previous tick", "0": "rest", "1": "edge avoidance", "2": "random wander", "3": "prey"},
    candidate_order="0=original; indices 1..36 use i=index-1, k=(i+1)//2, offset=(+1 if i even else -1)*k*pi/18",
    candidate_notes="-1 accepted means all candidates rejected (or no move; check mode). Bit n of rejected_candidate_mask refers to candidate n. Heading is updated by turn, independently of accepted fallback angle.",
    edge_frame="Selected edge coordinates and closest point are in the predator's observation frame before moving.",
    geometry="Obstacle IDs index map.json. The first four rectangles are boundaries. Collision rectangles are expanded by predator size using strict inequalities.",
    timing="Row tick t records the transition from t-1 to t; x/y/heading/energy/resting are after it. Birth rows have mode -1.")
