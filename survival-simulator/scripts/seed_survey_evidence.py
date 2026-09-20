"""Reconstruct seed evidence from public history, without simulator state.

Static edge pairs register observations across agents and time. Boundary sightings
anchor the resulting components; commands are used for heading and stationary
links only, never to undo a walk through collisions. Conflicting components and
ambiguous repeated edge-pair signatures are omitted rather than forced to fit.
"""

from collections import defaultdict, deque
from itertools import combinations
import math

from seed_biome_prefix_probe import boundary_pose
from seed_joint_constraints_probe import angle_close, heading_hypotheses


EPS = 1e-6
LINK_ERROR = 1e-5
LAND = {"forest", "swamp", "desert", "grassland"}


def add(a, b):
    return [a[0] + b[0], a[1] + b[1]]


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1]]


def rotate(point, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [c * point[0] - s * point[1], s * point[0] + c * point[1]]


def rock_edges(observations, heading):
    unique = {}
    for obj in observations:
        if obj["type"] != "Edge":
            continue
        a, b = obj["coords"]
        if not 30 + EPS < math.dist(a, b) < 100 - EPS:
            continue
        a, b = rotate(a, heading), rotate(b, heading)
        key = tuple(round(v, 6) for p in (a, b) for v in p)
        unique[key] = (a, b)
    return sorted(unique.values(), key=lambda edge: (*[round(v, 5) for v in sub(edge[1], edge[0])], *edge[0]))


def _nodes(frames, actions):
    """Resolve headings using fresh observations and the actual commanded turns."""
    turns, ages, hypotheses, records = {}, {}, {}, []
    for index, frame in enumerate(frames):
        for state in frame["observations"]:
            agent_id = state["agent_id"]
            if ages.get(agent_id) == state["age"]:
                continue  # Native agent-death iteration can leave cached sightings.
            ages[agent_id] = state["age"]
            turn = turns.get(agent_id, 0.)
            found = heading_hypotheses(state["observations"], turn)
            boundary = boundary_pose(state)
            if boundary is not None:
                # A map-width/height wall disambiguates its axis even when all
                # observed rock edges happen to be parallel.
                found = [(boundary[2] - turn) % math.tau]
            if found is not None:
                old = hypotheses.get(agent_id)
                hypotheses[agent_id] = found if old is None else [a for a in old if any(angle_close(a, b) for b in found)]
            records.append(dict(frame=index, sim_time=frame["sim_time"], state=state, turn=turn))
        if index < len(actions):
            for action in actions[index]:
                agent_id = action["agent_id"]
                turns[agent_id] = turns.get(agent_id, 0.) + action["turn_angle"]
    nodes = []
    for record in records:
        found = hypotheses.get(record["state"]["agent_id"], [])
        if len(found) != 1:
            continue
        record["heading"] = found[0] + record["turn"]
        record["edges"] = rock_edges(record["state"]["observations"], record["heading"])
        nodes.append(record)
    return nodes


def reconstruct_poses(frames, actions):
    """Share static geometry and solve translations back to first observations.

Quantized fingerprints only propose links; unrounded descriptors validate them.
Rounding misses lose coverage, never manufacture a link. Inconsistent anchored
components are discarded. Exact repeated constellations remain an identification
assumption; retain raw frames and require native replay before declaring recovery.
"""
    nodes = _nodes(frames, actions)
    graph = [[] for _ in nodes]
    pairs, ambiguous = defaultdict(list), set()
    by_frame = {(node["frame"], node["state"]["agent_id"]): i for i, node in enumerate(nodes)}

    def link(i, j, delta):
        graph[i].append((j, delta))  # position[j] = position[i] + delta
        graph[j].append((i, [-delta[0], -delta[1]]))

    for i, node in enumerate(nodes):
        seen = set()
        for first, second in combinations(node["edges"], 2):
            descriptor = (*sub(first[1], first[0]), *sub(second[1], second[0]), *sub(second[0], first[0]))
            key = tuple(round(value, 4) for value in descriptor)
            if key in seen:
                ambiguous.add(key)
            seen.add(key)
            pairs[key].append((i, first[0], descriptor))
        frame, agent_id = node["frame"], node["state"]["agent_id"]
        previous = by_frame.get((frame - 1, agent_id))
        if previous is not None:
            moves = [action for action in actions[frame - 1] if action["agent_id"] == agent_id]
            # Zero distance in the native generator cannot produce displacement.
            if all(action["move_distance"] <= 0 for action in moves):
                link(previous, i, [0., 0.])
        for obj in node["state"]["observations"]:
            target = by_frame.get((frame, obj.get("id"))) if obj["type"] == "Agent" else None
            if target is not None:
                offset = rotate([obj["distance"], 0.], node["heading"] + obj["angle"])
                link(i, target, offset)
    for key, occurrences in pairs.items():
        if key in ambiguous:
            continue
        first, offset, descriptor = occurrences[0]
        for other, other_offset, other_descriptor in occurrences[1:]:
            if max(abs(a - b) for a, b in zip(descriptor, other_descriptor)) <= EPS:
                link(first, other, sub(offset, other_offset))

    positions, relative_positions, visited, conflicts = {}, {}, set(), 0
    for root in range(len(nodes)):
        if root in visited:
            continue
        local, radii, queue = {root: [0., 0.]}, {root: LINK_ERROR}, deque([root])
        consistent = True
        while queue:
            current = queue.popleft()
            visited.add(current)
            for other, delta in graph[current]:
                proposed = add(local[current], delta)
                if other in local:
                    if math.dist(local[other], proposed) > radii[other] + radii[current] + LINK_ERROR:
                        consistent = False
                else:
                    local[other] = proposed
                    radii[other] = radii[current] + LINK_ERROR
                    queue.append(other)
        anchors = []
        for i, position in local.items():
            pose = boundary_pose(nodes[i]["state"])
            if pose is not None:
                anchors.append((sub(pose[:2], position), radii[i] + LINK_ERROR))
        if anchors:
            offset, anchor_radius = min(anchors, key=lambda item: item[1])
            if any(math.dist(offset, other) > anchor_radius + radius for other, radius in anchors):
                consistent = False
        if not consistent:
            conflicts += 1
            continue
        for i, position in local.items():
            relative_positions[i] = dict(xy=position, radius=radii[i], component=root)
        if not anchors:
            continue
        for i, position in local.items():
            radius = radii[i] + anchor_radius
            if radius <= .05:
                positions[i] = dict(xy=add(position, offset), radius=radius)
    return nodes, positions, relative_positions, dict(fresh_poses=len(nodes), localized_poses=len(positions),
                                                      conflicting_components=conflicts, ambiguous_edge_pairs=len(ambiguous))


def _rectangles(edges):
    """Infer paired width/height only from adjacent full perpendicular sides."""
    for first, second in combinations(edges, 2):
        av, bv = sub(first[1], first[0]), sub(second[1], second[0])
        if abs(av[0] * bv[0] + av[1] * bv[1]) > EPS:
            continue
        if min(math.dist(a, b) for a in first for b in second) > EPS:
            continue
        points = (*first, *second)
        x, y = min(p[0] for p in points), min(p[1] for p in points)
        width, height = max(p[0] for p in points) - x, max(p[1] for p in points) - y
        if 30 + EPS < width < 100 - EPS and 30 + EPS < height < 100 - EPS:
            yield dict(x=x, y=y, width=width, height=height)


def extract_evidence(frames, actions):
    if not frames or len(actions) != len(frames) - 1:
        raise ValueError("Evidence needs one public frame per tick and the intervening action lists")
    if any(abs(frame["sim_time"] - .1 * (i + 1)) > EPS for i, frame in enumerate(frames)):
        raise ValueError("History must start at the native first empty tick (0.1s) and contain every 0.1s frame")
    nodes, poses, relative_poses, diagnostics = reconstruct_poses(frames, actions)
    founders, samples, transitions, rectangles, edges = [], [], [], {}, {}
    # A candidate's known heading suffices to check these vectors, even when
    # public edge observations have not resolved the target's heading yet.
    initial_trees = [dict(agent_id=state["agent_id"], distance=obj["distance"], angle=obj["angle"],
                          sim_time=frames[0]["sim_time"], frame=0)
                     for state in frames[0]["observations"] for obj in state["observations"] if obj["type"] == "Tree"]
    previous = {}
    for i, node in enumerate(nodes):
        state, frame = node["state"], node["frame"]
        agent_id, heading = state["agent_id"], node["heading"]
        pose = poses.get(i)
        if frame == 0:
            if pose is not None:
                founders.append(dict(agent_id=agent_id, xy=pose["xy"], radius=pose["radius"],
                                     heading=heading, source="registered_first_observation"))
        for rectangle in _rectangles(node["edges"]):
            shape = (round(rectangle["width"], 6), round(rectangle["height"], 6))
            if pose is not None:
                rectangle["x"], rectangle["y"] = add((rectangle["x"], rectangle["y"]), pose["xy"])
                rectangle["radius"] = pose["radius"]
                key = (*shape, round(rectangle["x"], 4), round(rectangle["y"], 4))
            else:
                rectangle.pop("x")
                rectangle.pop("y")
                key = shape
            rectangle.update(frame=frame, agent_id=agent_id, sim_time=node["sim_time"])
            if key not in rectangles or rectangle.get("radius", 0.) < rectangles[key].get("radius", 0.):
                rectangles[key] = rectangle
        if pose is None:
            continue
        xy, radius = pose["xy"], pose["radius"]
        for a, b in node["edges"]:
            a, b = add(a, xy), add(b, xy)
            key = tuple(round(v, 4) for p in (a, b) for v in p)
            if key not in edges or radius < edges[key]["radius"]:
                edges[key] = dict(start=a, end=b, radius=radius, frame=frame, agent_id=agent_id)
        sample = dict(x=xy[0], y=xy[1], biome=state["biome"], uncertainty=radius,
                      # Biome labels use truncated pixel coordinates, not the
                      # continuous agent point: include a sqrt(2) pixel margin.
                      radius=radius + math.sqrt(2), position_radius=radius,
                      sim_time=node["sim_time"], frame=frame, agent_id=agent_id,
                      source="registered_public_geometry")
        samples.append(sample)
        old = previous.get(agent_id)
        if old is not None and old["frame"] == frame - 1 and old["biome"] != sample["biome"]:
            transitions.append(dict(before=old, after=sample, distance=math.dist((old["x"], old["y"]), xy)))
        previous[agent_id] = sample
    founders.sort(key=lambda row: row["agent_id"])
    origins = {nodes[i]["state"]["agent_id"]: pose for i, pose in relative_poses.items() if nodes[i]["frame"] == 0}
    distances = [dict(agent_ids=[first, second], distance=math.dist(a["xy"], b["xy"]),
                      radius=a["radius"] + b["radius"])
                 for (first, a), (second, b) in combinations(sorted(origins.items()), 2)
                 if a["component"] == b["component"]]
    return dict(version=1, founder_positions=founders, founder_distances=distances,
                rock_rectangles=list(rectangles.values()), world_rock_edges=list(edges.values()),
                initial_tree_sightings=initial_trees, biome_samples=samples,
                biome_transitions=transitions, diagnostics=diagnostics)


def informative_samples(evidence, fallback, spread, limit=64):
    """Reserve half the budget for short, observed biome transition brackets."""
    selected, seen = [], set()

    def retain(sample):
        key = (sample["x"], sample["y"], sample["biome"])
        if sample["biome"] in LAND and key not in seen:
            seen.add(key)
            selected.append(sample)

    for transition in sorted(evidence["biome_transitions"], key=lambda row: row["distance"]):
        if len(selected) + 2 > limit // 2:
            break
        retain(transition["before"])
        retain(transition["after"])
    # Spatially spread the rest; do not turn a single biome's dense trail into
    # dozens of ostensibly independent observations.
    pool = [row for row in evidence["biome_samples"] if row["biome"] in LAND] + fallback
    for sample in spread(pool, limit=limit):
        if len(selected) >= limit:
            break
        retain(sample)
    return selected


def candidate_geometry_matches(env, evidence):
    """Compare public constraints only to a generated *candidate* environment."""
    def near(a, b, radius):
        return math.dist(a, b) <= radius + EPS

    founders = evidence.get("founder_positions", [])
    founder_match = all(row["agent_id"] in env.agents_dict and near(
        row["xy"], (env.agents_dict[row["agent_id"]].x, env.agents_dict[row["agent_id"]].y), row["radius"])
        for row in founders)
    distance_match = True
    for row in evidence.get("founder_distances", []):
        first, second = [env.agents_dict.get(agent_id) for agent_id in row["agent_ids"]]
        if first is None or second is None or abs(math.hypot(first.x - second.x, first.y - second.y)
                                                  - row["distance"]) > row["radius"] + EPS:
            distance_match = False
            break
    rectangles = evidence.get("rock_rectangles", [])
    rectangle_match = all(any(abs(row["width"] - rock.width) <= EPS and abs(row["height"] - rock.height) <= EPS
        and ("x" not in row or near((row["x"], row["y"]), (rock.x, rock.y), row["radius"]))
        for rock in env.obstacles) for row in rectangles)
    edges = evidence.get("world_rock_edges", [])
    edge_match = all(any(near(row["start"], a, row["radius"]) and near(row["end"], b, row["radius"])
                         for a, b in env.edges) for row in edges)
    biome_match = True
    for row in evidence.get("biome_samples", []):
        radius = row["position_radius"] + EPS
        xs = range(max(0, int(row["x"] - radius)), min(env.width - 1, int(row["x"] + radius)) + 1)
        ys = range(max(0, int(row["y"] - radius)), min(env.height - 1, int(row["y"] + radius)) + 1)
        if not any(env.biome_map[x, y].type == row["biome"] for x in xs for y in ys):
            biome_match = False
            break
    return dict(founder_position_match=founder_match, founder_distance_match=distance_match, rock_rectangle_match=rectangle_match,
                world_rock_edge_match=edge_match, observed_biome_match=biome_match)


def initial_trees_match(env, sightings):
    for row in sightings:
        agent = env.agents_dict.get(row["agent_id"])
        if agent is None:
            return False
        offset = rotate((row["distance"], 0.), agent.direction + row["angle"])
        expected = add((agent.x, agent.y), offset)
        if not any(math.dist(expected, (tree.x, tree.y)) <= EPS for tree in env.trees):
            return False
    return True


def main():
    """Rebuild derived evidence from an archived public history, with no world."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from seed_survey_probe import spread_samples

    parser = argparse.ArgumentParser(description="Re-extract seed evidence from saved raw public frames")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output filename")
    document = json.loads(args.input.read_text(encoding="utf-8"))
    try:
        for row in document["targets"]:
            frames, actions = row["public_frames"], row["action_log"]
            from seed_joint_constraints_probe import canonical_frame
            hashes = [hashlib.sha256(canonical_frame(frame).encode()).hexdigest() for frame in frames]
            if hashes != row["public_frame_sha256"]:
                raise ValueError("Raw public frames do not match their recorded hashes")
            row["seed_evidence"] = extract_evidence(frames, actions)
            row["schema_version"] = 2
            for checkpoint in row["checkpoints"]:
                end = next(i + 1 for i, frame in enumerate(frames)
                           if abs(frame["sim_time"] - checkpoint["sim_time"]) <= EPS)
                evidence = row["seed_evidence"] if end == len(frames) else extract_evidence(frames[:end], actions[:end - 1])
                checkpoint["shared_samples"] = informative_samples(evidence, checkpoint["shared_samples"], spread_samples)
                checkpoint.update(selected_samples=len(checkpoint["shared_samples"]),
                                  observed_biomes=sorted({sample["biome"] for sample in checkpoint["shared_samples"]}),
                                  recovered_spawn_positions=len(evidence["founder_positions"]),
                                  paired_rock_rectangles=len(evidence["rock_rectangles"]),
                                  positioned_rock_edges=len(evidence["world_rock_edges"]),
                                  biome_transitions=len(evidence["biome_transitions"]),
                                  geometry_diagnostics=evidence["diagnostics"])
                # Any results from the previous signature are no longer valid.
                for key in ("prefix_survivors", "true_seed_passes"):
                    checkpoint.pop(key, None)
            for key in ("native_verification", "recovered_seeds", "verification_skipped"):
                row.pop(key, None)
        document.update(complete=True, search_performed=False, verification_complete=False,
                        evidence_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        document.pop("prefix_search_seconds_all_checkpoints", None)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (KeyError, TypeError, ValueError, StopIteration) as error:
        parser.error(f"Cannot rebuild evidence: {error}. Raw public_frames and matching action/hash history are required.")


if __name__ == "__main__":
    main()
