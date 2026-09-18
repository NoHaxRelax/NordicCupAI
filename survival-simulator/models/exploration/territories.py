"""Small connected food territories, with local changes when owners change."""

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment


def connected_parts(cells, graph):
    remaining = set(cells)
    parts = []
    while remaining:
        queue = [min(remaining)]
        remaining.remove(queue[0])
        part = set(queue)
        while queue:
            current = queue.pop()
            found = graph[current] & remaining
            remaining.difference_update(found)
            part.update(found)
            queue.extend(sorted(found))
        parts.append(part)
    return parts


def corridor_graph(gaps, points, valid):
    """Flood from coarse samples through observed free space to find neighbors.

    Unlike connecting centers by straight lines, this preserves connections
    around small rocks. Each coarse cell represents a connected fine-grid area.
    Only observed rock edges constrain the flood; unseen terrain stays unknown.
    """
    labels = np.full(gaps.seen.shape, -1, dtype=int)
    queue = deque()
    for index in np.flatnonzero(valid):
        x, y = np.floor((points[index] - gaps.origin) / gaps.pitch).astype(int)
        labels[y, x] = index
        queue.append((y, x))
    graph = [set() for _ in points]
    height, width = labels.shape
    while queue:
        y, x = queue.popleft()
        owner = int(labels[y, x])
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not (0 <= yy < height and 0 <= xx < width) or not gaps.reachable[yy, xx]:
                continue
            neighbor = int(labels[yy, xx])
            if neighbor < 0:
                labels[yy, xx] = owner
                queue.append((yy, xx))
            elif neighbor != owner:
                graph[owner].add(neighbor)
                graph[neighbor].add(owner)
    return graph, labels


def divide(cells, count, points, weights, graph):
    """Recursively cut connected regions into approximately equal food shares.

    Prefer compact axis cuts. Around complex obstacles a spanning-tree cut
    still guarantees both resulting regions are connected. Individual coarse
    cells are indivisible, so exact equality is neither required nor possible.
    """
    cells = set(cells)
    count = min(count, len(cells))
    if count <= 1:
        return [cells] if cells else []
    total = float(weights[list(cells)].sum())
    candidates = []

    def candidate(first, second, preference):
        if not first or not second:
            return
        fraction = float(weights[list(first)].sum()) / total
        left_count = min(len(first), count - 1, max(1, count - len(second), round(fraction * count)))
        if not 1 <= count - left_count <= len(second):
            return
        error = abs(fraction - left_count / count)
        candidates.append((error, abs(count - 2 * left_count), preference, min(first), first, second, left_count))

    # Fewer and squarer regions win near ties in food balance.
    extent = np.ptp(points[list(cells)], axis=0)
    for preference, axis in enumerate(np.argsort(-extent)):
        coordinates = sorted(set(points[list(cells), axis]))
        for cut in coordinates[:-1]:
            first = {i for i in cells if points[i, axis] <= cut}
            second = cells - first
            if len(connected_parts(first, graph)) == len(connected_parts(second, graph)) == 1:
                candidate(first, second, preference)
    if not candidates:
        # Removing a spanning-tree subtree cannot disconnect its complement.
        start = min(cells, key=lambda i: (points[i, 0] + points[i, 1], i))
        parent, order = {start: None}, [start]
        for current in order:
            for neighbor in sorted(graph[current] & cells):
                if neighbor not in parent:
                    parent[neighbor] = current
                    order.append(neighbor)
        descendants = {i: {i} for i in cells}
        for current in reversed(order[1:]):
            first = descendants[current]
            candidate(first.copy(), cells - first, 2)
            descendants[parent[current]].update(first)
    best = min(candidates, key=lambda item: item[:4])
    return (divide(best[4], best[6], points, weights, graph)
            + divide(best[5], count - best[6], points, weights, graph))


def _home(cells, points, weights):
    indices = sorted(cells)
    center = np.average(points[indices], axis=0, weights=weights[indices])
    return points[min(indices, key=lambda i: (np.linalg.norm(points[i] - center), i))].copy()


def allocate(points, weights, graph, valid, ids, poses, owners, imbalance, agent_cells):
    """Allocate independently in each component an agent can actually reach."""
    result = np.full(len(points), -1, dtype=int)
    homes = {}
    for component in connected_parts(np.flatnonzero(valid), graph):
        residents = [i for i in ids if agent_cells.get(i, -1) in component]
        if not residents:
            continue
        component_valid = np.zeros(len(points), dtype=bool)
        component_valid[list(component)] = True
        local_owners, local_homes = _allocate_component(
            points, weights, graph, component_valid, residents, poses, owners, imbalance)
        result[component_valid] = local_owners[component_valid]
        homes.update(local_homes)
    return result, homes


def _allocate_component(points, weights, graph, valid, ids, poses, owners, imbalance):
    """Preserve existing regions, split for births, absorb deaths, then balance."""
    ids = sorted(ids)
    valid_cells = set(np.flatnonzero(valid))
    owners = owners.copy() if len(owners) == len(points) else np.full(len(points), -1, dtype=int)
    owners[~valid] = -1
    owners[~np.isin(owners, ids)] = -1
    if not ids or not valid_cells:
        return np.full(len(points), -1, dtype=int), {}

    # New obstacles can cut a region; keep its largest piece and reassign the
    # separated cells by connected growth instead of leaving disjoint duties.
    for agent_id in ids:
        parts = connected_parts(np.flatnonzero(owners == agent_id), graph)
        if len(parts) > 1:
            keep = max(parts, key=lambda part: (len(part), -min(part)))
            owners[(owners == agent_id) & ~np.isin(np.arange(len(points)), list(keep))] = -1

    components = connected_parts(valid_cells, graph)
    unassigned = [i for i in ids if not np.any(owners == i)]
    empty_components = [part for part in components if np.all(owners[list(part)] < 0)]
    if empty_components and unassigned:
        # Allocate at least one owner to each reachable component when possible.
        # A component with no available agent remains explicitly unowned.
        allocations = {min(part): 1 for part in empty_components[:len(unassigned)]}
        by_key = {min(part): part for part in empty_components}
        for _ in range(len(unassigned) - len(allocations)):
            possible = [key for key, n in allocations.items() if n < len(by_key[key])]
            if not possible:
                break
            key = max(possible, key=lambda key: (weights[list(by_key[key])].sum() / (allocations[key] + 1), -key))
            allocations[key] += 1
        regions = []
        for key, count in allocations.items():
            regions.extend(divide(by_key[key], count, points, weights, graph))
        centers = np.array([_home(part, points, weights) for part in regions])
        costs = np.linalg.norm(np.array([poses[i].position for i in unassigned])[:, None] - centers[None], axis=2)
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            owners[list(regions[column])] = unassigned[row]
        unassigned = [i for i in ids if not np.any(owners == i)]

    # With births, split a food-rich existing region; all other owners keep
    # their exact cells. The parent's old side stays as close as possible.
    for newcomer in unassigned:
        donors = [i for i in ids if np.sum(owners == i) >= 2]
        if not donors:
            break
        donor = max(donors, key=lambda i: (weights[owners == i].sum(), -i))
        regions = divide(np.flatnonzero(owners == donor), 2, points, weights, graph)
        centers = [_home(part, points, weights) for part in regions]
        change = [np.linalg.norm(centers[j] - poses[newcomer].position)
                  + np.linalg.norm(centers[1 - j] - poses[donor].position) for j in range(2)]
        owners[list(regions[int(np.argmin(change))])] = newcomer

    # Absorb abandoned cells only through an owned neighboring cell, keeping
    # every surviving territory connected and leaving all other cells alone.
    remaining = set(np.flatnonzero(valid & (owners < 0)))
    loads = {i: float(weights[owners == i].sum()) for i in ids}
    while remaining:
        candidates = []
        for cell in sorted(remaining):
            neighbors = {int(owners[n]) for n in graph[cell] if owners[n] >= 0}
            if neighbors:
                owner = min(neighbors, key=lambda i: (loads[i], i))
                candidates.append((loads[owner], cell, owner))
        if not candidates:
            break
        _, cell, owner = min(candidates)
        owners[cell] = owner
        loads[owner] += float(weights[cell])
        remaining.remove(cell)

    # Small boundary transfers improve a substantially unequal food budget.
    # Every accepted move lowers squared imbalance and preserves connectivity.
    target = sum(loads.values()) / max(1, len([v for v in loads.values() if v > 0]))
    for _ in range(min(128, len(points))):
        moves = []
        for cell in np.flatnonzero(owners >= 0):
            donor = int(owners[cell])
            if loads[donor] <= target * (1 + imbalance):
                continue
            for recipient in sorted({int(owners[n]) for n in graph[cell] if owners[n] >= 0} - {donor}):
                weight = float(weights[cell])
                benefit = 2 * weight * (loads[donor] - loads[recipient] - weight)
                if loads[recipient] < target * (1 - imbalance / 2) and benefit > 0:
                    moves.append((-benefit, int(cell), donor, recipient))
        moved = False
        for _, cell, donor, recipient in sorted(moves):
            rest = set(np.flatnonzero(owners == donor)) - {cell}
            if rest and len(connected_parts(rest, graph)) == 1:
                owners[cell] = recipient
                loads[donor] -= float(weights[cell])
                loads[recipient] += float(weights[cell])
                moved = True
                break
        if not moved:
            break
    homes = {i: _home(np.flatnonzero(owners == i), points, weights) for i in ids if np.any(owners == i)}
    return owners, homes
