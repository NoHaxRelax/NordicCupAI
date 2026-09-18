"""Adapt observed geometry to OUR existing trap detector, never hidden map data."""
import math

from models.entrapment.entrapment_sites import _Geometry, enumerate_sites, enumerate_corner_sites


def observed_rectangles(group, tolerance=1.5):
    """Closed, observed rectangles plus observed arena walls of known thickness.

    Interior obstacles require all four sides. Boundary thickness (30) is a
    public generator rule; a boundary is included only after its face is seen.
    Unknown space remains unknown, so a candidate can be invalidated later.
    """
    size = group.world_size
    if size is None and group.known_width and group.known_height:
        size = (group.known_width, group.known_height)
    if not group.anchored or size is None:
        return None
    width, height = map(float, size)
    horizontal, vertical = [], []
    for edge in group.edges:
        a, b = edge.start, edge.end
        if abs(a[1] - b[1]) < tolerance:
            horizontal.append((min(a[0], b[0]), max(a[0], b[0]), (a[1] + b[1]) / 2))
        elif abs(a[0] - b[0]) < tolerance:
            vertical.append((min(a[1], b[1]), max(a[1], b[1]), (a[0] + b[0]) / 2))
    rects = []
    for index, (x0, x1, y0) in enumerate(horizontal):
        for u0, u1, y1 in horizontal[index + 1:]:
            if abs(x0-u0) > tolerance or abs(x1-u1) > tolerance or abs(y1-y0) <= tolerance:
                continue
            lo, hi = sorted((y0, y1))
            if all(any(abs(v0-lo) <= tolerance and abs(v1-hi) <= tolerance and abs(x-vx) <= tolerance
                       for v0, v1, vx in vertical) for x in (x0, x1)):
                rects.append(((x0+u0)/2, lo, (x1+u1-x0-u0)/2, hi-lo))
    for lo, hi, y in horizontal:
        if abs(lo) <= tolerance and abs(hi-width) <= tolerance:
            if min(abs(y), abs(y-30)) <= tolerance: rects.append((0., 0., width, 30.))
            if min(abs(y-height), abs(y-(height-30))) <= tolerance: rects.append((0., height-30, width, 30.))
    for lo, hi, x in vertical:
        if abs(lo) <= tolerance and abs(hi-height) <= tolerance:
            if min(abs(x), abs(x-30)) <= tolerance: rects.append((0., 0., 30., height))
            if min(abs(x-width), abs(x-(width-30))) <= tolerance: rects.append((width-30, 0., 30., height))
    unique = []
    for rect in rects:
        if not any(max(abs(a-b) for a, b in zip(rect, other)) <= tolerance for other in unique):
            unique.append(tuple(map(float, rect)))
    return dict(width=width, height=height, obstacles=unique)


def our_sites(static, *, corner_only=False):
    """Exactly the short-overlap/boundary-gap acceptance used in guide_lab."""
    geometry = _Geometry(static['width'], static['height'], static['obstacles'])
    result = []
    candidates = (enumerate_corner_sites(static) if corner_only else
                  enumerate_sites(static, min_gap=10.1, min_overlap=10.3))
    for site in candidates:
        accepted = None
        offsets = [0.] if not corner_only else [0.]+[sign*d for d in (2.,4.,6.,8.,10.,12.,15.,18.,22.,26.) for sign in (1.,-1.)]
        for distance in (25., 20., 30.):
            for shift in offsets:
                handoff = [site['mouth'][i] - site['inward'][i]*distance
                           + site['cross'][i]*(site['approach_lane_offset']+shift) for i in range(2)]
                if (geometry.free(handoff, 11.) and math.dist(handoff, site['goal']) <= 44.
                        and (not corner_only or geometry.clear(site['hold'], handoff, 11.))):
                    accepted = dict(site, handoff=handoff)
                    if corner_only:
                        accepted['handoff_lane_shift'] = shift
                    break
            if accepted is not None:
                result.append(accepted)
                break
    return result
