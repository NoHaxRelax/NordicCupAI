"""Small pure contracts for the strict per-site target-chain wrapper."""
from score_any_bait import TargetChain, _physical


def feed(chain, rows, guides={1}):
    result = None
    for time, target, physical in rows:
        result = chain.update(time=time, target=target,
                              eligible_guide_ids=guides,
                              physical=physical,
                              guide_chase_qualified=True) or result
    return result


def tail(start, target, ticks=20):
    return [(start + i / 10, target, True) for i in range(ticks)]


def test_only_actual_bait_chain_confirms():
    rows = [(0., 1, False), (.1, 2, False)] + tail(.2, 2)
    assert feed(TargetChain(2), rows)["handoff_guide_id"] == 1
    assert feed(TargetChain(0), rows) is None


def test_switching_baits_breaks_site_chain():
    rows = [(0., 1, False), (.1, 2, False), (.2, 3, False),
            (3.2, 3, False)] + tail(3.3, 2)
    assert feed(TargetChain(2), rows) is None


def test_native_rest_can_continue_same_site_confirmation():
    rows = [(0., 1, False), (.1, 2, True)] + tail(.2, "resting")
    assert feed(TargetChain(2), rows) is not None


def test_site_physical_geometry():
    site = {"mouth": [10., 10.], "inward": [1., 0.]}
    assert _physical({"x": 5., "y": 10.}, site)
    assert not _physical({"x": 21., "y": 10.}, site)  # too far inward
    assert not _physical({"x": -70., "y": 10.}, site)  # outside radius


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} multi-site scorer contracts passed")
