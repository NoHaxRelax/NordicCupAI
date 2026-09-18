"""Tiny deterministic contracts for v4 target-chain semantics."""
from score_receipt_v4 import TargetChain


def feed(rows, *, guides=(1,), bait=0, unqualified=()):
    chain = TargetChain(bait)
    result = None
    for time, target, physical in rows:
        result = chain.update(time=time, target=target,
                              eligible_guide_ids=set(guides), physical=physical,
                              guide_chase_qualified=time not in unqualified) or result
    return result


def contained_tail(start, target=0, ticks=20):
    return [(start + index / 10, target, True) for index in range(ticks)]


def test_distant_physical_delay_with_continuous_bait_allowed():
    rows = [(0., 1, False), (.1, 0, False), (5., 0, False)]
    rows += contained_tail(5.1)
    assert feed(rows)["handoff_time"] == .1


def test_active_none_or_other_breaks_chain():
    for broken in (None, 9):
        rows = [(0., 1, False), (.1, 0, False), (.2, broken, False)]
        rows += contained_tail(4.0)
        assert feed(rows) is None


def test_rest_continues_but_cannot_initiate():
    continuing = [(0., 1, False), (.1, 0, False), (.2, "resting", False)]
    continuing += contained_tail(.3, target="resting")
    assert feed(continuing) is not None
    initiating = [(0., 1, False), (.1, "resting", False)]
    initiating += contained_tail(.2, target="resting")
    assert feed(initiating) is None


def test_long_active_guide_gap_rejected():
    rows = [(0., 1, False), (3.2, None, False), (3.3, 0, False)]
    rows += contained_tail(3.4)
    assert feed(rows) is None


def test_living_guide_is_neutral():
    rows = [(0., 1, False), (.1, 0, False)] + contained_tail(.2)
    # The state machine has no death/alive input by design.
    assert feed(rows) is not None


def test_identified_reserve_guide_can_handoff():
    rows = [(0., 8, False), (.1, 0, False)] + contained_tail(.2)
    result = feed(rows, guides=(1, 8))
    assert result is not None and result["handoff_guide_id"] == 8


def test_pivot_branch_guide_label_is_not_handoff_evidence():
    rows = [(0., 1, False), (.1, 0, False)] + contained_tail(.2)
    assert feed(rows, unqualified=(0.,)) is None


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} v4 target-chain contracts passed")
