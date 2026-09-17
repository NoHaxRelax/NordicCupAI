from replaceable_sites.selector import SiteSelectionError, enumerate_sites, select_site


def _map(obstacles, width=600, height=600):
    return {"width": width, "height": height,
            "obstacles": [{"x": x, "y": y, "width": w, "height": h}
                          for x, y, w, h in obstacles]}


def test_compatible_site_and_second_entry():
    site = select_site(_map([(100, 300, 40, 80), (155, 300, 40, 80)]))
    assert {"mouth", "inward", "cross", "goal", "far", "hold", "gap",
            "overlap", "obstacle_indices", "axis"} <= site.keys()
    assert site["gap"] == 15 and site["bait_depth"] == 5
    assert site["second_access_clear"] is True
    assert site["geometric_replacement_access_only"] is True


def test_dead_opposite_entry_is_rejected():
    # A third obstacle seals only the opposite end; one-mouth geometry remains
    # visible when the replacement requirement is disabled.
    m = _map([(100, 300, 40, 80), (155, 300, 40, 80),
              (130, 385, 35, 20)])
    assert enumerate_sites(m, require_second_access=False)
    assert not enumerate_sites(m)


def test_native_boundary_rectangles_participate_as_obstacles():
    obstacles = [(0, 0, 600, 30), (0, 570, 600, 30),
                 (0, 0, 30, 600), (570, 0, 30, 600),
                 (45, 300, 40, 80)]
    old = enumerate_sites(_map(obstacles), require_second_access=False)
    legacy = enumerate_sites(_map(obstacles), require_second_access=False,
                             allow_offset_approach=False)
    assert not any(site["boundary_indices"] for site in legacy)
    boundary = [site for site in old if site["boundary_indices"]]
    assert boundary and boundary[0]["offset_approach"]
    assert any(site["boundary_indices"] for site in enumerate_sites(_map(obstacles)))


def test_bad_payload_and_no_site_contract():
    try:
        select_site({"width": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("bad payload accepted")
    try:
        select_site(_map([]))
    except SiteSelectionError:
        pass
    else:
        raise AssertionError("missing site accepted")
