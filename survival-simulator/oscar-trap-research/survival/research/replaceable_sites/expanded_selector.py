"""Opt-in frontier selector; frozen selector defaults remain unchanged."""
from .selector import SiteSelectionError, enumerate_sites as _enumerate

EXPANDED_DEFAULTS = dict(min_gap=10.1, max_gap=19.1, min_overlap=10.3,
                         bait_depth=5.0, require_second_access=True,
                         allow_offset_approach=True)


def enumerate_sites(static_map, **overrides):
    options = {**EXPANDED_DEFAULTS, **overrides}
    return _enumerate(static_map, **options)


def select_site(static_map, **overrides):
    sites = enumerate_sites(static_map, **overrides)
    if not sites:
        raise SiteSelectionError("static map has no expanded-frontier replaceable site")
    return sites[0]
