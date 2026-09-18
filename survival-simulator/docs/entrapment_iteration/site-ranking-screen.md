# Screen-only static site-ranking audit

This audit uses only the 276 completed `phase=screen` cases from
`site-coverage-baseline20`, plus the 20 static inventories and the precomputed
screen selection file. Held-out validation outcomes were not downloaded or
opened. One map had no candidate; the other 19 maps contributed 46 candidate
sites with six screen encounters each.

## Static findings

The clearest interpretable split is boundary support. Twenty sites using a map
boundary as one channel support passed 102/120 screen encounters (85.0%). The
26 sites supported only by interior obstacles passed 99/156 (63.5%). This is a
development association, not an independent estimate. A boundary-supported
channel plausibly removes geometry behind one support and constrains escape
directions without relying on where the site lies relative to map center.

Approach geometry points in the same direction but is weaker. Across sites,
screen passes correlate positively with minimum non-supporting-obstacle
clearance along the runup-to-handoff line (Pearson 0.23) and negatively with
the number of non-supporting obstacles within 100 units of that line (-0.25).
Ranking solely by either measure selected 84/114 and 87/114 screen passes,
respectively, so neither is strong enough to justify a more complex score.
Rear clearance, overlap, gap, and approach offset each had weaker marginal
correlations. The earlier centrality experiment's regression is respected:
the candidate below contains no center distance or central-site preference.

## Frozen candidate

The isolated scorer in `candidate_site_rank.py` is lexicographic:

1. prefer a site with any `boundary_indices`;
2. among equal boundary classes, prefer the wider `gap`;
3. retain `overlap` as the final tie breaker.

Boundary support is the evidence-backed change. Wider gap is a narrow,
predeclared tie breaker motivated by physical channel tolerance; its apparent
screen benefit is development-selected and needs validation. The implementation
uses only fields already derived from observed static geometry.

On the screen sample, choosing inventory index 0 passed 76/114 encounters.
The frozen candidate rank passed 93/114. The per-map screen oracle passed
100/114. The candidate changed eight of 19 eligible map choices: it improved
six maps, tied one, and lost one pass on one map. It matched the best observed
screen pass count on 15/19 maps. These comparisons reuse the same six outcomes
that motivated the rank and therefore make no reliability claim.

All raw screen counts were independently regrouped from `screen-results.json`
and exactly matched `selection.json`. The frozen scorer SHA-256 is
`9ae5fd654c7c1c15e18818d574e41c3c55dc4d0c1d70d06b689ada64915c41dc`.
Detailed features and choices are in `analysis.json`.

## Prospective validation plan

Freeze this exact rank before evaluation. On a new map list that does not
overlap these 20 development maps, enumerate candidates from static geometry
only. For every map having at least two candidates, compare the frozen current
choice and candidate-ranked choice on identical independently generated
encounter seeds. Include single-candidate and no-site maps in the overall
accounting without pretending ranking can change them.

Report paired map-level wins, losses, ties, strict success totals, and failure
types (`initial_predators_escaped`, delivery failure, and wrong side). Keep the
existing success gate unchanged. A useful first checkpoint is at least 20 new
multi-candidate maps with several encounters per chosen site; expand rather
than promote if the interval on the paired difference remains wide. Do not tune
gap thresholds, boundary weights, or clutter terms after opening those results.

The already-recorded held-out validation cases in the remote batch were run for
the screen-selected site, not necessarily both choices from this frozen rank,
so they cannot provide a clean paired test of this candidate on changed maps.
