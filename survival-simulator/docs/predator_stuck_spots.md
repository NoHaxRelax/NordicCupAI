# Analyze all predator trap locations

`scripts/predator_stuck_spots.py` analyzes every saved confinement event and
unflagged control in the 10,000-game Runpod dataset. It runs locally using NumPy
and the standard library. It does not require pygame, a native build or a pod.

From the repository root, using the existing environment:

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" survival-simulator/scripts/predator_stuck_spots.py --workers 8
```

By default, input is `survival-simulator/runs/predator-stuck-diagnostics-10000`
and output is `survival-simulator/runs/predator-stuck-spots-10000`. Override these
with `--input` and `--output`. The program streams the compressed archives and
buffers only a small number of games. It never expands the entire dataset to
disk. Completed game features are saved atomically in `game-features`, so the
same command resumes an interrupted run. Changed extraction code or settings
require a new output directory to prevent mixing incompatible measurements.

Use `--report-only` to rebuild CSV tables and reports from the completed cache.
Use `--limit-games 10 --output <separate-pilot-directory>` for a small test;
limited runs are explicitly marked partial. An uncompressed scanner output
directory, such as the existing pilot, also works as input.

## Places predators enter but do not leave

The `--entrances` extension checks recorded entrances and tests whether an exit
route exists under the native walking geometry. It reuses the completed base
feature cache and streams the original approach traces and biome maps:

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" survival-simulator/scripts/predator_stuck_spots.py --entrances --workers 8
```

Results are saved separately in `runs/predator-stuck-entrances-10000`, including
[the entrance report](../runs/predator-stuck-entrances-10000/entrance_report.md),
`entry_spots.csv` and `entry_traps.csv`. Use `--entrances --report-only` to
regenerate reports from that cache. Custom datasets need `--base-features`
pointing to the matching completed spot analysis. The original output remains
available in its original directory.

A confirmed entrance requires an observed, accepted, unclamped step from
outside the original 15-unit circle to inside it, with both endpoints legally
outside collision rectangles. At least 60 seconds of continuous residence must
follow. A predator trapped from birth is excluded unless a later legal entry
was recorded. Missing approach evidence is marked unknown rather than inferred
from the spawn position. The code also records whether the crossing segment
itself intersects an expanded obstacle, because the native engine checks only
endpoints and can accept a step that clips a corner.

Each confirmed entry is separated into later observed escape or no departure
observed before game end. For the latter, the analyzer distinguishes:

- **A legal exit route exists:** a fully checked sequence of native-length steps
  reaches outside the circle, using different heading choices. The normal
  predator policy still did not leave during the recorded observation.
- **No legal first step exists at any heading:** the entire step circle is
  blocked by the union of expanded obstacles. Angular intervals and their
  boundary points are checked, with a positive margin, rather than assuming
  that a finite grid of blocked directions proves impossibility.
- **Unresolved:** no route was found within the bounded search. This is not a
  proof that a route is impossible.

The first exit test reverses the recorded steps, verifying the step length
against the biome at each reversed position. Different biome penalties can
make a forward step impossible to reverse exactly. If necessary, a bounded
search tries other native-length steps with arbitrary headings. Defaults are
256 expanded nodes, five-degree directions plus narrow-opening samples, and a
0.5-unit visitation grid. Coordinates are never snapped; the grid only limits
exploration. Route witnesses are stored in the per-game compressed feature
files. Reversed-route witnesses reference the archived trace and entry tick;
searched routes include their full point sequence.

These are movement-geometry tests under alternate heading choices, not a claim
that the native controller would choose those actions. Follow-up remains
censored at 600 seconds. Increase `--escape-max-nodes` or decrease
`--escape-angle-step` in a **new output directory** for a broader search.

The completed extension checked all 248,090 findings. It confirmed 231,325 legal
entries: 231,001 had no observed departure before game end, and 324 later
escaped. The no-departure cases occupy 52,999 approximate spots; 25,062 of those
spots received at least two such predators. Of the no-departure cases, 212,037
had at least 60 additional seconds of follow-up after the confinement detector
triggered.

A legal exit route under alternate headings was verified for 230,880 of the
231,001 no-departure cases. No entered case was proven to lack every legal first
step. The remaining 121 cases, spread across 48 approximate spots, are unresolved
at the default search budget. They must not be described as proven one-way
geometric traps. Of all confirmed entries, 211,590 had clear crossing segments
and 19,735 clipped an expanded obstacle between legal endpoints.

Additional tests: `python -m unittest discover -s survival-simulator/scripts -p test_predator_stuck_entrances.py`.

## Completed analysis

The full run analyzed all 248,090 findings and 9,993 unflagged controls from
10,000 maps, grouping findings into 69,630 approximate spots. Reconstruction
of the late movement's collision outcome matched every saved log: zero
mismatches and zero borderline windows. Nine geometry, recurrence and
held-out-selection tests passed.

Every moving-confinement case had two expanded collision rectangles within
15 units at the measured late pose, including at least one internal obstacle.
This is an observed requirement, not a sufficient condition for capture.
65,874 moving cases lacked nearby perpendicular collision faces, and only
2.68% pointed within one degree of a map diagonal. Corners and exact diagonal
headings are therefore not required. 229,866 of 231,516 moving cases (99.29%)
showed a repeating position-and-heading pattern in the final active moves.

Of 16,139 fully blocked cases, 14,885 (92.23%) were deeper inside a single
expanded obstacle than their current step length. The whole step circle was
inside that obstacle, giving a geometric sufficient condition for blocking at
any heading. In 60 other fully blocked cases, a one-degree direction sample
found a legal endpoint missed by the native 10-degree candidate grid at the
measured pose. This establishes a possible step, not escape from the test circle.

The results also identify a spawn/collision mismatch in the native code:
spawning checks a square whose upper-left corner is `(x, y)`, while movement
uses `(x, y)` as the center and expands obstacles by the predator radius.
The analysis did not change this behavior or any movement rules.

## Measurements

Each finding and control gets an entry measurement and a late measurement
immediately before its final active move in the 60-second window. Measurements
include the pose, biome-scaled step length, signed clearances to the nearest
three square-expanded collision rectangles, nearby obstacle counts, overlap,
nearby perpendicular/opposing faces, world boundaries and corners. The program
checks endpoints in all 36 distinct native fallback directions and independently
compares the reconstructed all-blocked outcome with the movement log. Contacts
within 1e-8 units of a collision boundary are marked as borderline.

For the continuous range of headings, 360 one-degree samples describe the
fraction of legal endpoints; this is explicitly a sample, not proof that a tiny
opening does not exist. A separate conservative geometric test establishes
that an entire step circle lies strictly inside one expanded rectangle. That
test proves no heading has a legal step endpoint against that obstacle.

Trajectory measurements use every active tick in each detected interval. They
record movement branches, 90-degree turns, fallbacks, blocking, biome changes,
travel distance and approximate recurrence. Recurrence checks position and
heading modulo 2*pi separately: a position return alone can hide a changed
heading. The smallest accepted lag from 1 to 16 must match at least 99% of
comparisons in the last 128 active ticks, with tolerances of 1e-6 units/radians.
This is measured recurrence, not proof of an exactly periodic trajectory.

Approximate spots are connected components of late path centers within five
units on the same map. Their component ID links every predator to `spots.csv`.
The link distance is configurable using `--cluster-radius`. Connected components
can chain across a larger area, so component span is also reported. These are
practical groups of nearby cases, not exact unique attractors or a census of
every place on a map that could trap a predator.

## Conditions and controls

The report tests explicit candidate requirements, such as a corner, two nearby
obstacles, overlap, tight clearance, diagonal heading, deterministic avoidance
and repeated poses. It reports every hypothesis's coverage and counterexample
count separately for moving, mixed and completely blocked findings. The
counterexample table retains seeds, predator IDs, trace paths and clearances.

Simple geometry/pose rules are selected on seeds whose remainder modulo five
is nonzero, then evaluated on the remaining maps. No seed appears in both
sets. Rules use one predicate or two predicates joined by AND/OR; thresholds
are declared in the script. Selection balances case coverage, giving each
approximate spot equal total weight, against control specificity. Counts and
raw predator coverage are also reported.

The negative examples are the saved lowest-index unflagged predator's final
window from each game. They are selected controls, not random or matched
observations. Rule coverage and specificity describe this dataset, not the
population probability that entering a location will cause trapping. Late
geometry describes occupied traps; it does not by itself establish capture
probability from an arbitrary approach. A condition with zero observed
counterexamples is an empirical invariant, not a universal theorem.

## Outputs

- `report.md`: readable findings and rules evaluated on separate maps.
- `analysis.json`: full counts, hypothesis results, rules, distributions and provenance.
- `windows.csv`: every finding/control and its measured geometry and trajectory features.
- `spots.csv`: approximate locations, associated predator IDs and example trace paths.
- `conditions.csv`: every candidate requirement, by mechanism, with counterexample counts.
- `counterexamples.csv`: up to five examples per rejected requirement and mechanism.
- `coverage.json`: completed map, event and control counts and reconstruction mismatches.
- `settings.json`: extraction settings and source hashes; `progress.json`: current status.

The full analysis is linked from
[the generated report](../runs/predator-stuck-spots-10000/report.md).
The [original experiment report](predator_stuck_results_2026-09-19.md) describes
simulation parity checks, source provenance and the 15-unit/60-second detector.

Validation: `python -m unittest discover -s survival-simulator/scripts -p test_predator_stuck_spots.py`.
