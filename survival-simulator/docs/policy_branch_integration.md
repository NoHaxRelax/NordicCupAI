# Survival policy branch integration

This document records the earlier `803bdd57` import. The later user-authorized
Lucas `53f1a4c7` and C++ engine integration is documented in
[Lucas/fastsim integration](lucas_fastsim_integration.md), including executed
checks. Its newer guide delivery and corner fallback behavior supersedes the
earlier behavior described below.

Reviewed 2026-09-18. Remote branches were fetched from `origin`; the production
source selected for this integration is
`survival-simulator/lucas-experimental` at
`803bdd57b860416d6ee189495c2b93e59f0ad6e3` (20:00:15 +02:00).
The branch advanced during review, so its later navigation/corner changes
were included after the first import of `6bebca41`.

**No simulation, test, optimizer, service, dependency installer or cloud Pod
was started.** Files, history, diffs and hashes were inspected. Added and
imported regression tests remain unexecuted. This is source integration,
not evidence of better whole-game scores.

## Branch comparison

| Branch | Reviewed head | Finding and decision |
| --- | --- | --- |
| `survival-simulator/lucas-experimental` | `803bdd57` | Latest development of the integrated native colony. Imported its production modules and wired the new functions into current runners and tuning. |
| `survival-simulator/entrapment-9059-benchmark` | `dfef3c5a` | Frozen reference benchmark and completion-status viewer updates. Preserve benchmark sources; import the viewer's unfinished-run labels. |
| `survival-simulator/entrapment` | `91437ef0` | Earlier guide/delivery work already represented in the later integrated lineage. No second coordinator added. |
| `survival-simulator/oscar-orchard-population` | `f6ab0357` | Its Orchard source has the same Git blob as both the local policy and Lucas's survival module: `7c2b4b45d2d14d264f77b3bab92769754c7ebb26`. No newer Orchard algorithm was found. |
| `survival-simulator/oscar-trapper` | `2c4b1195` | Alternative older integration built around Society and its own estimator/manager. Includes both observed and oracle development paths; not a drop-in update to the Orchard colony. Not imported wholesale. |
| `survival-simulator/entrapment-AI-attempt`, `survival-simulator/lucas-trap-slopsesh1` | `df3ef96d` | Earlier experimental/handoff lineage, already superseded by the later native integration. |
| `codex/survival-trap-handoff-2026-09-17` | `6a21ffec` | Earlier trap research rather than a newer production coordinator. |

Unrelated medical/drone changes and large research recordings were not brought
into the active simulator. The existing repository cleanup and Runpod search
work were retained. This is a selective source integration in the current
working branch, not a wholesale merge of unrelated branch history.

## Current runtime

`models/core.py` coordinates three modules:

- `models/exploration/`: public-observation localization, map sharing and exploration.
- `models/survival/`: the existing Orchard food and population policy.
- `models/entrapment/`: trap geometry, guide navigation/tracking and bystander avoidance.

`run.py trapping` now selects `scripts/trapping_game.py` and this coordinator.
`run.py serve` selects the same coordinator. The no-predator runner and its
observed-bounds adapter import the relocated, unchanged Orchard implementation.
The tuning worker's baseline and experimental subclass both use the current
coordinator. Its saved protocol identifies `models.core.EntrapmentPolicy` and
uses a new protocol version/source fingerprint.

The graphically recorded runner now has an independent `--policy-seed`, default
0; its map seed is not passed into the policy. The optimization worker retains
its spawned JSON-only observation boundary. Replay and evaluator data remain
outside policy inputs. No oracle world or hidden predator identity is used.

## New behavior and integration adjustments

1. **Ordinary-agent avoidance.** Upstream added local steering around observed
   predator hearing/vision/contact zones and the estimated bait area. It checks
   observed edges and suppresses births while avoiding. Guides and bait retain
   their dedicated controllers. Its default distances and steering resolution
   are unchanged; `bystander.*` exposes these settings and an enable switch to
   the optimizer. Turning all experimental additions off keeps the current
   avoidance rather than silently returning to the old 130-unit override.
2. **Observed newcomer association.** Upstream's guide tracks successive local
   sightings and avoids acquiring an already held predator. The colony used to
   always pass `target_predator`, bypassing that new association path. The
   coordinator now opts into association explicitly. Its observed target is an
   initial hint outside the held group; subsequent association follows the
   guide's own observation history. Overlapping sightings remain ambiguous.
3. **Delivery handoff.** Upstream now requires the selected predator to be sensed
   within the bait's target hearing distance before the guide stops for sacrifice.
   Merely reaching the handoff point is insufficient.
4. **Navigation recovery.** A guide displaced close to a wall can use an
   agent-width route to rejoin a position from which predator-width navigation
   is possible, instead of remaining blocked at the planner's starting point.
5. **Trap geometry.** Ordinary gap acceptance is now 10.1-19.9 units. The new
   corner-pocket selector checks contact exclusion and rear access on supplied
   geometry. It stays disabled by default. The coordinator can add these sites
   to normal gaps with `include_corner_pockets=True`; the recorder exposes
   `--corner-pockets`, and the optimizer compares the `corner_pockets` feature.
   It does not replace normal gap selection with a corner-only search.
6. **Viewer status.** The benchmark viewer now reads `final_status.json` and
   labels unfinished games, carrying the latest completed/incomplete reporting
   through to the existing viewer command.

The optional feature suite now has eleven additions, producing 25 initial
control/feature combinations before parameter tuning. Raw simulator mechanics
are unchanged. The branch's lab/fixture results are not treated as proof that
the observation-only colony survives the full 3,000 seconds.

## Frozen reference and recovery

All 39 source files in `entrapment_native_seed0/manifest.json` were compared
by SHA-256 and remain unchanged. The old flat modules and `models/nikolaj/`
therefore remain reference-only dependencies of `run.py benchmark` and the
historical `scripts/entrapment_game.py`. Current runners use the new directories;
[the module guide](../models/README.md) makes this distinction explicit.

The earlier files touched by integration and exact branch source exports are
saved locally under the ignored directory
`runs/branch-integration-20260918/`. The larger earlier cleanup backup is also
retained. Nothing was committed or pushed as part of this update.

Source inspection found no imports of the old reference policy in the current
coordinator, modular policies, observation adapters, recording runner or tuning
worker. Regression cases cover the coordinator/guide association connection,
held-crowd confusion, sensed handoff proximity, blocked-start recovery, optional
corner selection, bystander steering and the configurable control's default
behavior. Their runtime outcome is unknown until execution is authorized.
