# Real-map intake harness

This directory supplies a generated-map fixture and evaluator for an
observation-only guide policy. It uses the unmodified vendored
`SimulationCore`, including the complete generated biome grid, obstacles,
trees, fruit, and native future spawning.

The qualifying setup is `station_bait=False`: two agents and every initial
predator are independently sampled across the full obstacle-valid map. A
policy must move its designated bait to its selected site, search for a
predator, engage it through native observations, and guide it to that site.
`--station-bait` is an explicitly arranged diagnostic mode for isolating the
guide route; its receipt and replay metadata disclose the exception.

Policy contract:

```python
policy = PolicyClass(static_map, bait_id=0, guide_id=1)
site = policy.site
actions = policy.act(native_dto_list, public_sim_time)
```

`static_map` is a detached JSON payload containing exact arena dimensions,
indexed obstacle rectangles, and a losslessly compressed per-pixel terrain
grid. `site` must contain `mouth`, unit `inward`, `cross`, `goal`, `gap`,
`overlap`, and `obstacle_indices`. The harness checks that the map payload is
not mutated. No predator coordinates, IDs, energy, rest state, or fixture
starts cross the policy boundary. Hidden state is used only after actions for
evaluation. Bait deployment is reported independently: deployment requires the
bait center to remain within 6 units of `site.goal` for one continuous second.
Guided delivery requires that same tracked predator to have appeared in the
guide's native DTO, actively selected the guide as its target, and subsequently
remained physically at the site for two seconds while resting or targeting the
bait. Overall success also requires no later physical/target loss and all
tracked predators present and contained through the final 30 seconds.

If a generated map has no eligible policy-selected gap, the harness writes an
`unsupported_map_no_eligible_site` setup receipt with its map seed and static
map hash. No simulation has begun in that case, so there is no replay. These
receipts must remain in the denominator of any map-seed result summary.

`--mode encounter_child_guide` evaluates the encounter-triggered workflow.
Initial bait, parent, and predator starts remain independent. Agent 1 may
request native reproduction only after a native Predator DTO. The policy may
keep the parent as guide, assign the native near-parent child, or reassign the
role later from ordinary observations. Receipts record fruit consumption, the
tracked encounter, spawn requests, actual child IDs and birth locations,
target switches, guide-role epochs, deaths, and conditional delivery.
A seed with no tracked encounter is retained in the requested-seed denominator
but labeled `no_tracked_encounter_excluded` outside the conditional delivery
denominator.

`--mode oracle_approach_child_guide` keeps all initial starts random but uses a
disclosed setup oracle to move only the parent toward the initial tracked
predator until the parent's first ordinary Predator DTO. The receipt records
the handoff time and oracle tick count. The setup movement uses a radius-5.01
visibility graph over the permitted static obstacle map so a blocked direct
line does not waste a pilot. From the handoff instant onward the oracle is
retired permanently; policy actions use only native DTOs, public time, and the
static map. This mode isolates post-encounter child spawning and delivery from
predator discovery.

Policies may explicitly hand the guide role to a native child using
`active_guide_id`, `guide_role_ids`, and a `lineage_guide_role` event derived
from ordinary agent DTOs. In that case the evaluator's chase prerequisite
follows the declared active lineage guide for each tick. Every new role epoch
must obtain fresh pursuit evidence before it can satisfy guided delivery.
Receipts retain the role timeline, pursuit evidence, explicit parent/child
roles, and deaths; no hidden predator target is passed back to the policy.

All living agents are refilled to maximum energy immediately before each
policy call. Predator energy and rest state are untouched. Native predators
therefore begin asleep at zero energy and wake through the ordinary simulator
cycle. Ambient predators may spawn normally and are reported separately.

Every run records the initial state and every 0.1-second simulation frame with
`ReplayRecorder(every=1)`. Native images are enabled by default and are
required for visual qualifying evidence. `--state-only` is reserved for an
explicitly nonvisual diagnostic; it retains every state frame but is not a
visual result. A deterministic rerun made to add native images must use
`--reproduction-of <prior-run-id>` so metadata states that the new replay is a
reproduction rather than reconstructed footage. A stop sentinel ends the run
and still saves the partial replay.

After each receipt is safely written, the harness makes a best-effort call to
`research/real_map_visualization/build_manifest.py`. Viewer failures or
timeouts are reported and never invalidate the saved simulation artifacts.

The bundled `SmokePolicy` only verifies the harness contract; it is not an
intake candidate. Example:

```bash
/home/Ucals/projects/NordicCupAI/.venv/bin/python \
  survival/research/real_map_intake_sol/run.py \
  --policy smoke_policy:SmokePolicy --seconds 0.3 --map-seed 5101 \
  --fixture-seed 9101
```
