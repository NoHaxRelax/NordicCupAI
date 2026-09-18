# Observation-only colony integration

This combines **Nikolaj's exploration and shared map**, **Oscar's orchard population policy**, and **our entrapment detector and guide** on `survival-simulator/entrapment`.

The policy accepts only ordinary per-agent observation dictionaries and simulation time. The runner uses the unmodified native `SimulationCore`: default starting population, natural predator spawns, normal energy consumption, aging, reproduction, fruit growth, and tree loss. There are no placed predators, immortal bait, teleports, full-map inputs, or disabled mechanics.

## Who decides what

| Component | Source | Responsibility |
|---|---|---|
| Exploration, map estimation, boundary anchoring | NikolajHJ's `challenge1_nikolaj`, `77eb2a21817231d6c4f8812a8e469cec28df229f` | Discover geometry using normal observations and executed actions |
| Gathering and population selection | `survival-simulator/oscar-orchard-population`, `f6ab0357869c114b7f6bd0b1eb93f907b3747da1` | Orchard posts, fruit timing, fitness-based reproduction |
| Trap acceptance | Our `guide_lab_sites.py`, copied to `models/entrapment_sites.py` | Gap 10.1–19.1, overlap ≥10.3, bait depth 5, clear predator approach and opposite-end replacement access |
| Guide movement | Our existing `models/my_guide.py` and helpers | Follow/reacquire an observed predator, travel to the bait, hold at delivery |
| Colony roles | `models/entrapment_policy.py` | Select bait/replacements, assign guides, avoid guided predators, feed back actual actions |

**Nikolaj's policy does not select traps or set their conditions.** His harvesting subsystem is disabled. `models/observed_trap_sites.py` adapts the estimated map to **our** existing detector, using the same thresholds and handoff checks as the earlier guide experiments.

Interior obstacle rectangles require four observed sides. Arena walls are included only after their faces are observed; their 30-unit thickness is a public generator constant. Unobserved boundaries are not inserted. Map dimensions must have been inferred by the mapper before trap selection.

## Roles and replacement

The colony uses Nikolaj's exploration until our detector finds a candidate. It then uses Oscar's orchard decisions, with guide and bait roles taking priority. The mapper continues updating throughout the game.

Agents older than 55 seconds or showing observed senescence are preferred for bait and guide work. Bait candidates must have enough energy for a conservative route through the rear entrance and at least 15 further useful seconds. Lifespan estimates assume the earliest possible senescence threshold, normal passive loss and increasing age loss; they do not read hidden maximum age or assume future fruit.

Replacement dispatch compares the active bait's conservative remaining life with estimated travel time plus a 20-second margin. A replacement reaches the same bait location before promotion. The previous bait remains stationary and dies naturally. When no candidate is viable, the controller cannot guarantee continuity; the replay and summary expose gaps rather than masking them.

Predator observations have no public IDs. Tracks are therefore associated by estimated position, group and time. An already assigned track keeps its guide. Unassigned sightings choose a nearby observing agent, preferring old agents; an isolated observer can take the role immediately. Dead guides release the assignment. Gatherers avoid nearby observed predators, including those another agent is guiding.

Regular reproduction prefers young gatherers/explorers, using the source policies' fitness decisions. When fewer than two young workers remain, an emergency parent is selected: an eligible young worker first, otherwise a fit old worker with enough energy to retain more than 100 after reproduction. Bait, guides and fleeing agents are excluded. This fallback was added after the first full-game run exposed the original strict age cutoff's failure to renew an aging colony. Policy randomness has a fixed seed independent of the world's generation seed.

Both Nikolaj's and Oscar's odometry receive the actions actually executed, including guide/bait overrides. The guide accepts an optional `context.target_predator` that must be an ordinary observed predator; callers omitting it retain the earlier guide behavior.

## Run and inspect

From the repository root:

```bash
.venv/bin/python survival-simulator/scripts/entrapment_game.py \
  --seed 0 --seconds 3000 \
  --out survival-simulator/logs/entrapment_game/my-run

.venv/bin/python survival-simulator/scripts/entrapment_viewer.py \
  survival-simulator/logs/entrapment_game/my-run --port 9059
```

Open <http://localhost:9059>. The viewer provides a timeline, one-tick stepping, event jumps, roles, each agent's inputs and actions, guide diagnostics and estimated poses. It reuses the native game drawing methods; it does not run an approximate simulator. Each recorded tick can be rendered on demand. The full-world spectator replay is evaluator output, never policy input.

The run folder contains:

- `manifest.json`: seed, native defaults, source hashes and policy input contract.
- `summary.json`: progress/final metrics and role events.
- `chunks/*.json.gz`: **every tick**, with ordinary policy input, output actions, policy diagnostics and separate spectator state.
- `background.png`, `static.json`: spectator rendering assets.

Large replay files remain under ignored `logs/`. Output paths with an existing summary are rejected to preserve previous runs.

## Current limitations

- A candidate is valid against **observed** geometry. New observations can invalidate it. Unseen obstacles and mapping/localization errors remain real failure modes.
- Guides currently snapshot known static edges on assignment to keep the existing guide's coordinate frame stable. Later wall discoveries are not yet integrated into that guide's route cache. Native collisions still apply.
- Position-based predator association is ambiguous when predators overlap; one logical track per predator is not guaranteed in a crowd.
- Arrival at the delivery point is an event, **not proof of capture**. “Within 40 units of bait for 30 seconds” is a separate spectator proximity proxy, not permanent retention or a full capture success rate.
- Bait occupancy and reported gaps use the controller's estimated arrival location. Energy values and predator distances in spectator metrics are native world state. Localization error could overstate correct positioning inside the gap.
- Guide survival does not yet automatically release a guide back to fruit gathering after a confirmed transfer; holding until eaten remains allowed.
- Oscar's original research used predator-free evaluation. Its fruit and late-game survival limitations remain, and diverting workers to bait/guiding increases pressure on the colony.
- This is a first integrated full-game test, not a new 1,000-map reliability evaluation. The previous fixed-fixture statistics do not establish this controller's full-game reliability.

## Verification

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m unittest discover \
  -s survival-simulator/scripts -p test_entrapment_policy.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m unittest discover \
  -s survival-simulator/scripts -p 'test_guide*.py'
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m unittest discover \
  -s survival-simulator/scripts -p test_predator_following.py
```

The integration checks cover conservative energy/lifetime estimates, observed-only rectangle construction, our trap criteria, overlapping replacement, duplicate sighting assignment and associated-predator selection. Existing guide tests preserve the earlier default behavior.

## First native full-game result: seed 0

Replay: `logs/entrapment_game/native-seed-0-v1`, served at <http://localhost:9059>.
Compact results and source hashes are in `docs/entrapment_native_seed0/`.

| Measure | Result |
|---|---:|
| Our trap discovered from observations | 26.5 s |
| First bait arrival | 29.0 s |
| Guide assignments | 32 |
| Assignments reaching delivery point | 15 (46.9%) |
| Overlapping bait replacements | 23 |
| Maximum predators simultaneously within 40 units of bait for 30 s | 11 |
| Naturally spawned predators | 11 |
| Colony extinction | 1,073.6 s |
| Recorded bait absence after first arrival | 8.0 s, at the end |
| Peak / total observed agent population | 28 / 144 |
| Native score | 1,116.60 |
| Runtime | 354.6 s |
| Saved frames | 10,737, every tick |

The 15/32 ratio measures arrival events, not independent predator-capture trials. Predators can require multiple guides, and proximity to bait alone does not prove permanent capture. The bait service ran for most of this game, but colony survival and continued replacement failed near the end. The final gatherer was old and still had 744 energy at 1,000 s; the strict age-55 reproduction cutoff prevented renewal. **This does not meet whole-game reliability.** The current emergency fallback addresses that specific policy restriction.

The post-run audit verified the physical bait location (maximum projected goal error 0.099 units), 24 distinct bait agents and the final 8-second gap. No predator occupied the rear entrance under the recorded evaluator criterion. All 10,737 frames were contiguous. Replaying all 10,736 public action inputs against the boundary-fixed version produced identical actions; this comparison predates the subsequent emergency-reproduction change.

After this run, a boundary reconstruction edge case was fixed: all four outer arena edges must not be mistaken for one enormous solid rectangle. A regression test covers it. The exact source used during the run is preserved in `logs/entrapment_game/native-seed-0-v1/source_snapshot.tar.gz`, verified against every recorded source hash. The audit script verifies saved frames; `--recheck-policy` additionally compares public inputs with the current policy and reports any divergence:

```bash
.venv/bin/python survival-simulator/scripts/entrapment_audit.py \
  survival-simulator/logs/entrapment_game/native-seed-0-v1
```
