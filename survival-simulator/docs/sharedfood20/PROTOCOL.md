# Shared information and economic decisions: third 20-family campaign

Branch: codex/sharedfood20. Parent: 45f4551, expanded_local_food winner from docs/localfood20/pod-7. No engine rules, observations, energy dynamics or hidden-state interfaces change. No official competition submission.

## Panels frozen before running

- Training: 200 seeds 17001–17200. Run unchanged expanded local food; checkpoint 250 seconds before extinction, preserving full engine/policy/RNG/memory with Linux fork. Exact baseline score AND extinction-time replay required. 32 trials per family (seeded, five random, 26 GP expected improvement), 20 families =128,000 continuations.
- After every winner is frozen: independent checkpoint validation on 200 seeds 18001–18200, initialized using the SAME expanded-food policy and checkpoint rule. Evaluate ALL winners and controls. Do not tune/select trials using validation.
- Fresh full-game test: 1000 seeds 19001–20000 per winner/control, starting at game time zero. Evaluate all 20 winners, expanded-food baseline, shared-food control (sharing only, no retuning), prior local-food baseline and scheduled-breeding baseline: 24,000 full games.
- For rare baseline lifetimes under 250 seconds, keep the prespecified seed with a time-zero checkpoint; report these counts in both checkpoint panels. Do not silently replace maps. If the baseline reaches the 3000-second horizon, stop and report: that is not an extinction checkpoint.
- Inputs scaled to [0,1], GP objective standardized; displayed scores remain original score units. Each family retunes vision/hearing/energy/speed breeding weights (four dimensions) plus 2–3 mechanism parameters. No convergence claim from 32 trials.

Report training gain versus baseline continuation, held-out continuation gain and paired 95% CI, training-minus-validation gap, full-game gain and paired CI, mean score/CI, policy and loop compute per tick, and predator/energy death counts. Training-minus-validation estimates optimism under the same initial-state procedure; validation-minus-full-game also changes deployment conditions and is not pure overfitting. All intervals pointwise, not multiplicity corrected.

## Shared map and observation provenance

Reuse the existing native orchard odometry/food map. Adapt Nikolaj's independently anchored-group merge and shared-rock matching from models/nikolaj/world_estimator.py at upstream 77eb2a21817231d6c4f8812a8e469cec28df229f (unchanged in fetched sim-optimization-no-trapping 9f9159a). The entire Python global planner is NOT transplanted. Native stone matching uses exact directed lengths, nonparallel endpoint consensus, a 128-candidate bound and ambiguity rejection. Excludes boundary faces/caps; 2-unit consensus tolerance and an anchored world-bounds check. Also adapt nonparallel-stone relocalization to correct collision drift before adding reports. Wall memory is capped at1024 edges, with original observation timestamps; each agent refreshes its nearest150 walls within350units once per second.

Additional exact shared fruit IDs provide two-point frame alignment; direct agent sightings already align groups. All anchored groups merge without requiring physical contact. Unsupported frame alignments remain separate until evidence arrives.

Trees/fruits/exploration/biome memory and allocation are shared within aligned frames. Static wall memory and current predator sightings are now shared too, without a communication-distance limit. Predator heading is reconstructed from the public relative-direction convention and transformed for each recipient; no hidden predator ID/energy/intent is read. Stale predator sightings are not propagated. Nearby reports are deduplicated within 5 units while own observations remain present. All agents' public statistics are already centrally available. Compile-time policy/engine isolation is retained.

## Economic reasoning

Agents are a cooperative team: competition for the same fruit is an allocation problem, not an incentive to maximize individual consumption. Compare marginal food value with travel, aging, congestion and danger costs. At full fruit maturity, each forest/grassland tree has nominal output 0.1 fruits/s ×60 energy=6 energy/s; swamp 4.8, desert 3.0. This is an upper expectation before tree death, blocked spawns, spoilage, early harvest and missed collection. A young idle agent consumes 1 energy/s; walking costs 0.05/unit plus time, sprinting has an extra 0.5/unit above walk speed. An aged agent also loses 0.1×age energy/s. Its private aging threshold is never read: use observed aging or the public threshold distribution. Fruit starts at20 energy, reaches60 after20seconds, and expires at about50seconds. Tree age is estimated from observation history, not engine age.

Food budgets sum estimated current fruit and near-term tree production. Capacity divides this by a tunable horizon with a 1.5-energy/s allowance and25 net reproduction cost (100 parent energy becomes75 child energy). This is a planning heuristic, not an exact carrying-capacity theorem. Allocation can auction fruit by useful energy minus travel cost per travel time, urgency and reproductive value. Post optimization tests nine local locations plus the current location, minimizing weighted collection distance with observed-wall costs. Relocation uses depleted local food and elapsed time; renewal uses remembered biome productivity and the known time-dependent spawn trend. Phase controls rely on public elapsed time, never future extinction.

## Families

1. shared_alerts: share information, tune ordinary evasion thresholds.
2. shared_one_tick: shared threats plus one-step escape approximation.
3. food_capacity: population capacity from observed food budget.
4. late_food_capacity: enable that capacity only after a tuned time.
5. fruit_auction: centrally allocate fruit by net value/travel time.
6. net_energy_fruit: reject trips with poor net energy or likely expiry.
7. congestion_pricing: penalize competing posts harvesting the same area.
8. safe_food: price observed predator exposure in food assignments.
9. rock_aware_posts: stand locations and detour costs account for rocks.
10. fruit_centroid_posts: choose a collection-distance-minimizing stand point.
11. leave_empty_area: relocate despite ordinary waiting when local food is depleted.
12. renewal_sites: value remembered productive biomes/food when waiting or exploring.
13. budget_breeding: change reproduction reserves with local food per agent.
14. aging_efficiency: prioritize food using expected aging drain and travel cost.
15. late_auction: switch allocation objectives late.
16. late_positioning: switch stand-position optimization late.
17. cooperative_harvest: combine fruit auctions and congestion costs.
18. safe_harvest: combine shared food danger, auctions and one-step escapes.
19. mobile_colony: combine depletion-triggered relocation, renewal and rock costs.
20. scarcity_economy: combine capacity, breeding reserves and allocation.

Use the same ten idle Oscar CPU pods, 32 workers for BO/full-game tests. Keep pods running after completion. Raw results, hashes, frozen parameters and source are committed/pushed before interpretation.
