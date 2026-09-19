# Nikolaj's pushes checked after the frozen experiment

Fetched origin on 2026-09-19. Relevant branch:
`origin/sim-optimization-no-trapping`, head `9ea41240a518ac22dabf9ef0c108f5ed16fb3a82`.

- `9ea4124` (14:45 CEST): adds the all-C++ engine/policy backend to the optimizer.
  Commit reports 388.79 seconds down to 17.41 seconds for one full game, about
  22x faster. This is a runtime measurement, not a mean score benchmark.
- `7c3b8ba` (14:21 CEST): newer vendored Orchard, no-trapping campaign, and two
  correctness fixes: ExpertPolicy fleeing now avoids walls; Orchard evasion
  releases a fleeing/facing agent's fruit reservation.

Pushed score evidence found:

- `docs/orchard_reference_results.json` lists 2834.033, 2726.8146, and
  2037.9156. All three explicitly have **zero predators** and are labelled
  historical original Orchard runs, not rerun during cleanup.
- `fastsim/README.md` documents a one-seed native performance benchmark surviving
  to 2143.9 seconds. It does not state a >2000 mean score across maps.
- `docs/research_notrap_seeds_20260919.md` distinguishes upstream no-predator
  Orchard results from predator-enabled screening. It contains no verified
  full-horizon >2000 mean with predators.

Therefore I could not verify the claimed >2000 predator-enabled mean from the
results included in these pushes. An active campaign may have newer unpushed
results; no such results were assumed.

Relevance to this experiment: our native evasion override at
`nightsim/_npolicy.hpp` changes the action after fruit assignment and retains the
claim. Releasing it is a concrete follow-up candidate. This review did not change
any evaluated policy or the frozen baseline. The sixteen-family experiment did
not benchmark Nikolaj's newly pushed implementation.
