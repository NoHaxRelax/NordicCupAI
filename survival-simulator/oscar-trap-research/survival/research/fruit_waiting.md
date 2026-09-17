# Fruit maturation: when waiting helps, and what the agent can know

Waiting can triple a newly spawned fruit's nutrition from 20 to 60, but “wait twenty seconds after first seeing every fruit” is unsafe. First sighting does not reveal its age. The useful rule is to defer a **known recent spawn**, collect already mature or unknown-age fruit first, and abandon the wait when energy becomes scarce.

This study uses upstream source commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`, actual source movement, fruit growth/removal, tree turnover, automatic eating, aging, and reproduction. There are no predators in these experiments, so safe waiting is an assumption, not a demonstrated predator defence. No external validation was run.

## Observable rule and hidden-age comparator

The normal fruit payload contains type, distance and angle. It has no age, nutrition or ID. The observed policy reconstructs local positions from its own movement and associates fruit geometrically. A first sighting is treated as a recent spawn only if that position was within a region observed on the preceding tick and no matching fruit existed there. Otherwise its age remains unknown and the policy collects it immediately.

The experiment compares:

- **Immediate:** collect the available fruit without delaying for maturation.
- **Observed:** wait for twenty seconds of known recent-spawn age; unknown-age fruit is immediately eligible.
- **Oracle:** the same movement and reserve rules, but with explicitly supplied true fruit ages. This is a diagnostic comparator, not a deployable policy.
- **Blind:** wait twenty seconds from any first sighting. This is included only in finite-fruit probes to illustrate rot risk.

While waiting, the agent stays 15.5 units from the fruit. A fruit's radius grows from 5 to about 9, and an agent has radius 5; standing closer than about 14 can collect it automatically before maturity. The observed policy scans for other fruit and prefers fruit available sooner, so it can collect other ripe or unknown-age food while a tracked new fruit grows.

Waiting is aborted when energy cannot cover the remaining wait, travel and a 25-energy reserve. For a young agent and a fruit 30 units away, a full twenty-second wait therefore needs roughly 47 energy at decision time. After age 60 the policy conservatively budgets the possible old-age surcharge, because the actual senescence threshold is hidden. These thresholds are simple chosen safeguards, not optimized values. Low-energy agents may benefit from partial maturation waits that this conservative implementation does not attempt.

Local positions are initialized in an arbitrary frame at the first agent, and newborn positions/headings are reconstructed from public Agent observations containing IDs and relative directions. No true position corrections are supplied. The final patch comparison restricts foraging to a 250-unit radius around the initial frame inside an arranged clear forest patch. This keeps the test away from walls; it does not solve localization after collisions in a generated map.

## Finite-fruit results

The initial agent has 150 energy, lives in flat forest and has a senescence threshold of 90 seconds. Each comparison ends at the same 60-second horizon. A single fruit lies 30 units away; no other food or reproduction is present.

| Fruit condition | Immediate outcome | Observed waiting outcome | Hidden-age oracle |
| --- | --- | --- | --- |
| Fresh but already present in first observation | 20.2 nutrition; 109.1 energy left | Same: its age is unknown | 60 nutrition; 148.9 energy left |
| Fresh appearance in previously observed empty space | 20.4 nutrition; 109.3 energy left | 60 nutrition; 148.9 energy left | 60 nutrition; 148.9 energy left |
| First observed at age 40 s | 60 nutrition | 60 nutrition, collected immediately | 60 nutrition |
| First observed at age 49 s | 60 nutrition | 60 nutrition, collected immediately | 60 nutrition |

Thus the observation-based rule gained **39.6 stored energy at the same time horizon** for a known recent spawn. It did not pretend the initial unknown-age fruit was fresh. Blind waiting lost both the 40-second-old and 49-second-old fruit to rot and ended with only about 89.3 energy.

With initial energies 15, 30 and 45, the reserve rule collected the fresh fruit promptly instead of waiting. With 60 it safely waited and ended with 58.9 energy, compared with 19.3 for immediate collection. An already 100-second-old agent also collected immediately: its high passive drain made a full maturation wait unaffordable. These safeguards prevented the waiting rule from turning the tested low-energy cases into starvation-before-pickup cases, but they are not a proof against every environment.

## Economics

A fruit's value is approximately `min(60, 20 + 2 × age_in_seconds)`. It reaches full value at twenty simulated seconds and disappears around fifty, leaving about thirty seconds of mature shelf life.

- A newly appeared fruit gains 40 nutrition over its first twenty seconds.
- A young idle agent burns 20 energy during that wait, before movement and turning. The meal can therefore leave it about 40 above its pre-wait energy, versus about 20 from eating immediately.
- At a common later time, both agents have paid the same ordinary living cost, so waiting's advantage is the extra nutrition, less any differing actions or wasted capacity.
- If immediate collection would free the agent to obtain other fruit, that foregone collection must be charged against the gain. One additional mature fruit is worth 60, exceeding the 40 gained by waiting for one fresh fruit.
- An old agent at age 100 can burn about 11 energy per second. A single mature fruit cannot fund a twenty-second wait at that rate. Prefer a younger collector or early harvest when reserve is tight.
- Near maximum capacity, extra nutrition can be wasted. Deferring fresh fruit may conserve it for later or for offspring, but collecting more fruit still increases the source score even when its energy cannot be stored.

These are reasons to keep a maturation queue while doing useful work, rather than universally stopping beside one fruit.

## Native tree patch comparison

The final comparison uses preselected seeds 3, 11 and 29, a fixed 300-second horizon, 150 starting energy, three initially mature trees, source-native global tree creation/death and fruit generation, and reproduction either disabled or capped at three living agents. The world is all forest, with no predators or internal obstacles. It is a deliberately favourable controlled ecology, not the contest's generated mixed-biome map.

The first exploratory version allowed unrestricted travel and recorded 1.61 units of localization drift in one reproducing run near a boundary. Its results are preserved as `fruit-waiting-patch-uncapped-drift.json`; they are excluded from the final comparison. The final operating radius was introduced to address that measurement failure, before interpreting the final results.

The final eighteen runs had zero measured position drift and no unlocalized newborn decisions. Results at the common 300-second limit:

| Reproduction | Policy | Mean survival time, capped at 300 s | Runs still alive at 300 s | Mean score |
| --- | --- | ---: | ---: | ---: |
| Off | Immediate | 243.6 s | 1 / 3 | 249.781 |
| Off | Observed waiting | 217.0 s | 1 / 3 | 222.583 |
| Off | Hidden-age oracle | 291.9 s | 1 / 3 | 300.271 |
| On | Immediate | 300.0 s | 3 / 3 | 306.997 |
| On | Observed waiting | 300.0 s | 3 / 3 | 308.079 |
| On | Hidden-age oracle | 300.0 s | 3 / 3 | 308.371 |

With reproduction, observed waiting increased mean score by about **1.082, or 0.35%**, entirely in secondary scoring because every run reached the same survival horizon. It improved two seeds and regressed one. Without reproduction, it failed badly on seed 29: 168.6 seconds versus 253.8 for immediate collection. Slowing collection and changing which fruit to visit can outweigh maturation gains, particularly when old agents become expensive.

The seeds control initial conditions; births and mutations consume the environment's shared random stream, so reproducing policies can also induce different later tree/fruit histories. Three favourable controlled seeds cannot establish a competition advantage or statistical reliability. The hidden-age oracle is not uniformly best either.

**Recommendation:** retain maturation as an optional local rule for known recent spawns, with reserve and danger overrides, and keep collecting other food while they grow. Do not apply blind first-sighting waits or replace the working foraging policy solely on this small score difference. Reproduction and keeping young foragers mattered more consistently in this patch test than waiting itself.

A generated-map nursery wrapper was deliberately not claimed: the exploratory drift demonstrates that obstacle-aware localization and fruit association need additional work. The final controlled policy's assumptions are explicit and audited.

## Reproduction

```sh
survival/.venv/bin/python survival/research/fruit_waiting.py --phase all
```

`--phase finite` runs 46 isolated age, reserve and rot cases. `--phase patch` runs the eighteen controlled tree-patch comparisons. Optional `--workers 2` runs cases in parallel. Result files contain source/script provenance, collection times and ages, energy, score, births/deaths, waiting decisions and measured coordinate drift. The oracle's true ages are passed through a separate input; the observed policy never receives them.

All 64 final runs passed the accounting audit: measured position drift below one millionth of a unit, zero unlocalized decisions, harvested nutrition matching the fruit component of score, and stored food energy no greater than consumed nutrition. Removal classification respects the source's eat-before-rot order, so an over-age fruit eaten on its final tick is counted as food rather than rot.
