# Best no-trapping policies (campaign notrap-search-20260919)

The three best configurations found, ranked. All are for
`models.orchard_evasion_policy.OrchardEvasionPolicy` (coordinated orchard
foraging + own-sighting predator evasion, **no trapping**).

**Use rank 1 unless you have a reason not to.**

| Rank | File | Objective | Mean score | Mean survival | Worst survival | Reached 3000s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **1** | `rank1_pod-03.json` | **0.8383** | **2038.7** | 2046 s | 1875 s | 0 / 4 |
| 2 | `rank2_pod-04.json` | 0.8308 | 2002.7 | 2020 s | 1889 s | 0 / 4 |
| 3 | `rank3_pod-00-v2.json` | 0.8262 | 2020.2 | 2003 s | 1901 s | 0 / 4 |

Objective is the campaign's ranking rule,
`mean_survival/horizon + 0.25*worst_survival/horizon`. Every game ran the full
3,000 s horizon with natural predator spawning enabled (never disabled, never
forced), on the reference Python engine, over training seeds 0-3, with the
policy receiving only public per-agent observations and simulation time.

## How to load one

```python
import json
from models.notrap_config import validate, orchard_kwargs
from models.orchard_evasion_policy import OrchardEvasionPolicy

config = json.load(open('models/best_policies/rank1_pod-03.json'))
validate(config)
policy = OrchardEvasionPolicy(seed=0, **orchard_kwargs(config))
```

Each file is a complete configuration with four blocks (`orchard`, `evasion`,
`expert`, `planner`) and passes `models.notrap_config.validate()` unchanged.

## Read this before trusting the ranking

**The three are not meaningfully different from each other.** They span 0.8383
to 0.8262 on four training seeds. The project's own analysis concluded that
**at least twelve seeds** are needed to separate candidates from RNG noise, so
treat rank 1/2/3 as a tie and prefer rank 1 only as a tiebreak.

**Ranks 1 and 2 come from before two correctness fixes** (fleeing agents had no
wall avoidance; fleeing agents kept their claimed fruit reserved). Rank 3 is
from the post-fix run. They are not strictly comparable, and the fact that the
pre-fix configurations score no worse is itself inside the noise band.

**No configuration reaches the 3,000 s horizon (0 / 4 in every case).** That is
expected, not a failure: standing tree count halves roughly every 600 s, so by
t=3000 only about 2.5 trees remain, producing ~9 energy/s - barely one agent's
idle drain. Reaching the horizon is roughly a 1-in-60 outcome. Survival is
therefore a weak discriminator, which is why later runs switch the objective to
`mean_score + 0.25*worst_score` (`--objective score`).

**The search never moved the evasion parameters.** All three keep the inherited
defaults exactly (`pred_r` 70, `pred_face_r` 80, `pred_sprint_r` 40,
`pred_dodge_r` 80, `pred_dodge_ang` 1.4). Two measurements explain why the
tuning so far has been foraging-only:

- Evasion *existing* is worth a great deal: disabling it (`pred_mode=0`) costs
  about 36% of mean score (921.0 -> 676.6 over six seeds at 900 s).
- Evasion *parameter* changes measured flat. Raising `pred_face_r` alone is
  neutral-to-worse (80 -> 933.4, 100 -> 905.5, 120 -> 905.4, 160 -> 904.2), and
  a face-and-retreat variant was within noise (+15 on a ~50-point spread).

One known, unexploited detail: the engine only charges unconditionally when the
agent looks away **or** is closer than `hearing_radius * 1.5` = 90 units
(`src/elements/predator.py`). `pred_face_r` is 80, i.e. inside that zone, so the
face branch never reaches the range where facing actually deters a charge.
Changing the radius alone does not help; it likely needs a different behaviour,
not a different number.

## Provenance

Campaign `notrap-search-20260919`: 12 RunPod `cpu3c` machines, independent
`tune` studies with distinct search seeds over a shared 269-parameter space
(`models/notrap_config.py`), merged afterwards - the pods did not share search
state while running. Full per-trial history for all 24 studies is archived
locally under `runs/runpod-notrap-20260919/archive/` (gitignored; these three
files are the part worth keeping in version control).

Related: [research_notrap_seeds_20260919.md](../../docs/research_notrap_seeds_20260919.md).
