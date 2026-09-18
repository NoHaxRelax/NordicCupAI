# Coordinator reproduction and successor-scarcity experiment

Both variants start from the exact source frozen with full-game pilot seed
1266967689. They use native games, ordinary policy inputs, native energy and
aging, and at most two local workers. No survival implementation or simulator
mechanic was changed.

## Variants

`spawn-guard-source` adds one narrow rule: if Oscar's action for an agent asks
to spawn this tick, `_select_bait` cannot assign that same agent as bait and
silently replace the spawn action. Oscar still decides whether spawning is
appropriate. The coordinator may consider the agent again on a later tick.

`scarcity-source` includes the spawn guard, counts no-successor ticks while a
bait is still alive, never assigns the sole remaining young parent as guide,
and pauses new guide assignments whenever no conservatively viable bait
successor exists. It does not change food selection or reproduction decisions.

Frozen `models/core.py` SHA-256:

- spawn guard: `94e4aca87220c3e0a99e22d419ec7b538d78807e16068b9db17a6a0159b19847`
- scarcity plus guard: `f0bfca0acd4141eddc6a39f17dbfbc80a2bed4d56a77e5b592e85dce69061df8`

## Results

| seed / policy | end | final agents | born | peak | bait gap | guides / deaths | held 30s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1266967689 frozen baseline | extinction 284.2s | 0 | 27 | 22 | 55.9s | 4 / 3 | 0 |
| 1266967689 spawn guard | extinction 376.9s | 0 | 49 | 34 | 79.3s | 9 / 7 | 3 |
| 1266967689 scarcity + guard | extinction 325.4s | 0 | 42 | 22 | 124.6s | 1 / 1 | 1 |
| 3 spawn guard | extinction 208.8s | 0 | 46 | 24 | 0.1s | 19 / 15 | 2 |
| 3 scarcity + guard | 300s horizon | 20 | 86 | 24 | 0.0s | 7 / 6 | 3 |

The guard-only pilot produced 22 more agents and lasted 92.7 seconds longer
than the known baseline, which supports the concern that coordinator role
assignment can interfere with renewal. It did not solve extinction or bait
continuity: its final bait gap was 23.4 seconds longer. The scarcity rule
substantially reduced guide assignments, but its pilot bait gap was worse and
its survival result fell between baseline and guard-only. On seed 3 it was much
better than guard-only, reaching the horizon with 20 agents and continuous
bait, but this two-seed result is not stable enough for promotion.

The native full-game runner is not bit-reproducible across separate same-seed
processes in these observations: the two variants' populations diverged before
their scarcity-specific guide behavior could act. Therefore differences cannot
be attributed entirely to either patch, and these runs are mechanism probes,
not paired estimates. The exact traces and manifests are retained in the four
result directories.

## Recommendation alongside the tuned orchard

Use the newly tuned upstream orchard reproduction policy first. The frozen
coordinator instantiated `OrchardPolicy(extra_old=False, heir_age=1e9)` and
also suppressed every spawn for agents marked old or age at least 55, even
though the native engine has no such age fertility prohibition. Those choices
are a more direct explanation of the baseline renewal collapse than late
successor scarcity.

Keep the narrow deferred-bait guard when integrating the tuned orchard. It is
compatible with the orchard's decisions and prevents a coordinator role change
from erasing an explicit spawn action on that tick. Remove or revise the broad
age/old spawn suppression separately so tuned heir spawning can actually reach
the engine.

Do not promote the scarcity guide freeze from these runs. Retain its corrected
pre-death scarcity metric for diagnosis, then reassess guide conservation only
after the tuned orchard is integrated. A healthy renewing colony changes the
meaning and frequency of successor scarcity.
