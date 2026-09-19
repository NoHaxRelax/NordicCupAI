# Parallel capture studies

Three additional CPU Pods run independent bounded studies alongside the existing
campaign. The existing supervisor has a local worker pool, not a cross-Pod queue;
the extra Pods accelerate exploration of complementary hypotheses. They do not
make an individual simulation or the LLM review run four times faster.

The studies are guide delivery/trap geometry, bait replacement continuity, and
capture versus feeding-worker allocation. Each uses the frozen incumbent source
and configuration captured at preparation, 16 simulation workers, at most 16
candidate configurations, and development seeds only. The inherited baseline is
evaluated separately within each study. Available existing feature switches are
compared explicitly before tuning relevant scalar parameters.

Native screening is followed by a matched Python comparison of the incumbent and
the highest-ranked eligible finalist on eight development maps, if time remains.
A native candidate with more worst-case survival loss than the promotion
allowance is not selected for validation. These results are advisory: the main
supervisor retains selection authority and must compare a proposed import with
its current incumbent. A satellite study never changes the main best or reads
the final holdout.

Plans, results, diagnostics, progress and shutdown journals live under
`/workspace/research-capture-fleet/<study>/`. Each study has a separate cache and
is the only writer to its own directory. Source snapshots and the shared Python
environment are read-only in this workflow. The operational runner is deployed
as `/workspace/research-parallel.py`, outside the frozen source.

Each additional Pod costs $0.96/hour for compute. Its shutdown guard is armed
before the study launches and stops it when its launcher finishes, its study
ends, or 2.1 hours after Pod creation. Workers also stop when the primary campaign
leaves development, loses its heartbeat, or their 4 GiB storage allowance is
exhausted. The guard requires its container and API connectivity. Results persist
on the shared network volume. Stopped container disks and retained network
storage still incur storage charges.

The fleet reservation is $6.50, including a buffer over 3 × 2.1 × $0.965.
Together with the primary Pod's $9.65 conservative compute ceiling, $16 maximum
LLM allowance, $5 safety reserve and $1 storage/rounding allowance, the reserved
total is $38.15 against the authorized $40. This is a budget ceiling estimate,
not an invoice. The main campaign's immutable internal budget file still describes
only its own Pod; fleet monitoring adds the satellite estimates separately.

The network volume was expanded from 20 to 30 GB for the additional diagnostics;
the increase costs $0.70/month at the standard rate while retained. The volume
cannot be shrunk in place. No extra LLM jobs run on these worker Pods.

Future LLM reviews receive a campaign-scoped instruction to inspect the fleet's
reports and acknowledge their source, seeds, engine and frozen incumbent. The
review may include compatible configurations in its next variants and identify
unresolved interactions, but must not treat native screening as Python evidence.
