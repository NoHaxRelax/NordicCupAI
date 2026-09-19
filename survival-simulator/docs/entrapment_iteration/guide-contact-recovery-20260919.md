# Recovering a guide's lost assignment

The earlier swamp failure has an upstream association problem. At 547.1 s,
guide 160's normal DTO contains a predator 67.60 units away, but its assigned
track provides no target observation. Its guide debug therefore says
`not_observed` and routes toward a stale contact about 163 units away.
Another guide, 164, sees the same predator at 115.80 units. Both public poses
have uncertainty 1 and transform those sightings to (1342.12, 1119.92).
Guide 160 had a localization uncertainty spike to 60.36 at 545.0 s; global
tracking ignores observations above uncertainty 12. Recovering localization
does not automatically reconnect the old assigned track to its visible prey.

The optional `--guide-contact-recovery` experiment repairs that assignment
before choosing actions. An existing guide with no assigned sighting can
transfer to the sole local predator's observed track when:

- Its pose uncertainty is at most 8 and it has sprint energy plus a move reserve.
- The predator can detect it under public hearing/vision geometry.
- The contact lies beyond bait hearing range plus delayed-motion uncertainty.
- An existing owner also sees it, is well localized, and is farther away by
  more than two predator steps plus both pose uncertainties.

An unowned observed track can also be recovered. The old stale assignment
is released; the observed track retains its inferred position and gains the
new guide. Route/following memory resets using observed walls. A replaced
owner returns to ordinary survival with predator avoidance and is excluded
from another guide assignment that tick. `guide_contact_recovered` records
each switch. No native predator identity or target state is used.

The pilot completed at **730.12 score / 678.0 seconds**, compared with
805.35 / 752.0 for the default on the same development seed. One contact was
recovered; there was one delivery arrival from 27 assignments, 25.4 estimated
bait-gap seconds, and no ownership handovers. This remains **off by default**
and is not carried into the final C++ adaptation. It does not prove which
agent a predator chooses or repair general identity ambiguity. Every frame is
retained in `logs/entrapment-iteration/guide-contact-recovery-20260919`.
Adjacent summary and manifest record the completed run.
