# Hearing handoff and isolated predator tracking

Stopping 55 units from bait can leave a predator almost 70 units away when
it eats the guide: capture occurs within 15 units, hearing within 60. Use
44 units (60 - 15 - 1 margin), on the front side of the crevice. Reaching a
handoff waypoint alone no longer permits stopping outside that range.
This is a nominal geometric condition; localization error and replacement
continuity can still invalidate retention.

Seeing multiple predators no longer disables following checks if the selected
target is isolated. Another sighting within 31.5 units still makes identity
ambiguous; the previous following latch is retained. Seven focused motion and
latch checks passed, including distant and overlapping second predators.

Native full games on seed 1883894846, 3000-second maximum, every frame saved:

| Variant | Score | Lifetime (s) | Delivery arrivals / assignments | Sprint-available premature captures |
| --- | ---: | ---: | ---: | ---: |
| Corrected escape, previous 55-unit delivery | 1077.50 | 1027.5 | 3 / 28 | 0 |
| 44-unit front-side delivery | 842.56 | 791.6 | 1 / 16 | 0 |
| Plus isolated-target following checks | 842.83 | 791.6 | 1 / 17 | 0 |

Both new runs had 52.5 estimated bait-gap seconds and became extinct. These
changes repair specific handoff/detection conditions, but **did not improve
the dev-seed score or delivery count**. Arrivals are not confirmed captures;
do not describe this as improved reliability. Source hashes, summaries and
parameters are saved in the adjacent hearing-handoff JSON files. Replay
folders have the same names under `logs/entrapment-iteration/`, with suffix
`-20260919`. No survival-module code changed.
