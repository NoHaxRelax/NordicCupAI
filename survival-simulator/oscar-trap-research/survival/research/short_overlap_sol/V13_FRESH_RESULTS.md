# Frozen v13 fresh cases

Four map/fixture pairs were fixed before execution: 10028/9528 through
10031/9531. Frozen `policy_v13_close_reacquire.py` source hash:
`f6bed4e62dfc8f385f3f5ebd2cbca39764e62429b26c613512859d8376afd88f`.
Each case ran 180 seconds in a fresh process with the event-streaming recorder,
320-pixel native rendering, and every 0.1-second frame.

| Map / fixture | Result | Entry / delivery | Guide | Retention |
| --- | --- | --- | --- | --- |
| 10028 / 9528 | Pass | 11.5 / 11.9 s | Alive | zero loss; final-30 s contained |
| 10029 / 9529 | Fail | none | Died 0.8 s | no capture |
| 10030 / 9530 | Pass | 26.5 / 26.5 s | Died 26.3 s | zero loss; final-30 s contained |
| 10031 / 9531 | Pass | 8.4 / 9.0 s | Alive | zero loss; final-30 s contained |

The fixed fresh result is **3/4**, far below the evidence needed for a
greater-than-95% reliability claim. No case is excluded.

Map 10029 failed during the initial observation/localization transient. The
first tick had no Predator DTO, so the guide held while the awake predator
closed from 55 to 40. The next policy input carried a one-movement-stale
distance of 55. Even after a nominally away move, the actual gap was only 45.
The emergency controller then chose diagonal translations: actual separation
fell to 27.64 at 0.3 seconds, stayed near 27, reached 17.83 at 0.7, and the
guide died at 0.8. The initial guide-predator segment was collision-clear at
radii 0, 5, and 10, so this was not a wall collision or localization mirror.

This establishes two distinct observation-only needs: native-relative
prelocalization spacing that accounts for the guaranteed blank/stale opening,
and an emergency endpoint chosen to maximize separation from the raw last
observation rather than route reward or velocity extrapolation.

