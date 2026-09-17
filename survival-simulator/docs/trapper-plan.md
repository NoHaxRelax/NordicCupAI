# Trapper development plan (Oscar's order) and progress

Work the steps in this order. Wall trap and narrow-gap trap are developed in parallel because
luring and bait replenishment are shared. The society keeps foraging and breeding throughout.
"Validated" means measured on repeated runs, not a single run (the engine is not reproducible).

## Phase A: full knowledge of the game state (oracle world, development only)

| Step | Goal | Wall | Gap |
| --- | --- | --- | --- |
| 1 | Bait one predator into the trap while society continues; two agents cooperate (bait behind the wall, a guide that lures) | fixtures 4/4 when the predator is within ±45° of the approach axis, 0/4 otherwise; real games 0 deliveries so far | fixtures 3/3 (guide enters and becomes the bait); real games: chased agents run in on their own |
| 2 | Replace the bait (agent 1) | implemented (second slot), not validated | implemented (far mouth, rest window), not validated |
| 3 | Guard (agent 2) protects the bait from other predators; escaped predators are simply lured back by others | implemented (intercept from the protected side), not validated | not needed |
| 4 | Several predators in the same trap: every new predator is lured in by the first capable agent | not validated | happens naturally at the mouth; max 2 held so far |

## Phase B: no full knowledge

| Step | Goal | Status |
| --- | --- | --- |
| 5 | Fuse the agents' observations into a full game-state estimate | estimator exact (positions, headings), predators within one step; manager runs on it; agent server serves it |
| 6 | Redo steps 1–4 on the estimate | pending steps 1–4 |

## Refocus (Oscar, 18 September ~00:30)

Narrow-gap traps first; wall traps stay in the code as a backup (`site_kinds=('wall','gap')`) to be
tested after Phase A and B are done on gaps. Reasons: no guard is needed, a guide that dies in front
of the mouth still hands the predator to the bait, and relaxed gap requirements give a site on almost
every map. Replacement baits enter from behind (the far mouth).

Gap-site rules now: passage width 10.5-19.5 (strict engine tests), passage length >= 30, bait depth
chosen so the bait is > 16.5 from any point the predator can reach from either mouth, straight
approach >= 100 clear for the predator. Survey over 60 generated maps: 59/60 have a gap site
(mean 6.1 per map); walls: 50/60.

Endgame on an arranged passage (`scripts/trapper/gap_endgame.py`, bait inside, guide brings one
predator down the axis): the "flyby" (guide sprints along the obstacle face when the predator is
within 45) hands the predator to the bait within 3 ticks and the guide survives; the sacrifice (guide
stands at 25 out) also hands it over, guide dead. Bait depth 5 or 9.4 both survive 40 s.

## Overnight work log (18 September 2026)

Entries are appended as milestones complete; each names the commit and the evidence file.
