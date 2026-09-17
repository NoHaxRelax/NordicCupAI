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

## Overnight work log (18 September 2026)

Entries are appended as milestones complete; each names the commit and the evidence file.
