# Local game-mechanics edge cases

Tested 17 September 2026 against unmodified public source commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. These are small synthetic scenarios, not official-server or final-score results. All requests parse with the upstream `ActionRequest` schema. Schema acceptance alone does not establish competition permission: the published contract calls for one action per observed agent per tick. The ordinary benchmark policies follow that contract.

Run `survival/.venv/bin/python survival/research/edge_cases.py`. Assertions cover the results in `survival/results/edge_cases.json`.

## Most consequential findings

| Finding | Local evidence | Practical meaning |
| --- | --- | --- |
| Repeated agent IDs execute repeatedly within one world tick | Ten walk-10 actions move 100 units, cost 5 movement energy, and age the agent only 0.1 seconds; one sprint-20 action moves 20 and costs 5.5 | Major missing action-count validation. This violates the documented one-action-per-agent contract; excluded from strategy benchmarks. Official handling/permission unverified. |
| Newborn can act in the same request list | Birth updates `agents_dict` immediately; a following action for its next sequential ID turns the child and charges its turn cost | Also outside the observed-agent contract. Predicting an unseen ID is not required for ordinary play. |
| Collision tests only the endpoint | A synthetic evolved agent with sprint40 moves across a width30 wall, from x195 to235, with radius5 at each endpoint | Thin-wall crossing is mechanically possible at evolved speeds; exact tangency and obstacle geometry matter. No swept path check. |
| Boundary clamping happens after collision test | An evolved sprint40 agent moves from x35 to x5 through the outer wall and ends inside it | Potential refuge, but getting food, replacing agents, reaching the required trait and safely leaving are unsolved. No full-game benefit established. |
| Movement and looking are independent | Agent moves backwards10 without changing its heading or paying turn energy | Useful for kiting while watching a predator. `move_direction` is **relative** to facing despite README saying absolute. Normal gameplay, not a malformed request. |
| Resting predators skip contact checks | Agent overlapping a resting predator survives; predator energy rises0→3 | Sleep is a real escape window, but rest/energy are hidden from observations and wake-up contact is lethal. Avoid treating this as permanent safety. |
| Predator observations precede predator movement | Reported predator distance40 while its actual post-tick distance is25 | Maintain at least a movement-step safety buffer. A controller treating observations as current can react too late. |

Source locations: `src/utils/simulation.py` action loop; `src/elements/environment.py` movement, spawning, and `non_agent_step`; `src/elements/creature.py` relative coordinates and observation payload.

## Smaller bugs and scoring details

- Removing a dead agent while iterating `self.agents` skips the next agent's age, passive energy, food and observation update for that tick. The fixture's second agent stays age0/energy150 after the first dies. This is an engine correctness issue, not a strong repeatable strategy.
- Old-age energy is deducted after the death check. An age100.1 agent can remain alive at energy−0.601 until the next tick. If eaten immediately, subtracting negative energy **adds** 0.00601 score and drains the predator slightly. Tiny reward, unreliable and not worth sacrificing survival.
- A fruit crossed mid-move is not collected unless the endpoint overlaps it. The test moves past a fresh fruit and finishes exactly at the strict contact boundary. Plan endpoints to collect fruit rather than assuming collection along a path.
- Eating ripe fruit still yields its full score bonus even when energy storage is full. The corrected fixture grows a normal fruit to60 energy and verifies +0.06 score. This creates a survival-versus-score tradeoff: leave food for later survival until the species can reliably finish the game.
- Repeated births still cost100 each and require energy strictly greater than100. Starting at500 yields four children, not five; there is no free population or energy creation.

## Ideas falsified

- Negative movement does not grant energy: it is clamped to zero.
- An oversized single movement is capped at sprint speed.
- Omitting an agent's action does not freeze aging or living energy cost; it matches a normal idle action.

The probes configure scenario positions, energy, traits and terrain to isolate rules. Such internal state editing is a research tool only. They do not establish that a policy can create those situations from ordinary observations, nor that the hosted competition uses this exact version. No remote validation/evaluation was submitted.
