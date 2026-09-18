# Overnight plan, 19 Sept 2026 (Claude, orchard session) — keep this short and current

## Oscar's instructions (23:55 and 00:20, paraphrased; also in memory overnight-directives-2026-09-19)
- First ~2 h (to ~02:00): make the population live as long as possible (no predators), everything in C++,
  run on Runpod only (never simulate on the laptop). Be creative; think at several abstraction layers.
- NEVER submit anything through the competition API.
- Then (and while waiting): study Lucas's latest bait/trap work, rewrite it in C++, and combine it with the
  survival policy: maximise lifetime while baiting/trapping predators as well as possible.
  Policy ideas: non-contributing agents run the survival policy; keepers stay near the trap to replenish
  bait; old agents (dying anyway) become guides/bait; agents keep enough energy to spawn a child when a
  predator is near; predators chase the closest agent; escapes by sprint, circling behind (turn-rate
  exploit), walking backwards, hiding behind walls; losing agents to predators should be rare; many
  predators late in the game is the most important regime; test the hypothesis that late game lacks the
  resources to keep loading the trap. When something fails, build a narrow test and fix it step by step.
- Keep working all night (cron tick every 10 min). Budget $100 new spend, up to 8 pods, until spent.
  Shut pods down when idle; boot more when needed within budget.
- Version control locally; push only to survival-simulator/oscar-* branches. Few replays; compact data;
  always save the best policy so we can return to it. Keep a structured plan; avoid rabbit holes.

## Layers and experiments
P1 no-predator survival (C++ fork survival/nightsim, runner nightsim/run.py, driver night.sh)
- L1 measure: 1000-seed baseline best vs r21s0c2 (base1k); harvest per 250 s (fruit spawned vs eaten),
  death classes by trees left. Question: late game supply-limited or discovery-limited?
- L2 parameters: paired search around the leader, 200-500 seeds per candidate, confirm on 1000+.
- L3 mechanisms in C++ (each behind a parameter, default off, A/B on 500+ seeds): chosen from L1 data.
- L4 population strategy (colony count/spread, late-game size).
P2 predators (after 02:00) — ONE problem at a time, nail it, then build on it (Oscar 00:57):
  1. Walking ONE predator to the trap (guide + narrow-gap crevice bait; Lucas's work as the base, improve it).
     Narrow scenario tests first (nightsim hooks), then in full games. Done = high delivery rate, guide rarely dies.
  2. Multiple predators into the same trap (holding several; delivering while others are held).
  3. Not losing agents to predators in general (evasion: detection, facing/backing, dodge, speed, refuges).
  4. Then: bait replacement by old agents, keepers near the trap, many-predator late game, resource limits.
  Parked code ready for reuse: evasion (pred_mode/share/dodge), crevice site finder (trap_mode=1), bait role
  (trap_mode=2), scenario hooks (dbg_*), perfect-trap model (test_pred_life), escape.py, trapsite.py.
  Crevice (narrow gap) baits only; never wall/corner baits.

## P2 step 1 design (02:00): walk ONE predator into a narrow-gap crevice
Engine facts used: predators kill any agent whose centre is < 15 away after the predator's move (agents move
first, then predators); agents never collide with agents or predators; a predator (r=10) cannot put its centre
within 10 of any wall face, so in a 10.1-19.9 gap it stays >= 10 outside the mouth line => a bait 9 deep is >= 19
away (safe); an agent at the mouth line can be 10 away (killed). Predator senses: hearing 60 through walls,
vision 250 in a pi/3 cone with line of sight; it chases the CLOSEST sensed agent. Direct chase (15/tick,
turn cap 0.3) when the agent faces away or is < 90 away; otherwise a 45-degree pivot approach closing ~10.6/tick.
Predator energy 200 max, sprint 2.55/tick, walk 0.55/tick, at <= 0 it rests ~3.4 s (no move, no kill).
Guide (trap_mode >= 3, one per trap, prefers old/high-energy/fast agents):
  ACQUIRE: approach the predator to < 60 (hearing) so it locks on; LEAD: face it, walk BACKWARDS along the
  straight line to the lane point 60 outside the mouth, keep 40-90 distance (sprint backwards if < 40; wait
  if > 120; use its rest windows); DELIVER: continue backwards through the mouth without stopping (never
  linger at the mouth), pass the bait, exit through the rear (rear_ok) or stop 8 deeper than the bait and
  become a second bait. Success = predator within 25 of the mouth with the bait alive, guide alive.
Test (nightsim/guide.py, hooks dbg_keep_agents / dbg_load_walls / dbg_freeze): true wall map loaded into
the policy (isolates guiding from site finding), bait frozen at the goal, guide at 150/300/500 units from
the lane with a clear line, predator at 100-250 from the guide, awake or resting; guide walk speed 10/13/16.
Measure delivery rate, guide survival, ticks, energy. Baseline to beat: Lucas 794/1000 single deliveries,
guides almost always die.

## Rules of thumb
- Judge by paired means on >=500 seeds (per-run sd ~330 s). Report survival relative to trees left.
- Every accepted change: BEST.md + git commit in teamrepo branch survival-simulator/oscar-overnight-cpp.
