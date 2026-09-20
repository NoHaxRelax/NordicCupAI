# Seed recovery to mode 144 checkpoint

The live server searches the full 32-bit seed space while continuing to return
the public orchard policy. Responses are paced during the search. Once a unique
seed is recovered, a shadow engine replays every public frame and action. The
server changes control to Oscar's mode 144 only after the shadow engine matches
the current public state.

## Fresh-seed score check

- 100 deterministic fresh seeds (`random.Random(20260920)`)
- public policy before simulated time 180; mode 144 afterward
- horizon 3000 seconds
- mean score: **2101.2727**
- sample standard deviation: **323.1465**
- 95% t interval for the mean: **[2037.1604, 2165.3850]**
- range: **1532.7739–2955.0653**
- failures: **0/100**

The batch ran on five 32-vCPU Runpod workers, 20 games per worker. This test
uses the reconstructed-world handoff time but bypasses HTTP serialization.

## End-to-end integrity check

Seed `123456789` was recovered from live public observations. The shadow engine
replayed the stream without a dynamic-state mismatch and began returning mode
144 actions through the HTTP response path. The long HTTP run is recorded
separately from the 100-game score estimate so that transport performance and
policy quality are not conflated.

The imported mode 144 policy is the `codex/seed-aware-policy-2200` checkpoint:
one-tick predictive collision safety and child food priority 60. Its original
500-seed evidence reports mean 2076.2007 with model availability at simulated
time 180.
