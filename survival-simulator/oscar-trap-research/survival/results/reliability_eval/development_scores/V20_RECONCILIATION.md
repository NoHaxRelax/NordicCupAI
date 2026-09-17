# v20 fresh16 count reconciliation

The original v2 harness reports 12 successes in 16 declared cases. The strict
causal scorer reports 11 passes in the same 16-case denominator.

The sole difference is map 10136 / fixture 20136. Its immutable receipt is
`2632039f`. The harness records success because the predator followed the guide
near the beginning and eventually acquired bait at 64.2 seconds. Target history
shows that it last targeted the guide at 5.1 seconds, lost that target at 5.2,
and first targeted bait at 62.3. Excluding two native rest intervals, about 50.1
predator-active seconds elapsed without a guide target. The guide remained alive.
The strict result is therefore `no_recent_guide_to_bait_handoff`, rather than an
attributable delivery.

The strict sidecar was written on 2026-09-17 under frozen protocol v3 hash
`429cc7b2d0439ac1756acea3f0fe521e68fced933f95bb7ebabe175a257f8016`.
The earlier v2 hash
`c31253764f7b6a741635d3cc10ca0d8b885e2f07f086f4157b92c285956767eb`
used a three-second wall-clock handoff window. It was superseded before any
59-case run because native rest should not consume the causal window. That
amendment cannot rescue 10136: its excess gap is awake, active wandering.

The other four failures agree between the harness denominator and strict result:
10134 unsupported, 10138 and 10139 without active bait acquisition, and 10142
too late to retain for 30 seconds (active acquisition at 273.7).
