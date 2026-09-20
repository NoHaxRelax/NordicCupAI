# Action-bug harvest layer

This is the experimental duplicate-action layer for the native nightsim policy.
It tests repeated finite actions at the HTTP boundary and records whether a
predator actually consumes the drained agent with negative energy.

The safe validation configuration is a 20,000-action burst with at most 12
harvests per run. Larger bursts are retained for bounded interaction tests but
are not the default deployment configuration.

The server files in this directory are experimental and are not a final
evaluation submission.
