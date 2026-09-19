# Late-game checkpoint search

20 parameter families, 32 trials each, same 200 training maps (11001–11200).
Each map runs scheduled_breeding to extinction under normal rules. Reproduce its
state exactly 250 seconds (2500 ticks) before extinction; refuse censored games
or games shorter than 250 seconds rather than silently relabeling checkpoints.
Resume the unchanged baseline and require bit-identical final score and time.

Checkpoints are complete Linux process snapshots held in memory, including all
native engine and policy objects, RNG states, public observations and agent memory.
Workers retain the original snapshot and fork a disposable continuation for every
candidate. Only configuration parameters are changed in that child. Durable
metadata records map seed, tick, score and baseline ending, allowing exact snapshot
regeneration from frozen code. Snapshots themselves are not portable disk images.
Baseline death time selects training states only; it is never passed to a policy.

Objective: mean additional game score after checkpoint, continuing until extinction
or the original game horizon of 3000 seconds. Do not truncate at baseline death.
This gives the same candidate ranking as mean final score on identical checkpoints.
Unit-scaled hyperparameters and standardized trial objectives feed the GP; scores
remain in actual game units. Each family uses one seeded trial, five random startup
trials and 26 expected-improvement trials. Candidate parameters preserve policy
memory; native ABI still exposes only public observations.

Reuse ten Oscar CPU pods, each holding all 200 checkpoints and optimizing two
families sequentially (index i and i+10), with 32 actors. Exactly 128,000 training
continuations, plus snapshot construction and verification. Retain raw per-map
gains, timings, every configuration, checkpoint metadata and source hashes.

After ALL winners are frozen, evaluate each FROM GAME START on 1000 new maps,
12001–13000, alongside scheduled_breeding and original baseline: 22,000 full games.
This measures whether tail-state tuning transfers to deployable full-game behavior;
there is no privileged future-death trigger. Distribute 100 maps with every model
per pod, preserving paired maps and balanced hardware. Report score means, 95%
bootstrap CIs, paired differences and per-tick policy/CPU/wall timings as before.

No official competition validation; no dollar cap currently. Leave CPU pods running.
Use a separate worktree/branch codex/late250-checkpoints to avoid interfering with
concurrent native-serving edits in the earlier research worktree.
