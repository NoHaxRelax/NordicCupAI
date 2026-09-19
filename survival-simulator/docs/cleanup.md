# Simulator cleanup

This records the earlier cleanup and its checks. The subsequent
[branch integration](policy_branch_integration.md) introduces the current
modular policy and has not been executed or tested locally.

The supported entry point is `run.py` (`run.cmd` on Windows). See the
[quickstart](../README.md) for trapping, replay, no-predator Orchard, and
benchmark commands.

## Preserved

- The integrated trapping coordinator, shared map, trap detection, predator
  guides, bait replacement, and Orchard gathering/reproduction.
- All 39 policy, engine, and recording-runner source hashes in
  `entrapment_native_seed0/manifest.json`.
- The strict Orchard observation boundary: a spawned process receives public
  DTO fields and time, uses an independent RNG seed, and infers arena bounds
  from observed edges. It also clears inherited command-line arguments.
- Native recording and browser replay, benchmark recording/resume/merge,
  and the local HTTP simulator client. The HTTP agent endpoint now selects
  the maintained trapping policy.
- Historical baseline measurements, clearly separated from current test runs.

The research copy of Orchard and its vendored engine matched
`models/oscar_orchard.py` and `src/`. Both supported modes now use those shared
implementations. The audited runner and its tests moved out of the research
folder. Orchard's native window retains the 1x/2x/5x/10x/20x controls.

`models/nikolaj/` remains because trapping imports its map and exploration
logic. The engine's upstream dummy policy remains part of the frozen manifest.
Historical comments inside frozen source may mention retired experiments;
they are not supported launch instructions.

## Removed from the active tree

- Earlier standalone Simple/Expert policies, planners, playground, renderers,
  tests, and population/harvest/mapping benchmark runners.
- Their Bayesian tuning scripts, configurations, HPC submission files,
  dependency file, and outdated findings.
- Duplicate Orchard and trap research trees, experimental guide/trap labs,
  root predator handoff, and generated research recordings/results.

The maintained tests now live together in `tests/`. New scripts belong in
`scripts/`, behavior belongs in `models/`, and notes belong in `docs/`.
Other challenge folders were outside the cleanup scope.

## Recovery

Retired files, including uncommitted work and local research results, were
moved to the ignored local directory
`runs/cleanup-backup-20260918-185556/`. Paths beneath it retain their original
repository-relative structure. This backup is not shipped with Git.
Previously committed versions also remain in Git history. The existing stash
was preserved; applying it wholesale would bring back retired code.

## Verification

- All 49 maintained unit tests pass, including source hashes, guide behavior,
  bait replacement, observation isolation, inferred bounds, speed controls,
  and the HTTP policy contract.
- A 60-second seed-0 trapping run completed through the Windows launcher,
  recorded 601 frames, discovered a trap, and established bait. No natural
  predators appeared in this short run; it does not measure capture success.
- A 20-second seed-1 headless Orchard run completed with five agents and zero
  predators. Its worker audit confirms that simulator modules were not loaded.
- A two-second seed-2 Orchard rendering check completed using SDL's dummy
  video driver and the 20x setting.
- The trapping replay renderer produced a native frame from the new recording;
  its image was inspected, including the bait role ring. All seven launcher
  commands accept `--help`, and all 57 active Python files parse successfully.
- A two-second benchmark case completed through the launcher's spawned worker;
  rerunning the same command reused its completed result without rewriting it.

These are integration checks, not new 3,000-second score benchmarks. Historical
Orchard scores and trapping results remain reference evidence only.
