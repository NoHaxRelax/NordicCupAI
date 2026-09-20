# Review of Nikolaj's fast seed-recovery push

Reviewed `e84abd4ea66505ed259d1cfb94ce424dc2414063` on
`origin/claude/seed-recovery-fast` (20 September 2026). This is a source/document
review, not an independently repeated GPU or policy benchmark.

## Useful additions beyond our current implementation

- CUDA search: reports 9.86 seconds for the full uint32 range on one RTX 4090
  with a 128-word window. The 96-word variant reports 7.21 seconds but requires
  exact CPU rechecking of 83 window-overflow seeds. Candidate-buffer truncation
  and window-overflow receipts must be checked; a successful process exit alone
  does not mean complete usable results. GPU offload changes the deployment:
  it does not establish sub-minute search on the Hetzner CPU itself.
- CPU prefix kernel recomputes its first seeding chain and fuses it with the
  second, avoiding storage of the full 624-word state. This differs from our
  interleaved64-seed implementation, which still stores that state. Interleaving
  and short output windows overlap our work; no incremental speedup over our
  233.9-second Hetzner result has yet been measured.
- Dispatch with weak early evidence, cache survivors' terrain prefixes, and
  refine them as observations arrive. Their 2-second dispatch example retained
  26.1 million candidates and took22.1 seconds/1.46GB to rebuild the cache.
  Emitting prefixes directly from the GPU could avoid that second seeding pass.
  This differs from our fixed-landmark offline index and avoids its16GiB build.
- Position uncertainty radii with a sound relaxed nearest-site condition. Useful
  for integration with approximate shared-map localization. Our current boundary-
  anchored extractor measured near-machine-precision poses over1000 maps, so the
  benefit applies when adding less certain pose sources, not evidence that our
  existing exact samples were already invalid.
- Dynamic engine snapshot/restore (reported2us/1us) makes branching cheaper than
  full clones (reported1.48ms). A valuable missing primitive for real lookahead.
  The production policy's shared_ptr state still needs independent snapshots;
  engine snapshots alone do not make policy rollouts independent.

## Evidence limits / claims not to inherit

The 23/24 unique recovery result is a terrain-filter test: reliability.py obtains
labels directly from biome_map at synthetic waypoint coordinates. It does not
prove live navigation, localization, public acquisition, or synchronized replay.
The acquisition curve also labels its timings as a true-position floor.

The report's claim that the entire future food supply is action-independent is
not supported by the shipped engine: spawn_agent consumes the same e.rng used
by tree spawning, tree fruit checks, and fruit placement. Changing birth actions
can shift future food draws. Exact branch-and-replay is conditional on the action
sequence; it does not remove our cross-host replay divergence.

Likewise, measured lack of overflow in the128-word scan is not a general proof
that rejection sampling cannot overrun a fixed window. Retain fallback/overflow
handling. And its +163% planning result uses three seeds and a weak baseline;
policy-state copying is explicitly unresolved. Do not treat it as a gain over
our strongest policy.

## Next priorities

1. Benchmark the fused CPU kernel against our fast64 on identical public samples
   and current Hetzner hardware; keep correctness/fallback checks.
2. Use early dispatch plus incremental filtering with justified pose radii.
3. Consider GPU offload if external hardware fits deployment constraints.
4. Add snapshot-safe planning only while preserving replay rejection/resync.

The user cancelled the1M switch in this task. Commit618e9db was reverted without
rewriting history. Other tasks' deployments were not changed.
