# Native validation coverage review

This folder is an auditable object-discovery pass, not a claim of organizer ground truth.

## Review contract

- `raw/` is the authoritative visual input. Every source tile is shown at native 1:1 pixels.
- `overlay/` adds known candidates: green means already represented by a manual annotation; magenta means a score-confirmed seed still needs a manual track.
- Frame 5 is reviewed completely. Later sheets cover the top 540 pixels at a stride of 4 frames.
- The measured-motion gate must stay below 460 px so consecutive top bands overlap by at least 80 px.
- A failed motion estimate adds a complete-frame recovery review for the target sample.
- A sheet is not complete until both `primary_review` and `small_object_review` are recorded in `coverage-ledger.json`.
- Do not use unreviewed or partially reviewed image regions as detector background.

## Completed state

- Unresolved score-confirmed candidates: 0
- Primary review: 69 of 69 sheets complete
- Small-object review: 69 of 69 sheets complete
- Maximum measured vertical displacement between samples: 285 px
- Motion gate: passed
- Manual annotations: 995 boxes across 230 frames
- Negative detector loss is certified only inside the reviewed regions under the top-entry invariant. The lower part of later full frames remains outside the negative-label contract.

The second small-object pass added four missed tracks: a tank in frames 29-61, a tank in frames 66-75, a small tower in frames 155-164, and a mine roller in frames 197-206.
