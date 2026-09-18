# Stable drone snapshot – 2026-09-18

This snapshot intentionally contains the latest completed and frozen drone work,
not experiments that were still changing when it was assembled.

## Contents

- `solution/drone/`: exact source tree from the frozen
  `asset-precision-20260918-v3` archive.
- `release/asset-precision-20260918-v3/`: packaged model, selection settings,
  checksums, data-use ledger, evaluation receipts, and the 44-test passing log.
- `validation/reconstructed-validation/`: 249 reconstructed native-resolution
  validation frames plus their capture manifest. The PNG files are stored with
  Git LFS.
- `validation/annotations/coverage-ledger.json`: participant-created validation
  annotations and coverage state (1,009 candidates, including 995 boxes across
  230 frames).
- `validation/annotations/candidate-union.json`: compact candidate union used by
  the validation review workflow.

## Boundaries

The annotations are participant-reviewed data, not organizer ground truth. The
coverage README records which regions were reviewed and which regions remain
outside the negative-label contract. The release data-use ledger must be
consulted before training or evaluating because some validation frames were used
for calibration or development.

No final competition evaluation is included or implied by this snapshot.
