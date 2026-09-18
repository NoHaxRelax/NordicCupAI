# Precision follow-up

Frozen v7 found 12/13 reviewed test targets versus the original hybrid's 4/13,
but emitted 233 proposals. Its full results are retained in the v2 directory.
Three extra geometric detections are real objects missing from the sparse labels;
most broad CNN proposals were obvious orange-roof or vegetation mistakes.

Those eight test images are now development data for the next iteration. A fresh
four-frame check, 187/201/224/247, was predeclared before viewing its images. It
contains seven participant labels across five classes. The candidate was frozen
in `selection-v3.json` before their image-only annotation review. Original and
reviewed boxes are both retained. They share physical tracks with calibration.
Training/calibration still uses 41 validation frames.

## Bounded experiments

- Patch CNN v5 trains for 800 steps (23 seconds on mypc), correcting background
  crop sizes/aspects and removing retained orange pixels from gray ta-ta masks.
  The masks were visually audited in `neutral-tata-audit.jpg`. No new training
  images or negative tiles were added.
- Weak feature matches use score 0.70. The missed tower at frame 220 has five
  consistent local matches, score 0.743, aligned pixel agreement 0.289 and v5 CNN
  class probability 0.366. Joint checks (pixel >=0.25, same-class CNN >=0.30)
  recover it and reject the remaining lower-scoring false hypotheses in that
  whole-image diagnostic. This is a consumed-data diagnostic, not a test score.
- `asset-precision-20260918-v2` evaluated a stricter full detector: high-score
  geometry stays unchanged, weaker matches require pixel and CNN agreement,
  tiny silhouettes require the v5 CNN verifier at 0.80, and both broad CNN
  proposal branches are disabled. Masked pixel correlation remains at 0.70.
  It found 23/26 reference and 14/19 validation targets. It was stopped after
  that complete development scan and three consumed images; partial results
  are preserved in `precision-v2-stopped.tar.gz`, with an explicit stop receipt.
- The selected `asset-precision-20260918-v3` restores scratch-CNN proposals at
  verifier confidence 0.95 and dense small-tower proposals at same-class verifier
  confidence 0.40. It preserves 26/26 reference targets with 26 proposals and
  16/19 development targets with 20 proposals. The eight consumed images and
  four fresh images were evaluated by separate saved launchers. Results are
  13/13 with 22 proposals on consumed images and 7/7 with 12 proposals on the
  reserved images. The baseline is 0/7 with 330 proposals on those same images.

The feature-fusion implementation keeps weak hypotheses from shifting stronger
boxes. New tests cover that contract and aligned-pixel validation. Results and
checkpoint identities are saved beside the exact source archives and launchers.
All 44 local tests passed after the final class-preserving heatmap verification
check. `model/manifest.json` packages this exact candidate; it records the
completed fresh result and its background-error limitations. `bundle-smoke.json` records the CLI check on
a known calibration crop, which is an interface check rather than a test score.
