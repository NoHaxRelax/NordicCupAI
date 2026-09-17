# Manual validation pseudo-labels

## Contents

- `annotated-frames/` contains 1920 x 1080 visual QA overlays. These are for review only, not model inputs.
- `finetune-metadata/annotations/` contains one JSON file per annotated source frame. Its `annotations` entries use the reference training shape: `object_id` plus 4K `[x1, y1, x2, y2]` boxes.
- `*-crops/` directories contain expanded positive crops and a manifest for each manually reviewed track. Use these with the original class name for sprite-classifier fine-tuning.
- `finetune-metadata/dataset.json` is the combined index, class counts, image path convention, and constraints.

## Method

1. Compare each candidate against the labelled Helsinki reference crops.
2. Manually place a seed box in a reconstructed 3840 x 2160 validation image.
3. Track adjacent frames with CSRT and visually review the resulting boxes.
4. Preserve reviewed boxes in source-pixel coordinates and render them using the same class-colour, box, and label convention as the training video.

## Important limitation

These are participant-created training pseudo-labels, not organiser annotations. They are reviewed positive examples, but the current set is not exhaustive: a second helicopter was found after the first pass. Do not treat an unannotated full frame as an image containing no target object. For immediate fine-tuning, use the class-specific crops as positive classifier data. Before detector fine-tuning on full frames, complete the stricter small-sprite audit and add any missing objects.
