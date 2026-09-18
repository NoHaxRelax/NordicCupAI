#!/usr/bin/env python3
"""Classify class-agnostic validation proposals with an ensemble of crop models.

This is a discovery tool. Its output is candidate evidence, never an annotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.models import convnext_small, convnext_tiny, resnet50
from torchvision.transforms import functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    spec = checkpoint.get("spec", {})
    architecture = spec.get("architecture", "resnet50")
    if architecture == "convnext_small":
        model = convnext_small(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, len(classes))
    elif architecture == "convnext_tiny":
        model = convnext_tiny(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, len(classes))
    elif architecture == "resnet50":
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    return model, classes, int(spec.get("input_size", 224)), architecture


def crop_tensor(image: Image.Image, bbox: list[float], size: int) -> torch.Tensor:
    width, height = image.size
    x1, y1, x2, y2 = bbox
    pad = max(2.0, 0.12 * max(x2 - x1, y2 - y1))
    crop = image.crop(
        (
            max(0, math.floor(x1 - pad)),
            max(0, math.floor(y1 - pad)),
            min(width, math.ceil(x2 + pad)),
            min(height, math.ceil(y2 + pad)),
        )
    )
    crop = ImageOps.pad(crop, (size, size), method=Image.Resampling.BILINEAR, color=(114, 114, 114))
    return F.normalize(F.to_tensor(crop), [.485, .456, .406], [.229, .224, .225])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--classifiers", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-proposals", type=int, help="Benchmark/debug limit after filtering.")
    parser.add_argument(
        "--min-proposal-confidence",
        type=float,
        default=0.0,
        help="Discard foreground proposals below this detector confidence before classification.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for report_path in args.reports:
        report = json.loads(report_path.read_text())
        rows.extend(
            row
            for row in report["predictions"]
            if float(row["confidence"]) >= args.min_proposal_confidence
        )
    rows.sort(key=lambda row: (int(row["frame"]), -float(row["confidence"])))
    if args.max_proposals is not None:
        rows = rows[: args.max_proposals]

    device = (
        ("mps" if torch.backends.mps.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    models = []
    class_order = None
    for path in args.classifiers:
        model, classes, size, architecture = load_model(path, device)
        if class_order is None:
            class_order = classes
        if classes != class_order:
            raise ValueError(f"Classifier class mismatch: {path}")
        models.append((path, model, size, architecture))
    assert class_order is not None

    # Batch across frame boundaries. The earlier per-frame implementation left
    # the accelerator mostly idle for the usual 5-25 proposals per frame.
    per_model = []
    for model_index, (_, model, size, _) in enumerate(models, start=1):
        probabilities = []
        image_cache: dict[int, Image.Image] = {}
        for offset in range(0, len(rows), args.batch_size):
            batch_rows = rows[offset : offset + args.batch_size]
            tensors = []
            for row in batch_rows:
                frame = int(row["frame"])
                if frame not in image_cache:
                    image_cache[frame] = Image.open(
                        args.images / f"frame_{frame:06d}.png"
                    ).convert("RGB")
                    while len(image_cache) > 8:
                        oldest = next(iter(image_cache))
                        del image_cache[oldest]
                tensors.append(crop_tensor(image_cache[frame], row["bbox_source_xyxy"], size))
            batch = torch.stack(tensors).to(device)
            with torch.inference_mode():
                probabilities.extend(model(batch).softmax(1).cpu().numpy())
            print(
                json.dumps(
                    {
                        "model": model_index,
                        "models": len(models),
                        "processed": min(offset + len(batch_rows), len(rows)),
                        "proposals": len(rows),
                    }
                ),
                flush=True,
            )
        per_model.append(np.asarray(probabilities, dtype=np.float32))

    stacked = np.stack(per_model)
    mean_probs = stacked.mean(axis=0)
    predictions = stacked.argmax(axis=2)
    outputs = []
    for index, row in enumerate(rows):
        mean_index = int(mean_probs[index].argmax())
        model_indices = predictions[:, index].tolist()
        agreement = max(model_indices.count(value) for value in set(model_indices))
        outputs.append(
            {
                **row,
                "ensemble_class": class_order[mean_index],
                "ensemble_probability": float(mean_probs[index, mean_index]),
                "model_classes": [class_order[value] for value in model_indices],
                "model_probabilities_for_ensemble_class": [float(values[index, mean_index]) for values in stacked],
                "agreement": int(agreement),
                "class_probabilities": {
                    label: float(mean_probs[index, class_index])
                    for class_index, label in enumerate(class_order)
                },
            }
        )

    outputs.sort(key=lambda row: row["ensemble_probability"], reverse=True)
    payload = {
        "description": "Class-agnostic foreground proposals classified by independent crop models; candidates only.",
        "classes": class_order,
        "device": device,
        "proposal_reports": [{"path": str(path), "sha256": sha256(path)} for path in args.reports],
        "classifiers": [
            {"path": str(path), "sha256": sha256(path), "architecture": architecture}
            for path, _, _, architecture in models
        ],
        "proposal_count": len(outputs),
        "min_proposal_confidence": args.min_proposal_confidence,
        "predictions": outputs,
        "training": False,
        "live_queries": 0,
        "competition_evaluation": False,
        "limitations": [
            "Classifier agreement is proposal evidence, not validation ground truth.",
            "Models share training data and are not statistically independent.",
            "All candidate tracks require native-pixel temporal and visual review before annotation.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output / 'report.json'), "proposals": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
