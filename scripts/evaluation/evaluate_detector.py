"""Operating-point person detection against independent COCO annotations."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from attention_tracking.config import ROOT, Settings
from attention_tracking.detection.person_detector import PersonDetector
from attention_tracking.geometry import bbox_iou


def counts(predictions, references, threshold=0.5):
    if not predictions or not references:
        return 0, len(predictions), len(references)
    iou = np.array([[bbox_iou(p, r) for r in references] for p in predictions])
    # Prioritize maximum valid match count, then IoU, without greedy ordering bias.
    score = (iou >= threshold).astype(float) * (min(iou.shape) + 1) + iou
    rows, columns = linear_sum_assignment(-score)
    true = sum(iou[r, c] >= threshold for r, c in zip(rows, columns))
    return int(true), len(predictions) - int(true), len(references) - int(true)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["opencv", "onnx"], default="opencv")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confidence", type=float, default=Settings().person_confidence
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "input/coco-validation-64/manifest.json"
    )
    args = parser.parse_args()
    manifest_path = args.manifest
    manifest = json.loads(manifest_path.read_text())
    frames = []
    for row in manifest["cases"]:
        path = manifest_path.parent / row["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Input hash mismatch")
        frames.append(cv2.imread(str(path)))
    cv2.setNumThreads(4)
    settings = replace(
        Settings(),
        person_confidence=args.confidence,
        person_backend=args.backend,
        person_device=args.device,
        person_tiles=args.tiles,
        tile_workers=4,
    )
    started = time.perf_counter()
    model = PersonDetector(settings)
    load = time.perf_counter() - started
    for _ in range(3):
        model.detect(frames[0])
    rows = []
    for reference, frame in zip(manifest["cases"], frames):
        start = time.perf_counter()
        predictions = model.detect(frame)
        seconds = time.perf_counter() - start
        tp, fp, fn = counts(predictions, reference["person_boxes_xyxy"])
        rows.append(
            {
                "image_id": reference["image_id"],
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "seconds": seconds,
                "predicted_boxes_xyxy": predictions,
            }
        )
    tp = sum(x["true_positive"] for x in rows)
    fp = sum(x["false_positive"] for x in rows)
    fn = sum(x["false_negative"] for x in rows)
    seconds = [x["seconds"] for x in rows]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    summary = {
        "backend": args.backend,
        "device": args.device,
        "cases": len(rows),
        "person_tiles": args.tiles,
        "confidence": settings.person_confidence,
        "iou_threshold": 0.5,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "warm_median_seconds": statistics.median(seconds),
        "warm_p95_seconds": float(np.percentile(seconds, 95)),
        "load_seconds": load,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "scope": "Frozen COCO val subset; human boxes; operating-point matching, not official COCO AP. No crowd ignore metadata in mirror. Tiling and confidence are fixed across runtime comparisons.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
