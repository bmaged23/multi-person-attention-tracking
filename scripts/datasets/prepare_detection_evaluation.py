"""Freeze the first64 COCO validation rows before any model evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def load(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", choices=["development", "validation"], default="development"
    )
    args = parser.parse_args()
    validation = args.subset == "validation"
    destination = ROOT / (
        "input/coco-holdout-36" if validation else "input/coco-validation-64"
    )
    destination.mkdir(parents=True, exist_ok=True)
    metadata = load("https://huggingface.co/api/datasets/detection-datasets/coco")
    revision = metadata["sha"]
    data = load(
        "https://datasets-server.huggingface.co/first-rows?dataset=detection-datasets%2Fcoco&config=default&split=val"
    )
    feature = next(x for x in data["features"] if x["name"] == "objects")
    (destination / "feature-schema.json").write_text(
        json.dumps(feature, indent=2) + "\n"
    )
    rows = []
    selected = data["rows"][64:100] if validation else data["rows"][:64]
    for item in selected:
        row = item["row"]
        image_id = row["image_id"]
        path = destination / f"{image_id:012}.jpg"
        if not path.exists():
            urllib.request.urlretrieve(row["image"]["src"], path)
        objects = row["objects"]
        boxes = [b for b, c in zip(objects["bbox"], objects["category"]) if c == 0]
        rows.append(
            {
                "image_id": image_id,
                "row_index": item["row_idx"],
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": row["width"],
                "height": row["height"],
                "person_boxes_xyxy": boxes,
            }
        )
    if (
        load("https://huggingface.co/api/datasets/detection-datasets/coco")["sha"]
        != revision
    ):
        raise RuntimeError("Dataset revision changed during provisioning")
    manifest = {
        "dataset": "detection-datasets/coco",
        "revision": revision,
        "split": "val",
        "selection": (
            "Rows64-99 reserved before threshold selection; no inference-driven filtering."
            if validation
            else "First64 dataset-viewer validation rows, fixed before inference; no filtering on model output."
        ),
        "boxes": "Human COCO person annotations mirrored as xyxy; category0=person. Crowd flags unavailable in this mirror; this is a simple operating-point regression, not official COCO AP.",
        "cases": rows,
    }
    frozen_path = ROOT / (
        "evaluation/datasets/coco-person-holdout-36.json"
        if validation
        else "evaluation/datasets/coco-person-validation-64.json"
    )
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != manifest:
        raise RuntimeError(
            "Provisioned data differs from frozen evaluation manifest; refusing to replace references"
        )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(rows),
                "person_boxes": sum(len(x["person_boxes_xyxy"]) for x in rows),
                "revision": revision,
            }
        )
    )


if __name__ == "__main__":
    main()
