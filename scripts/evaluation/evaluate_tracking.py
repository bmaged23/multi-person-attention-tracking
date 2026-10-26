"""Score IoU association on frozen synthetic identity trajectories, separately from detection."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from attention_tracking.config import ROOT, Settings
from attention_tracking.tracking.face_tracks import Tracker
from attention_tracking.detection.mediapipe_backend import Face


def evaluate():
    data = json.loads(
        (ROOT / "evaluation/datasets/tracking-trajectories-v1.json").read_text()
    )
    results = []
    for case in data["cases"]:
        tracker = Tracker(Settings())
        last = {}
        switches = 0
        counts = {}
        observations = 0
        for frame in case["frames"]:
            faces = [
                Face(tuple(o["box"]), np.empty((0, 2)), None) for o in frame["objects"]
            ]
            output = tracker.update(faces, frame["seconds"], (360, 640, 3))
            for reference, (track, _) in zip(frame["objects"], output):
                identity = reference["identity"]
                observations += 1
                switches += identity in last and last[identity] != track.id
                last[identity] = track.id
                key = (identity, track.id)
                counts[key] = counts.get(key, 0) + 1
        gt = sorted({k[0] for k in counts})
        pred = sorted({k[1] for k in counts})
        matrix = np.array([[counts.get((a, b), 0) for b in pred] for a in gt])
        r, c = linear_sum_assignment(-matrix)
        idtp = int(matrix[r, c].sum())
        idf1 = idtp / max(1, observations)
        results.append(
            {
                "case_id": case["case_id"],
                "observations": observations,
                "identity_switches": int(switches),
                "predicted_tracks": len(pred),
                "idf1": idf1,
            }
        )
    return {
        "cases": len(results),
        "results": results,
        "scope": "Synthetic perfect boxes with independently generated identity paths; no missed detections. IDF1 uses global one-to-one identity matching. Crossing can be unidentifiable from IoU geometry alone; not a real-world MOT benchmark.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))
