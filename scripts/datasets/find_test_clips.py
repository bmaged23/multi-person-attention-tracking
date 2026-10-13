"""Select real TVHI clips containing multiple detected faces, without identities."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402

from attention_tracking.config import Settings  # noqa: E402
from attention_tracking.detection.mediapipe_backend import FaceBackend  # noqa: E402

if __name__ == "__main__":
    backend = FaceBackend(Settings())
    timestamp = 0
    candidates = []
    try:
        videos = sorted((ROOT / "input/tvhi").rglob("*.avi"))
        for path in videos:
            capture = cv2.VideoCapture(str(path))
            counts = []
            for fraction in (0.2, 0.5, 0.8):
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(capture.get(cv2.CAP_PROP_FRAME_COUNT) * fraction),
                )
                ok, frame = capture.read()
                if ok:
                    timestamp += 1
                    counts.append(len(backend.detect(frame, timestamp)))
            capture.release()
            if counts and max(counts) >= 2:
                candidates.append(
                    {
                        "video": str(path.relative_to(ROOT)),
                        "sampled_face_counts": counts,
                    }
                )
                print(candidates[-1], flush=True)
            if len(candidates) >= 4:
                break
    finally:
        backend.close()
    (ROOT / "input/tvhi/selected_clips.json").write_text(
        json.dumps(candidates, indent=2)
    )
