"""Download the versioned official Google MediaPipe Face Landmarker bundle."""

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

EXPECTED_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

if __name__ == "__main__":
    destination = ROOT / "models/face_landmarker.task"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(".download")
        try:
            urllib.request.urlretrieve(URL, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != EXPECTED_SHA256:
                raise RuntimeError("Downloaded model checksum mismatch")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("Model checksum mismatch")
    print(destination)
    print("SHA-256:", hashlib.sha256(destination.read_bytes()).hexdigest())
