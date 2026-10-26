"""Download the revision-pinned YOLOv8n ONNX export for person-only detection."""

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://huggingface.co/webml/yolov8n/resolve/85bc8d7ab30d4065a41909d756c42819b67b4388/onnx/yolov8n.onnx"

EXPECTED_SHA256 = "190ba5f1e61411a001683e349d6b2cdb0804c0dc67a5e34cd8ff6fd00ee54b4d"

if __name__ == "__main__":
    destination = ROOT / "models/yolov8n.onnx"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".download")
        try:
            urllib.request.urlretrieve(URL, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != EXPECTED_SHA256:
                raise RuntimeError("Person model checksum mismatch")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("Person model checksum mismatch")
    print(destination)
    print("SHA-256:", hashlib.sha256(destination.read_bytes()).hexdigest())
