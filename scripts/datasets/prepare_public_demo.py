"""Download a verified CC BY classroom source and extract the public demo clips."""

import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = json.loads((ROOT / "docs/airc_provenance.json").read_text())
    destination = ROOT / "input/airc"
    destination.mkdir(parents=True, exist_ok=True)
    original = destination / "original018.avi"
    if not original.exists():
        temporary = original.with_suffix(".download")
        try:
            urllib.request.urlretrieve(source["url"], temporary)
            with temporary.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != source["sha256"]:
                raise RuntimeError("Classroom source checksum mismatch")
            temporary.replace(original)
        finally:
            temporary.unlink(missing_ok=True)
    with original.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != source["sha256"]:
        raise RuntimeError("Classroom source checksum mismatch")
    for excerpt in source["excerpts"]:
        output = destination / excerpt["filename"]
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                str(excerpt["start_seconds"]),
                "-i",
                str(original),
                "-t",
                str(excerpt["duration_seconds"]),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "16",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
        )
        print(output)
    print("AIRC-SMARTCLASS CC BY 4.0; attribution: docs/DEMO_SOURCE.md")


if __name__ == "__main__":
    main()
