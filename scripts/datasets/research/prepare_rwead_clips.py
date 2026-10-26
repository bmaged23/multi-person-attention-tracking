"""Download checksum-verified CC BY 4.0 single-camera RWEAD audience clips."""

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[3]


def main():
    provenance = json.loads((ROOT / "docs/rwead_provenance.json").read_text())
    destination = ROOT / "input/rwead"
    destination.mkdir(parents=True, exist_ok=True)
    for source in provenance["files"]:
        output = destination / source["filename"]
        if not output.exists():
            temporary = output.with_suffix(".download")
            try:
                urllib.request.urlretrieve(source["url"], temporary)
                with temporary.open("rb") as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                if checksum != source["sha256"]:
                    raise RuntimeError(f"Checksum mismatch: {source['filename']}")
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        with output.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != source["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {source['filename']}")
        print(output)
    print("RWEAD CC BY 4.0; source attribution: docs/DEMO_SOURCE.md")


if __name__ == "__main__":
    main()
