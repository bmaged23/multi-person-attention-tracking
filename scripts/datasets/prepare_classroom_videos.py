"""Download CDED-7 research clips and normalize Windows archive member names."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "https://zenodo.org/api/records/21207208"

if __name__ == "__main__":
    folder = ROOT / "input/classroom"
    folder.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API) as response:
        metadata = json.load(response)
    asset = next(
        f for f in metadata["files"] if f["key"] == "CDED-7-v1.1.0-dataset.zip"
    )
    archive = folder / "dataset.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        try:
            urllib.request.urlretrieve(asset["links"]["self"], temporary)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    algorithm, expected = asset["checksum"].split(":", 1)
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, algorithm).hexdigest()
    if actual != expected:
        raise RuntimeError("Classroom archive checksum mismatch")
    (folder / "zenodo.json").write_text(json.dumps(metadata, indent=2))
    with zipfile.ZipFile(archive) as data:
        for member in data.infolist():
            name = member.filename.replace("\\", "/").split("/")[-1]
            if name in {"class_6.mp4", "class_7.mp4", "README.txt"}:
                with (
                    data.open(member) as source,
                    (folder / name).open("wb") as destination,
                ):
                    import shutil

                    shutil.copyfileobj(source, destination)
    print("Prepared two university-classroom clips under", folder)
    print("Dataset is for research and non-commercial use; see source documentation.")
