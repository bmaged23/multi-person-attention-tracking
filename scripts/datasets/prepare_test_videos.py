"""Download Oxford TVHI research clips and retain their original provenance."""

import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://www.robots.ox.ac.uk/~alonso/data/"

if __name__ == "__main__":
    folder = ROOT / "input/tvhi"
    folder.mkdir(parents=True, exist_ok=True)
    for remote, name in [
        ("tv_human_interactions_videos.tar.gz", "videos.tar.gz"),
        ("readme.txt", "SOURCE_README.txt"),
    ]:
        destination = folder / name
        if not destination.exists():
            temporary = destination.with_suffix(".download")
            try:
                urllib.request.urlretrieve(BASE + remote, temporary)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    with tarfile.open(folder / "videos.tar.gz") as archive:
        archive.extractall(folder, filter="data")
    print("Prepared TVHI research videos:", folder)
    print("Research use only; original copyrights remain with their owners.")
