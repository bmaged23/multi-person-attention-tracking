"""Stage only reviewed public demo assets for GitHub Pages."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    destination = ROOT / "_site"
    assets = destination / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ["index.html", "style.css", "player.js"]:
        shutil.copy2(ROOT / "docs/site" / name, destination / name)
    for name in ["attention-demo.mp4", "annotated-demo-poster.jpg", "pipeline.svg"]:
        shutil.copy2(ROOT / "docs/media" / name, assets / name)
    (destination / ".nojekyll").touch()
    print("Staged annotated MP4 demo and static site in _site/")


if __name__ == "__main__":
    main()
