"""Fetch two CC BY 4.0 AMI meeting excerpts for a redistributable demo."""

from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IS1004d/video/"


def main():
    destination = ROOT / "input/ami"
    destination.mkdir(parents=True, exist_ok=True)
    for view in ["L", "R"]:
        output = destination / f"IS1004d_{view}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-rw_timeout",
                "30000000",
                "-ss",
                "60",
                "-i",
                BASE + f"IS1004d.{view}_orig.avi",
                "-t",
                "12",
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
    urllib.request.urlretrieve(
        "https://groups.inf.ed.ac.uk/ami/download/CCBY4.0.txt",
        destination / "CCBY4.0.txt",
    )
    print("Prepared native-resolution AMI CC BY 4.0 meeting excerpts:", destination)
    print("Attribution and modification details: docs/DEMO_SOURCE.md")


if __name__ == "__main__":
    main()
