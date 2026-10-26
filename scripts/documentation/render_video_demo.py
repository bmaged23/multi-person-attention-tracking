"""Combine annotated videos and build an infinitely looping, local GIF preview."""

import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos",
        type=Path,
        nargs="+",
        default=[
            ROOT / "output/classroom/class_6_annotated.mp4",
            ROOT / "output/classroom/class_7_annotated.mp4",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/demo")
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-fps", type=int, default=8)
    args = parser.parse_args()
    if args.preview_width < 2 or args.preview_fps < 1:
        parser.error("Preview size and FPS must be positive")
    if any(not p.is_file() for p in args.videos):
        parser.error("Generate all annotated source videos first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video = args.output_dir / "attention-demo.mp4"
    preview = args.output_dir / "attention-demo.gif"
    if video.resolve() in [p.resolve() for p in args.videos]:
        parser.error("Output cannot replace a source video")
    inputs = [item for p in args.videos for item in ["-i", str(p)]]
    filters = "".join(
        f"[{i}:v]setpts=PTS-STARTPTS[v{i}];" for i in range(len(args.videos))
    )
    filters += (
        "".join(f"[v{i}]" for i in range(len(args.videos)))
        + f"concat=n={len(args.videos)}:v=1:a=0[v]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            *inputs,
            "-filter_complex",
            filters,
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-filter_complex",
            f"fps={args.preview_fps},scale={args.preview_width}:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1:dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(preview),
        ],
        check=True,
    )
    (args.output_dir / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Attention tracking demo</title><body style="margin:0;background:#0b1220;color:white;font-family:sans-serif"><h1>Attention tracking · local demo</h1><video src="attention-demo.mp4" autoplay muted loop playsinline controls style="width:100%;max-height:90vh"></video></body></html>'
    )
    print(
        f"Combined original-resolution MP4, looping GIF and local player: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
