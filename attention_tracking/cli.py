"""Shared command-line configuration for video files and cameras."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from attention_tracking.config import Settings
from attention_tracking.video.pipeline import run


def main(live=False):
    defaults = Settings()
    parser = argparse.ArgumentParser(
        description="Multi-person camera-relative visual attention; no identity recognition"
    )
    if live:
        parser.add_argument(
            "--source", default="0", help="camera index or RTSP/video URL"
        )
    else:
        parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/attention.json"))
    parser.add_argument("--annotated-video", type=Path)
    parser.add_argument("--names", type=Path, help='JSON map: {"person_001": "Alex"}')
    parser.add_argument("--device", choices=["cpu", "gpu"], default=defaults.device)
    parser.add_argument(
        "--person-backend", choices=["opencv", "onnx"], default=defaults.person_backend
    )
    parser.add_argument(
        "--person-device", choices=["cpu", "cuda"], default=defaults.person_device
    )
    parser.add_argument("--max-faces", type=int, default=defaults.max_faces)
    parser.add_argument(
        "--tiles",
        type=int,
        default=defaults.tile_grid,
        help="overlapping grid per frame; 1 for webcam, 3 for wide classrooms",
    )
    parser.add_argument("--tile-workers", type=int, default=defaults.tile_workers)
    parser.add_argument(
        "--detection-confidence", type=float, default=defaults.detection_confidence
    )
    parser.add_argument(
        "--detector", choices=["people", "faces"], default=defaults.detection_mode
    )
    parser.add_argument(
        "--person-confidence", type=float, default=defaults.person_confidence
    )
    parser.add_argument("--person-tiles", type=int, default=defaults.person_tiles)
    parser.add_argument("--iou-threshold", type=float, default=defaults.iou_threshold)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--display", action="store_true")
    parser.add_argument(
        "--landmarks",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_landmarks,
    )
    args = parser.parse_args()
    settings = replace(
        Settings(),
        device=args.device,
        person_backend=args.person_backend,
        person_device=args.person_device,
        max_faces=args.max_faces,
        tile_grid=args.tiles,
        tile_workers=args.tile_workers,
        detection_confidence=args.detection_confidence,
        detection_mode=args.detector,
        person_confidence=args.person_confidence,
        person_tiles=args.person_tiles,
        iou_threshold=args.iou_threshold,
        show_landmarks=args.landmarks,
    )
    names = json.loads(args.names.read_text()) if args.names else None
    if names is not None and (
        not isinstance(names, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in names.items())
    ):
        parser.error(
            "--names must contain a JSON object mapping track IDs to display names"
        )
    source = (
        int(args.source)
        if live and args.source.isdecimal()
        else args.source
        if live
        else str(args.video)
    )
    report = run(
        source,
        settings,
        args.output,
        args.annotated_video,
        names,
        live,
        args.max_seconds,
        args.display,
    )
    print(
        f"Finished: {report['frames']} frames, {len(report['people'])} tracks. JSON: {args.output}"
    )
