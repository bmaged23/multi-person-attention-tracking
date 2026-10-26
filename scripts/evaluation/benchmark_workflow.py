"""Measure real AMI video workflows, stages and sampled local resource usage."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import psutil

from attention_tracking.config import ROOT, Settings
from attention_tracking.detection.people_backend import PeopleBackend
from attention_tracking.video.pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["opencv", "onnx"], default="opencv")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confidence", type=float, default=Settings().person_confidence
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = replace(
        Settings(),
        person_confidence=args.confidence,
        person_backend=args.backend,
        person_device=args.device,
        tile_workers=args.workers,
    )
    samples = []
    stop = threading.Event()
    process = psutil.Process()
    process.cpu_percent()

    def sample():
        while not stop.is_set():
            row = {
                "rss_mib": process.memory_info().rss / 1024**2,
                "cpu_percent_one_core": process.cpu_percent(),
                "system_ram_used_mib": psutil.virtual_memory().used / 1024**2,
            }
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    memory, util = result.stdout.splitlines()[0].split(",")
                    row.update(gpu_vram_mib=float(memory), gpu_util_percent=float(util))
            except (OSError, subprocess.TimeoutExpired):
                pass
            samples.append(row)
            stop.wait(0.5)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    backend = None
    try:
        backend = PeopleBackend(settings)
        backend.detect(__import__("numpy").zeros((320, 320, 3), dtype="uint8"), 0)
        loading = time.perf_counter() - started
        rows = []
        for view in ["L", "R"]:
            source = ROOT / f"input/ami/IS1004d_{view}.mp4"
            result = run(
                str(source),
                settings,
                args.output / f"{view}.json",
                args.output / f"{view}.mp4",
                max_seconds=args.seconds,
                backend=backend,
            )
            labels = {}
            for person in result["people"]:
                for key, value in person["label_counts"].items():
                    labels[key] = labels.get(key, 0) + value
            rows.append(
                {
                    "view": view,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "frames": result["frames"],
                    "tracks": len(result["people"]),
                    "label_counts": labels,
                    "performance": result["performance"],
                }
            )
    finally:
        if backend:
            backend.close()
        stop.set()
        thread.join(5)
    metrics = {
        key: {
            "mean": sum(s[key] for s in samples if key in s)
            / sum(key in s for s in samples),
            "peak": max(s[key] for s in samples if key in s),
        }
        for key in samples[-1]
    }
    report = {
        "backend": args.backend,
        "person_device": args.device,
        "landmark_device": "cpu",
        "workers": args.workers,
        "seconds_per_clip": args.seconds,
        "load_and_warmup_seconds": loading,
        "clips": rows,
        "resources": {
            "samples": len(samples),
            "metrics": metrics,
            "scope": "Process RSS and CPU (100%=one core); whole-device GPU and system RAM, samples may miss peaks. Includes initialization and FFmpeg work; CPU excludes FFmpeg child.",
        },
        "attention_accuracy_measured": False,
        "scope": "Two CC-BY4.0 AMI meeting excerpts, fixed source hashes. Observed labels are not attention ground truth. Real inference, tracking, JSON/JSONL and H264 output.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
