"""Process arbitrary video or camera input into per-track JSON and annotations."""

import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import cv2

from attention_tracking.analysis.face_metrics import measure
from attention_tracking.detection.mediapipe_backend import FaceBackend
from attention_tracking.detection.people_backend import PeopleBackend
from attention_tracking.tracking.face_tracks import Tracker
from attention_tracking.video.annotation import COLORS, annotate, face_bounds
from attention_tracking.video.encoding import finalize
from attention_tracking.video.reporting import write_summary


def run(
    source,
    settings,
    output,
    annotated=None,
    names=None,
    live=False,
    max_seconds=None,
    display=False,
    backend=None,
    max_frame_pixels=None,
):
    output = Path(output)
    annotated = Path(annotated) if annotated else None
    observations_path = output.with_name(output.stem + "_observations.jsonl")
    targets = [
        p.resolve() for p in (output, annotated, observations_path) if p is not None
    ]
    if len(set(targets)) != len(targets):
        raise ValueError("JSON output and annotated video paths must differ")
    if not live:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Input video not found: {path}")
        if path.resolve() in targets:
            raise ValueError("Outputs must not overwrite the input")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive")
    if annotated and not shutil.which("ffmpeg"):
        raise RuntimeError("Install FFmpeg to produce playable H.264 annotations")
    output.parent.mkdir(parents=True, exist_ok=True)
    if annotated:
        annotated.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(settings.tile_workers)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Cannot open video/camera source")
    own_backend = backend is None
    model = None
    writer = None
    temporary = None
    observation_stream = None
    summaries = {}
    tracker = Tracker(settings, names)
    stage_seconds = {
        "decode": 0.0,
        "detection_and_landmarks": 0.0,
        "scoring_tracking_reporting": 0.0,
        "annotation_encoding": 0.0,
        "final_encoding": 0.0,
        "person_detection": 0.0,
        "landmarks": 0.0,
    }
    count = 0
    started = time.monotonic()
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        if not live:
            capture.release()
            raise ValueError("Video has no valid FPS")
        fps = 30
    try:
        model = backend or (
            PeopleBackend(settings)
            if settings.detection_mode == "people"
            else FaceBackend(settings)
        )
        observation_stream = observations_path.open("w")
        while True:
            stage_start = time.perf_counter()
            ok, frame = capture.read()
            stage_seconds["decode"] += time.perf_counter() - stage_start
            if not ok:
                if count == 0:
                    raise RuntimeError("No readable frames")
                break
            if (
                max_frame_pixels is not None
                and frame.shape[0] * frame.shape[1] > max_frame_pixels
            ):
                raise ValueError("Decoded frame exceeds configured pixel limit")
            now = time.monotonic() - started if live else count / fps
            if max_seconds is not None and now >= max_seconds:
                break
            stage_start = time.perf_counter()
            faces = model.detect(frame, now)
            stage_seconds["detection_and_landmarks"] += (
                time.perf_counter() - stage_start
            )
            for key, value in getattr(model, "last_timings", {}).items():
                stage_seconds[key.removesuffix("_seconds")] += value
            stage_start = time.perf_counter()
            observations = []
            for track, face in tracker.update(faces, now, frame.shape):
                result = track.attention.update(measure(face, settings), now, settings)
                observations.append((track, face, result))
                record = {
                    "id": track.id,
                    "name": track.name,
                    "time": round(now, 3),
                    "bbox": list(track.bbox),
                    "face_bbox": face_bounds(face, frame.shape),
                    **result,
                }
                observation_stream.write(json.dumps(record) + "\n")
                summary = summaries.setdefault(
                    track.id,
                    {
                        "id": track.id,
                        "name": track.name,
                        "first_seen": round(now, 3),
                        "last_seen": round(now, 3),
                        "observations": 0,
                        "label_counts": {key: 0 for key in COLORS},
                    },
                )
                summary["last_seen"] = round(now, 3)
                summary["observations"] += 1
                summary["label_counts"][result["label"]] += 1
            stage_seconds["scoring_tracking_reporting"] += (
                time.perf_counter() - stage_start
            )
            stage_start = time.perf_counter()
            if annotated or display:
                shown = annotate(frame.copy(), observations, settings)
                if annotated and writer is None:
                    with tempfile.NamedTemporaryFile(
                        dir=annotated.parent, suffix=".mp4", delete=False
                    ) as fd:
                        temporary = Path(fd.name)
                    writer = cv2.VideoWriter(
                        str(temporary),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create video writer")
                if writer:
                    writer.write(shown)
                if display:
                    cv2.imshow("Visual attention - Q to stop", shown)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        count += 1
                        break
            stage_seconds["annotation_encoding"] += time.perf_counter() - stage_start
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} frames; {len(summaries)} tracks", flush=True)
    except KeyboardInterrupt:
        if not live:
            raise
    except Exception:
        if temporary is not None:
            if writer is not None:
                writer.release()
            temporary.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        if writer:
            writer.release()
        if observation_stream is not None:
            observation_stream.close()
        if model is not None and own_backend:
            model.close()
        if display:
            cv2.destroyAllWindows()
    if temporary:
        try:
            stage_start = time.perf_counter()
            finalize(temporary, annotated)
            stage_seconds["final_encoding"] += time.perf_counter() - stage_start
        finally:
            temporary.unlink(missing_ok=True)
    for summary in summaries.values():
        known = summary["observations"] - summary["label_counts"]["no_face"]
        summary["attentive_fraction_of_known"] = (
            round(summary["label_counts"]["attentive"] / known, 4) if known else None
        )
    report = {
        "source": str(source),
        "reference": "original attention algorithm",
        "attention_algorithm": "original_weighted",
        "iou_threshold": settings.iou_threshold,
        "attention_thresholds": {
            "yaw": settings.yaw_limit,
            "pitch_down": settings.pitch_down,
            "pitch_up": settings.pitch_up,
            "ear": settings.ear_closed,
            "iris_side": settings.iris_side,
            "attentive": settings.attentive_min,
            "partial": settings.partial_min,
        },
        "detection_mode": settings.detection_mode,
        "frames": count,
        "fps": fps,
        "device": settings.device,
        "person_backend": settings.person_backend,
        "person_device": settings.person_device,
        "performance": {
            "elapsed_seconds": time.monotonic() - started,
            "processed_fps": count / max(1e-9, time.monotonic() - started),
            "stage_seconds": stage_seconds,
            "scope": "Full run includes initialization when backend is owned; detection sub-stages are nested, not additive with detection_and_landmarks.",
        },
        "tile_grid": settings.tile_grid,
        "detection_confidence": settings.detection_confidence,
        "people": list(summaries.values()),
        "observations_file": observations_path.name,
    }
    write_summary(output, report)
    return report
