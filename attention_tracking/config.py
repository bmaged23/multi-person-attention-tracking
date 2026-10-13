"""General camera-relative settings, independent of classroom or identity schemas."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def _setting(name, default):
    """Environment overrides .env; paths are relative to the repository root."""
    key = "ATTENTION_" + name.upper()
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        if isinstance(default, bool):
            if raw.lower() not in {"true", "false", "1", "0"}:
                raise ValueError("expected true/false or 1/0")
            return raw.lower() in {"true", "1"}
        if isinstance(default, Path):
            path = Path(raw).expanduser()
            return path if path.is_absolute() else ROOT / path
        return type(default)(raw)
    except ValueError as error:
        raise ValueError(f"Invalid {key}: {raw!r}") from error


@dataclass(frozen=True)
class Settings:
    model_path: Path = _setting("model_path", ROOT / "models/face_landmarker.task")
    max_faces: int = _setting("max_faces", 20)
    tile_grid: int = _setting("tile_grid", 1)
    tile_workers: int = _setting("tile_workers", 8)
    detection_confidence: float = _setting("detection_confidence", 0.35)
    presence_confidence: float = _setting("presence_confidence", 0.25)
    person_model_path: Path = _setting(
        "person_model_path", ROOT / "models/yolov8n.onnx"
    )
    person_confidence: float = _setting("person_confidence", 0.15)
    person_tiles: int = _setting("person_tiles", 2)
    detection_mode: str = _setting("detection_mode", "people")
    device: str = _setting("device", "cpu")
    yaw_limit: float = _setting("yaw_limit", 39.4)
    pitch_down: float = _setting("pitch_down", -24.6)
    pitch_up: float = _setting("pitch_up", 41.6)
    ear_closed: float = _setting("ear_closed", 0.092)
    iris_side: float = _setting("iris_side", 0.37)
    head_weight: float = _setting("head_weight", 0.50)
    gaze_weight: float = _setting("gaze_weight", 0.10)
    eye_weight: float = _setting("eye_weight", 0.15)
    visibility_weight: float = _setting("visibility_weight", 0.25)
    attentive_min: float = _setting("attentive_min", 0.50)
    partial_min: float = _setting("partial_min", 0.25)
    iou_threshold: float = _setting("iou_threshold", 0.5)
    track_timeout: float = _setting("track_timeout", 1.0)
    show_landmarks: bool = _setting("show_landmarks", False)

    def __post_init__(self):
        if not isinstance(self.tile_grid, int) or not 1 <= self.tile_grid <= 6:
            raise ValueError("tile_grid must be an integer from 1 to 6")
        if not isinstance(self.tile_workers, int) or not 1 <= self.tile_workers <= 16:
            raise ValueError("tile_workers must be an integer from 1 to 16")
        weights = (
            self.head_weight,
            self.gaze_weight,
            self.eye_weight,
            self.visibility_weight,
        )
        if not all(math.isfinite(v) and v >= 0 for v in weights) or not math.isclose(
            sum(weights), 1
        ):
            raise ValueError("Attention weights must be non-negative and sum to one")
        if self.detection_mode not in {"people", "faces"}:
            raise ValueError("detection_mode must be people or faces")
        if not isinstance(self.person_tiles, int) or not 1 <= self.person_tiles <= 6:
            raise ValueError("person_tiles must be in [1,6]")
        if (
            not self.pitch_down < self.pitch_up
            or not 0 <= self.partial_min <= self.attentive_min
        ):
            raise ValueError("invalid original pitch or score bounds")
        if not math.isfinite(self.iou_threshold) or not 0 <= self.iou_threshold <= 1:
            raise ValueError("iou_threshold must be in [0,1]")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be cpu or gpu")
        for key in (
            "yaw_limit",
            "ear_closed",
            "iris_side",
            "track_timeout",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.max_faces < 1:
            raise ValueError("max_faces must be positive")
        for value in (
            self.detection_confidence,
            self.presence_confidence,
            self.person_confidence,
            self.attentive_min,
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("confidence/score thresholds must be in (0, 1]")
        if not all(math.isfinite(v) for v in (self.pitch_down, self.pitch_up)):
            raise ValueError("pitch bounds must be finite")
        if not 0 < self.iris_side < 0.5:
            raise ValueError("iris_side must be between zero and .5")
