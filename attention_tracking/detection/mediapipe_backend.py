"""MediaPipe detects all faces independently; tracking never implies recognition."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from attention_tracking.geometry import bbox_iou


@dataclass
class Face:
    bbox: tuple
    landmarks: np.ndarray
    rotation: np.ndarray | None
    measurement_box: tuple | None = None
    measurement_landmarks: np.ndarray | None = None


class FaceBackend:
    def __init__(self, settings, image_mode=False):
        import os

        from attention_tracking.config import ROOT

        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
        import mediapipe as mp

        if not settings.model_path.is_file():
            raise FileNotFoundError(
                "Run python scripts/setup/download_face_landmarker.py first"
            )
        self.mp = mp
        self.settings = settings
        self.image_mode = image_mode or settings.tile_grid > 1
        delegate = (
            mp.tasks.BaseOptions.Delegate.GPU
            if settings.device == "gpu"
            else mp.tasks.BaseOptions.Delegate.CPU
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(settings.model_path), delegate=delegate
            ),
            running_mode=(
                mp.tasks.vision.RunningMode.VIDEO
                if not self.image_mode
                else mp.tasks.vision.RunningMode.IMAGE
            ),
            num_faces=settings.max_faces,
            min_face_detection_confidence=settings.detection_confidence,
            min_face_presence_confidence=settings.presence_confidence,
            min_tracking_confidence=settings.presence_confidence,
            output_facial_transformation_matrixes=False,
        )
        try:
            self.model = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except RuntimeError as error:
            if settings.device == "gpu":
                raise RuntimeError(
                    "MediaPipe GPU initialization failed. Its delegate requires EGL/OpenGL; CUDA availability alone is insufficient. Use --device cpu on this system."
                ) from error
            raise
        self.models = [self.model]
        self.executor = None
        workers = (
            min(settings.tile_workers, settings.tile_grid**2)
            if settings.device == "cpu"
            else 1
        )
        try:
            for _ in range(workers - 1):
                self.models.append(
                    mp.tasks.vision.FaceLandmarker.create_from_options(options)
                )
        except Exception:
            for model in self.models:
                model.close()
            raise
        if workers > 1:
            self.executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="face-tile"
            )
        self.last_ms = -1

    def detect(self, frame, seconds):
        timestamp = max(self.last_ms + 1, round(seconds * 1000))
        self.last_ms = timestamp
        height, width = frame.shape[:2]
        regions = list(tile_regions(width, height, self.settings.tile_grid))
        if self.executor:
            jobs = [
                (model, regions[index :: len(self.models)])
                for index, model in enumerate(self.models)
            ]
            groups = list(
                self.executor.map(
                    lambda job: self._detect_regions(frame, job[0], job[1], timestamp),
                    jobs,
                )
            )
            faces = [face for group in groups for face in group]
        else:
            faces = self._detect_regions(frame, self.model, regions, timestamp)
        return suppress_duplicates(faces)[: self.settings.max_faces]

    def _detect_regions(self, frame, model, regions, timestamp):
        faces = []
        for x1, y1, x2, y2 in regions:
            image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB),
            )
            result = (
                model.detect_for_video(image, timestamp)
                if not self.image_mode
                else model.detect(image)
            )
            faces.extend(result_faces(result, x2 - x1, y2 - y1, x1, y1))
        return faces

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        for model in self.models:
            model.close()


def suppress_duplicates(faces, threshold=0.6):
    """Keep one landmark result per face when MediaPipe returns overlapping copies."""
    kept = []

    def area(face):
        x1, y1, x2, y2 = face.bbox
        return (x2 - x1) * (y2 - y1)

    for face in sorted(faces, key=area, reverse=True):
        if all(bbox_iou(face.bbox, other.bbox) < threshold for other in kept):
            kept.append(face)
    return kept


def tile_regions(width, height, grid, overlap=0.2):
    """Return clipped, overlapping regions covering the complete original frame."""
    for row in range(grid):
        for column in range(grid):
            pad_x = int(width / grid * overlap) if grid > 1 else 0
            pad_y = int(height / grid * overlap) if grid > 1 else 0
            yield (
                max(0, int(column * width / grid) - pad_x),
                max(0, int(row * height / grid) - pad_y),
                min(width, int((column + 1) * width / grid) + pad_x),
                min(height, int((row + 1) * height / grid) + pad_y),
            )


def result_faces(result, width, height, offset_x=0, offset_y=0):
    """Map each crop's landmarks back to source pixels, retaining its rotation."""
    faces = []
    for index, landmarks in enumerate(result.face_landmarks):
        points = np.array([[p.x * width, p.y * height] for p in landmarks], dtype=float)
        if len(points) < 468 or not np.isfinite(points).all():
            continue
        low, high = points[:468].min(axis=0), points[:468].max(axis=0)
        pad = (high - low) * 0.08
        low, high = (
            np.maximum(0, low - pad),
            np.minimum([width - 1, height - 1], high + pad),
        )
        if np.any(high <= low):
            continue
        rotation = None
        if index < len(result.facial_transformation_matrixes):
            matrix = np.asarray(result.facial_transformation_matrixes[index])[:3, :3]
            if np.isfinite(matrix).all():
                u, _, vt = np.linalg.svd(matrix)
                rotation = u @ vt
                if np.linalg.det(rotation) < 0:
                    u[:, -1] *= -1
                    rotation = u @ vt
        offset = np.array([offset_x, offset_y])
        faces.append(
            Face(
                tuple(map(int, (*(low + offset), *(high + offset)))),
                points + offset,
                rotation,
            )
        )
    return faces
