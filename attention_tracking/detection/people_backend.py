"""Person-first detection and per-person MediaPipe head analysis on the original CPU backend."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import time

import cv2
import numpy as np

from attention_tracking.detection.mediapipe_backend import Face, FaceBackend
from attention_tracking.detection.person_detector import PersonDetector


def head_region(box, shape):
    """Retain the original upper-40-percent rule and 15-percent crop padding."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    head_y2 = y1 + int(height * (0.40 if height > width * 1.3 else 1))
    pad_x = int(width * 0.15)
    pad_y = int((head_y2 - y1) * 0.15)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(shape[1], x2 + pad_x),
        min(shape[0], head_y2 + pad_y),
    )


class PeopleBackend:
    def __init__(self, settings):
        self.settings = settings
        self.last_timings = {}
        self.detector = PersonDetector(settings)
        self.models = []
        self.executor = None
        workers = settings.tile_workers if settings.device == "cpu" else 1
        try:
            for _ in range(workers):
                self.models.append(
                    FaceBackend(
                        replace(settings, max_faces=1, tile_grid=1, tile_workers=1),
                        image_mode=True,
                    )
                )
        except Exception:
            for model in self.models:
                model.close()
            raise
        if workers > 1:
            self.executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="person-head"
            )

    def _analyze(self, frame, box, model, seconds):
        x1, y1, x2, y2 = head_region(box, frame.shape)
        crop = frame[y1:y2, x1:x2]
        empty = Face(box, np.empty((0, 2)), None)
        if crop.size == 0:
            return empty
        original_h, original_w = crop.shape[:2]
        # This is the original small-head resizing and retry strategy.
        if min(original_h, original_w) < 120:
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
        work_h, work_w = crop.shape[:2]
        for target, enhance in ((320, False), (640, False), (320, True)):
            factor = max(1, target / min(work_h, work_w))
            image = cv2.resize(
                crop,
                (round(work_w * factor), round(work_h * factor)),
                interpolation=cv2.INTER_CUBIC,
            )
            if enhance:
                ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
                image = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
            faces = model.detect(image, seconds)
            if not faces:
                continue
            face = faces[0]
            analysis_points = face.landmarks.copy()
            analysis_box = face.bbox
            scaling = np.array(
                [original_w / image.shape[1], original_h / image.shape[0]]
            )
            points = face.landmarks * scaling + np.array([x1, y1])
            # Reject a neighboring face outside the intended person's head region.
            center = points[:468].mean(axis=0)
            if not (box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]):
                continue
            return Face(box, points, face.rotation, analysis_box, analysis_points)
        return empty

    def _group(self, frame, boxes, model, seconds):
        return [self._analyze(frame, box, model, seconds) for box in boxes]

    def detect(self, frame, seconds):
        started = time.perf_counter()
        boxes = self.detector.detect(frame)
        detected = time.perf_counter()
        self.last_timings = {
            "person_detection_seconds": detected - started,
            "landmarks_seconds": 0.0,
        }
        if not boxes:
            return []
        if not self.executor:
            faces = self._group(frame, boxes, self.models[0], seconds)
            self.last_timings["landmarks_seconds"] = time.perf_counter() - detected
            return faces
        jobs = [
            (model, boxes[i :: len(self.models)]) for i, model in enumerate(self.models)
        ]
        groups = self.executor.map(
            lambda job: self._group(frame, job[1], job[0], seconds), jobs
        )
        faces = [person for group in groups for person in group]
        self.last_timings["landmarks_seconds"] = time.perf_counter() - detected
        return faces

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        for model in self.models:
            model.close()
