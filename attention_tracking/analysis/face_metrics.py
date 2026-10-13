"""Original solvePnP head pose, eye-aspect ratio, and averaged iris-distance ratio."""

import math

import cv2
import numpy as np

FACE_INDICES = [1, 152, 263, 33, 287, 57]
FACE_MODEL = np.array(
    [
        [0, 0, 0],
        [0, -330, -65],
        [-225, 170, -135],
        [225, 170, -135],
        [-150, -150, -125],
        [150, -150, -125],
    ],
    dtype=np.float64,
)


def head_pose(points, width, height):
    """Apply the original six-point template and crop-width focal approximation."""
    camera = np.array(
        [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64
    )
    try:
        ok, rotation, _ = cv2.solvePnP(
            FACE_MODEL,
            np.asarray(points[FACE_INDICES], dtype=np.float64),
            camera,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        matrix, _ = cv2.Rodrigues(rotation)
        angles = cv2.RQDecomp3x3(matrix)[0]
        return tuple(float(a) for a in angles)
    except cv2.error:
        return None


def ear(points, indices):
    p = points[indices]
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return float((a + b) / (2 * c + 1e-6))


def iris_ratio(points, iris, corners):
    center = points[iris].mean(axis=0)
    outer, inner = points[corners]
    return float(
        np.clip(
            np.linalg.norm(center - outer) / (np.linalg.norm(outer - inner) + 1e-6),
            0,
            1,
        )
    )


def measure(face, settings):
    if len(face.landmarks) < 478 or not np.isfinite(face.landmarks).all():
        return None
    # The original pose function operates in the tightened landmark crop.
    box = getattr(face, "measurement_box", None) or face.bbox
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    raw_points = getattr(face, "measurement_landmarks", None)
    points = (raw_points if raw_points is not None else face.landmarks) - np.array(
        [x1, y1]
    )
    pose = head_pose(points, x2 - x1, y2 - y1)
    result = {
        "pitch": pose[0] if pose else None,
        "yaw": pose[1] if pose else None,
        "ear": (
            ear(points, [362, 385, 387, 263, 373, 380])
            + ear(points, [33, 160, 158, 133, 153, 144])
        )
        / 2,
        "iris_avg": (
            iris_ratio(points, [474, 475, 476, 477], [362, 263])
            + iris_ratio(points, [469, 470, 471, 472], [33, 133])
        )
        / 2,
    }
    return {k: v if v is None or math.isfinite(v) else None for k, v in result.items()}
