"""Readable, text-sized labels for every observed face."""

import cv2
import numpy as np

COLORS = {
    "attentive": (40, 200, 40),
    "partial": (0, 210, 255),
    "non_attentive": (40, 40, 235),
    "drowsy": (210, 80, 210),
    "no_face": (160, 160, 160),
}


def face_bounds(face, shape):
    """Visible face extent from the facial mesh, independent of the tracking box."""
    points = face.landmarks[:468]
    if len(points) == 0 or not np.isfinite(points).all():
        return None
    height, width = shape[:2]
    low = np.floor(points.min(axis=0)).astype(int)
    high = np.ceil(points.max(axis=0)).astype(int)
    x1, y1 = max(0, int(low[0])), max(0, int(low[1]))
    x2, y2 = min(width - 1, int(high[0])), min(height - 1, int(high[1]))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def annotate(frame, observations, settings):
    for track, face, result in observations:
        bounds = face_bounds(face, frame.shape)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        color = COLORS[result["label"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{track.name} | {result['label'].replace('_', ' ')}"
        scale = max(0.35, min(0.6, frame.shape[1] / 1200))
        font = cv2.FONT_HERSHEY_SIMPLEX
        while text and cv2.getTextSize(text, font, scale, 1)[0][0] > frame.shape[1] - 8:
            text = text[:-1]
        (tw, th), base = cv2.getTextSize(text, font, scale, 1)
        left = min(x1, max(0, frame.shape[1] - tw - 8))
        bottom = max(th + base + 8, y1)
        cv2.rectangle(
            frame,
            (left, bottom - th - base - 8),
            (left + tw + 8, bottom),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (left + 4, bottom - base - 4),
            font,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        if settings.show_landmarks:
            for x, y in face.landmarks[::8]:
                cv2.circle(frame, (int(x), int(y)), 1, color, -1)
    return frame
