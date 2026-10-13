"""Shared bounding-box geometry for detection and spatial tracking."""


def bbox_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )

    def area(x):
        return max(0, x[2] - x[0]) * max(0, x[3] - x[1])

    return overlap / max(1, area(a) + area(b) - overlap)
