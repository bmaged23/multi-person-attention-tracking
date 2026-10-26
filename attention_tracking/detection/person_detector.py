"""Locate people with a small YOLOv8 ONNX detector; no identity recognition."""

from pathlib import Path

import cv2
import numpy as np

from attention_tracking.detection.mediapipe_backend import tile_regions


def decode_predictions(output, scale, padding, offset, shape, confidence):
    """Decode YOLOv8's 4 box coordinates and 80 class scores in source pixels."""
    predictions = np.asarray(output).squeeze()
    if predictions.ndim != 2:
        raise ValueError("Expected a two-dimensional YOLOv8 output")
    if predictions.shape[0] == 84:
        predictions = predictions.T
    if predictions.shape[1] != 84:
        raise ValueError("Expected YOLOv8 COCO output with 84 channels")
    scores = predictions[:, 4:]
    selected = np.flatnonzero(
        (scores.argmax(axis=1) == 0) & (scores[:, 0] >= confidence)
    )
    height, width = shape[:2]
    boxes, values = [], []
    for index in selected:
        cx, cy, w, h = predictions[index, :4]
        x1 = (cx - w / 2 - padding[0]) / scale + offset[0]
        y1 = (cy - h / 2 - padding[1]) / scale + offset[1]
        x2 = (cx + w / 2 - padding[0]) / scale + offset[0]
        y2 = (cy + h / 2 - padding[1]) / scale + offset[1]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 - x1 >= 10 and y2 - y1 >= 10:
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            values.append(float(scores[index, 0]))
    return boxes, values


class PersonDetector:
    def __init__(self, settings):
        if not Path(settings.person_model_path).is_file():
            raise FileNotFoundError(
                "Run python scripts/setup/download_person_detector.py first"
            )
        self.settings = settings
        self.session = None
        if settings.person_backend == "onnx":
            import onnxruntime as ort

            if settings.person_device == "cuda":
                from attention_tracking.runtime import preload_cuda

                preload_cuda()
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError("ONNX Runtime CUDA provider is unavailable")
            options = ort.SessionOptions()
            options.intra_op_num_threads = settings.tile_workers
            options.inter_op_num_threads = 1
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if settings.person_device == "cuda"
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(
                str(settings.person_model_path), options, providers=providers
            )
            self.session.disable_fallback()
            if self.session.get_providers()[0] != providers[0]:
                raise RuntimeError(
                    "Requested inference provider failed to initialize; refusing silent CPU fallback"
                )
            self.input_name = self.session.get_inputs()[0].name
        else:
            self.net = cv2.dnn.readNetFromONNX(str(settings.person_model_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def forward(self, blob):
        if self.session is not None:
            return self.session.run(None, {self.input_name: blob})[0]
        self.net.setInput(blob)
        return self.net.forward()

    def detect(self, frame):
        height, width = frame.shape[:2]
        regions = [(0, 0, width, height)]
        if self.settings.person_tiles > 1:
            regions += list(tile_regions(width, height, self.settings.person_tiles))
        boxes, scores = [], []
        for x1, y1, x2, y2 in regions:
            crop = frame[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            scale = 640 / max(h, w)
            resized = cv2.resize(crop, (round(w * scale), round(h * scale)))
            pad_x = (640 - resized.shape[1]) // 2
            pad_y = (640 - resized.shape[0]) // 2
            image = cv2.copyMakeBorder(
                resized,
                pad_y,
                640 - resized.shape[0] - pad_y,
                pad_x,
                640 - resized.shape[1] - pad_x,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            output = self.forward(
                cv2.dnn.blobFromImage(
                    image, 1 / 255, (640, 640), swapRB=True, crop=False
                )
            )
            found, values = decode_predictions(
                output,
                scale,
                (pad_x, pad_y),
                (x1, y1),
                frame.shape,
                self.settings.person_confidence,
            )
            for box, value in zip(found, values):
                # A tile that cuts off the head can create a second torso-only detection.
                if y1 > 0 and box[1] <= y1 + 4:
                    continue
                boxes.append(box)
                scores.append(value)
        if not boxes:
            return []
        xywh = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes]
        indices = cv2.dnn.NMSBoxes(xywh, scores, self.settings.person_confidence, 0.5)
        return deduplicate_people([boxes[int(i)] for i in np.asarray(indices).ravel()])


def deduplicate_people(boxes):
    """Merge head/upper-body copies from overlapping tiles, retaining distinct heads."""
    kept = []
    for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        duplicate = False
        for other in kept:
            intersection = max(0, min(box[2], other[2]) - max(box[0], other[0])) * max(
                0, min(box[3], other[3]) - max(box[1], other[1])
            )
            minimum = min(
                (box[2] - box[0]) * (box[3] - box[1]),
                (other[2] - other[0]) * (other[3] - other[1]),
            )
            widths = (box[2] - box[0], other[2] - other[0])
            close_x = abs((box[0] + box[2] - other[0] - other[2]) / 2) <= 0.25 * min(
                widths
            ) + 0.15 * max(widths)
            close_top = abs(box[1] - other[1]) <= max(
                8, 0.15 * min(box[3] - box[1], other[3] - other[1])
            )
            if intersection / max(1, minimum) >= 0.7 and close_x and close_top:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept
