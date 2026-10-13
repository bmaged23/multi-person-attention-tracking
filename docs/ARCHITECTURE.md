# Architecture

The core analysis follows the original project's per-face calculations and
weighted score. Detection/identity assignment are organized independently so
attention does not require classroom filenames or an external recognition service.

## Person-first observation

OpenCV DNN runs a YOLOv8n COCO ONNX export, retaining person-class boxes only.
Full-frame and overlapping-region observations are merged before tracking;
head/upper-body copies from different regions are suppressed with containment
and aligned-head geometry, retaining distinct neighboring heads.

Each current person box gets the original head-zone crop. Independent MediaPipe
0.10.35 CPU model instances analyze crops in parallel. The integrated landmark
bundle includes a face detector; this replaces the absent standalone detector
asset rather than fabricating recognition output. Face-finding retries retain
320/640 upscaling and contrast enhancement. The old small-crop square resizing
is preserved and measurements use that analysis space before landmarks are
mapped back to source pixels for drawing.

A missing face produces the original no_face observation without removing the
person's body box. Current person boxes are associated one-to-one through
Hungarian assignment, requiring the configured IoU threshold. Center distance
only breaks ties among already eligible matches; it cannot bypass IoU.

## Restored attention core

face_metrics.py uses the original six-point solvePnP template, crop-width camera
approximation, eye-aspect ratio, and averaged Euclidean iris-distance ratio.
scoring.py uses original neutral fallbacks, component weights, and label order.
The refactor's transformation-matrix pose, per-eye iris deviation, higher EAR
threshold, hard pose veto, and temporal closure/smoothing were removed.
The configured source defaults are frozen in original_attention_defaults.json.

## Reporting

The shared frame loop powers video/camera commands and streams observations to
JSONL. Per-track summaries are written atomically after processing. Annotation
uses face-mesh bounds only, measured text widths, and original labels. Missing
faces remain in tracking and JSON but are not drawn. H.264 video
encoding retains source frame count and removes its temporary recording.

Dummy IDs are session-local spatial tracks. Missing detections, occlusion, and
large box changes can fragment them. No identity embedding, enrollment service,
room/date filename parser, or classroom-specific trainee schema is required.
