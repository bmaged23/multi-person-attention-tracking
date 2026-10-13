# Multi-Person Attention Tracking

[![MediaPipe · Face Landmarker](https://img.shields.io/badge/MediaPipe-Face%20Landmarker-blue)](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
[![YOLOv8n · Person detection](https://img.shields.io/badge/YOLOv8n-Person%20Detection-38bda7)](https://docs.ultralytics.com/models/yolov8/)

Follow multiple people in a **video or live camera stream**, estimate visual
attention from head pose and eye landmarks, and export face-only annotations
with per-person results. One shared pipeline serves both entry points.

**YOLOv8n** locates people, **MediaPipe Face Landmarker** measures faces,
**IoU association** retains session-local IDs, and an explicit weighted rule
combines head direction, eye opening, iris position and face visibility.
Names such as Person 1 are spatial labels; no face recognition or enrollment is required.

## Demo

![Looping single-camera classroom audience attention demo](docs/media/attention-demo.gif)

**Five people together in one camera view**, across two amphitheatre seating
arrangements. Face-only boxes, session-local names and attention labels;
the combined preview repeats continuously.

Source: **RWEAD 1.0**, Lemos, Cardoso, Rodrigues and collaborators, **CC BY 4.0**.
Excerpts are trimmed, annotated and combined; labels are our heuristic results.
[Attribution and modifications](docs/DEMO_SOURCE.md) · [Full-resolution MP4](docs/media/attention-demo.mp4)

## Pipeline

![Animated video and camera attention pipeline](docs/media/pipeline.gif)

[Scalable SVG](docs/media/pipeline.svg) · [Architecture](docs/ARCHITECTURE.md)

Every decoded frame receives fresh detections and attention measurements.
Person boxes drive tracking; only the current face bounds are drawn in the
annotated video. A person whose face cannot be measured remains in the report
as `no_face`, with no drawn body box or guessed attention label.

The animation and static diagram share one layout:
`python scripts/build_pipeline_diagram.py`.

## From cloning to the first run

Requires **Python 3.12**, **Git**, and **FFmpeg**. The tested inference path is CPU;
no external recognition API, database, PyTorch installation or training step is needed.

### 1. Clone and create the environment

```bash
git clone https://github.com/bmaged23/multi-person-attention-tracking.git
cd multi-person-attention-tracking
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
cp .env.example .env
```

On Windows, create the environment with `py -3.12 -m venv .venv`, activate it
with `.venv\Scripts\Activate.ps1`, and copy the configuration using
`Copy-Item .env.example .env`. Install FFmpeg and add it to PATH.

Ubuntu/WSL system libraries, if not already installed:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libglib2.0-0
```

Alternatively, an installed `uv` can create the environment and install the same pins:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
cp .env.example .env
```

`requirements.txt` is the exact 22-package snapshot of this project's own environment,
including transitive dependencies and Ruff. `requirements-dev.txt` uses the same snapshot.
FFmpeg is a system executable, not a Python requirement.

### 2. Download the two models

```bash
python scripts/download_model.py
python scripts/download_person_model.py
```

These download Google's versioned float16 **Face Landmarker task bundle** and a
revision-pinned **YOLOv8n COCO ONNX export**. Both scripts verify SHA-256 checksums.
Weights go into ignored `models/`; cloning an external model repository is unnecessary.
See [model provenance](docs/person_detector_provenance.json) and
[third-party terms](docs/THIRD_PARTY.md).

### 3. Process your video

Put a recording at `input/example.mp4`, or supply any accessible video path:

```bash
python video_attention.py --video input/example.mp4 \
  --output output/example.json \
  --annotated-video output/example_annotated.mp4
```

For a quick check, append `--max-seconds 5`. Processing uses every frame and can
be slower than real time for crowded scenes with multiple landmark retries.

### 4. Use a live camera

```bash
python live_camera_attention.py --source 0 --display \
  --output output/camera.json \
  --annotated-video output/camera_annotated.mp4
```

Press **Q** to end a displayed session. `--source` accepts a camera index or
stream URL; `--max-seconds 60` stops after a fixed duration. Display requires
a graphical desktop. WSL needs camera passthrough; a saved webcam recording
can use the video entry point. Physical-camera and RTSP capture were not verified
in the current environment.

## Configuration

Copy `.env.example` to `.env` once. Settings load from the repository root,
with precedence **CLI flags → shell environment → `.env` → code defaults**.
The example contains working defaults; credentials are not required.
`.env` stays private and `.env.example` is committed.

| Setting / flag | Default | Purpose |
|---|---:|---|
| `ATTENTION_IOU_THRESHOLD` / `--iou-threshold` | 0.5 | Required overlap to reuse a track ID |
| `ATTENTION_PERSON_CONFIDENCE` / `--person-confidence` | 0.15 | Person detector confidence |
| `ATTENTION_PERSON_TILES` / `--person-tiles` | 2 | Full-frame plus overlapping regions |
| `ATTENTION_DETECTION_CONFIDENCE` / `--detection-confidence` | 0.35 | Face detector confidence |
| `ATTENTION_TILE_WORKERS` / `--tile-workers` | 8 | Independent CPU landmark instances |
| `ATTENTION_DEVICE` / `--device` | cpu | MediaPipe delegate |
| `ATTENTION_TRACK_TIMEOUT` | 1.0 s | Retain a missing track before expiration |

One previous track cannot match two current boxes. Overlap below the threshold
creates a new ID; occlusion and changing box sizes can fragment identities.
Increasing detection counts does not establish complete person recall.

`--names names.json` supplies optional labels:

```json
{"person_001": "Alex", "person_002": "Jordan"}
```

These names apply to track IDs within a session, not biometric identities.
Use `--landmarks` to draw sampled face points. `--detector faces` is an optional
face-only inspection mode with `--tiles` and `--max-faces`; person-first mode
is the default. Run either entry point with `--help` for all options.

## How attention is estimated

For each person, the pipeline crops the upper 40% of a tall body box, applies
15% padding, and measures the face with MediaPipe. Small crops are enlarged;
failed landmarks retry at a larger size and with contrast enhancement.

Facial landmarks produce three explicit signals:

1. **Head pose:** six-point `solvePnP` estimates yaw and pitch.
2. **Eye opening:** eye aspect ratio (EAR) compares eyelid distances.
3. **Iris position:** a Euclidean-distance ratio is averaged across both eyes.

| Established setting | Value |
|---|---:|
| Yaw limit | 39.4° |
| Pitch interval | −24.6° to 41.6° |
| EAR closed-eye threshold | 0.092 |
| Central iris interval | 0.37–0.63 |
| Head / EAR / iris / visibility weights | 0.50 / 0.15 / 0.10 / 0.25 |
| Attentive / partial score thresholds | 0.50 / 0.25 |

The head component is 1 when both direction tests pass, 0.5 when one passes,
and 0 when neither passes. Eye opening contributes 0 or 1; central iris position
contributes 1, otherwise 0.3. Unavailable individual signals contribute the
established neutral value of 0.5. Face visibility contributes 0.25.

A low EAR takes label priority as `drowsy`; otherwise the weighted score selects
`attentive`, `partial`, or `non_attentive`. Missing facial measurements yield
`no_face`. The established rule is preserved without score smoothing, a hard
head-direction veto, or a sustained eye-closure timer.

These are **visual heuristics**, not confidence probabilities or evidence of
comprehension or sleep. Camera placement, teacher direction, occlusion, small
faces and glasses affect measurements. No ground-truth attention accuracy is
claimed. Thresholds are explicit in `.env.example` and
[the recorded baseline](docs/original_attention_defaults.json).

## Outputs

```text
output/example.json                # settings and per-track summary
output/example_observations.jsonl  # one current-frame observation per person
output/example_annotated.mp4        # source resolution, H.264/yuv420p, no audio
```

Observations include `id`, `name`, source `time`, person `bbox`, visible
`face_bbox`, `label`, `is_attentive`, `score`, and head/eye signals. Missing faces
have `face_bbox: null`, score 0 and `is_attentive: null`. Summary fractions
exclude `no_face` and count observations rather than elapsed duration.

JSONL is streamed to limit memory use; the final summary is written atomically.
Face labels use text-sized black backgrounds. Person/body boxes stay in the data
and are never used as the visible annotation rectangle.

## Reproduce the public demo

The public demo uses **RWEAD 1.0**, a CC BY 4.0 amphitheatre attention dataset.
Both excerpts show five people together from **camera1_audience**, with
[source attribution and modification details](docs/DEMO_SOURCE.md). The earlier
research-only classroom footage stays local and is not part of the README demo.

```bash
python scripts/datasets/prepare_public_demo.py
python video_attention.py --video input/rwead/Set2_camera1_film8.mp4 \
  --max-seconds 8 --output output/rwead/film8.json \
  --annotated-video output/rwead/film8_annotated.mp4
python video_attention.py --video input/rwead/Set2_camera1_film51.mp4 \
  --max-seconds 8 --output output/rwead/film51.json \
  --annotated-video output/rwead/film51_annotated.mp4
python scripts/build_video_demo.py \
  --videos output/rwead/film8_annotated.mp4 output/rwead/film51_annotated.mp4 \
  --output-dir docs/media --preview-width 1920 --preview-fps 6
```

The combined MP4 retains **1920×1080 at 20 FPS**. The looping GIF retains the
same resolution at 6 FPS. IDs restart for the second recording; people are not
identified across clips. The GIF loops where animation playback is enabled.
The generator also creates a local autoplay/muted/loop HTML player.

For your own recordings, supply `--videos first.mp4 second.mp4` and an output
folder such as `output/demo`. Keep the attribution with any RWEAD-derived media.

## Structure

```text
video_attention.py                 # input-video entry point
live_camera_attention.py           # camera/stream entry point
config.py                          # shared Settings export
.env.example                       # reproducible configuration
attention_tracking/
  config.py                        # validated environment/default settings
  detection/                       # person detection, head crops, landmarks
  analysis/                        # pose, eye metrics, weighted scoring
  tracking/                        # one-to-one IoU association
  video/                           # frame loop, annotation, encoding, reports
  cli.py                           # shared command-line configuration
scripts/
  download_model.py                # checksum-verified MediaPipe bundle
  download_person_model.py         # checksum-verified YOLO ONNX export
  build_pipeline_diagram.py        # matching SVG and looping GIF
  build_video_demo.py              # combine videos and generate looping preview
  datasets/                        # optional research sample utilities
tests/                             # scoring, tracking, drawing and configuration
docs/                              # architecture, provenance, validation
  media/                           # redistributable diagram assets
  evaluation/                      # current and historical measurements
description/                       # ignored local project/change notes
```

Environments, model files, footage, outputs, caches, and local descriptions stay
out of Git. The standalone attention project does not depend on the separate
face recognition project.

## Validation

```bash
python -m unittest discover -s tests -v
ruff check .
ruff format --check attention_tracking scripts tests config.py video_attention.py live_camera_attention.py
python -m pip check
```

Tests cover established score thresholds, pose geometry, strict IoU matching,
missing faces, streamed reports, face-only annotation, environment precedence,
and input protection. The two complete classroom videos passed full FFmpeg
decode checks at 1920×1080, 30 FPS. [Validation and reproducibility](docs/VALIDATION.md)
distinguish implementation checks from unmeasured attention accuracy.
