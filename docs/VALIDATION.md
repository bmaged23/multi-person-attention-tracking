# Validation

## Current baseline

The original attention calculations and configured defaults are restored.
The active implementation is protected by 30 regression/integration tests.
Original thresholds are persistently recorded in original_attention_defaults.json.

Python 3.12.15, MediaPipe 0.10.35, the original CPU delegate, and the pinned
22-package environment is pinned exactly, including Ruff and python-dotenv.
The inference-library versions are retained. The person-box model uses
existing OpenCV DNN CPU; python-dotenv only adds configuration-file loading.

Tests verify original score weights and labels, head-direction component rules
without the replacement veto, EAR threshold boundaries, iris Euclidean distance,
missing-signal neutral fallbacks, six-point solvePnP, original head cropping,
strict configurable IoU, one-to-one matching, current IDs, no-face person
coverage, streaming summaries, source protection, and detector coordinate mapping.
Formatting, lint, and dependency compatibility checks also pass.

## Reorganization regression

After reorganizing utilities and adding .env configuration, the first 30 frames
of each classroom source were processed again. All 2,187 observation records
matched their saved baseline exactly, including IDs, names, times, person boxes,
face bounds, scores, labels and measured signals. Evidence is recorded in
evaluation/reorganization_regression.json. This verifies the compared segments;
it is not a guarantee across every machine, input or camera.

A clean staged checkout in a separate directory and new Python 3.12.15 virtual
environment installed all 22 pinned packages from the package index. Both model
downloads passed checksum verification; all 30 tests, lint, formatting and
dependency checks passed. A two-frame video run produced 68 observations exactly
equal to the recorded baseline, and its annotated MP4 decoded successfully.

## Actual video checks

Two 1080p classroom clips are regenerated with person-first detection and the
restored attention rules. Each source/output has 300 frames at 30 FPS. Videos
use H.264/yuv420p and are decoded completely with FFmpeg before completion.
Current observations include person boxes even without face landmarks. Video
annotations draw only landmark-derived face bounds; no-face observations are
retained in JSON but have no drawn box. JSONL records both bbox and face_bbox.
Current measurements are written to evaluation/classroom_restored_validation.json.

evaluation/historical/classroom_validation.json records the previous face-only experiment. Its counts
are face detections and must not be presented as the current person counts.
evaluation/historical/validation_results.json records older television smoke tests. Neither historical
report establishes ground-truth attention accuracy for the restored rule.

## Detection versus identity and attention

More located person boxes do not prove perfect person recall. Head/upper-body
copies from overlapping regions need deduplication; occlusion and changing box
extent can still fragment a track. Dummy IDs are spatial session-local labels,
not face-recognized identities. Missing face measurements produce no_face;
they must not be guessed to be attentive or non-attentive from a stale score.

The original source weighted rule is deliberately preserved, including its
visibility term and one-observation drowsy threshold. This restoration is not a
clinical sleep assessment or an attention-accuracy benchmark. No prior per-run
calibration file existed in the supplied checkout, so the recovered configured
defaults are used without inventing a calibration history.

## Model provenance

Face Landmarker: official Google float16 task bundle, version 1.
SHA-256: 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

Person detector: revision-pinned YOLOv8n COCO ONNX export; see
person_detector_provenance.json for source revision and checksum.

The original processing.py used CPU; the separate original benchmark offered
an optional GPU mode. A local diagnostic initialized EGL outside the sandbox
but reported llvmpipe software rendering. That did not replace the original
CPU path or establish NVIDIA hardware acceleration. Physical-camera/RTSP and
hardware GPU capture/inference remain unverified here.

## Public AMI demo

Two 12-second native 720×576, 25 FPS excerpts from AMI meeting IS1004d
were processed with the unchanged default settings. Their annotated output
is combined into a 600-frame, 24-second H.264 MP4 and an infinitely looping
192-frame, 8 FPS GIF. Full combined-video decoding, dimensions, frame counts,
GIF loop metadata and repository asset sizes are verified.

Source credit and CC BY 4.0 modification details are in DEMO_SOURCE.md;
measurements are in evaluation/public_demo_validation.json. These clips are
qualitative demonstrations, not a ground-truth attention-accuracy benchmark.
