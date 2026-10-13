# Third-party models and footage

## Models

- MediaPipe Face Landmarker: official Google version-1 float16 task bundle.
  Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
  Upstream MediaPipe source: https://github.com/google-ai-edge/mediapipe
  The source project uses Apache-2.0. Model files retain their upstream terms.
- YOLOv8n COCO ONNX: revision-pinned webml export of Ultralytics YOLOv8n.
  Source revision and SHA-256: person_detector_provenance.json.
  Export: https://huggingface.co/webml/yolov8n
  Upstream: https://github.com/ultralytics/ultralytics
  Ultralytics supplies AGPL-3.0 and enterprise licensing options. Model use retains
  those terms; this repository does not relicense upstream model weights.

The application performs inference. It does not train either model. Model weights
are downloaded directly from their recorded upstream sources and excluded from Git.

## Classroom research clips

Source: https://github.com/fangsheng223/classroom-distraction-tracker
Release: https://zenodo.org/records/21207208
Terms: https://github.com/fangsheng223/classroom-distraction-tracker/blob/main/DATASET_LICENSE.md

The research-only license applies to frames, crops and annotated visual derivatives.
It explicitly prohibits public web folders and publicly indexed video embeds,
restricts redistribution and commercial use, and prohibits re-identification.
Local footage and combined previews are excluded from Git. Do not copy them into
public README media without the maintainers' written permission.

TVHI clips are also research-only copyrighted footage; see TEST_DATA.md.
Use your own or suitably licensed videos for a public portfolio demonstration.

## Redistributable public demo

AIRC-SMARTCLASS Part 2, version 1:
https://data.mendeley.com/datasets/fw5hs57z78/1
Contributors: Van Anh Nguyen Thi, Tu Chu Pham Dinh, Son Le Hoang (2026).
The dataset's own page and metadata specify CC BY 4.0. The README uses annotated
excerpts from recording 018.avi, showing more than ten students in one camera
view. Source credit and modification details: DEMO_SOURCE.md. License text:
licenses/CC_BY_4.0.txt. Provenance and original source checksum:
airc_provenance.json.

Older RWEAD 1.0 and AMI samples remain local under their CC BY 4.0 terms. Their
metadata and historical validation reports are retained separately. CDED-7
restricted classroom footage is not used in public media.
