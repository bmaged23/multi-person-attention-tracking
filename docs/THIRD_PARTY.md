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

AMI Meeting Corpus, AMI Consortium: https://groups.inf.ed.ac.uk/ami/download/
Video signals are CC BY 4.0. The public README media uses trimmed, annotated
left/right camera excerpts from IS1004d, with credit and modification details
in DEMO_SOURCE.md. The official license text is under licenses/AMI_CC_BY_4.0.txt.
