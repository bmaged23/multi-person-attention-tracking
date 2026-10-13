# Seated classroom test footage

## Source

Classroom Distraction Evaluation Dataset (CDED-7):
https://github.com/fangsheng223/classroom-distraction-tracker

Raw video archive: https://zenodo.org/records/21207208

The source includes seven classroom clips. Class 6 and class 7 show seated
university students in a classroom, with multiple visible faces. Both are
1920x1080, 30 FPS, and 10 seconds long. They replace the TVHI clips as the
preferred context-specific test examples; the earlier samples remain available.

The authors restrict footage to research and non-commercial use. Downloads and
annotated outputs are kept locally under ignored input/output directories.

## Reproduce with restored attention rules

```sh
python scripts/datasets/prepare_classroom_videos.py
python scripts/download_model.py
python scripts/download_person_model.py
python video_attention.py --video input/classroom/class_6.mp4 \
  --output output/classroom/class_6.json \
  --annotated-video output/classroom/class_6_annotated.mp4 --iou-threshold 0.5
python video_attention.py --video input/classroom/class_7.mp4 \
  --output output/classroom/class_7.json \
  --annotated-video output/classroom/class_7_annotated.mp4 --iou-threshold 0.5
```

The original score formula, configured thresholds, solvePnP pose, eye-aspect
ratio, and iris-distance average are restored. Person-first detection locates
bodies before per-person MediaPipe analysis. A missing face retains a person
box/ID with the original no_face label. The original MediaPipe 0.10.35 CPU
delegate is retained. IoU >= 0.5 is required to reuse an ID/name.

The earlier refactor's face-only default and stronger head-direction veto
caused reduced coverage and different labels. Old face-only counts are retained
as historical evidence, not a benchmark of the current person-first detector.
Current validation is recorded separately after regeneration.

These examples remain qualitative. The source camera is offset from the teacher,
small/profile faces are difficult, and the original score is a visual heuristic.
No ground-truth classroom attention accuracy or complete person recall is claimed.

## Webcam learner alternative

A Windows Camera recording of a learner watching a lecture can use the same
pipeline. This WSL environment currently exposes no camera device. Pass a saved
recording to video_attention.py. The default original rules also apply here;
no separate unrequested webcam scoring rule is used.
