# Public classroom demo source and attribution

**Source:** AIRC-SMARTCLASS: A Multi-Modal Classroom Dataset for Automatically
Controlling Devices in Smart Classroom Part 2, version 1.

**Contributors:** Van Anh Nguyen Thi, Tu Chu Pham Dinh, Son Le Hoang (2026).

- Dataset: https://data.mendeley.com/datasets/fw5hs57z78/1
- DOI: https://doi.org/10.17632/fw5hs57z78.1
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
  https://creativecommons.org/licenses/by/4.0/
- License text: licenses/CC_BY_4.0.txt.
- Source file, download URL and publisher's SHA-256: airc_provenance.json.

## Exact footage

Original recording **018.avi**, from a single wall-mounted camera in the
Computer Laboratory, University of Transport Technology, Hanoi, Vietnam.
The demo contains excerpts from **30–38 seconds** and **90–98 seconds** of the
same recording. Both show substantially more than ten students in one physical
camera view; people and camera views are not composited.

Each excerpt retains native 1920×1080 resolution at 24.75 FPS and is processed
on every decoded frame. The combined MP4 lasts 16 seconds. The GIF retains the
same resolution at 5 FPS and loops indefinitely. Names/IDs restart for the second
excerpt and are session-local spatial labels, not recognized identities.

## Modifications and inference settings

Trimmed to eight seconds per excerpt, removed audio, re-encoded as H.264, added
face boxes and heuristic attention labels, combined both excerpts, and generated
a reduced-frame-rate looping GIF.

The existing --person-tiles 4 option is used for this crowded, occluded view;
other default model, tracking and attention-score settings are unchanged.
Monitors hide bodies/faces and distant heads are small, so not every visible
student receives a valid face measurement on every frame. Detection counts
and fragmented track IDs must not be reported as participant counts.

Displayed labels are this application's heuristic predictions, not source
attention ground truth, verified cognitive attention, or a clinical assessment.
The dataset contributors do not endorse this application or its predictions.

Source recordings remain local and excluded from Git. The credited derivative
MP4 and GIF are publicly included under CC BY 4.0. Earlier AMI, RWEAD and
research-restricted CDED-7 samples remain local; they are not the current demo.


## Upload-interface screenshot

`media/upload-api.png` shows the new API processing an AMI IS1004d left-camera
fixture under a two-second test cap. This is separate from the AIRC classroom
preview above. AMI source, CC-BY4.0 license and lossless modifications are recorded
in [evaluation/fixtures/README.md](../evaluation/fixtures/README.md).
