# Public demo source and attribution

**Source:** Real World Engagement Attention Dataset (RWEAD 1.0), Marco Lemos,
Pedro J. S. Cardoso, João M. F. Rodrigues and collaborators.

- Official page: https://sites.google.com/view/rwead/rwead-1-0
- Dataset: https://osf.io/453sh/
- DOI: https://doi.org/10.17605/OSF.IO/453SH
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
  https://creativecommons.org/licenses/by/4.0/
- Official OSF license text: licenses/RWEAD_CC_BY_4.0.txt.
- Exact file paths, download URLs and SHA-256: rwead_provenance.json.

## Exact footage

Engagement Amphitheatre / Set2 / camera1_audience, **film_8.mp4** and
**film_51.mp4**. Each scene shows five participants together in the same physical
camera view. The public demo uses the first eight seconds of each recording,
retaining native 1920×1080 resolution at 20 FPS. The first arrangement is followed
by the second; the preview then repeats. IDs restart for the second recording
and are session-local spatial labels, not matched identities across clips.

RWEAD uses instructed/simulated engagement behaviours. Our displayed labels
are the application's current head/eye heuristic outputs, not source ground-truth
labels and not a claim about the participants' actual cognitive attention.

## Modifications

Trimmed each recording to eight seconds, removed audio, re-encoded as H.264,
added face boxes and heuristic attention labels, combined the annotated clips,
and generated a native-resolution looping GIF at a reduced frame rate (6 FPS).
The combined MP4 remains 20 FPS. No faces, people or camera views were composited
into either scene. The dataset authors do not endorse these predictions.

Source recordings are excluded from Git; only the credited derivative demo is
committed. Older AMI samples and restricted CDED-7 footage remain local.

## Requested dataset citation

Lemos, Marco, Cardoso, Pedro J. S., & Rodrigues, João M. F. (2026).
MiE: A Microscopic Model for Real-Time Group Engagement Estimation Using Gaze
and Posture. Journal of Computational Science, 96.
https://doi.org/10.1016/j.jocs.2026.102856
