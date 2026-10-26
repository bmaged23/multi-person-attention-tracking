# Multi-Person Visual Attention Tracking

[GitHub source, demo and full documentation](https://github.com/bmaged23/multi-person-attention-tracking)

YOLOv8n person detection runs with ONNX Runtime CUDA; MediaPipe head and eye
landmarks run on CPU. One shared pipeline tracks session-local spatial IDs,
estimates a visual heuristic score and produces JSON/JSONL and face-only H.264
annotations. An included web UI accepts video uploads and plays the result.
No cloud account or API key is required.

## Pull and run

Requires Linux x86-64 NVIDIA GPU containers. Tested native GPU: GTX 1660 Ti with 6 GiB VRAM.
The complete image contains both production model bundles and pinned libraries;
no Git clone or startup model download is needed.

```bash
docker pull bmaged23/multi-person-attention-tracking:gpu
docker run --rm --name attention --gpus all \
  -p 127.0.0.1:8770:8770 \
  -v attention-results:/app/data \
  bmaged23/multi-person-attention-tracking:gpu
```

After initialization, open **http://localhost:8770**. Upload a clip, select its
processing duration, and download the summary, observations or annotated MP4.
The volume retains bounded results. Stop with Ctrl+C. For CPU mode of this same
bundle omit `--gpus all` and add `-e ATTENTION_PERSON_DEVICE=cpu`.

## API and quality gate

HTTP endpoints expose readiness/configuration and sequential upload jobs with
status, summary, observations and video artifacts. Open `/docs` for interactive
examples. Defaults limit uploads to 64 MiB, source duration to 10 seconds and video
resolution to 1080p. This is a local single-user service without authentication.

[API guide](https://github.com/bmaged23/multi-person-attention-tracking/blob/main/docs/api/README.md) ·
[Docker guide](https://github.com/bmaged23/multi-person-attention-tracking/blob/main/deployment/docker/README.md) ·
[CI/CD guide](https://github.com/bmaged23/multi-person-attention-tracking/blob/main/docs/deployment/README.md)

The release workflow tests the candidate image with 100 human-annotated COCO images,
synthetic tracking trajectories, real API video jobs and browser upload/playback.
Only passing candidates are published; public distribution is then tested through
anonymous pull and actual GPU/CPU inference. Detailed measured results and release
evidence are recorded in the
[evaluation report](https://github.com/bmaged23/multi-person-attention-tracking/blob/main/docs/evaluation/README.md).

Native observations: detector median 0.401 s OpenCV CPU versus 0.070 s CUDA on 64 images,
with identical detection counts. Two 50-frame AMI workflows took 48.9 s versus 14.8 s
with identical observation JSONL. Component and workflow comparisons are distinct.
These small exposed suites do not establish broad accuracy or real-time throughput.

Attention labels are head/eye visual heuristics, not comprehension measurements.
IoU tracking can swap IDs at crossings and detection can miss occluded people.
Source is AGPL-3.0-only; models and sample media retain their upstream terms.
[Notices and attribution](https://github.com/bmaged23/multi-person-attention-tracking/blob/main/docs/THIRD_PARTY.md).


## Verified public release

[Passing build, GPU gate, publication and public inference](https://github.com/bmaged23/multi-person-attention-tracking/actions/runs/40317220913).
Immutable reference: `bmaged23/multi-person-attention-tracking@sha256:989a26171b94321e4e4541b0ae46789ca6b82e69b986667e49b3e394b5a65956`.

The image passed **41 tests**, **100 human-annotated detection cases**, four synthetic
tracking thresholds, two 50-frame API jobs and real Chromium upload/playback, with
zero JavaScript errors. Anonymous public pull and actual GPU/CPU-mode video jobs
both passed. Native detection results remained unchanged in the container.

The two GPU API jobs took **7.95/8.35 s**. Sampled cgroup RAM peaked at **1,160 MiB**,
whole-device VRAM at **179 MiB**; CPU averaged **237%**, peaked at **330%**, where 100%
is one core. Thirteen samples cover the API/worker/FFmpeg phase; peaks can be missed.
The image is ~2.20 GB compressed in the registry and reports 6.68 GB locally.
These are development observations on one setup. The heuristic attention score
has no human attention-accuracy estimate, and the synthetic crossing limitation
remains visible. See the linked evaluation report for the full method and failures.
