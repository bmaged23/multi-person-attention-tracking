# Video analysis API

[Main README](../../README.md) · [Docker](../../deployment/docker/README.md) · [CI/CD](../deployment/README.md)

Run the image or start `python -m uvicorn attention_tracking.api:app --host 127.0.0.1 --port 8770 --workers 1` from the repository root.
Open **http://localhost:8770** to upload a clip and play its annotated output.
Interactive HTTP documentation is at `/docs`; the schema is `/openapi.json`.
Model initialization completes before readiness. There are no external API keys.

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /` | Video upload and annotated playback UI |
| `GET /health` | Readiness, busy state, person runtime/device and landmark device |
| `GET /config` | Public upload/job limits and attention-method description |
| `POST /jobs?max_seconds=2` | Multipart `video` upload; returns 202 and job ID |
| `GET /jobs/{id}` | `running`, `completed` or `failed` status |
| `GET /jobs/{id}/result` | Completed per-track summary and stage timings |
| `GET /jobs/{id}/observations` | JSONL with each frame's spatial tracks and measured signals |
| `GET /jobs/{id}/video` | Inline H.264/yuv420p MP4, with browser range support |

```bash
curl --fail http://localhost:8770/health
curl --fail -F 'video=@evaluation/fixtures/ami-L.mkv' \
  'http://localhost:8770/jobs?max_seconds=2'
# Copy the returned id into JOB_ID, then poll until completed:
curl --fail http://localhost:8770/jobs/JOB_ID
curl --fail http://localhost:8770/jobs/JOB_ID/result
curl --fail -o annotated.mp4 http://localhost:8770/jobs/JOB_ID/video
curl --fail -o observations.jsonl http://localhost:8770/jobs/JOB_ID/observations
```

`max_seconds` limits the beginning of the video, not elapsed inference time.
The browser lets you select that duration. Processing every frame can be slower
than playback speed. CLI camera capture remains separate; this HTTP API processes
uploaded files and does not implement a live-camera WebSocket protocol.

## Bounds and lifecycle

`ServiceSettings` in `attention_tracking/config.py` exposes environment settings:

| Environment key (prefix `ATTENTION_`) | Default |
| --- | --- |
| `API_MAX_UPLOAD_MB` | 64 MiB of file content |
| `API_UPLOAD_TIMEOUT_SECONDS` | 60 seconds to receive the request body |
| `API_MAX_CLIP_SECONDS` | 10 seconds of source video |
| `API_MAX_FRAME_PIXELS` | 2,073,600 pixels |
| `API_MAX_SOURCE_FPS` | 60 |
| `API_RETAINED_JOBS` | 10 |
| `API_DATA_DIR` | `data/api` |

Only one upload or inference job is admitted at a time. HTTP 409 indicates busy
capacity or a requested unfinished artifact; 413 indicates an oversized upload;
415 an unsupported container; 422 an invalid video or limit; 404 an unknown job.
A receive deadline returns 408. A separate ASGI limiter bounds multipart bytes
before parsing, with 64 KiB reserved for framing, including chunked requests.
MP4, MOV, AVI, WebM and MKV signatures are accepted; source FPS/resolution and each
decoded frame's pixel count are checked. Remote URLs are not accepted as sources.

Models stay loaded in one inference worker. Completed jobs expose all three
artifacts. Input uploads are deleted after success or failure; output retention is
bounded, with the oldest jobs removed on new submissions. Marked job directories
are recovered on restart and incomplete jobs become failed. A Docker volume keeps
results across container replacement. Report downloads omit the server input path.

Bind to localhost: this is a single-user local API with no authentication or tenant
isolation. Run one web worker; multiple workers would duplicate models and break
the in-memory admission rule. There is no job-cancellation endpoint or wall-clock
inference deadline. Stopping a process interrupts its job; clip limits bound source
duration, not the worst-case processing time of every possible video.

## Meaning of outputs

A person box is a detector observation; track IDs are session-local spatial labels.
`no_face` means a person box had no measurable face, with `face_bbox:null` and
`is_attentive:null`. The labels `attentive`, `partial`, `non_attentive` and `drowsy`
come from the explicit visual score. They do not establish comprehension, identity
or a clinical assessment. See [evaluation and limitations](../evaluation/README.md).

Per-track summary fractions exclude `no_face` and count observations rather than
elapsed time. Video annotations contain no source audio.
