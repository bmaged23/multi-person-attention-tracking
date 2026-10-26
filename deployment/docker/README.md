# Docker — complete video analysis application

[Main README](../../README.md) · [Docker Hub](https://hub.docker.com/r/bmaged23/multi-person-attention-tracking) · [API](../../docs/api/README.md) · [CI/CD](../../docs/deployment/README.md)

The image packages the upload UI, API, CLI, exact Python requirements, YOLOv8n ONNX
and MediaPipe Face Landmarker. Models are checksum-verified at build time. No Git
clone, cloud account, API key or startup weight download is required for image use.
Public release qualification is being prepared; final digest/results will be
recorded after the complete GPU gate and public pull checks pass.

## GPU pull and run

Requires Linux x86-64 containers, working NVIDIA drivers and Docker GPU support.
On WSL2, enable Docker Desktop integration for the distro. The native benchmark
uses a GTX 1660 Ti with 6 GiB VRAM; face landmarks remain on CPU. Allow 15 GiB disk
headroom for downloading/unpacking CUDA libraries and models.

```bash
docker pull bmaged23/multi-person-attention-tracking:gpu
docker run --rm --name attention --gpus all \
  -p 127.0.0.1:8770:8770 \
  -v attention-results:/app/data \
  bmaged23/multi-person-attention-tracking:gpu
```

Wait for model initialization and open **http://localhost:8770**. Upload a video,
choose its processing duration, then play the output and download reports. No
camera-device passthrough is needed for uploaded files. Stop with Ctrl+C or
`docker stop attention`. The named volume retains results, bounded by job retention.

```bash
curl --fail http://localhost:8770/health
curl --fail http://localhost:8770/config
docker logs attention
```

Health reports the selected backend and devices; startup fails rather than silently
using CPU when CUDA person detection was requested but cannot initialize.
This measures successful startup, not guaranteed attention accuracy on new scenes.

## CPU mode of the same bundle

The image also contains the ONNX CPU execution provider. Override the person device
and omit `--gpus`:

```bash
docker run --rm --name attention \
  -e ATTENTION_PERSON_DEVICE=cpu \
  -p 127.0.0.1:8770:8770 \
  -v attention-results:/app/data \
  bmaged23/multi-person-attention-tracking:gpu
```

The bundle still includes CUDA libraries; CPU mode does not make it a smaller image.
Both modes must pass actual public-image video-job checks before being called
verified. Native macOS/ARM containers are outside this release's tested scope.

## Try a licensed sample without cloning

The source repository provides a short, lossless AMI CC-BY 4.0 fixture, separate
from the inference image:

```bash
curl --fail -L -o ami-L.mkv \
  https://raw.githubusercontent.com/bmaged23/multi-person-attention-tracking/main/evaluation/fixtures/ami-L.mkv
curl --fail -F 'video=@ami-L.mkv' \
  'http://localhost:8770/jobs?max_seconds=2'
```

Poll and download the returned job's artifacts as shown in the
[API README](../../docs/api/README.md). Attribution and transformations are recorded
in [the fixture README](../../evaluation/fixtures/README.md).

## Configuration, updates and local builds

Pass runtime settings with `-e ATTENTION_...=...`, or use a private `.env` and
`--env-file .env`. For example `-e ATTENTION_API_MAX_CLIP_SECONDS=5` shortens the
service's source-duration cap. GPU person detection uses `PERSON_BACKEND=onnx`,
`PERSON_DEVICE=cuda`; `DEVICE=cpu` controls the separate landmark delegate.
A copied native `.env.example` defaults person detection to OpenCV CPU, so adjust
these values if supplying it to the GPU container. Do not change worker count above 1.

For an update, stop the old container, pull `:gpu` again and repeat the run command.
For a local build, clone and enter the source repository, then:

```bash
docker build -f deployment/docker/Dockerfile \
  --build-arg VCS_REF="$(git rev-parse HEAD)" -t attention:local .
docker run --rm --gpus all -p 127.0.0.1:8770:8770 attention:local
```

The image includes upstream model/license notices and a corresponding-source label.
It excludes uploads, raw evaluation images/clips and private local experiments.
See [third-party terms](../../docs/THIRD_PARTY.md) and the root AGPL license.
