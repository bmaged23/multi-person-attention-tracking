# Recorded measurements

These are useful reproducibility evidence, not temporary run outputs. Raw videos,
per-image predictions, service job folders and exploratory logs stay local/ignored.

| Files | Meaning |
| --- | --- |
| `public-release.json` | Exact source, public digest, size and verified distribution modes |
| `qualified-container.json` | Full image quality-gate aggregate and frozen policy |
| `container-resources.json` | Sampled cgroup CPU/RAM and whole-device GPU usage |
| `detector-opencv-baseline.json`, `detector-onnx-cpu.json`, `detector-onnx-cuda-v2.json` | Matched 64-image runtime comparison at original confidence |
| `detector-cuda-confidence035.json` | Exposed development threshold candidate |
| `holdout-baseline.json`, `holdout-confidence035.json` | Reserved 36-image threshold comparison |
| `workflow-original-cpu.json`, `workflow-cuda-8.json`, `workflow-cuda-4.json` | Native 100-frame route/landmark-worker comparison |
| `tracking-baseline.json` | Synthetic known-identity association stress |

`v2` in the recorded CUDA filename identifies the successful initialization run,
not a second model or changed weights. Native and container metrics have different
resource scopes. See the [evaluation method](../README.md) before comparing values.
