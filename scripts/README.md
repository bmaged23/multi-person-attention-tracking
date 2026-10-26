# Project tools

Run commands from the repository root with the project's environment activated.
The [main README](../README.md) contains application setup; these folders separate
provisioning, evaluation, documentation and release responsibilities.

| Folder | Purpose | Main command |
| --- | --- | --- |
| `setup/` | Checksum-verified production models | `python -m scripts.setup.download_face_landmarker`; `python -m scripts.setup.download_person_detector` |
| `datasets/` | Current COCO evaluation inputs, AMI benchmark clips and original public demo source | `python -m scripts.datasets.prepare_detection_evaluation` |
| `datasets/research/` | Optional older TVHI/CDED/RWEAD experiments | See its research README; these are not current release inputs |
| `evaluation/` | Independent person metrics, synthetic tracking and real workflow profiling | `python -m scripts.evaluation.evaluate_detector --help` |
| `documentation/` | Diagrams, videos and charts from existing results | `python -m scripts.documentation.render_pipeline_diagram` |
| `release/` | Exact-image quality gate, public smoke checks and overview synchronization | `python -m scripts.release.qualify_image --help` |

A source tool's move does not move model files, change scoring thresholds or rename
published media. The [path migration map](../docs/reference/tool-path-migration.json)
records former filenames; duplicate implementations and obsolete wrappers are not
kept. The three root application/config entry points remain for CLI compatibility.

Downloads can use storage/network access. Dataset source terms are documented in
[third-party notices](../docs/THIRD_PARTY.md). Release publication uses stored CI
credentials; normal inference uses no account or API key.
