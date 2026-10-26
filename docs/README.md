# Documentation index

| Guide | Scope |
| --- | --- |
| [Main README](../README.md) | Overview, demos, measured results and source setup |
| [Architecture](ARCHITECTURE.md) | Current detection, landmarks, association, scoring and outputs |
| [API](api/README.md) | Upload jobs, formats, endpoints, limits and result lifecycle |
| [Docker](../deployment/docker/README.md) | Complete image, GPU/CPU pull/run, sample and updates |
| [CI/CD](deployment/README.md) | Hosted build, GPU quality gates and public verification |
| [Evaluation](evaluation/README.md) | Current independent detection metrics, tracking stress and workflow measurements |
| [Media attribution](DEMO_SOURCE.md) | Sources and modifications for the published demo/screenshot |
| [Third-party terms](THIRD_PARTY.md) | Application, weights and footage notices |
| [Tool path migration](reference/tool-path-migration.json) | Descriptive setup/media/research tool moves |

## Evidence and reference files

`evaluation/charts/` contains figures generated from recorded measurements.
`evaluation/measurements/` contains current native and exact-container aggregate
JSON. Its [index](evaluation/measurements/README.md) explains each comparison.
`evaluation/historical/` and the older compatibility report paths are indexed
[separately](evaluation/historical/README.md). Historical reports are not current
GPU performance or behavioral attention accuracy.

The small provenance JSON files and `original_attention_defaults.json` retain their
stable locations because setup/research tools and published references use them.
`media/` contains reviewed redistributable demo/diagram assets; `licenses/` holds
upstream texts. Private recordings, model weights, caches and raw local runs are
outside this tracked documentation tree.

## Earlier research guides

[CPU validation](VALIDATION.md), [classroom research](CLASSROOM_TESTS.md) and
[TVHI research](TEST_DATA.md) are clearly labelled historical. Their original URLs
remain available. Current users should follow the setup and evaluation guides above.
