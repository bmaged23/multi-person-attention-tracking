"""Atomic summary writing; frame observations are streamed separately as JSONL."""

import json
import tempfile
from pathlib import Path


def write_summary(path, report):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".json", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
