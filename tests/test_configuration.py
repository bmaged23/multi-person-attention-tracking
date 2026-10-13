"""Check real .env loading, shell precedence, and CLI precedence in clean processes."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def execute(self, code, overrides=None):
        environment = {
            k: v for k, v in os.environ.items() if not k.startswith("ATTENTION_")
        }
        environment.update(overrides or {})
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def test_example_preserves_established_defaults(self):
        from dotenv import dotenv_values
        from attention_tracking.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            defaults = Settings()
        values = dotenv_values(ROOT / ".env.example")
        self.assertEqual(float(values["ATTENTION_EAR_CLOSED"]), defaults.ear_closed)
        self.assertEqual(
            float(values["ATTENTION_IOU_THRESHOLD"]), defaults.iou_threshold
        )
        self.assertEqual(
            float(values["ATTENTION_ATTENTIVE_MIN"]), defaults.attentive_min
        )

    def test_shell_overrides_dotenv(self):
        result = self.execute(
            "from attention_tracking.config import Settings; print(Settings().iou_threshold)",
            {"ATTENTION_IOU_THRESHOLD": "0.73"},
        )
        self.assertEqual(result, "0.73")

    def test_dotenv_loaded_from_repository_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "attention_tracking"
            package.mkdir()
            config = package / "config.py"
            config.write_text((ROOT / "attention_tracking/config.py").read_text())
            (root / ".env").write_text("ATTENTION_IOU_THRESHOLD=0.61\n")
            code = f"import runpy; data=runpy.run_path({str(config)!r}); print(data['Settings']().iou_threshold)"
            self.assertEqual(self.execute(code), "0.61")

    def test_cli_flag_overrides_environment(self):
        code = "from unittest.mock import patch; import sys; import attention_tracking.cli as cli; sys.argv=['video_attention.py','--video','dummy.mp4','--iou-threshold','0.82']; fake={'frames':0,'people':[]};\nwith patch.object(cli,'run',return_value=fake) as run:\n cli.main(); print(run.call_args.args[1].iou_threshold)"
        self.assertEqual(
            self.execute(code, {"ATTENTION_IOU_THRESHOLD": "0.73"}).splitlines()[-1],
            "0.82",
        )
