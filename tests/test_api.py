import tempfile
from pathlib import Path
from dataclasses import replace
import time
import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient
from attention_tracking.api import create_app
from attention_tracking.config import Settings, ServiceSettings


class EmptyBackend:
    def __init__(self, settings):
        pass

    def detect(self, frame, seconds):
        return []

    def close(self):
        pass


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        limits = replace(
            ServiceSettings(),
            data_dir=self.folder / "jobs",
            max_upload_mb=1,
            max_clip_seconds=1,
        )
        self.client = TestClient(create_app(Settings(), limits, EmptyBackend))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_health_and_limits_do_not_leak_paths(self):
        self.assertTrue(self.client.get("/health").json()["ready"])
        self.assertNotIn("data_dir", self.client.get("/config").json()["limits"])
        self.assertEqual(self.client.get("/jobs/not-a-uuid").status_code, 404)

    def test_invalid_upload_and_clip_limit_release_capacity(self):
        self.assertEqual(
            self.client.post(
                "/jobs", files={"video": ("bad.mp4", b"not video")}
            ).status_code,
            415,
        )
        self.assertEqual(
            self.client.post(
                "/jobs?max_seconds=2", files={"video": ("bad.mp4", b"x")}
            ).status_code,
            422,
        )
        self.assertFalse(self.client.get("/health").json()["busy"])
        self.assertEqual(list((self.folder / "jobs").iterdir()), [])

    def test_oversized_upload_rejected(self):
        r = self.client.post(
            "/jobs", files={"video": ("big.mp4", b"x" * (1024**2 + 1))}
        )
        self.assertEqual(r.status_code, 413)
        self.assertFalse(self.client.get("/health").json()["busy"])

    def test_full_job_produces_real_video_and_reports_with_injected_empty_detector(
        self,
    ):
        path = self.folder / "source.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64)
        )
        for _ in range(3):
            writer.write(np.zeros((64, 64, 3), np.uint8))
        writer.release()
        r = self.client.post("/jobs", files={"video": ("clip.mp4", path.read_bytes())})
        self.assertEqual(r.status_code, 202, r.text)
        job = r.json()["id"]
        for _ in range(100):
            status = self.client.get("/jobs/" + job).json()
            if status["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(status["status"], "completed", status)
        self.assertEqual(
            self.client.get("/jobs/" + job + "/result").json()["frames"], 3
        )
        self.assertNotIn("source", self.client.get("/jobs/" + job + "/result").json())
        self.assertEqual(
            self.client.get("/jobs/" + job + "/video").headers["content-type"],
            "video/mp4",
        )
        self.assertEqual(self.client.get("/jobs/" + job + "/observations").text, "")

    def test_busy_job_is_rejected_before_parsing_upload(self):
        self.client.app.state.busy = True
        response = self.client.post("/jobs", files={"video": ("clip.mp4", b"x")})
        self.assertEqual(response.status_code, 409)
        self.client.app.state.busy = False

    def test_chunked_oversized_body_is_rejected_before_multipart_parsing(self):
        def body():
            yield b"x" * (600 * 1024)
            yield b"x" * (600 * 1024)

        response = self.client.post(
            "/jobs",
            content=body(),
            headers={"Content-Type": "multipart/form-data; boundary=test"},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(list((self.folder / "jobs").iterdir()), [])

    def test_service_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            replace(ServiceSettings(), max_upload_mb=0)
        with self.assertRaises(ValueError):
            replace(ServiceSettings(), max_clip_seconds=float("nan"))
