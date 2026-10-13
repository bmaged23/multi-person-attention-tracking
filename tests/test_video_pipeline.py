"""Exercise source video decoding and streamed reports with controlled detections."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from attention_tracking.config import Settings
from attention_tracking.video.pipeline import run


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root / "input.mp4"
        writer = cv2.VideoWriter(
            str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120)
        )
        assert writer.isOpened()
        for _ in range(5):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()

    def test_two_people_streamed_with_separate_ids(self):
        faces = [
            SimpleNamespace(
                bbox=(10, 10, 60, 80), landmarks=np.array([[20, 20], [40, 40]])
            ),
            SimpleNamespace(
                bbox=(90, 10, 140, 80), landmarks=np.array([[100, 20], [120, 40]])
            ),
        ]
        backend = SimpleNamespace(detect=lambda frame, now: faces)
        signals = {"pitch": 0, "yaw": 0, "ear": 0.3, "iris_avg": 0.5}
        output = self.root / "result.json"
        with patch("attention_tracking.video.pipeline.measure", return_value=signals):
            report = run(str(self.video), Settings(), output, backend=backend)
        rows = [
            json.loads(s)
            for s in (self.root / report["observations_file"]).read_text().splitlines()
        ]
        self.assertEqual(report["frames"], 5)
        self.assertEqual(len(report["people"]), 2)
        self.assertEqual(len(rows), 10)
        self.assertEqual({r["id"] for r in rows}, {"person_001", "person_002"})
        self.assertNotIn("observations", report)
        self.assertTrue(all(r["is_attentive"] for r in rows))

    def test_person_without_face_landmarks_remains_in_results(self):
        person = SimpleNamespace(bbox=(10, 10, 60, 80), landmarks=np.empty((0, 2)))
        backend = SimpleNamespace(detect=lambda frame, now: [person])
        report = run(
            str(self.video), Settings(), self.root / "person.json", backend=backend
        )
        rows = [
            json.loads(s)
            for s in (self.root / report["observations_file"]).read_text().splitlines()
        ]
        self.assertEqual(len(report["people"]), 1)
        self.assertEqual(len(rows), 5)
        self.assertTrue(
            all(r["label"] == "no_face" and r["bbox"] == [10, 10, 60, 80] for r in rows)
        )
        self.assertIsNone(report["people"][0]["attentive_fraction_of_known"])

    def test_no_faces_produces_valid_empty_summary(self):
        backend = SimpleNamespace(detect=lambda frame, now: [])
        report = run(
            str(self.video), Settings(), self.root / "empty.json", backend=backend
        )
        self.assertEqual(report["people"], [])
        self.assertEqual((self.root / report["observations_file"]).read_text(), "")

    def test_output_cannot_replace_input(self):
        before = self.video.read_bytes()
        with self.assertRaises(ValueError):
            run(str(self.video), Settings(), self.video)
        self.assertEqual(self.video.read_bytes(), before)
