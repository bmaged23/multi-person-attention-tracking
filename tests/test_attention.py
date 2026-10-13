"""Behavioral checks for tracking, unknown evidence, and temporal classification."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from attention_tracking.analysis.face_metrics import iris_ratio
from attention_tracking.analysis.scoring import AttentionState
from attention_tracking.config import Settings
from attention_tracking.tracking.face_tracks import Tracker


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.signals = {"pitch": 0, "yaw": 0, "ear": 0.3, "iris_avg": 0.5}

    def test_frontal_open_eyes_are_attentive(self):
        r = AttentionState().update(self.signals, 0, self.settings)
        self.assertEqual(r["label"], "attentive")
        self.assertTrue(r["is_attentive"])

    def test_original_head_rule_has_no_new_hard_veto(self):
        result = AttentionState().update(
            dict(self.signals, yaw=90, pitch=90), 0, self.settings
        )
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["label"], "attentive")

    def test_original_ear_threshold_restored(self):
        state = AttentionState()
        self.assertEqual(
            state.update(dict(self.signals, ear=0.1), 0, self.settings)["label"],
            "attentive",
        )
        self.assertEqual(
            state.update(dict(self.signals, ear=0.091), 0, self.settings)["label"],
            "drowsy",
        )
        self.assertEqual(
            state.update(dict(self.signals, ear=0.092), 0, self.settings)["label"],
            "attentive",
        )

    def test_original_partial_score(self):
        result = AttentionState().update(
            dict(self.signals, yaw=90, pitch=90, iris_avg=0.9), 0, self.settings
        )
        self.assertEqual(result["score"], 0.43)
        self.assertEqual(result["label"], "partial")

    def test_original_neutral_missing_signal_fallbacks(self):
        signals = {"pitch": None, "yaw": None, "ear": None, "iris_avg": None}
        self.assertEqual(
            AttentionState().update(signals, 0, self.settings)["score"], 0.625
        )

    def test_original_no_face_result(self):
        result = AttentionState().update(None, 0, self.settings)
        self.assertEqual(result["label"], "no_face")
        self.assertEqual(result["score"], 0)
        self.assertIsNone(result["is_attentive"])

    def test_original_iris_distance_ratio(self):
        points = np.array([[0, 0], [10, 0], [3, 4]], dtype=float)
        self.assertAlmostEqual(iris_ratio(points, [2], [0, 1]), 0.5, places=5)

    def test_iou_threshold_is_required_even_for_nearby_centers(self):
        tracker = Tracker(replace(self.settings, iou_threshold=0.9))
        a = SimpleNamespace(bbox=(10, 10, 80, 80))
        b = SimpleNamespace(bbox=(30, 10, 100, 80))
        tracker.update([a], 0, (300, 400, 3))
        self.assertEqual(tracker.update([b], 0.1, (300, 400, 3))[0][0].id, "person_002")

    def test_multiface_order_changes_do_not_change_ids(self):
        tracker = Tracker(self.settings)

        def face(box):
            return SimpleNamespace(bbox=box)

        a, b = face((10, 10, 80, 80)), face((150, 10, 220, 80))
        first = tracker.update([a, b], 0, (300, 400, 3))
        second = tracker.update([b, a], 0.1, (300, 400, 3))
        self.assertEqual([x[0].id for x in first], ["person_001", "person_002"])
        self.assertEqual([x[0].id for x in second], ["person_002", "person_001"])

    def test_expired_track_gets_new_id_and_manual_name_is_optional(self):
        tracker = Tracker(self.settings, {"person_001": "Alex"})
        face = SimpleNamespace(bbox=(10, 10, 80, 80))
        self.assertEqual(tracker.update([face], 0, (300, 400, 3))[0][0].name, "Alex")
        self.assertEqual(
            tracker.update([face], 2, (300, 400, 3))[0][0].id, "person_002"
        )

    def test_one_track_cannot_be_assigned_to_two_faces(self):
        tracker = Tracker(self.settings)

        def face(box):
            return SimpleNamespace(bbox=box)

        tracker.update([face((10, 10, 80, 80))], 0, (300, 400, 3))
        observed = tracker.update(
            [face((12, 10, 82, 80)), face((30, 10, 100, 80))], 0.1, (300, 400, 3)
        )
        self.assertEqual(len({t.id for t, _ in observed}), 2)

    def test_duplicate_detection_is_suppressed_without_removing_second_person(self):
        from attention_tracking.detection.mediapipe_backend import suppress_duplicates

        boxes = [(10, 10, 80, 80), (12, 10, 82, 80), (150, 10, 220, 80)]
        faces = [SimpleNamespace(bbox=box) for box in boxes]
        self.assertEqual(len(suppress_duplicates(faces)), 2)

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            Settings(track_timeout=float("nan"))


if __name__ == "__main__":
    unittest.main()
