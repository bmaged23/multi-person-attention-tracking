"""Face-only drawing must not use the larger person tracking box."""

import unittest
from types import SimpleNamespace

import numpy as np

from attention_tracking.config import Settings
from attention_tracking.detection.mediapipe_backend import Face
from attention_tracking.video.annotation import annotate, face_bounds


class AnnotationTests(unittest.TestCase):
    def test_face_extent_is_drawn_instead_of_body(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        face = Face((10, 10, 190, 190), np.array([[60, 70], [90, 100]]), None)
        track = SimpleNamespace(bbox=face.bbox, name="Person 1")
        self.assertEqual(face_bounds(face, frame.shape), (60, 70, 90, 100))
        annotated = annotate(frame, [(track, face, {"label": "attentive"})], Settings())
        self.assertTrue(annotated[100, 75].any())
        self.assertFalse(annotated[190, 100].any())

    def test_no_face_has_no_drawn_body_or_label(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        face = Face((10, 10, 190, 190), np.empty((0, 2)), None)
        track = SimpleNamespace(bbox=face.bbox, name="Person 1")
        self.assertIsNone(face_bounds(face, frame.shape))
        self.assertFalse(
            annotate(frame, [(track, face, {"label": "no_face"})], Settings()).any()
        )
