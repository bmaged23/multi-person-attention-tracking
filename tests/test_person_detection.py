"""Person detection decoding, duplicate control, and no-face coverage checks."""

import unittest

import cv2
import numpy as np

from attention_tracking.analysis.face_metrics import FACE_MODEL, head_pose
from attention_tracking.detection.people_backend import head_region
from attention_tracking.detection.person_detector import (
    decode_predictions,
    deduplicate_people,
)


class PersonTests(unittest.TestCase):
    def test_yolo_person_class_and_source_coordinate_mapping(self):
        output = np.zeros((1, 84, 2), dtype=float)
        output[0, :4, 0] = [100, 100, 40, 60]
        output[0, 4, 0] = 0.9
        output[0, :4, 1] = [100, 100, 40, 60]
        output[0, 5, 1] = 0.95
        boxes, scores = decode_predictions(
            output, 0.5, (10, 20), (30, 40), (400, 400, 3), 0.15
        )
        self.assertEqual(boxes, [(170, 140, 250, 260)])
        self.assertEqual(len(scores), 1)

    def test_upper_body_duplicate_does_not_create_another_person(self):
        boxes = [(10, 10, 110, 210), (20, 12, 100, 90), (130, 10, 210, 180)]
        self.assertEqual(len(deduplicate_people(boxes)), 2)

    def test_neighboring_heads_are_not_merged(self):
        boxes = [(0, 0, 200, 300), (130, 0, 250, 180)]
        self.assertEqual(len(deduplicate_people(boxes)), 2)

    def test_original_head_crop_rule(self):
        self.assertEqual(
            head_region((100, 100, 200, 400), (500, 500, 3)), (85, 82, 215, 238)
        )

    def test_original_solve_pnp_pose(self):
        rotation = np.deg2rad([10, 15, 0]).reshape(3, 1)
        camera = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=float)
        projected, _ = cv2.projectPoints(
            FACE_MODEL, rotation, np.array([0.0, 0.0, 1200.0]), camera, np.zeros((4, 1))
        )
        points = np.zeros((478, 2))
        points[[1, 152, 263, 33, 287, 57]] = projected.reshape(-1, 2)
        expected = cv2.RQDecomp3x3(cv2.Rodrigues(rotation)[0])[0]
        np.testing.assert_allclose(head_pose(points, 640, 480), expected, atol=1e-5)
