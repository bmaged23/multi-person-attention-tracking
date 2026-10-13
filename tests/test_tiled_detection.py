"""Check full-frame coverage and coordinate mapping for distant-face detection."""

import unittest
from types import SimpleNamespace

import numpy as np

from attention_tracking.detection.mediapipe_backend import result_faces, tile_regions


class TileTests(unittest.TestCase):
    def test_overlapping_tiles_cover_every_source_pixel_without_leaving_frame(self):
        mask = np.zeros((121, 203), dtype=bool)
        for x1, y1, x2, y2 in tile_regions(203, 121, 6):
            self.assertTrue(0 <= x1 < x2 <= 203 and 0 <= y1 < y2 <= 121)
            mask[y1:y2, x1:x2] = True
        self.assertTrue(mask.all())

    def test_crop_landmarks_are_mapped_back_to_original_coordinates(self):
        landmarks = [
            SimpleNamespace(x=0.25, y=0.25),
            SimpleNamespace(x=0.75, y=0.75),
        ] * 239
        result = SimpleNamespace(
            face_landmarks=[landmarks], facial_transformation_matrixes=[np.eye(4)]
        )
        face = result_faces(result, 100, 80, offset_x=300, offset_y=200)[0]
        np.testing.assert_allclose(face.landmarks[0], [325, 220])
        self.assertTrue(face.bbox[0] >= 300 and face.bbox[1] >= 200)
        np.testing.assert_allclose(face.rotation, np.eye(3))
