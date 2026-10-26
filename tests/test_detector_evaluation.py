import unittest
from dataclasses import replace
from attention_tracking.config import Settings
from scripts.evaluation.evaluate_detector import counts


class EvaluationTests(unittest.TestCase):
    def test_one_prediction_cannot_satisfy_two_references(self):
        self.assertEqual(
            counts([(0, 0, 100, 100)], [(0, 0, 100, 100), (0, 0, 100, 100)]), (1, 0, 1)
        )

    def test_assignment_prioritizes_valid_matches(self):
        self.assertEqual(
            counts(
                [(0, 0, 100, 100), (100, 0, 200, 100)],
                [(100, 0, 200, 100), (0, 0, 100, 100)],
            ),
            (2, 0, 0),
        )

    def test_empty_cases_and_false_positive(self):
        self.assertEqual(counts([], []), (0, 0, 0))
        self.assertEqual(counts([(0, 0, 1, 1)], []), (0, 1, 0))

    def test_cuda_cannot_be_requested_for_opencv_cpu_backend(self):
        with self.assertRaises(ValueError):
            replace(Settings(), person_backend="opencv", person_device="cuda")
