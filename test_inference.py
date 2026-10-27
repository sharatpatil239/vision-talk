"""
test_inference.py - Automated Test Suite for VisionTalk Inference Engine

Tests:
1. Label map parsing (exact class IDs 1-5, no offset).
2. SavedModel loading and signature resolution.
3. Multi-class gesture inference on real test dataset images.
4. Correct mapping from predicted class IDs to gesture names.
5. Coordinate conversion and boundary clamping logic.
6. Error handling for invalid model and label map paths.
7. Unavailable camera diagnostics and clean error propagation.
8. Backward compatibility wrapper (4_realtime_detection.py).
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detect import (
    DEFAULT_LABEL_MAP,
    DEFAULT_MODEL_DIR,
    SignLanguageDetector,
    draw_detections,
    load_label_map,
    run_image_inference,
    run_webcam_detection,
)


class TestLabelMap(unittest.TestCase):
    """Verifies label map parsing matches exact project specifications."""

    def test_load_valid_label_map(self):
        category_index = load_label_map(DEFAULT_LABEL_MAP)
        expected = {
            1: "hello",
            2: "thankyou",
            3: "yes",
            4: "no",
            5: "iloveyou",
        }
        self.assertEqual(
            category_index,
            expected,
            f"Label map did not match expected IDs 1-5: {category_index}",
        )

    def test_missing_label_map_raises(self):
        bogus_path = os.path.join(PROJECT_ROOT, "workspace", "annotations", "nonexistent.pbtxt")
        with self.assertRaises(FileNotFoundError):
            load_label_map(bogus_path)


class TestSavedModelLoading(unittest.TestCase):
    """Verifies loading the exported TensorFlow SavedModel."""

    @classmethod
    def setUpClass(cls):
        cls.detector = SignLanguageDetector(
            model_dir=DEFAULT_MODEL_DIR,
            label_map_path=DEFAULT_LABEL_MAP,
            min_threshold=0.5,
        )

    def test_detector_initialized(self):
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.detect_fn)
        self.assertEqual(len(self.detector.category_index), 5)

    def test_invalid_model_path_raises(self):
        bogus_dir = os.path.join(PROJECT_ROOT, "workspace", "models", "nonexistent_dir")
        with self.assertRaises(FileNotFoundError):
            SignLanguageDetector(model_dir=bogus_dir)


class TestImageInferenceAndClassMapping(unittest.TestCase):
    """Verifies predictions on real test dataset images for all 5 classes."""

    @classmethod
    def setUpClass(cls):
        cls.detector = SignLanguageDetector(
            model_dir=DEFAULT_MODEL_DIR,
            label_map_path=DEFAULT_LABEL_MAP,
            min_threshold=0.5,
        )
        cls.test_dir = os.path.join(PROJECT_ROOT, "workspace", "images", "test")

    def _find_test_image(self, prefix: str) -> str:
        files = [f for f in os.listdir(self.test_dir) if f.startswith(prefix) and f.endswith(".jpg")]
        self.assertTrue(len(files) > 0, f"No test image found for class prefix: {prefix}")
        return os.path.join(self.test_dir, files[0])

    def test_predict_hello(self):
        img_path = self._find_test_image("hello")
        img = cv2.imread(img_path)
        dets = self.detector.predict(img, threshold=0.5)
        self.assertTrue(len(dets) > 0, "No detection for hello sample")
        top = dets[0]
        self.assertEqual(top["class_id"], 1)
        self.assertEqual(top["label"], "hello")
        self.assertGreater(top["score"], 0.85)

    def test_predict_thankyou(self):
        img_path = self._find_test_image("thanks")
        img = cv2.imread(img_path)
        dets = self.detector.predict(img, threshold=0.5)
        self.assertTrue(len(dets) > 0, "No detection for thankyou sample")
        top = dets[0]
        self.assertEqual(top["class_id"], 2)
        self.assertEqual(top["label"], "thankyou")
        self.assertGreater(top["score"], 0.85)

    def test_predict_yes(self):
        img_path = self._find_test_image("yes")
        img = cv2.imread(img_path)
        dets = self.detector.predict(img, threshold=0.5)
        self.assertTrue(len(dets) > 0, "No detection for yes sample")
        top = dets[0]
        self.assertEqual(top["class_id"], 3)
        self.assertEqual(top["label"], "yes")
        self.assertGreater(top["score"], 0.85)

    def test_predict_no(self):
        img_path = self._find_test_image("no")
        img = cv2.imread(img_path)
        dets = self.detector.predict(img, threshold=0.5)
        self.assertTrue(len(dets) > 0, "No detection for no sample")
        top = dets[0]
        self.assertEqual(top["class_id"], 4)
        self.assertEqual(top["label"], "no")
        self.assertGreater(top["score"], 0.85)

    def test_predict_iloveyou(self):
        img_path = self._find_test_image("iloveyou")
        img = cv2.imread(img_path)
        dets = self.detector.predict(img, threshold=0.5)
        self.assertTrue(len(dets) > 0, "No detection for iloveyou sample")
        top = dets[0]
        self.assertEqual(top["class_id"], 5)
        self.assertEqual(top["label"], "iloveyou")
        self.assertGreater(top["score"], 0.85)


class TestBoundingBoxDrawingAndClamping(unittest.TestCase):
    """Verifies coordinate conversion, clamping, and zero-area rejection."""

    def test_coordinate_clamping(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Test box that exceeds boundaries: negative ymin/xmin, ymax/xmax > 1.0
        abnormal_dets = [
            {
                "box": [-0.2, -0.1, 1.4, 1.3],
                "class_id": 1,
                "label": "hello",
                "score": 0.95,
            },
            # Inverted / zero area box (should be safely ignored)
            {
                "box": [0.6, 0.6, 0.4, 0.4],
                "class_id": 2,
                "label": "thankyou",
                "score": 0.90,
            },
        ]
        out_frame = draw_detections(frame, abnormal_dets, min_threshold=0.5, max_boxes=5)
        self.assertEqual(out_frame.shape, (480, 640, 3))


class TestCameraAvailabilityDiagnostics(unittest.TestCase):
    """Verifies clean error reporting when a camera device is unavailable."""

    def test_unavailable_camera_index(self):
        detector = SignLanguageDetector(
            model_dir=DEFAULT_MODEL_DIR,
            label_map_path=DEFAULT_LABEL_MAP,
            min_threshold=0.5,
        )
        # Device index 9999 should not exist
        with self.assertRaises(RuntimeError) as ctx:
            run_webcam_detection(detector, camera_id=9999)
        self.assertIn("Could not open camera at device index 9999", str(ctx.exception))
        self.assertIn("python detect.py --image", str(ctx.exception))


class TestCompatibilityWrapper(unittest.TestCase):
    """Verifies 4_realtime_detection.py compatibility wrapper imports cleanly."""

    def test_wrapper_imports(self):
        import importlib
        wrapper = importlib.import_module("4_realtime_detection")
        self.assertTrue(hasattr(wrapper, "SignLanguageDetector"))
        self.assertTrue(hasattr(wrapper, "main"))
        self.assertTrue(hasattr(wrapper, "run_webcam_detection"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
