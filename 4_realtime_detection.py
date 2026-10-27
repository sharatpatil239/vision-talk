"""
4_realtime_detection.py - Compatibility Wrapper for VisionTalk Sign Language Detection

This script provides backward compatibility with earlier project workflows by forwarding
all execution directly to the unified, production inference engine in detect.py.

Why this wrapper exists:
- Eliminates duplicate / conflicting inference implementations.
- Fixes the previous label_id_offset bug (+1 offset that corrupted class mappings).
- Removes legacy dependency on external TensorFlow Models / Object Detection API.
- Automatically leverages the exported SavedModel and OpenCV visualization in detect.py.
"""

import os
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detect import (  # noqa: E402
    DEFAULT_LABEL_MAP,
    DEFAULT_MODEL_DIR,
    SignLanguageDetector,
    draw_detections,
    load_label_map,
    main,
    run_image_inference,
    run_webcam_detection,
)

if __name__ == "__main__":
    print("=" * 70)
    print(" [VisionTalk] Running 4_realtime_detection.py (Compatibility Wrapper)")
    print(" Forwarding execution to production SavedModel detector in detect.py...")
    print("=" * 70)
    main()