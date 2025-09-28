"""
OCR text detection module for sign recognition.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from .config import PerceptionConfig
from .utils import is_valid_bbox, extract_directional_text  # Assuming these are still useful

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not available. Install with: pip install opencv-python")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not available. Install with: pip install easyocr")


class SimpleOCRDetector:
    """EasyOCR detector for text/sign recognition."""

    def __init__(self, config: PerceptionConfig):
        """
        Initialize OCR detector.

        Args:
            config: Perception configuration
        """
        self.config = config
        self.reader = None
        self.frame_count = 0

        if EASYOCR_AVAILABLE:
            self._load_reader()

    def _load_reader(self):
        """Load EasyOCR reader."""
        try:
            self.reader = easyocr.Reader(['en'], gpu=False)  # CPU-only for compatibility
            print("Loaded EasyOCR reader")
        except Exception as e:
            print(f"Failed to load EasyOCR: {e}")
            self.reader = None

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Simplified OCR detection for hackathon demo.
        Processes the upper half of the frame and looks for whole words.
        """
        self.frame_count += 1

        # Only run OCR every N frames to save compute
        if self.frame_count % self.config.ocr_stride != 0:
            return []

        if not EASYOCR_AVAILABLE or self.reader is None:
            return []

        try:
            # 1. Simple ROI: Just use the upper half of the frame where signs are expected.
            h, w = frame_bgr.shape[:2]
            upper_half_roi = frame_bgr[:int(h * 0.5), :]

            if upper_half_roi.size == 0:
                return []

            # 2. Let EasyOCR do the work. It's good at finding words.
            results = self.reader.readtext(upper_half_roi, detail=1)

            detections = []
            print(f"DEBUG: Found {len(results)} raw text results in upper half.")

            for bbox_rel, text, conf in results:
                # 3. Simple text cleaning and validation.
                cleaned_text = text.upper().strip()

                # Check confidence threshold
                if conf < self.config.ocr_conf_text:
                    continue

                # 4. Simple keyword check. No complex logic needed.
                label = None
                if "STOP" in cleaned_text:
                    label = "STOP"
                elif "EXIT" in cleaned_text:
                    label = "EXIT"

                # If a target word was found, create the detection object
                if label:
                    # Bbox from readtext is [[TL], [TR], [BR], [BL]]
                    # We need the absolute min/max coordinates
                    x_coords = [point[0] for point in bbox_rel]
                    y_coords = [point[1] for point in bbox_rel]

                    abs_bbox = [
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords)),
                        int(max(y_coords))
                    ]

                    # Validate the final bounding box size
                    if not is_valid_bbox(abs_bbox, self.config.min_box_side):
                        continue

                    print(f"  → ACCEPTED: Found '{label}' with conf {conf:.2f}")

                    detection = {
                        "label": label,
                        "conf": float(conf),
                        "bbox": abs_bbox,  # Coordinates are already relative to the frame top-left
                        "source": "ocr",
                        "full_text": cleaned_text
                    }
                    detections.append(detection)

            return detections

        except Exception as e:
            print(f"Simplified OCR detection error: {e}")
            return []

    def is_available(self) -> bool:
        """Check if OCR detector is available and loaded."""
        return EASYOCR_AVAILABLE and self.reader is not None


class MockSimpleOCRDetector:
    """Mock OCR detector for testing when EasyOCR is not available."""

    def __init__(self, config: PerceptionConfig):
        self.config = config
        self.frame_count = 0

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Return empty list for mock detector."""
        self.frame_count += 1
        return []

    def is_available(self) -> bool:
        """Mock detector is always 'available' but returns no detections."""
        return True


def create_simple_ocr_detector(config: PerceptionConfig):
    """
    Factory function to create appropriate OCR detector.

    Args:
        config: Perception configuration

    Returns:
        OCRDetector instance (real or mock)
    """
    if EASYOCR_AVAILABLE:
        return SimpleOCRDetector(config)
    else:
        print("Using mock OCR detector (EasyOCR not available)")
        return MockSimpleOCRDetector(config)