"""
OCR text detection module for sign recognition.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from .config import PerceptionConfig
from .utils import is_valid_bbox, extract_directional_text

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


class OCRDetector:
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

    def _extract_candidate_rois(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Extract candidate ROIs for OCR processing.
        Focus on rectangular regions in upper half with text-like properties.

        Args:
            frame_bgr: Input frame in BGR format

        Returns:
            List of ROI bounding boxes (x1, y1, x2, y2)
        """
        if not CV2_AVAILABLE:
            # Fallback: just use upper half of frame as single ROI
            h, w = frame_bgr.shape[:2]
            return [(0, 0, w, int(h * 0.67))]

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Focus on upper 2/3 of frame (signs typically not on ground)
        roi_gray = gray[:int(h * 0.67), :]

        # Edge detection to find rectangular regions
        edges = cv2.Canny(roi_gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rois = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w_rect, h_rect = cv2.boundingRect(contour)

            # More permissive size filtering for better text capture
            if (w_rect >= 60 and h_rect >= 30 and  # Larger minimum size
                w_rect <= w * 0.8 and h_rect <= h * 0.5 and  # Allow larger regions
                0.3 <= w_rect / h_rect <= 12.0):  # Wider aspect ratio range

                # Expand ROI to capture surrounding text
                padding = 20
                x_exp = max(0, x - padding)
                y_exp = max(0, y - padding)
                x2_exp = min(w, x + w_rect + padding)
                y2_exp = min(int(h * 0.67), y + h_rect + padding)

                rois.append((x_exp, y_exp, x2_exp, y2_exp))

        # If no ROIs found, create larger fallback ROIs for signs
        if len(rois) == 0:
            print("DEBUG: No contour-based ROIs found, using large grid fallback")
            # Create larger 2x1 grid to capture full signs
            grid_w, grid_h = w // 2, int(h * 0.5)  # Larger regions
            for i in range(2):
                x_start = i * grid_w
                y_start = 0  # Start from top
                rois.append((x_start, y_start, x_start + grid_w, y_start + grid_h))

            # Also add a full-width region for horizontal signs
            rois.append((0, 0, w, int(h * 0.4)))

        # Sort by area (largest first) and limit count
        rois.sort(key=lambda roi: (roi[2] - roi[0]) * (roi[3] - roi[1]), reverse=True)
        print(f"DEBUG: Found {len(rois)} ROIs for OCR processing")
        return rois[:self.config.max_ocr_rois_per_frame]

    def _combine_nearby_text(self, results, max_x_gap=30, max_y_diff=15):
        """
        Combine nearby text detections into words with spatial validation.

        Args:
            results: List of (bbox, text, conf) from EasyOCR
            max_x_gap: Maximum horizontal gap between letters (pixels)
            max_y_diff: Maximum vertical difference for same line (pixels)

        Returns:
            List of combined text results
        """
        if len(results) <= 1:
            return []

        # Filter out empty/whitespace-only results
        valid_results = [(bbox, text.strip(), conf) for bbox, text, conf in results
                        if len(text.strip()) > 0]

        print(f"DEBUG: _combine_nearby_text - {len(results)} total, {len(valid_results)} valid")
        for i, (bbox, text, conf) in enumerate(valid_results):
            print(f"  Text {i}: '{text}' at position ({bbox[0][0]}, {bbox[0][1]})")

        if len(valid_results) <= 1:
            print(f"DEBUG: Not enough valid results for combination")
            return []

        # Sort by position (left to right, top to bottom)
        sorted_results = sorted(valid_results, key=lambda r: (r[0][0][1], r[0][0][0]))

        combined_groups = []
        current_group = [sorted_results[0]]

        for i in range(1, len(sorted_results)):
            prev_bbox = current_group[-1][0]
            curr_bbox = sorted_results[i][0]

            # Calculate spatial relationship
            prev_right = prev_bbox[1][0]  # Right edge of previous
            curr_left = curr_bbox[0][0]   # Left edge of current

            prev_y = (prev_bbox[0][1] + prev_bbox[2][1]) / 2  # Vertical center
            curr_y = (curr_bbox[0][1] + curr_bbox[2][1]) / 2  # Vertical center

            x_gap = curr_left - prev_right
            y_diff = abs(curr_y - prev_y)

            # Check if letters are close enough to be part of same word
            if x_gap <= max_x_gap and y_diff <= max_y_diff:
                current_group.append(sorted_results[i])
            else:
                # Process current group if it has multiple letters
                if len(current_group) > 1:
                    combined = self._merge_text_group(current_group)
                    if combined:
                        combined_groups.append(combined)
                current_group = [sorted_results[i]]

        # Process final group
        if len(current_group) > 1:
            combined = self._merge_text_group(current_group)
            if combined:
                combined_groups.append(combined)

        return combined_groups

    def _merge_text_group(self, group):
        """
        Merge a group of spatially-close text detections into a single word.

        Args:
            group: List of (bbox, text, conf) detections

        Returns:
            (bbox, combined_text, avg_conf) or None if invalid
        """
        if len(group) < 2:
            return None

        # Sort by x-coordinate for reading order
        group_sorted = sorted(group, key=lambda r: r[0][0][0])

        # Combine text
        combined_text = ''.join([text for _, text, _ in group_sorted])

        # Validate combined result
        if len(combined_text) < 3 or len(combined_text) > 10:  # Reasonable word length
            return None

        if len(group_sorted) > 6:  # Too many fragments (likely noise)
            return None

        # Calculate spatial bounds for validation
        first_bbox = group_sorted[0][0]
        last_bbox = group_sorted[-1][0]

        word_width = last_bbox[1][0] - first_bbox[0][0]
        if word_width > 200:  # Word too wide (likely false combination)
            return None

        # Calculate confidence (weighted by text length)
        total_chars = sum(len(text) for _, text, _ in group_sorted)
        weighted_conf = sum(conf * len(text) for _, text, conf in group_sorted) / total_chars

        # Create combined bounding box
        min_x = min(bbox[0][0] for bbox, _, _ in group_sorted)
        min_y = min(bbox[0][1] for bbox, _, _ in group_sorted)
        max_x = max(bbox[1][0] for bbox, _, _ in group_sorted)
        max_y = max(bbox[2][1] for bbox, _, _ in group_sorted)

        combined_bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

        print(f"OCR combined: '{combined_text}' (weighted_conf={weighted_conf:.3f}, width={word_width:.0f}px)")

        return (combined_bbox, combined_text, weighted_conf)

    def _expand_search_around_letter(self, frame_bgr, roi_bbox, letter_result, roi_x, roi_y):
        """
        Expand search around a detected letter to find adjacent letters of similar size.

        Args:
            frame_bgr: Full frame
            roi_bbox: Original ROI coordinates
            letter_result: (bbox, text, conf) of detected letter
            roi_x, roi_y: ROI offset in full frame

        Returns:
            List of additional letter detections
        """
        letter_bbox, letter_text, letter_conf = letter_result

        # Calculate letter dimensions
        letter_w = letter_bbox[1][0] - letter_bbox[0][0]
        letter_h = letter_bbox[2][1] - letter_bbox[0][1]

        # Convert to full frame coordinates
        letter_center_x = roi_x + (letter_bbox[0][0] + letter_bbox[1][0]) / 2
        letter_center_y = roi_y + (letter_bbox[0][1] + letter_bbox[2][1]) / 2

        print(f"DEBUG: Letter '{letter_text}' size: {letter_w}x{letter_h}px at ({letter_center_x:.0f}, {letter_center_y:.0f})")

        # Create expanded search area around the letter
        # For STOP sign: expect 4 letters horizontally
        search_width = letter_w * 6  # Space for ~6 letters
        search_height = letter_h * 2  # Some vertical tolerance

        # Calculate expanded ROI
        expand_x1 = max(0, int(letter_center_x - search_width // 2))
        expand_y1 = max(0, int(letter_center_y - search_height // 2))
        expand_x2 = min(frame_bgr.shape[1], int(letter_center_x + search_width // 2))
        expand_y2 = min(frame_bgr.shape[0], int(letter_center_y + search_height // 2))

        # Extract expanded region
        expanded_roi = frame_bgr[expand_y1:expand_y2, expand_x1:expand_x2]

        print(f"DEBUG: Expanding search to {expand_x2-expand_x1}x{expand_y2-expand_y1}px region")

        try:
            # Run OCR on expanded region
            expanded_results = self.reader.readtext(expanded_roi, detail=1)

            additional_letters = []
            for bbox_rel, text, conf in expanded_results:
                text = text.strip()

                # Skip the original letter (avoid duplicates)
                if text == letter_text:
                    continue

                # Look for STOP sign letters
                if text in ['S', 'T', 'O', 'P', '0'] and conf >= self.config.ocr_conf_text:
                    # Convert back to original ROI coordinates for consistency
                    abs_bbox = [
                        [bbox_rel[0][0] + expand_x1 - roi_x, bbox_rel[0][1] + expand_y1 - roi_y],
                        [bbox_rel[1][0] + expand_x1 - roi_x, bbox_rel[1][1] + expand_y1 - roi_y],
                        [bbox_rel[2][0] + expand_x1 - roi_x, bbox_rel[2][1] + expand_y1 - roi_y],
                        [bbox_rel[3][0] + expand_x1 - roi_x, bbox_rel[3][1] + expand_y1 - roi_y]
                    ]

                    additional_letters.append((abs_bbox, text, conf))
                    print(f"DEBUG: Found additional letter: '{text}' (conf={conf:.3f})")

            return additional_letters

        except Exception as e:
            print(f"DEBUG: Error in letter expansion: {e}")
            return []

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        COMPLEX OCR COMMENTED OUT - Using simple_ocr.py instead for hackathon
        This was production-grade but too complex for demo
        """
        # Complex OCR disabled - see simple_ocr.py for hackathon version
        return []

        # OLD COMPLEX IMPLEMENTATION - COMMENTED OUT
        # """
        # Run OCR detection on frame.
        #
        # Args:
        #     frame_bgr: Input frame in BGR format
        #
        # Returns:
        #     List of raw text detections with format:
        #     {
        #         "label": str,
        #         "conf": float,
        #         "bbox": [x1, y1, x2, y2],
        #         "source": "ocr"
        #     }
        # """

    def is_available(self) -> bool:
        """Check if OCR detector is available and loaded."""
        return EASYOCR_AVAILABLE and self.reader is not None


class MockOCRDetector:
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


def create_ocr_detector(config: PerceptionConfig) -> OCRDetector:
    """
    Factory function to create appropriate OCR detector.

    Args:
        config: Perception configuration

    Returns:
        OCRDetector instance (real or mock)
    """
    if EASYOCR_AVAILABLE:
        return OCRDetector(config)
    else:
        print("Using mock OCR detector (EasyOCR not available)")
        return MockOCRDetector(config)