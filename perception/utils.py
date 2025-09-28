"""
Utility functions for perception pipeline.
"""
import math
from typing import List, Tuple, Optional
import re


def bearing_from_bbox(bbox: List[int], frame_w: int, hfov_deg: float) -> float:
    """
    Calculate bearing (horizontal angle) from bounding box center.

    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        frame_w: Frame width in pixels
        hfov_deg: Horizontal field of view in degrees

    Returns:
        Bearing in degrees (-hfov/2 to +hfov/2, negative=left, positive=right)
    """
    x1, y1, x2, y2 = bbox
    bbox_center_x = (x1 + x2) / 2

    # Convert pixel position to normalized position (-0.5 to 0.5)
    norm_x = (bbox_center_x / frame_w) - 0.5

    # Convert to bearing angle
    bearing_deg = norm_x * hfov_deg

    return bearing_deg


def approx_distance_from_bbox(bbox: List[int], label: str = "person") -> Optional[float]:
    """
    Approximate distance using simple bbox height heuristic.
    Only works for people currently - returns None for other objects.

    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        label: Object label

    Returns:
        Estimated distance in meters, or None if not estimable
    """
    if label != "person":
        return None

    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1

    # Simple heuristic: assume average person height ~1.7m
    # Distance inversely proportional to bbox height
    # Rough calibration: person at 3m appears ~80px tall in 720p
    if bbox_height < 10:  # Too small, unreliable
        return None

    estimated_distance = (80 * 3.0) / bbox_height
    return max(0.5, min(estimated_distance, 50.0))  # Clamp to reasonable range


def iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: [x1, y1, x2, y2] first bounding box
        bbox2: [x1, y1, x2, y2] second bounding box

    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Calculate intersection area
    if x2_i <= x1_i or y2_i <= y1_i:
        intersection = 0
    else:
        intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def label_to_intent(label: str, bearing_deg: float, text_context: str = "") -> str:
    """
    Map detection labels to canonical intents.

    Args:
        label: Detected label (e.g., "person", "STOP", "EXIT")
        bearing_deg: Bearing angle in degrees
        text_context: Full OCR text for context (e.g., "EXIT →")

    Returns:
        Canonical intent string
    """
    label_upper = label.upper()

    # Obstacle intents
    if label_upper in ["PERSON", "CAR", "BUS", "TRUCK", "POLE"]:
        return f"OBSTACLE_{label_upper}"

    # Stop sign intent
    if label_upper == "STOP" or "STOP" in label_upper:
        return "STOP"

    # Exit sign intents with directional detection
    if label_upper == "EXIT" or "EXIT" in label_upper:
        # Check for directional arrows in text context
        if any(arrow in text_context for arrow in ["→", "➡", "➜", "RIGHT"]):
            return "EXIT_RIGHT"
        elif any(arrow in text_context for arrow in ["←", "⬅", "➜", "LEFT"]):
            return "EXIT_LEFT"
        else:
            # Use bearing to infer direction if no explicit arrow
            if bearing_deg > 10:
                return "EXIT_RIGHT"
            elif bearing_deg < -10:
                return "EXIT_LEFT"
            else:
                return "EXIT"

    # Arrow detection (optional scope)
    if label_upper in ["→", "➡", "➜"]:
        return "ARROW_RIGHT"
    elif label_upper in ["←", "⬅"]:
        return "ARROW_LEFT"

    # Default fallback
    return f"UNKNOWN_{label_upper}"


def extract_directional_text(text: str) -> Tuple[str, str]:
    """
    Extract base text and directional indicators from OCR text.

    Args:
        text: Raw OCR text

    Returns:
        Tuple of (base_text, full_context)
    """
    # Clean and normalize text
    text = text.strip().upper()

    # Extract main words (remove arrows for base text)
    base_text = re.sub(r'[→➡➜←⬅]', '', text).strip()

    return base_text, text


def bbox_area(bbox: List[int]) -> int:
    """Calculate bounding box area."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def bbox_center(bbox: List[int]) -> Tuple[float, float]:
    """Calculate bounding box center point."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def is_valid_bbox(bbox: List[int], min_side: int = 20) -> bool:
    """Check if bounding box meets minimum size requirements."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width >= min_side and height >= min_side