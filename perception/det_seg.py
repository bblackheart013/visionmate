"""
YOLOv8 segmentation detection module for obstacle detection.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from .config import COCO_OBSTACLE_CLASSES, PerceptionConfig
from .utils import is_valid_bbox

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")


class SegDetector:
    """YOLOv8 segmentation detector for obstacles."""

    def __init__(self, config: PerceptionConfig):
        """
        Initialize YOLOv8 segmentation detector.

        Args:
            config: Perception configuration
        """
        self.config = config
        self.model = None
        self.class_names = None

        if YOLO_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Load YOLOv8 model and extract class names."""
        try:
            self.model = YOLO(self.config.yolo_weights)
            self.class_names = self.model.names
            print(f"Loaded YOLOv8 model: {self.config.yolo_weights}")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 detection on frame.

        Args:
            frame_bgr: Input frame in BGR format

        Returns:
            List of raw detections with format:
            {
                "label": str,
                "conf": float,
                "bbox": [x1, y1, x2, y2],
                "source": "yolo",
                "mask": np.ndarray (optional)
            }
        """
        if not YOLO_AVAILABLE or self.model is None:
            return []

        try:
            # Run inference
            results = self.model(
                frame_bgr,
                imgsz=self.config.yolo_imgsz,
                conf=self.config.yolo_conf_obstacle,
                verbose=False
            )

            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confs = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()

                    # Extract masks if available
                    masks = None
                    if hasattr(result, 'masks') and result.masks is not None:
                        masks = result.masks.data.cpu().numpy()

                    for i, (box, conf, class_id) in enumerate(zip(boxes, confs, class_ids)):
                        class_name = self.class_names[int(class_id)]

                        # Filter to obstacle classes only
                        if class_name not in COCO_OBSTACLE_CLASSES:
                            continue

                        # Convert box to integer coordinates
                        bbox = [int(x) for x in box]

                        # Validate minimum box size
                        if not is_valid_bbox(bbox, self.config.min_box_side):
                            continue

                        detection = {
                            "label": class_name,
                            "conf": float(conf),
                            "bbox": bbox,
                            "source": "yolo"
                        }

                        # Add mask if available
                        if masks is not None and i < len(masks):
                            detection["mask"] = masks[i]

                        detections.append(detection)

            return detections

        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

    def is_available(self) -> bool:
        """Check if detector is available and loaded."""
        return YOLO_AVAILABLE and self.model is not None


class MockSegDetector:
    """Mock detector for testing when YOLO is not available."""

    def __init__(self, config: PerceptionConfig):
        self.config = config

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Return empty list for mock detector."""
        return []

    def is_available(self) -> bool:
        """Mock detector is always 'available' but returns no detections."""
        return True


def create_detector(config: PerceptionConfig) -> SegDetector:
    """
    Factory function to create appropriate detector.

    Args:
        config: Perception configuration

    Returns:
        SegDetector instance (real or mock)
    """
    if YOLO_AVAILABLE:
        return SegDetector(config)
    else:
        print("Using mock detector (YOLO not available)")
        return MockSegDetector(config)