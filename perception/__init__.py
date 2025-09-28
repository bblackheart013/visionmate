"""
GuidedSight Perception Pipeline - Public API

This module provides the main interface for Person 3 (Orchestrator) to integrate
the perception pipeline. The API is designed to be stable and allow backend swapping.
"""

from typing import List, Dict, Any
import numpy as np

from .config import PerceptionConfig
from .det_seg import create_detector
from .simple_ocr import create_simple_ocr_detector
from .custom_detector import create_custom_detector
from .navigation_filter import create_navigation_filter
from .nearby_filter import create_nearby_display_filter
from .events import EventBuilder, Event, event_to_dict, dict_to_event


class PerceptionPipeline:
    """
    Main perception pipeline for GuidedSight.

    This class provides a stable API for the orchestrator to process camera frames
    and receive structured events about obstacles and signs.
    """

    def __init__(self, config: PerceptionConfig = None):
        """
        Initialize perception pipeline.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or PerceptionConfig()

        # Initialize detectors
        self.seg_detector = create_detector(self.config)
        self.ocr_detector = create_simple_ocr_detector(self.config)
        self.custom_detector = create_custom_detector()
        self.navigation_filter = create_navigation_filter()
        self.nearby_display_filter = create_nearby_display_filter()

        # Initialize event builder
        self.event_builder = EventBuilder(self.config)

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "raw_detections": 0,
            "yolo_detections": 0,
            "ocr_detections": 0,
            "ocr_runs": 0,
            "custom_detections": 0,
            "filtered_detections": 0,
            "nearby_display_detections": 0,
            "events_promoted": 0
        }

    def process_frame(self, frame_bgr: np.ndarray, fps: float = 30.0) -> List[Event]:
        """
        Process a single frame and return promoted events.

        This is the main API method that Person 3 should call for each camera frame.

        Args:
            frame_bgr: Input frame in BGR format (OpenCV standard)
            fps: Current frame rate (for timing calculations)

        Returns:
            List of promoted Event objects that meet persistence requirements.
            Events are canonical and ready for guidance processing.
        """
        self.stats["frames_processed"] += 1

        # Run YOLO detection
        yolo_detections = self.seg_detector.detect(frame_bgr)

        # Run OCR detection (with stride)
        ocr_detections = self.ocr_detector.detect(frame_bgr)

        # Run custom detection for walls, poles, furniture
        custom_detections = self.custom_detector.detect(frame_bgr)
        self.stats["custom_detections"] += len(custom_detections)

        # Track OCR runs (when it actually processes, regardless of results)
        if self.ocr_detector.frame_count % self.config.ocr_stride == 0:
            self.stats["ocr_runs"] += 1

        # Combine raw detections
        raw_detections = yolo_detections + ocr_detections + custom_detections
        self.stats["raw_detections"] += len(raw_detections)

        # Apply navigation filter to reduce clutter and focus on relevant objects
        filtered_detections = self.navigation_filter.filter_detections(raw_detections, frame_bgr.shape)
        self.stats["filtered_detections"] = len(filtered_detections)

        # Build promoted events from filtered detections
        promoted_events = self.event_builder.build(filtered_detections, frame_bgr)
        self.stats["events_promoted"] += len(promoted_events)

        return promoted_events
    
    def get_display_detections(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Get detections that should be displayed (nearby objects only).
        This is separate from process_frame to avoid affecting the main perception pipeline.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            List of detections that should be displayed (nearby objects only)
        """
        # Get all raw detections (same as in process_frame)
        yolo_detections = self.seg_detector.detect(frame_bgr)
        self.stats["yolo_detections"] += len(yolo_detections)
        
        # OCR detection (every N frames)
        ocr_detections = []
        if self.ocr_detector.frame_count % self.config.ocr_stride == 0:
            ocr_detections = self.ocr_detector.detect(frame_bgr)
            self.stats["ocr_detections"] += len(ocr_detections)
        
        # Custom CV detection
        custom_detections = self.custom_detector.detect(frame_bgr)
        self.stats["custom_detections"] += len(custom_detections)
        
        # Combine all raw detections
        raw_detections = yolo_detections + ocr_detections + custom_detections
        
        # Apply navigation filter (basic confidence filtering)
        filtered_detections = self.navigation_filter.filter_detections(raw_detections, frame_bgr.shape)
        
        # Apply nearby display filter (only show nearby objects)
        display_detections = self.nearby_display_filter.filter_for_display(filtered_detections, frame_bgr.shape)
        self.stats["nearby_display_detections"] = len(display_detections)
        
        return display_detections

    def get_stats(self) -> dict:
        """
        Get pipeline statistics for monitoring.

        Returns:
            Dictionary with processing statistics
        """
        builder_stats = self.event_builder.get_stats()
        return {
            **self.stats,
            **builder_stats,
            "seg_detector_available": self.seg_detector.is_available(),
            "ocr_detector_available": self.ocr_detector.is_available()
        }

    def reset(self):
        """Reset pipeline state (useful for new video sequences)."""
        self.event_builder.reset()
        self.stats = {
            "frames_processed": 0,
            "raw_detections": 0,
            "ocr_runs": 0,
            "custom_detections": 0,
            "events_promoted": 0
        }

    def set_backend(self, backend: str):
        """
        Switch processing backend (for future Snapdragon optimization).

        Args:
            backend: Backend type ("cpu", "gpu", "qnn")

        Note:
            This is a placeholder for future backend switching.
            Person 3 can call this to optimize for different hardware.
        """
        print(f"Backend switching to {backend} - placeholder for future implementation")
        # TODO: Implement backend switching logic for Snapdragon deployment


# Convenience exports for external use
__all__ = [
    "PerceptionPipeline",
    "PerceptionConfig",
    "Event",
    "event_to_dict",
    "dict_to_event"
]


# Quick test function for development
def test_pipeline():
    """Test function for development/debugging."""
    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Initialize pipeline
    pipeline = PerceptionPipeline()

    # Process frame
    events = pipeline.process_frame(frame)

    print(f"Pipeline test: {len(events)} events detected")
    print(f"Stats: {pipeline.get_stats()}")

    return events


if __name__ == "__main__":
    test_pipeline()