"""
Event processing and persistence for perception pipeline.
"""
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .config import PerceptionConfig
from .utils import iou, bearing_from_bbox, approx_distance_from_bbox, label_to_intent


@dataclass
class Event:
    """Structured event for perception output."""
    schema: str
    ts: float
    type: str  # "obstacle" or "sign"
    label: str
    intent: str
    conf: float
    bbox: List[int]
    bearing_deg: float
    dist_m: Optional[float]
    sources: List[str]


class EventBuilder:
    """
    Builds and manages events with persistence and deduplication.
    """

    def __init__(self, config: PerceptionConfig):
        """
        Initialize event builder.

        Args:
            config: Perception configuration
        """
        self.config = config

        # Persistence tracking: {(label, region_key): frame_count}
        self.persistence_tracker = defaultdict(int)

        # Last seen events for deduplication: {(label, region_key): event_data}
        self.last_events = {}

        # Frame counter
        self.frame_count = 0

    def _get_region_key(self, bbox: List[int]) -> str:
        """Generate a region key for spatial deduplication."""
        # Discretize to grid for grouping nearby detections
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        grid_size = 50  # pixels
        grid_x, grid_y = cx // grid_size, cy // grid_size
        return f"{grid_x}_{grid_y}"

    def _merge_detections(self, raw_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping detections from different sources.
        Keep higher confidence detection and merge sources.
        """
        if not raw_detections:
            return []

        merged = []
        used_indices = set()

        for i, det1 in enumerate(raw_detections):
            if i in used_indices:
                continue

            # Find overlapping detections with same label
            overlapping = [det1]
            used_indices.add(i)

            for j, det2 in enumerate(raw_detections[i + 1:], i + 1):
                if (j not in used_indices and
                    det1["label"] == det2["label"] and
                    iou(det1["bbox"], det2["bbox"]) > 0.5):

                    overlapping.append(det2)
                    used_indices.add(j)

            # Merge overlapping detections
            if len(overlapping) == 1:
                merged.append(overlapping[0])
            else:
                # Keep detection with highest confidence
                best_det = max(overlapping, key=lambda x: x["conf"])

                # Merge sources
                all_sources = set()
                for det in overlapping:
                    all_sources.add(det["source"])

                merged_det = best_det.copy()
                merged_det["sources"] = sorted(list(all_sources))

                # Keep any additional context from OCR
                for det in overlapping:
                    if "full_text" in det:
                        merged_det["full_text"] = det["full_text"]

                merged.append(merged_det)

        return merged

    def _create_event(self, detection: Dict[str, Any], frame_bgr: np.ndarray) -> Event:
        """
        Create an Event from a raw detection.

        Args:
            detection: Raw detection dictionary
            frame_bgr: Current frame for spatial calculations

        Returns:
            Event object
        """
        label = detection["label"]
        bbox = detection["bbox"]
        conf = detection["conf"]
        sources = detection.get("sources", [detection["source"]])

        # Calculate spatial properties
        frame_h, frame_w = frame_bgr.shape[:2]
        bearing = bearing_from_bbox(bbox, frame_w, self.config.horiz_fov_deg)

        # Estimate distance (currently only for people)
        dist_m = approx_distance_from_bbox(bbox, label)

        # Determine event type
        event_type = "obstacle" if label in ["person", "car", "bus", "truck", "pole"] else "sign"

        # Map to canonical intent
        text_context = detection.get("full_text", "")
        intent = label_to_intent(label, bearing, text_context)

        return Event(
            schema="v1",
            ts=time.time(),
            type=event_type,
            label=label,
            intent=intent,
            conf=conf,
            bbox=bbox,
            bearing_deg=round(bearing, 1),
            dist_m=round(dist_m, 1) if dist_m is not None else None,
            sources=sources
        )

    def _should_promote_event(self, label: str, region_key: str) -> bool:
        """
        Check if an event should be promoted based on persistence.

        Args:
            label: Detection label
            region_key: Spatial region key

        Returns:
            True if event should be promoted
        """
        key = (label, region_key)
        self.persistence_tracker[key] += 1
        return self.persistence_tracker[key] >= self.config.persist_frames

    def _cleanup_stale_tracking(self):
        """Remove stale entries from persistence tracking."""
        # Clean up entries not seen in recent frames
        keys_to_remove = []
        for key in list(self.persistence_tracker.keys()):
            # If not updated this frame, decrement
            # (This is a simple cleanup - could be more sophisticated)
            pass

    def build(self, raw_detections: List[Dict[str, Any]], frame_bgr: np.ndarray) -> List[Event]:
        """
        Build promoted events from raw detections.

        Args:
            raw_detections: List of raw detections from YOLO/OCR
            frame_bgr: Current frame

        Returns:
            List of promoted Event objects
        """
        self.frame_count += 1

        # Merge overlapping detections
        merged_detections = self._merge_detections(raw_detections)

        promoted_events = []
        current_frame_keys = set()

        for detection in merged_detections:
            label = detection["label"]
            bbox = detection["bbox"]
            region_key = self._get_region_key(bbox)
            tracking_key = (label, region_key)

            current_frame_keys.add(tracking_key)

            # Check if this event should be promoted
            if self._should_promote_event(label, region_key):
                event = self._create_event(detection, frame_bgr)
                promoted_events.append(event)

                # Store for future deduplication
                self.last_events[tracking_key] = event

        # Decay persistence for unseen detections
        for key in list(self.persistence_tracker.keys()):
            if key not in current_frame_keys:
                self.persistence_tracker[key] = max(0, self.persistence_tracker[key] - 1)
                if self.persistence_tracker[key] == 0:
                    del self.persistence_tracker[key]
                    if key in self.last_events:
                        del self.last_events[key]

        return promoted_events

    def reset(self):
        """Reset internal state."""
        self.persistence_tracker.clear()
        self.last_events.clear()
        self.frame_count = 0

    def get_stats(self) -> Dict[str, int]:
        """Get current tracking statistics."""
        return {
            "frame_count": self.frame_count,
            "tracked_objects": len(self.persistence_tracker),
            "promoted_objects": len(self.last_events)
        }


def event_to_dict(event: Event) -> Dict[str, Any]:
    """Convert Event to dictionary for JSON serialization."""
    return {
        "schema": event.schema,
        "ts": event.ts,
        "type": event.type,
        "label": event.label,
        "intent": event.intent,
        "conf": event.conf,
        "bbox": event.bbox,
        "bearing_deg": event.bearing_deg,
        "dist_m": event.dist_m,
        "sources": event.sources
    }


def dict_to_event(data: Dict[str, Any]) -> Event:
    """Convert dictionary to Event object."""
    return Event(
        schema=data["schema"],
        ts=data["ts"],
        type=data["type"],
        label=data["label"],
        intent=data["intent"],
        conf=data["conf"],
        bbox=data["bbox"],
        bearing_deg=data["bearing_deg"],
        dist_m=data["dist_m"],
        sources=data["sources"]
    )