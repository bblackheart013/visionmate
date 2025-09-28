"""
Configuration for GuidedSight perception pipeline.
All tunable parameters are defined here for easy adjustment without code changes.
"""
from dataclasses import dataclass
from typing import Set


@dataclass
class PerceptionConfig:
    """Configuration parameters for the perception pipeline."""

    # YOLO Detection
    yolo_conf_obstacle: float = 0.5
    yolo_weights: str = "yolov8n-seg.pt"
    yolo_imgsz: int = 512

    # OCR Detection
    ocr_conf_text: float = 0.3
    ocr_stride: int = 3  # Run OCR every N frames
    max_ocr_rois_per_frame: int = 2

    # Filtering
    min_box_side: int = 20  # Minimum bbox side length in pixels

    # Persistence
    persist_frames: int = 3  # Min consecutive frames to promote event

    # Camera/Spatial
    horiz_fov_deg: float = 60.0  # Horizontal field of view
    center_deadzone_deg: float = 5.0  # Center deadzone for bearing

    # Text whitelist
    whitelist_text: Set[str] = None

    def __post_init__(self):
        if self.whitelist_text is None:
            self.whitelist_text = {"STOP", "EXIT", "→", "➡", "➜"}


# COCO class mapping for obstacles - EXPANDED FOR HACKATHON DEMO
COCO_OBSTACLE_CLASSES = {
    # Core obstacles & vehicles
    "person": 0,
    "bicycle": 1,           # Bikes blocking paths/sidewalks
    "car": 2,
    "motorcycle": 3,        # Motorcycles
    "bus": 5,
    "train": 6,             # Trains at crossings
    "truck": 7,

    # Traffic & street infrastructure
    "traffic light": 9,     # Traffic signals (navigation context)
    "fire hydrant": 10,     # Street fixtures
    "stop sign": 11,        # STOP signs (backup for OCR)

    # Street furniture & obstacles
    "bench": 13,           # Seating obstacles
    "chair": 56,           # Outdoor chairs/furniture
    "dining table": 60,    # Outdoor tables
    "potted plant": 58,    # Large planters/landscaping

    # Dropped items & hazards (common in urban areas)
    "backpack": 24,        # Dropped/abandoned bags
    "handbag": 26,         # Purses/bags on ground
    "suitcase": 28,        # Luggage obstacles
    "umbrella": 25,        # Weather items
}

# For reference, common COCO class IDs:
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus, 6: train, 7: truck
# 8: boat, 9: traffic light, 10: fire hydrant, 11: stop sign, 12: parking meter