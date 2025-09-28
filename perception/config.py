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


# COCO class mapping for obstacles - COMPREHENSIVE REAL-WORLD OBSTACLES
COCO_OBSTACLE_CLASSES = {
    # Core obstacles & vehicles
    "person": 0,
    "bicycle": 1,           # Bikes blocking paths/sidewalks
    "car": 2,
    "motorcycle": 3,        # Motorcycles
    "bus": 5,
    "train": 6,             # Trains at crossings
    "truck": 7,
    "boat": 8,              # Boats/marine obstacles

    # Traffic & street infrastructure
    "traffic light": 9,     # Traffic signals (navigation context)
    "fire hydrant": 10,     # Street fixtures
    "stop sign": 11,        # STOP signs (backup for OCR)
    "parking meter": 12,    # Street fixtures

    # Street furniture & obstacles
    "bench": 13,           # Seating obstacles
    "bird": 14,            # Birds that might be obstacles
    "cat": 15,             # Animals
    "dog": 16,             # Animals
    "horse": 17,           # Animals
    "sheep": 18,           # Animals
    "cow": 19,             # Animals
    "elephant": 20,        # Animals
    "bear": 21,            # Animals
    "zebra": 22,           # Animals
    "giraffe": 23,         # Animals

    # Personal items & hazards
    "backpack": 24,        # Dropped/abandoned bags
    "umbrella": 25,        # Weather items
    "handbag": 26,         # Purses/bags on ground
    "tie": 27,             # Clothing items
    "suitcase": 28,        # Luggage obstacles
    "frisbee": 29,         # Sports equipment
    "skis": 30,            # Sports equipment
    "snowboard": 31,       # Sports equipment
    "sports ball": 32,     # Sports equipment
    "kite": 33,            # Outdoor items
    "baseball bat": 34,    # Sports equipment
    "baseball glove": 35,  # Sports equipment
    "skateboard": 36,      # Sports equipment
    "surfboard": 37,       # Sports equipment
    "tennis racket": 38,   # Sports equipment
    "bottle": 39,          # Trash/obstacles
    "wine glass": 40,      # Fragile items
    "cup": 41,             # Containers
    "fork": 42,            # Utensils
    "knife": 43,           # Sharp objects
    "spoon": 44,           # Utensils
    "bowl": 45,            # Containers
    "banana": 46,          # Food items (can be obstacles)
    "apple": 47,           # Food items
    "sandwich": 48,        # Food items
    "orange": 49,          # Food items
    "broccoli": 50,        # Food items
    "carrot": 51,          # Food items
    "hot dog": 52,         # Food items
    "pizza": 53,           # Food items
    "donut": 54,           # Food items
    "cake": 55,            # Food items

    # Furniture & indoor obstacles
    "chair": 56,           # All types of chairs
    "couch": 57,           # Sofas/couches
    "potted plant": 58,    # Large planters/landscaping
    "bed": 59,             # Beds
    "dining table": 60,    # All types of tables
    "toilet": 61,          # Bathroom fixtures
    "tv": 62,              # Electronics
    "laptop": 63,          # Electronics
    "mouse": 64,           # Electronics
    "remote": 65,          # Electronics
    "keyboard": 66,        # Electronics
    "cell phone": 67,      # Electronics
    "microwave": 68,       # Appliances
    "oven": 69,            # Appliances
    "toaster": 70,         # Appliances
    "sink": 71,            # Fixtures
    "refrigerator": 72,    # Appliances
    "book": 73,            # Objects
    "clock": 74,           # Objects
    "vase": 75,            # Objects
    "scissors": 76,        # Tools
    "teddy bear": 77,      # Objects
    "hair drier": 78,      # Appliances
    "toothbrush": 79,      # Personal items
}

# For reference, common COCO class IDs:
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus, 6: train, 7: truck
# 8: boat, 9: traffic light, 10: fire hydrant, 11: stop sign, 12: parking meter