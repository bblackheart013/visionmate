"""
Nearby Object Filter for Display
Detects everything but only shows boundaries for objects that are nearby.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class NearbyZone:
    """Nearby zones for display filtering."""
    immediate: float = 3.0    # 3 meters - show boundaries
    caution: float = 5.0      # 5 meters - show boundaries
    awareness: float = 8.0    # 8 meters - show boundaries


class NearbyDisplayFilter:
    """
    Filters object boundaries for display - only shows nearby objects.
    Keeps all detection data but filters what's visually displayed.
    """
    
    def __init__(self):
        """Initialize the nearby display filter."""
        self.zone = NearbyZone()
        
        # Walking obstacle thresholds - only show objects that block walking path
        self.walking_obstacle_thresholds = {
            # Immediate walking obstacles - show when close
            'person': 4.0,      # 4 meters - people in walking path
            'pole': 3.0,        # 3 meters - poles blocking path
            'wall': 2.5,        # 2.5 meters - walls blocking path
            'fire hydrant': 2.0, # 2 meters - street fixtures
            'bench': 3.0,       # 3 meters - benches blocking sidewalk
            
            # Furniture obstacles - show when blocking path
            'chair': 2.0,       # 2 meters - chairs in path
            'table': 3.0,       # 3 meters - tables blocking path
            'sofa': 3.0,        # 3 meters - sofas blocking path
            'couch': 3.0,       # 3 meters - couches blocking path
            'bed': 4.0,         # 4 meters - beds blocking path
            
            # Small walking obstacles - show when very close
            'backpack': 1.5,    # 1.5 meters - dropped items
            'suitcase': 2.0,    # 2 meters - luggage obstacles
            'umbrella': 1.5,    # 1.5 meters - dropped umbrellas
            'bottle': 1.0,      # 1 meter - trash obstacles
            'cup': 1.0,         # 1 meter - small obstacles
            
            # Vehicle obstacles - show when blocking path
            'car': 6.0,         # 6 meters - cars blocking crosswalk
            'bus': 8.0,         # 8 meters - buses at stops
            'truck': 8.0,       # 8 meters - trucks blocking path
            'bicycle': 3.0,     # 3 meters - bikes blocking sidewalk
            'motorcycle': 3.0,  # 3 meters - motorcycles blocking path
            
            # Navigation signs - show when in walking path
            'stop sign': 5.0,   # 5 meters - stop signs
            'traffic light': 6.0, # 6 meters - traffic lights
            'exit': 6.0,        # 6 meters - exit signs
        }
        
        # Objects that are NOT walking obstacles (don't show)
        self.non_walking_objects = {
            'book', 'laptop', 'mouse', 'keyboard', 'remote', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
            'apple', 'banana', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'sandwich', 'wine glass', 'fork',
            'knife', 'spoon', 'bowl', 'kite', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'tie', 'book', 'tv', 'laptop'
        }
    
    def filter_for_display(self, detections: List[Dict[str, Any]], 
                          frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Filter detections for display - only show walking path obstacles.
        
        Args:
            detections: All detections from perception pipeline
            frame_shape: (height, width) of the frame
            
        Returns:
            Detections that should be displayed (walking obstacles only)
        """
        if not detections:
            return []
        
        height, width = frame_shape[:2]
        display_detections = []
        
        for det in detections:
            label = det['label'].lower()
            
            # Skip non-walking objects completely
            if self._is_non_walking_object(label):
                continue
            
            # Skip if not a walking obstacle
            if not self._is_walking_obstacle(label):
                continue
            
            # Calculate distance estimate
            distance_estimate = self._estimate_distance(det, height, width)
            
            # Get walking obstacle threshold for this object type
            threshold = self._get_walking_obstacle_threshold(label)
            
            # Only show if object is close enough to be a walking obstacle
            if distance_estimate <= threshold:
                # Add distance info to detection for debugging
                det_with_distance = det.copy()
                det_with_distance['estimated_distance'] = distance_estimate
                det_with_distance['walking_threshold'] = threshold
                display_detections.append(det_with_distance)
        
        return display_detections
    
    def _estimate_distance(self, detection: Dict[str, Any], 
                          frame_height: int, frame_width: int) -> float:
        """Estimate distance to object based on position and size."""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Calculate object center and size
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        obj_width = x2 - x1
        obj_height = y2 - y1
        obj_area = obj_width * obj_height
        
        # Position-based distance estimation
        y_ratio = center_y / frame_height
        
        # Size-based distance estimation (larger objects are closer)
        frame_area = frame_height * frame_width
        size_ratio = obj_area / frame_area
        
        # Distance estimation formula
        if y_ratio > 0.8 and size_ratio > 0.01:  # Very close
            return 1.5
        elif y_ratio > 0.7 and size_ratio > 0.005:  # Close
            return 3.0
        elif y_ratio > 0.6 and size_ratio > 0.002:  # Medium distance
            return 5.0
        elif y_ratio > 0.5 and size_ratio > 0.001:  # Far
            return 8.0
        elif y_ratio > 0.4:  # Very far
            return 12.0
        else:  # Extremely far
            return 20.0
    
    def _is_non_walking_object(self, label: str) -> bool:
        """Check if object is not a walking obstacle."""
        return label in self.non_walking_objects
    
    def _is_walking_obstacle(self, label: str) -> bool:
        """Check if object is a walking path obstacle."""
        # Check if it's in our walking obstacle thresholds
        for obj_type in self.walking_obstacle_thresholds.keys():
            if obj_type in label:
                return True
        return False
    
    def _get_walking_obstacle_threshold(self, label: str) -> float:
        """Get walking obstacle threshold for object type."""
        label_lower = label.lower()
        
        # Check for exact matches first
        for obj_type, threshold in self.walking_obstacle_thresholds.items():
            if obj_type in label_lower:
                return threshold
        
        # Check for partial matches
        if 'person' in label_lower:
            return self.walking_obstacle_thresholds['person']
        elif any(vehicle in label_lower for vehicle in ['car', 'bus', 'truck']):
            return self.walking_obstacle_thresholds['car']
        elif 'bicycle' in label_lower or 'motorcycle' in label_lower:
            return self.walking_obstacle_thresholds['bicycle']
        elif any(sign in label_lower for sign in ['stop', 'exit', 'traffic']):
            return self.walking_obstacle_thresholds['stop sign']
        elif 'pole' in label_lower:
            return self.walking_obstacle_thresholds['pole']
        elif 'wall' in label_lower:
            return self.walking_obstacle_thresholds['wall']
        elif any(furniture in label_lower for furniture in ['chair', 'table', 'bench', 'sofa', 'couch', 'bed']):
            return self.walking_obstacle_thresholds['chair']
        elif any(item in label_lower for item in ['backpack', 'suitcase', 'umbrella', 'bottle', 'cup']):
            return self.walking_obstacle_thresholds['backpack']
        
        # Default threshold for unknown walking obstacles
        return 3.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        return {
            "nearby_thresholds": len(self.nearby_thresholds),
            "zones": {
                "immediate": self.zone.immediate,
                "caution": self.zone.caution,
                "awareness": self.zone.awareness
            }
        }


def create_nearby_display_filter():
    """Create and return a nearby display filter."""
    return NearbyDisplayFilter()


# Test function
def test_nearby_display_filter():
    """Test the nearby display filter."""
    print("Testing Nearby Display Filter...")
    
    # Create test detections - mix of walking obstacles and non-obstacles
    test_detections = [
        {'label': 'person', 'conf': 0.9, 'bbox': [100, 100, 150, 250], 'type': 'obstacle'},  # Walking obstacle
        {'label': 'car', 'conf': 0.8, 'bbox': [200, 200, 300, 300], 'type': 'obstacle'},    # Walking obstacle
        {'label': 'pole', 'conf': 0.7, 'bbox': [400, 100, 420, 400], 'type': 'obstacle'},   # Walking obstacle
        {'label': 'wall', 'conf': 0.6, 'bbox': [0, 0, 640, 50], 'type': 'obstacle'},        # Walking obstacle
        {'label': 'chair', 'conf': 0.5, 'bbox': [300, 400, 350, 500], 'type': 'obstacle'},  # Walking obstacle
        {'label': 'book', 'conf': 0.8, 'bbox': [500, 300, 550, 350], 'type': 'obstacle'},   # NOT walking obstacle
        {'label': 'laptop', 'conf': 0.7, 'bbox': [600, 200, 700, 300], 'type': 'obstacle'}, # NOT walking obstacle
    ]
    
    # Test filter
    filter_obj = NearbyDisplayFilter()
    filtered = filter_obj.filter_for_display(test_detections, (480, 640))
    
    print(f"Original: {len(test_detections)} detections")
    print(f"Walking obstacles (display): {len(filtered)} detections")
    print("Walking obstacles only:")
    for det in filtered:
        print(f"  - {det['label']}: {det['estimated_distance']:.1f}m (threshold: {det['walking_threshold']}m)")
    
    # Show what was filtered out
    filtered_labels = {det['label'] for det in filtered}
    original_labels = {det['label'] for det in test_detections}
    filtered_out = original_labels - filtered_labels
    if filtered_out:
        print(f"Filtered out (non-walking objects): {', '.join(filtered_out)}")


if __name__ == "__main__":
    test_nearby_display_filter()
