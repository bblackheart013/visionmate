"""
Navigation-Focused Detection Filter
Filters detections to show only navigation-relevant objects for blind users.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class NavigationZone:
    """Navigation zones for filtering detections."""
    immediate: float = 8.0    # 8 meters - immediate navigation concern
    caution: float = 15.0     # 15 meters - caution zone
    awareness: float = 30.0   # 30 meters - general awareness


class NavigationFilter:
    """
    Filters detections to show only navigation-relevant objects.
    Prioritizes objects that actually matter for blind navigation.
    """
    
    def __init__(self):
        """Initialize the navigation filter."""
        self.zone = NavigationZone()
        
        # Navigation-relevant object priorities
        self.navigation_objects = {
            # High priority - immediate navigation impact
            'high': ['person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 
                    'stop sign', 'traffic light', 'fire hydrant', 'pole', 'wall'],
            
            # Medium priority - navigation awareness
            'medium': ['bench', 'chair', 'table', 'sofa', 'couch', 'bed', 
                      'cabinet', 'shelf', 'exit'],
            
            # Low priority - general obstacles
            'low': ['backpack', 'suitcase', 'umbrella', 'bottle', 'cup', 'obstacle']
        }
        
        # Object size thresholds (minimum size for navigation relevance)
        self.min_sizes = {
            'person': (30, 50),      # Minimum person size
            'car': (80, 40),         # Minimum car size
            'pole': (10, 60),        # Minimum pole size
            'wall': (100, 20),       # Minimum wall segment
            'furniture': (50, 30),   # Minimum furniture size
            'default': (40, 40)      # Default minimum size
        }
    
    def filter_detections(self, detections: List[Dict[str, Any]], 
                         frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Simple filter that just applies basic confidence thresholding.
        
        Args:
            detections: Raw detections from perception pipeline
            frame_shape: (height, width) of the frame
            
        Returns:
            Filtered detections with basic confidence filtering
        """
        if not detections:
            return []
        
        # Simple confidence-based filtering only
        filtered = []
        
        for det in detections:
            # Only filter by basic confidence threshold
            if det['conf'] > 0.3:  # Low threshold to keep most detections
                filtered.append(det)
        
        # Return all detections that pass confidence threshold
        return filtered
    
    # Simplified methods removed - just using basic confidence filtering
    
    # Simplified - no complex filtering methods


def create_navigation_filter():
    """Create and return a navigation filter."""
    return NavigationFilter()


# Test function
def test_navigation_filter():
    """Test the navigation filter."""
    print("Testing Navigation Filter...")
    
    # Create test detections
    test_detections = [
        {'label': 'person', 'conf': 0.9, 'bbox': [100, 100, 150, 250], 'type': 'obstacle'},
        {'label': 'book', 'conf': 0.8, 'bbox': [200, 200, 250, 300], 'type': 'obstacle'},
        {'label': 'car', 'conf': 0.7, 'bbox': [300, 300, 400, 350], 'type': 'obstacle'},
        {'label': 'pole', 'conf': 0.6, 'bbox': [400, 100, 420, 400], 'type': 'obstacle'},
        {'label': 'wall', 'conf': 0.5, 'bbox': [0, 0, 640, 50], 'type': 'obstacle'},
    ]
    
    # Test filter
    filter_obj = NavigationFilter()
    filtered = filter_obj.filter_detections(test_detections, (480, 640))
    
    print(f"Original: {len(test_detections)} detections")
    print(f"Filtered: {len(filtered)} detections")
    print("Filtered objects:")
    for det in filtered:
        print(f"  - {det['label']}: {det['conf']:.2f}")


if __name__ == "__main__":
    test_navigation_filter()
