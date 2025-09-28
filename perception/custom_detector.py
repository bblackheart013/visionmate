"""
Custom Computer Vision Detector for Real-World Obstacles
Detects objects that YOLOv8 might miss: walls, poles, furniture, etc.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class CustomDetection:
    """Custom detection result."""
    label: str
    conf: float
    bbox: List[int]
    source: str = "custom"
    description: str = ""


class CustomObstacleDetector:
    """
    Custom detector for real-world obstacles using computer vision techniques.
    Detects walls, poles, furniture, and other objects that YOLOv8 might miss.
    """
    
    def __init__(self):
        """Initialize the custom detector."""
        self.frame_count = 0
        
        # Detection parameters
        self.min_contour_area = 500
        self.max_contour_area = 50000
        self.min_aspect_ratio = 0.1
        self.max_aspect_ratio = 10.0
        
    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect custom obstacles using computer vision techniques.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            List of custom detections
        """
        self.frame_count += 1
        
        # Run different detection methods
        detections = []
        
        # 1. Detect vertical structures (poles, walls)
        pole_detections = self._detect_vertical_structures(frame_bgr)
        detections.extend(pole_detections)
        
        # 2. Detect horizontal surfaces (tables, benches)
        surface_detections = self._detect_horizontal_surfaces(frame_bgr)
        detections.extend(surface_detections)
        
        # 3. Detect large furniture (sofas, cabinets)
        furniture_detections = self._detect_furniture(frame_bgr)
        detections.extend(furniture_detections)
        
        # 4. Detect walls and barriers
        wall_detections = self._detect_walls_barriers(frame_bgr)
        detections.extend(wall_detections)
        
        # 5. Detect dropped items and small obstacles
        item_detections = self._detect_small_obstacles(frame_bgr)
        detections.extend(item_detections)
        
        return detections
    
    def _detect_vertical_structures(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect vertical structures like poles, posts, columns."""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Group nearby vertical lines
            vertical_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly vertical
                if abs(x2 - x1) < 20 and abs(y2 - y1) > 50:
                    vertical_lines.append(line[0])
            
            # Group lines into potential poles
            pole_groups = self._group_vertical_lines(vertical_lines)
            
            for group in pole_groups:
                if len(group) >= 2:  # Need at least 2 lines to form a pole
                    # Calculate bounding box
                    x_coords = []
                    y_coords = []
                    for line in group:
                        x1, y1, x2, y2 = line
                        x_coords.extend([x1, x2])
                        y_coords.extend([y1, y2])
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Check if it's a reasonable pole size
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    if width < 100 and height > 100 and height / width > 3:
                        detections.append({
                            'label': 'pole',
                            'conf': 0.7,
                            'bbox': [x_min, y_min, x_max, y_max],
                            'source': 'custom',
                            'description': f'Vertical pole {width}x{height}'
                        })
        
        return detections
    
    def _detect_horizontal_surfaces(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect horizontal surfaces like tables, benches, platforms."""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=80, maxLineGap=10)
        
        if lines is not None:
            # Group nearby horizontal lines
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly horizontal
                if abs(y2 - y1) < 20 and abs(x2 - x1) > 80:
                    horizontal_lines.append(line[0])
            
            # Group lines into potential surfaces
            surface_groups = self._group_horizontal_lines(horizontal_lines)
            
            for group in surface_groups:
                if len(group) >= 2:
                    # Calculate bounding box
                    x_coords = []
                    y_coords = []
                    for line in group:
                        x1, y1, x2, y2 = line
                        x_coords.extend([x1, x2])
                        y_coords.extend([y1, y2])
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Check if it's a reasonable surface size
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    if width > 100 and height < 50 and width / height > 2:
                        # Determine surface type based on size and position
                        if width > 300:
                            surface_type = 'table'
                        elif width > 150:
                            surface_type = 'bench'
                        else:
                            surface_type = 'platform'
                        
                        detections.append({
                            'label': surface_type,
                            'conf': 0.6,
                            'bbox': [x_min, y_min, x_max, y_max],
                            'source': 'custom',
                            'description': f'Horizontal {surface_type} {width}x{height}'
                        })
        
        return detections
    
    def _detect_furniture(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect large furniture using contour analysis."""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_contour_area < area < self.max_contour_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter by aspect ratio
                if self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio:
                    # Determine furniture type based on size and shape
                    if area > 10000 and aspect_ratio > 1.5:
                        furniture_type = 'sofa'
                    elif area > 8000 and 0.8 < aspect_ratio < 1.2:
                        furniture_type = 'cabinet'
                    elif area > 5000 and aspect_ratio > 2:
                        furniture_type = 'shelf'
                    else:
                        furniture_type = 'furniture'
                    
                    detections.append({
                        'label': furniture_type,
                        'conf': 0.5,
                        'bbox': [x, y, x + w, y + h],
                        'source': 'custom',
                        'description': f'{furniture_type.title()} {w}x{h}'
                    })
        
        return detections
    
    def _detect_walls_barriers(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect walls and barriers using edge detection."""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply strong edge detection for walls
        edges = cv2.Canny(gray, 30, 100)
        
        # Detect long lines that could be walls
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=200, maxLineGap=20)
        
        if lines is not None:
            # Group lines into potential walls
            wall_segments = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 200:  # Long lines are likely walls
                    wall_segments.append(line[0])
            
            # Create wall detections
            for segment in wall_segments:
                x1, y1, x2, y2 = segment
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                
                # Expand bounding box to represent wall thickness
                thickness = 20
                detections.append({
                    'label': 'wall',
                    'conf': 0.6,
                    'bbox': [x_min-thickness, y_min-thickness, x_max+thickness, y_max+thickness],
                    'source': 'custom',
                    'description': f'Wall segment {x_max-x_min}px long'
                })
        
        return detections
    
    def _detect_small_obstacles(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect small obstacles and dropped items."""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to find small objects
        kernel = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Find contours of small objects
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Small objects (50-1000 pixels)
            if 50 < area < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's roughly square/circular (likely a dropped item)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:
                    detections.append({
                        'label': 'obstacle',
                        'conf': 0.4,
                        'bbox': [x, y, x + w, y + h],
                        'source': 'custom',
                        'description': f'Small obstacle {w}x{h}'
                    })
        
        return detections
    
    def _group_vertical_lines(self, lines: List) -> List[List]:
        """Group nearby vertical lines into potential poles."""
        if not lines:
            return []
        
        groups = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            group = [line1]
            used.add(i)
            
            for j, line2 in enumerate(lines):
                if j in used:
                    continue
                
                # Check if lines are close horizontally
                x1_1, y1_1, x1_2, y1_2 = line1
                x2_1, y2_1, x2_2, y2_2 = line2
                
                center1_x = (x1_1 + x1_2) / 2
                center2_x = (x2_1 + x2_2) / 2
                
                if abs(center1_x - center2_x) < 30:  # Close horizontally
                    group.append(line2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _group_horizontal_lines(self, lines: List) -> List[List]:
        """Group nearby horizontal lines into potential surfaces."""
        if not lines:
            return []
        
        groups = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            group = [line1]
            used.add(i)
            
            for j, line2 in enumerate(lines):
                if j in used:
                    continue
                
                # Check if lines are close vertically
                x1_1, y1_1, x1_2, y1_2 = line1
                x2_1, y2_1, x2_2, y2_2 = line2
                
                center1_y = (y1_1 + y1_2) / 2
                center2_y = (y2_1 + y2_2) / 2
                
                if abs(center1_y - center2_y) < 30:  # Close vertically
                    group.append(line2)
                    used.add(j)
            
            groups.append(group)
        
        return groups


def create_custom_detector():
    """Create and return a custom obstacle detector."""
    return CustomObstacleDetector()


# Test function
def test_custom_detector():
    """Test the custom detector."""
    print("Testing Custom Obstacle Detector...")
    
    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some test shapes
    cv2.rectangle(frame, (100, 100, 150, 400), (255, 255, 255), -1)  # Vertical rectangle (pole)
    cv2.rectangle(frame, (200, 300, 400, 320), (255, 255, 255), -1)  # Horizontal rectangle (table)
    
    # Test detector
    detector = CustomObstacleDetector()
    detections = detector.detect(frame)
    
    print(f"Detected {len(detections)} custom obstacles:")
    for det in detections:
        print(f"  - {det['label']}: {det['description']} (conf: {det['conf']:.2f})")


if __name__ == "__main__":
    test_custom_detector()
