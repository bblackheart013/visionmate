#!/usr/bin/env python3
"""
Enhanced Guidance System for VisionMate
Adds path analysis and walking directions to Person 2's guidance system.

Features:
- Path analysis (straight/turn detection)
- Object-based navigation guidance
- Comprehensive walking directions
- Integration with Person 2's existing guidance system
"""

import time
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

@dataclass
class PathAnalysis:
    """Results of path analysis for navigation guidance."""
    path_type: str  # "straight", "turn_left", "turn_right", "obstacle_ahead"
    confidence: float
    guidance_message: str
    urgency: str  # "high", "medium", "low"

class EnhancedGuidanceEngine:
    """
    Enhanced guidance engine that provides comprehensive walking directions.
    Integrates with Person 2's existing guidance system.
    """
    
    def __init__(self):
        """Initialize the enhanced guidance engine."""
        self.last_guidance_time = 0
        self.guidance_cooldown = 1.5  # Minimum time between guidance messages
        
        # Path analysis state
        self.recent_detections = deque(maxlen=30)  # Keep last 30 detections
        self.path_history = deque(maxlen=10)  # Keep last 10 path analyses
        
        # Object tracking for path analysis
        self.object_positions = defaultdict(list)
        
        # Import Person 2's guidance system
        try:
            from guidance import GuidancePolicy
            self.person2_guidance = GuidancePolicy()
            self.person2_available = True
            print("✅ Enhanced guidance with Person 2 integration")
        except ImportError:
            self.person2_guidance = None
            self.person2_available = False
            print("⚠️ Person 2 guidance not available, using fallback")
    
    def analyze_path(self, events: List[Any]) -> PathAnalysis:
        """
        Analyze the current path situation and provide guidance.
        
        Args:
            events: List of detected events from Person 1's perception
            
        Returns:
            PathAnalysis with guidance recommendation
        """
        if not events:
            return PathAnalysis("straight", 0.7, "Path appears clear, continue straight ahead", "low")
        
        # Convert events to standard format
        standard_events = self._convert_events(events)
        
        # Analyze different types of obstacles and situations
        analysis = self._analyze_obstacles(standard_events)
        
        # Update path history
        self.path_history.append(analysis)
        
        return analysis
    
    def _convert_events(self, events: List[Any]) -> List[Dict[str, Any]]:
        """Convert events to standard format for analysis."""
        standard_events = []
        
        for event in events:
            if hasattr(event, 'type'):
                # Person 1's Event object
                standard_event = {
                    'type': event.type,
                    'label': event.label,
                    'intent': event.intent,
                    'conf': event.conf,
                    'bbox': event.bbox,
                    'bearing_deg': event.bearing_deg,
                    'dist_m': getattr(event, 'dist_m', None)
                }
            else:
                # Mock dict event
                standard_event = {
                    'type': event.get('type', 'obstacle'),
                    'label': event.get('label', 'unknown'),
                    'intent': event.get('intent', 'OBSTACLE_UNKNOWN'),
                    'conf': event.get('conf', 0.5),
                    'bbox': event.get('bbox', [0, 0, 100, 100]),
                    'bearing_deg': event.get('bearing_deg', event.get('bearing', 0.0)),
                    'dist_m': event.get('dist_m', event.get('distance', None))
                }
            
            standard_events.append(standard_event)
        
        return standard_events
    
    def _analyze_obstacles(self, events: List[Dict[str, Any]]) -> PathAnalysis:
        """Analyze obstacles and determine path guidance."""
        
        # Categorize events by type
        people = [e for e in events if e['type'] == 'obstacle' and 'person' in e['label'].lower()]
        vehicles = [e for e in events if e['type'] == 'obstacle' and any(v in e['label'].lower() for v in ['car', 'bus', 'truck', 'bicycle', 'motorcycle', 'boat'])]
        static_obstacles = [e for e in events if e['type'] == 'obstacle' and e['label'].lower() in ['pole', 'bench', 'chair', 'fire hydrant', 'potted plant', 'wall', 'sofa', 'couch', 'table', 'bed', 'cabinet', 'shelf']]
        furniture = [e for e in events if e['type'] == 'obstacle' and e['label'].lower() in ['sofa', 'couch', 'bed', 'cabinet', 'shelf', 'table', 'bench', 'chair']]
        walls_barriers = [e for e in events if e['type'] == 'obstacle' and e['label'].lower() in ['wall', 'pole', 'fire hydrant']]
        small_obstacles = [e for e in events if e['type'] == 'obstacle' and e['label'].lower() in ['obstacle', 'bottle', 'cup', 'backpack', 'suitcase', 'umbrella']]
        signs = [e for e in events if e['type'] == 'sign']
        
        # Priority 1: STOP signs (immediate hazard)
        stop_signs = [e for e in signs if 'stop' in e['label'].lower()]
        if stop_signs:
            return PathAnalysis(
                "obstacle_ahead",
                0.95,
                "STOP sign detected ahead - please stop immediately",
                "high"
            )
        
        # Priority 2: People in immediate danger zone
        close_people = [p for p in people if p.get('dist_m') and p['dist_m'] < 2.0]
        if close_people:
            closest = min(close_people, key=lambda x: x.get('dist_m', 999))
            bearing = closest['bearing_deg']
            distance = closest['dist_m']
            
            if abs(bearing) < 10:  # Directly ahead
                return PathAnalysis(
                    "obstacle_ahead",
                    0.9,
                    f"Person {distance:.1f} meters directly ahead - please stop and wait",
                    "high"
                )
            elif bearing > 10:  # Right side
                return PathAnalysis(
                    "turn_left",
                    0.8,
                    f"Person on your right side {distance:.1f} meters away - steer left",
                    "medium"
                )
            else:  # Left side
                return PathAnalysis(
                    "turn_right",
                    0.8,
                    f"Person on your left side {distance:.1f} meters away - steer right",
                    "medium"
                )
        
        # Priority 3: EXIT signs and navigation
        exit_signs = [e for e in signs if 'exit' in e['label'].lower()]
        if exit_signs:
            exit_sign = exit_signs[0]
            bearing = exit_sign['bearing_deg']
            
            if bearing > 15:
                return PathAnalysis(
                    "turn_right",
                    0.85,
                    f"EXIT sign detected to your right - turn right at {exit_sign['label']}",
                    "medium"
                )
            elif bearing < -15:
                return PathAnalysis(
                    "turn_left",
                    0.85,
                    f"EXIT sign detected to your left - turn left at {exit_sign['label']}",
                    "medium"
                )
            else:
                return PathAnalysis(
                    "straight",
                    0.8,
                    f"EXIT sign ahead - continue straight towards {exit_sign['label']}",
                    "medium"
                )
        
        # Priority 4: Walls and barriers (high priority)
        central_walls = [w for w in walls_barriers if abs(w['bearing_deg']) < 15]
        if central_walls:
            wall = central_walls[0]
            bearing = wall['bearing_deg']
            wall_type = wall['label']
            
            if abs(bearing) < 5:  # Directly ahead
                return PathAnalysis(
                    "obstacle_ahead",
                    0.9,
                    f"{wall_type.title()} directly ahead - stop and find alternative path",
                    "high"
                )
            elif bearing > 0:  # Right side
                return PathAnalysis(
                    "turn_left",
                    0.8,
                    f"{wall_type.title()} on your right - steer left to avoid",
                    "medium"
                )
            else:  # Left side
                return PathAnalysis(
                    "turn_right",
                    0.8,
                    f"{wall_type.title()} on your left - steer right to avoid",
                    "medium"
                )
        
        # Priority 5: Large furniture in path
        central_furniture = [f for f in furniture if abs(f['bearing_deg']) < 20]
        if central_furniture:
            piece = central_furniture[0]
            bearing = piece['bearing_deg']
            furniture_type = piece['label']
            
            if abs(bearing) < 10:  # Directly ahead
                return PathAnalysis(
                    "obstacle_ahead",
                    0.8,
                    f"{furniture_type.title()} directly ahead - navigate around carefully",
                    "medium"
                )
            elif bearing > 0:  # Right side
                return PathAnalysis(
                    "turn_left",
                    0.7,
                    f"{furniture_type.title()} on your right - steer left to avoid",
                    "low"
                )
            else:  # Left side
                return PathAnalysis(
                    "turn_right",
                    0.7,
                    f"{furniture_type.title()} on your left - steer right to avoid",
                    "low"
                )
        
        # Priority 6: Other static obstacles
        central_obstacles = [o for o in static_obstacles if abs(o['bearing_deg']) < 20]
        if central_obstacles:
            obstacle = central_obstacles[0]
            bearing = obstacle['bearing_deg']
            obstacle_type = obstacle['label']
            
            if abs(bearing) < 10:  # Directly ahead
                return PathAnalysis(
                    "obstacle_ahead",
                    0.7,
                    f"{obstacle_type.title()} directly ahead - avoid by steering around",
                    "medium"
                )
            elif bearing > 0:  # Right side
                return PathAnalysis(
                    "turn_left",
                    0.6,
                    f"{obstacle_type.title()} on your right - steer left to avoid",
                    "low"
                )
            else:  # Left side
                return PathAnalysis(
                    "turn_right",
                    0.6,
                    f"{obstacle_type.title()} on your left - steer right to avoid",
                    "low"
                )
        
        # Priority 7: Small obstacles and dropped items
        central_small_obstacles = [s for s in small_obstacles if abs(s['bearing_deg']) < 15]
        if central_small_obstacles:
            obstacle = central_small_obstacles[0]
            bearing = obstacle['bearing_deg']
            obstacle_type = obstacle['label']
            
            if abs(bearing) < 8:  # Directly ahead
                return PathAnalysis(
                    "obstacle_ahead",
                    0.6,
                    f"Small {obstacle_type} directly ahead - step over or around",
                    "low"
                )
            elif bearing > 0:  # Right side
                return PathAnalysis(
                    "turn_left",
                    0.5,
                    f"Small {obstacle_type} on your right - steer left slightly",
                    "low"
                )
            else:  # Left side
                return PathAnalysis(
                    "turn_right",
                    0.5,
                    f"Small {obstacle_type} on your left - steer right slightly",
                    "low"
                )
        
        # Priority 8: Vehicles (awareness)
        central_vehicles = [v for v in vehicles if abs(v['bearing_deg']) < 30]
        if central_vehicles:
            vehicle = central_vehicles[0]
            bearing = vehicle['bearing_deg']
            vehicle_type = vehicle['label']
            
            if abs(bearing) < 15:  # Ahead
                return PathAnalysis(
                    "obstacle_ahead",
                    0.75,
                    f"{vehicle_type.title()} ahead - proceed with caution",
                    "medium"
                )
            elif bearing > 0:
                return PathAnalysis(
                    "turn_left",
                    0.6,
                    f"{vehicle_type.title()} on your right - stay left",
                    "low"
                )
            else:
                return PathAnalysis(
                    "turn_right",
                    0.6,
                    f"{vehicle_type.title()} on your left - stay right",
                    "low"
                )
        
        # Default: Clear path
        return PathAnalysis(
            "straight",
            0.7,
            "Path appears clear - continue straight ahead",
            "low"
        )
    
    def get_guidance(self, events: List[Any]) -> Optional[str]:
        """
        Get comprehensive guidance message integrating Person 2's system with path analysis.
        
        Args:
            events: List of detected events from Person 1's perception
            
        Returns:
            Guidance message string or None if no guidance needed
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_guidance_time < self.guidance_cooldown:
            return None
        
        # Analyze path situation
        path_analysis = self.analyze_path(events)
        
        # Convert events for Person 2's guidance system
        if self.person2_available:
            try:
                person2_events = self._convert_events_for_person2(events)
                person2_guidance = self.person2_guidance.choose(person2_events)
                
                # Combine Person 2's guidance with our path analysis
                if person2_guidance and path_analysis.urgency == "high":
                    # Person 2's high-priority guidance takes precedence
                    self.last_guidance_time = current_time
                    return person2_guidance
                elif path_analysis.urgency == "high" or path_analysis.urgency == "medium":
                    # Our path analysis for medium/high priority situations
                    self.last_guidance_time = current_time
                    return path_analysis.guidance_message
                elif person2_guidance:
                    # Person 2's guidance for low priority
                    self.last_guidance_time = current_time
                    return person2_guidance
                else:
                    # Our path analysis for low priority
                    if path_analysis.urgency == "low" and path_analysis.confidence > 0.8:
                        self.last_guidance_time = current_time
                        return path_analysis.guidance_message
            except Exception as e:
                print(f"Warning: Person 2 guidance error: {e}")
                # Fall back to our path analysis
        
        # Fallback to our path analysis
        if path_analysis.urgency in ["high", "medium"] or (path_analysis.urgency == "low" and path_analysis.confidence > 0.8):
            self.last_guidance_time = current_time
            return path_analysis.guidance_message
        
        return None
    
    def _convert_events_for_person2(self, events: List[Any]) -> List[Dict[str, Any]]:
        """Convert events to format expected by Person 2's guidance system."""
        person2_events = []
        
        for event in events:
            if hasattr(event, 'intent'):
                # Person 1's Event object
                person2_event = {
                    'intent': event.intent,
                    'conf': event.conf,
                    'bearing_deg': event.bearing_deg
                }
                if hasattr(event, 'dist_m') and event.dist_m is not None:
                    person2_event['dist_m'] = event.dist_m
                person2_events.append(person2_event)
            else:
                # Mock dict event
                person2_event = {
                    'intent': event.get('intent', 'OBSTACLE_UNKNOWN'),
                    'conf': event.get('conf', 0.5),
                    'bearing_deg': event.get('bearing_deg', event.get('bearing', 0.0))
                }
                if event.get('dist_m') or event.get('distance'):
                    person2_event['dist_m'] = event.get('dist_m', event.get('distance'))
                person2_events.append(person2_event)
        
        return person2_events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guidance statistics for monitoring."""
        return {
            "person2_available": self.person2_available,
            "path_history_length": len(self.path_history),
            "recent_detections": len(self.recent_detections),
            "last_guidance_time": self.last_guidance_time
        }


# Test function
def test_enhanced_guidance():
    """Test the enhanced guidance system."""
    print("Testing Enhanced Guidance System...")
    
    # Create test events
    test_events = [
        {
            'type': 'obstacle',
            'label': 'person',
            'intent': 'OBSTACLE_PERSON',
            'conf': 0.9,
            'bbox': [300, 200, 400, 500],
            'bearing_deg': 5,
            'dist_m': 1.5
        }
    ]
    
    # Test enhanced guidance
    engine = EnhancedGuidanceEngine()
    guidance = engine.get_guidance(test_events)
    
    print(f"Guidance: {guidance}")
    print(f"Stats: {engine.get_stats()}")


if __name__ == "__main__":
    test_enhanced_guidance()
