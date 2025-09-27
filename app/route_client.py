#!/usr/bin/env python3
"""
Route client for cloud-based waypoint navigation.

This module provides integration with cloud route services or local
prebaked waypoint data for navigation guidance.

Author: Person 3 (Integration & Snapdragon lead)
"""

import json
import logging
import requests
from typing import List, Dict, Any, Optional
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Waypoint:
    """Represents a navigation waypoint."""
    id: str
    lat: float
    lon: float
    name: str
    instructions: str
    distance_m: float = 0.0
    bearing_deg: float = 0.0

class RouteClient:
    """Client for fetching navigation routes from cloud or local sources."""
    
    def __init__(self, service_url: Optional[str] = None, timeout: int = 10):
        """
        Initialize route client.
        
        Args:
            service_url: URL of the route service (optional)
            timeout: Request timeout in seconds
        """
        self.service_url = service_url
        self.timeout = timeout
        self.current_route: List[Waypoint] = []
        self.current_waypoint_index = 0
        
    def get_route(self, start: str, goal: str) -> List[Waypoint]:
        """
        Get route waypoints from service or fallback to local data.
        
        Args:
            start: Starting location
            goal: Destination
            
        Returns:
            List of waypoints for navigation
        """
        if self.service_url:
            try:
                return self._fetch_cloud_route(start, goal)
            except Exception as e:
                logger.warning(f"Cloud route failed: {e}. Using local fallback.")
        
        return self._load_local_route(start, goal)
    
    def _fetch_cloud_route(self, start: str, goal: str) -> List[Waypoint]:
        """Fetch route from cloud service."""
        payload = {
            "start": start,
            "goal": goal,
            "format": "waypoints"
        }
        
        response = requests.post(
            self.service_url,
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        data = response.json()
        waypoints = []
        
        for wp_data in data.get("waypoints", []):
            waypoint = Waypoint(
                id=wp_data.get("id", ""),
                lat=wp_data.get("lat", 0.0),
                lon=wp_data.get("lon", 0.0),
                name=wp_data.get("name", ""),
                instructions=wp_data.get("instructions", ""),
                distance_m=wp_data.get("distance_m", 0.0),
                bearing_deg=wp_data.get("bearing_deg", 0.0)
            )
            waypoints.append(waypoint)
        
        logger.info(f"Fetched {len(waypoints)} waypoints from cloud service")
        return waypoints
    
    def _load_local_route(self, start: str, goal: str) -> List[Waypoint]:
        """Load route from local prebaked data."""
        # Try to load from samples directory
        samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
        route_file = os.path.join(samples_dir, "route.json")
        
        if os.path.exists(route_file):
            try:
                with open(route_file, 'r') as f:
                    data = json.load(f)
                
                # Find matching route
                route_key = f"{start.lower()}_to_{goal.lower()}"
                if route_key in data:
                    waypoints_data = data[route_key]["waypoints"]
                    waypoints = []
                    
                    for wp_data in waypoints_data:
                        waypoint = Waypoint(
                            id=wp_data.get("id", ""),
                            lat=wp_data.get("lat", 0.0),
                            lon=wp_data.get("lon", 0.0),
                            name=wp_data.get("name", ""),
                            instructions=wp_data.get("instructions", ""),
                            distance_m=wp_data.get("distance_m", 0.0),
                            bearing_deg=wp_data.get("bearing_deg", 0.0)
                        )
                        waypoints.append(waypoint)
                    
                    logger.info(f"Loaded {len(waypoints)} waypoints from local route file")
                    return waypoints
                
            except Exception as e:
                logger.warning(f"Failed to load local route: {e}")
        
        # Generate default waypoints if no route found
        return self._generate_default_route(start, goal)
    
    def _generate_default_route(self, start: str, goal: str) -> List[Waypoint]:
        """Generate a simple default route."""
        default_waypoints = [
            Waypoint(
                id="start",
                lat=0.0,
                lon=0.0,
                name=f"Starting from {start}",
                instructions=f"Begin navigation from {start}",
                distance_m=0.0,
                bearing_deg=0.0
            ),
            Waypoint(
                id="mid",
                lat=0.001,
                lon=0.001,
                name="Continue forward",
                instructions="Continue straight ahead",
                distance_m=50.0,
                bearing_deg=0.0
            ),
            Waypoint(
                id="goal",
                lat=0.002,
                lon=0.002,
                name=f"Arrived at {goal}",
                instructions=f"You have arrived at {goal}",
                distance_m=100.0,
                bearing_deg=0.0
            )
        ]
        
        logger.info(f"Generated default route with {len(default_waypoints)} waypoints")
        return default_waypoints
    
    def set_current_route(self, waypoints: List[Waypoint]):
        """Set the current active route."""
        self.current_route = waypoints
        self.current_waypoint_index = 0
        logger.info(f"Set current route with {len(waypoints)} waypoints")
    
    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Get the current waypoint."""
        if self.current_route and 0 <= self.current_waypoint_index < len(self.current_route):
            return self.current_route[self.current_waypoint_index]
        return None
    
    def advance_waypoint(self):
        """Move to the next waypoint."""
        if self.current_waypoint_index < len(self.current_route) - 1:
            self.current_waypoint_index += 1
            current = self.get_current_waypoint()
            if current:
                logger.info(f"Advanced to waypoint: {current.name}")
    
    def get_route_progress(self) -> Dict[str, Any]:
        """Get current route progress information."""
        if not self.current_route:
            return {"total_waypoints": 0, "current_index": 0, "progress": 0.0}
        
        progress = (self.current_waypoint_index / (len(self.current_route) - 1)) * 100 if len(self.current_route) > 1 else 100.0
        
        return {
            "total_waypoints": len(self.current_route),
            "current_index": self.current_waypoint_index,
            "progress": progress,
            "current_waypoint": self.get_current_waypoint()
        }

# Global route client instance
_route_client: Optional[RouteClient] = None

def get_route(start: str, goal: str, service_url: Optional[str] = None) -> List[Waypoint]:
    """
    Get route waypoints (convenience function).
    
    Args:
        start: Starting location
        goal: Destination
        service_url: Optional service URL
        
    Returns:
        List of waypoints
    """
    global _route_client
    
    if _route_client is None:
        _route_client = RouteClient(service_url)
    
    return _route_client.get_route(start, goal)

def set_current_route(waypoints: List[Waypoint]):
    """Set current route (convenience function)."""
    global _route_client
    
    if _route_client is None:
        _route_client = RouteClient()
    
    _route_client.set_current_route(waypoints)

def get_current_waypoint() -> Optional[Waypoint]:
    """Get current waypoint (convenience function)."""
    global _route_client
    
    if _route_client is None:
        return None
    
    return _route_client.get_current_waypoint()

def get_route_progress() -> Dict[str, Any]:
    """Get route progress (convenience function)."""
    global _route_client
    
    if _route_client is None:
        return {"total_waypoints": 0, "current_index": 0, "progress": 0.0}
    
    return _route_client.get_route_progress()

# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test route client")
    parser.add_argument("--service", type=str, help="Route service URL")
    parser.add_argument("--start", default="lobby", help="Start location")
    parser.add_argument("--goal", default="cafeteria", help="Goal location")
    
    args = parser.parse_args()
    
    # Test the route client
    client = RouteClient(args.service)
    waypoints = client.get_route(args.start, args.goal)
    
    print(f"Route from {args.start} to {args.goal}:")
    for i, wp in enumerate(waypoints):
        print(f"  {i+1}. {wp.name} - {wp.instructions}")
    
    # Test route progress
    client.set_current_route(waypoints)
    print(f"\nRoute progress: {client.get_route_progress()}")
    
    # Advance waypoints
    for i in range(len(waypoints)):
        current = client.get_current_waypoint()
        if current:
            print(f"Current waypoint: {current.name}")
        if i < len(waypoints) - 1:
            client.advance_waypoint()
