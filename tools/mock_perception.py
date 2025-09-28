#!/usr/bin/env python3
"""
Mock perception tool for generating synthetic events.

This tool emits realistic synthetic perception events at 30 FPS following the same
schema as the real perception pipeline. This allows Person 2 (Guidance) and
Person 3 (Orchestrator) to develop and test their components without needing
actual camera input or trained models.

Usage:
    python mock_perception.py [--duration 60] [--fps 30] [--out events.json]
"""
import argparse
import json
import time
import random
import math
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception import event_to_dict, Event


class MockPerceptionGenerator:
    """Generates synthetic perception events for testing."""

    def __init__(self, fps: float = 30.0):
        """
        Initialize mock generator.

        Args:
            fps: Target frame rate for event generation
        """
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.start_time = time.time()
        self.frame_count = 0

        # Simulation state
        self.active_objects = {}
        self.next_object_id = 1

    def _create_mock_event(self, event_type: str, label: str, intent: str,
                          bbox: List[int], bearing: float, dist: float = None) -> Event:
        """Create a mock Event object."""
        return Event(
            schema="v1",
            ts=time.time(),
            type=event_type,
            label=label,
            intent=intent,
            conf=random.uniform(0.7, 0.95),
            bbox=bbox,
            bearing_deg=round(bearing, 1),
            dist_m=round(dist, 1) if dist is not None else None,
            sources=["mock"]
        )

    def _generate_person_event(self) -> Event:
        """Generate a mock person obstacle event."""
        # Random position in frame (simulate 720p)
        frame_w, frame_h = 1280, 720

        # Person typically appears in center 80% of frame
        x_center = random.randint(int(frame_w * 0.1), int(frame_w * 0.9))

        # Person height varies with distance
        distance = random.uniform(1.5, 8.0)  # 1.5 to 8 meters
        person_height_px = int(80 * 3.0 / distance)  # Height heuristic
        person_width_px = person_height_px // 3

        # Calculate bbox
        x1 = x_center - person_width_px // 2
        x2 = x_center + person_width_px // 2
        y2 = random.randint(frame_h // 2, frame_h - 10)
        y1 = y2 - person_height_px

        # Calculate bearing
        bearing = ((x_center / frame_w) - 0.5) * 60.0  # 60° FOV

        return self._create_mock_event(
            event_type="obstacle",
            label="person",
            intent="OBSTACLE_PERSON",
            bbox=[x1, y1, x2, y2],
            bearing=bearing,
            dist=distance
        )

    def _generate_vehicle_event(self) -> Event:
        """Generate a mock vehicle obstacle event."""
        vehicle_types = ["car", "bus", "truck"]
        vehicle = random.choice(vehicle_types)

        frame_w, frame_h = 1280, 720

        # Vehicles can be larger and more varied in position
        if vehicle == "car":
            width = random.randint(80, 150)
            height = random.randint(50, 80)
        elif vehicle == "bus":
            width = random.randint(120, 200)
            height = random.randint(80, 120)
        else:  # truck
            width = random.randint(100, 180)
            height = random.randint(70, 110)

        x1 = random.randint(0, frame_w - width)
        y1 = random.randint(frame_h // 3, frame_h - height)
        x2 = x1 + width
        y2 = y1 + height

        x_center = (x1 + x2) / 2
        bearing = ((x_center / frame_w) - 0.5) * 60.0

        return self._create_mock_event(
            event_type="obstacle",
            label=vehicle,
            intent=f"OBSTACLE_{vehicle.upper()}",
            bbox=[x1, y1, x2, y2],
            bearing=bearing
        )

    def _generate_sign_event(self) -> Event:
        """Generate a mock sign event."""
        sign_types = [
            ("STOP", "STOP"),
            ("EXIT", "EXIT"),
            ("EXIT", "EXIT_RIGHT"),
            ("EXIT", "EXIT_LEFT")
        ]

        label, intent = random.choice(sign_types)

        frame_w, frame_h = 1280, 720

        # Signs typically in upper half of frame
        width = random.randint(60, 120)
        height = random.randint(40, 80)

        x1 = random.randint(0, frame_w - width)
        y1 = random.randint(0, frame_h // 2)
        x2 = x1 + width
        y2 = y1 + height

        x_center = (x1 + x2) / 2
        bearing = ((x_center / frame_w) - 0.5) * 60.0

        # Choose source (STOP can be yolo+ocr, EXIT only ocr)
        if label == "STOP" and random.random() < 0.5:
            sources = ["yolo", "ocr"]
        else:
            sources = ["ocr"]

        event = self._create_mock_event(
            event_type="sign",
            label=label,
            intent=intent,
            bbox=[x1, y1, x2, y2],
            bearing=bearing
        )
        event.sources = sources

        return event

    def generate_frame_events(self) -> List[Event]:
        """
        Generate events for a single frame.

        Returns:
            List of Event objects for this frame
        """
        self.frame_count += 1
        events = []

        # Simulate realistic event frequency
        # Not every frame has events (persistence filter effect)

        # Person events (most common)
        if random.random() < 0.15:  # 15% chance per frame
            events.append(self._generate_person_event())

        # Additional people (sometimes multiple)
        if random.random() < 0.05:
            events.append(self._generate_person_event())

        # Vehicle events (less common)
        if random.random() < 0.08:  # 8% chance per frame
            events.append(self._generate_vehicle_event())

        # Sign events (least common, more persistence)
        if random.random() < 0.03:  # 3% chance per frame
            events.append(self._generate_sign_event())

        # Occasionally no events (realistic)
        if random.random() < 0.7:  # 70% chance of no events
            events = []

        return events

    def run_simulation(self, duration_seconds: float, output_file: str = None):
        """
        Run the mock perception simulation.

        Args:
            duration_seconds: How long to run simulation
            output_file: Optional file to save events to
        """
        print(f"Starting mock perception simulation")
        print(f"Duration: {duration_seconds}s at {self.fps} FPS")
        print(f"Target frames: {int(duration_seconds * self.fps)}")

        all_events = []
        start_time = time.time()
        last_stats_time = start_time

        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            if elapsed >= duration_seconds:
                break

            # Generate events for this frame
            frame_events = self.generate_frame_events()

            # Add frame timing info
            for event in frame_events:
                event_dict = event_to_dict(event)
                event_dict["frame_num"] = self.frame_count
                event_dict["sim_time"] = elapsed
                all_events.append(event_dict)

            # Print stats every 5 seconds
            if current_time - last_stats_time >= 5.0:
                actual_fps = self.frame_count / elapsed
                print(f"Time: {elapsed:.1f}s | "
                      f"Frame: {self.frame_count} | "
                      f"FPS: {actual_fps:.1f} | "
                      f"Events: {len(all_events)}")
                last_stats_time = current_time

            # Sleep to maintain target FPS
            next_frame_time = start_time + (self.frame_count * self.frame_interval)
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Final stats
        total_time = time.time() - start_time
        actual_fps = self.frame_count / total_time

        print(f"\nSimulation complete!")
        print(f"Total time: {total_time:.1f}s")
        print(f"Frames generated: {self.frame_count}")
        print(f"Actual FPS: {actual_fps:.1f}")
        print(f"Total events: {len(all_events)}")

        # Event type breakdown
        event_types = {}
        for event in all_events:
            intent = event["intent"]
            event_types[intent] = event_types.get(intent, 0) + 1

        print("\nEvent breakdown:")
        for intent, count in sorted(event_types.items()):
            print(f"  {intent}: {count}")

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    "metadata": {
                        "simulation_type": "mock_perception",
                        "duration_seconds": duration_seconds,
                        "target_fps": self.fps,
                        "actual_fps": actual_fps,
                        "total_frames": self.frame_count,
                        "total_events": len(all_events)
                    },
                    "events": all_events
                }, f, indent=2)
            print(f"Events saved to: {output_file}")


def main():
    """Main entry point for mock perception tool."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic perception events for testing"
    )
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Simulation duration in seconds (default: 60)")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Target frame rate (default: 30)")
    parser.add_argument("--out",
                       help="Output JSON file for events (optional)")

    args = parser.parse_args()

    # Create and run simulation
    generator = MockPerceptionGenerator(args.fps)

    try:
        generator.run_simulation(args.duration, args.out)
        return 0
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 0
    except Exception as e:
        print(f"Error running simulation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())