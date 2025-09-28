#!/usr/bin/env python3
"""
Replay tool for processing videos through GuidedSight perception pipeline.

Usage:
    python replay.py --video input.mp4 --out events.json [--overlay output.mp4]

This tool processes a video file frame by frame, runs perception, and outputs:
1. events.json - All promoted events in chronological order
2. output.mp4 - Optional overlay video with bounding boxes and labels
3. Console stats every 5 seconds
"""
import argparse
import json
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cv2
    import numpy as np
    from perception import PerceptionPipeline, PerceptionConfig, event_to_dict
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install opencv-python ultralytics easyocr")
    DEPENDENCIES_AVAILABLE = False
    # Create dummy numpy for type hints
    class DummyNp:
        ndarray = object
    np = DummyNp()


def draw_event_overlay(frame, events):
    """
    Draw event overlays on frame for visualization.

    Args:
        frame: Input frame
        events: List of Event objects

    Returns:
        Frame with overlays drawn
    """
    overlay_frame = frame.copy()

    for event in events:
        bbox = event.bbox
        x1, y1, x2, y2 = bbox

        # Color coding by type
        if event.type == "obstacle":
            color = (0, 0, 255)  # Red for obstacles
        else:
            color = (0, 255, 0)  # Green for signs

        # Draw bounding box
        cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)

        # Draw label and confidence
        label_text = f"{event.intent} ({event.conf:.2f})"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # Background for text
        cv2.rectangle(overlay_frame,
                      (x1, y1 - text_size[1] - 10),
                      (x1 + text_size[0], y1),
                      color, -1)

        # Text
        cv2.putText(overlay_frame, label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # Draw bearing indicator
        if event.bearing_deg is not None:
            center_x = (x1 + x2) // 2
            arrow_y = y2 + 20
            arrow_length = 30

            # Calculate arrow end point
            bearing_rad = np.radians(event.bearing_deg)
            end_x = int(center_x + arrow_length * np.sin(bearing_rad))
            end_y = arrow_y

            cv2.arrowedLine(overlay_frame,
                           (center_x, arrow_y),
                           (end_x, end_y),
                           color, 2)

            # Bearing text
            bearing_text = f"{event.bearing_deg:.1f}°"
            cv2.putText(overlay_frame, bearing_text,
                       (center_x - 20, arrow_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       color, 1)

    return overlay_frame


def process_video(video_path: str, output_json: str, output_video: str = None):
    """
    Process video through perception pipeline.

    Args:
        video_path: Input video file path
        output_json: Output JSON file path for events
        output_video: Optional output video path for overlay
    """
    if not DEPENDENCIES_AVAILABLE:
        print("Cannot process video - missing dependencies")
        return

    # Initialize pipeline
    config = PerceptionConfig()
    pipeline = PerceptionPipeline(config)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing video: {video_path}")
    print(f"Properties: {width}x{height}, {fps:.1f} FPS, {frame_count} frames")

    # Setup video writer if overlay requested
    video_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Process frames
    all_events = []
    frame_num = 0
    start_time = time.time()
    last_stats_time = start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        frame_start = time.time()
        events = pipeline.process_frame(frame, fps)
        frame_end = time.time()

        # Store events with frame number
        for event in events:
            event_dict = event_to_dict(event)
            event_dict["frame_num"] = frame_num
            all_events.append(event_dict)

        # Draw overlay if requested
        if video_writer is not None:
            overlay_frame = draw_event_overlay(frame, events)
            video_writer.write(overlay_frame)

        frame_num += 1

        # Print stats every 5 seconds
        current_time = time.time()
        if current_time - last_stats_time >= 5.0:
            elapsed = current_time - start_time
            progress = frame_num / frame_count * 100
            pipeline_stats = pipeline.get_stats()

            print(f"Progress: {progress:.1f}% | "
                  f"Frame: {frame_num}/{frame_count} | "
                  f"Time: {elapsed:.1f}s | "
                  f"FPS: {frame_num/elapsed:.1f} | "
                  f"Events: {len(all_events)} | "
                  f"Raw dets: {pipeline_stats['raw_detections']} | "
                  f"OCR runs: {pipeline_stats['ocr_runs']}")

            last_stats_time = current_time

    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()

    # Final stats
    total_time = time.time() - start_time
    avg_fps = frame_num / total_time
    final_stats = pipeline.get_stats()

    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Total events promoted: {len(all_events)}")
    print(f"Total raw detections: {final_stats['raw_detections']}")
    print(f"OCR runs: {final_stats['ocr_runs']}")

    # Save events to JSON
    with open(output_json, 'w') as f:
        json.dump({
            "metadata": {
                "video_path": video_path,
                "total_frames": frame_num,
                "fps": fps,
                "processing_time": total_time,
                "avg_processing_fps": avg_fps,
                "pipeline_stats": final_stats
            },
            "events": all_events
        }, f, indent=2)

    print(f"Events saved to: {output_json}")
    if output_video:
        print(f"Overlay video saved to: {output_video}")


def main():
    """Main entry point for replay tool."""
    parser = argparse.ArgumentParser(
        description="Process video through GuidedSight perception pipeline"
    )
    parser.add_argument("--video", required=True,
                       help="Input video file path")
    parser.add_argument("--out", required=True,
                       help="Output JSON file for events")
    parser.add_argument("--overlay",
                       help="Optional output video file with overlays")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    # Process video
    try:
        process_video(args.video, args.out, args.overlay)
        return 0
    except Exception as e:
        print(f"Error processing video: {e}")
        return 1


if __name__ == "__main__":
    exit(main())