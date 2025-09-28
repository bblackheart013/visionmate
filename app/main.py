#!/usr/bin/env python3
"""
VisionMate - Blind Navigation Assistant
Main orchestrator application that integrates perception and guidance pipelines.

This module orchestrates the full vision processing loop:
- Video/webcam input → Perception → Guidance → Visual output + Audio
- CPU/QNN execution provider toggle for Snapdragon optimization
- Phone controller integration via WebSocket
- Performance monitoring and HUD overlay

Author: Vaibhav Chandgir (Integration & Snapdragon lead)
"""

import argparse
import cv2
import time
import threading
import logging
from typing import Optional, List, Dict, Any
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from perception import PerceptionPipeline
    PERCEPTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: {e}. Running with mock perception for testing.")
    PERCEPTION_AVAILABLE = False

from app.perf import Timer, perf_line
from app.ws_server import start_websocket_server_thread, get_controller_state
from app.phone_stream_server import start_phone_stream_server_thread, get_phone_stream_receiver
from app.route_client import get_route, RouteClient

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Person 2's guidance system
try:
    from guidance import GuidancePolicy
    from guidance.tts import TTS
    GUIDANCE_AVAILABLE = True
    TTS_AVAILABLE = True
    logger.info("Using Person 2's guidance and TTS systems")
except ImportError as e:
    print(f"Warning: {e}. Running with mock guidance for testing.")
    GUIDANCE_AVAILABLE = False
    TTS_AVAILABLE = False

class MockPerceptionPipeline:
    """Mock perception pipeline for testing when Person 1's module isn't available."""
    
    def __init__(self):
        self.frame_count = 0
        # Define test scenarios for different signs and obstacles
        self.test_scenarios = [
            {"type": "sign", "label": "STOP", "intent": "STOP", "bearing": 0, "distance": 8.0},
            {"type": "sign", "label": "EXIT RIGHT", "intent": "EXIT_RIGHT", "bearing": 20, "distance": 12.0},
            {"type": "sign", "label": "EXIT LEFT", "intent": "EXIT_LEFT", "bearing": -25, "distance": 10.5},
            {"type": "obstacle", "label": "person", "intent": "OBSTACLE_PERSON", "bearing": 5, "distance": 1.5},
            {"type": "obstacle", "label": "person", "intent": "OBSTACLE_PERSON", "bearing": -10, "distance": 3.2},
            {"type": "obstacle", "label": "person", "intent": "OBSTACLE_PERSON", "bearing": 0, "distance": 0.8},  # Very close
            {"type": "obstacle", "label": "car", "intent": "OBSTACLE_CAR", "bearing": 15, "distance": 6.0},
            {"type": "obstacle", "label": "pole", "intent": "OBSTACLE_POLE", "bearing": 2, "distance": 1.8},
        ]
        self.scenario_index = 0
        
    def process_frame(self, frame_bgr, fps=30) -> List[Dict[str, Any]]:
        """Generate mock events for testing with varied scenarios."""
        self.frame_count += 1
        
        # Generate events every 60 frames (2 seconds at 30fps)
        events = []
        if self.frame_count % 60 == 0:
            scenario = self.test_scenarios[self.scenario_index % len(self.test_scenarios)]
            self.scenario_index += 1
            
            if scenario["type"] == "sign":
                events.append({
                    "schema": "v1",
                    "ts": time.time(),
                    "type": "sign",
                    "label": scenario["label"],
                    "intent": scenario["intent"],
                    "conf": 0.9,
                    "bbox": [200, 50, 400, 120],
                    "bearing_deg": scenario["bearing"],
                    "dist_m": scenario["distance"],
                    "sources": ["ocr"]
                })
            else:  # obstacle
                events.append({
                    "schema": "v1",
                    "ts": time.time(),
                    "type": "obstacle",
                    "label": scenario["label"],
                    "intent": scenario["intent"],
                    "conf": 0.85,
                    "bbox": [100, 100, 250, 350],
                    "bearing_deg": scenario["bearing"],
                    "dist_m": scenario["distance"],
                    "sources": ["yolo"]
                })
            
        return events

class MockGuidanceEngine:
    """Mock guidance engine that integrates Person 2's guidance system when available."""
    
    def __init__(self):
        self.last_utterance = ""
        
        # Use Person 2's guidance system if available
        if GUIDANCE_AVAILABLE:
            self.guidance_policy = GuidancePolicy()
            logger.info("Initialized with Person 2's guidance system")
        else:
            self.guidance_policy = None
            logger.info("Using fallback mock guidance")
        
    def step(self, frame_bgr, events, fps=None, mode="CPU", latency_ms=None):
        """Generate guidance using Person 2's system or fallback mock."""
        utterance = None
        
        if self.guidance_policy:
            # Use Person 2's smart guidance system
            guidance_events = self._convert_events_for_guidance(events)
            utterance = self.guidance_policy.choose(guidance_events)
        else:
            # Fallback mock guidance
            if events:
                event = events[0]  # Use first event
                if event["type"] == "obstacle":
                    utterance = "Obstacle detected ahead, please slow down"
                elif event["type"] == "sign":
                    utterance = f"Sign detected: {event['label']}"
        
        if utterance:
            self.last_utterance = utterance
        
        # Add HUD overlay
        frame_with_hud = frame_bgr.copy()
        height, width = frame_with_hud.shape[:2]
        
        # Mode indicator
        mode_color = (0, 255, 0) if mode == "QNN" else (0, 255, 255)
        cv2.putText(frame_with_hud, f"Mode: {mode}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Performance info
        perf_text = perf_line()
        cv2.putText(frame_with_hud, perf_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Event overlay
        for i, event in enumerate(events):
            bbox = event.get("bbox", [0, 0, 100, 100])
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_with_hud, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_with_hud, f"{event['label']} ({event['conf']:.2f})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Controller state
        controller_state = get_controller_state()
        if controller_state.get("goal_str"):
            cv2.putText(frame_with_hud, f"Goal: {controller_state['goal_str']}", 
                       (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if controller_state.get("is_muted"):
            cv2.putText(frame_with_hud, "MUTED", (width-100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        self.last_utterance = utterance or ""
        return frame_with_hud, utterance
    
    def _convert_events_for_guidance(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert perception events to Person 2's guidance event format."""
        guidance_events = []
        
        for event in events:
            event_type = event.get('type')
            
            if event_type == 'obstacle':
                guidance_event = {
                    'intent': 'OBSTACLE_PERSON',  # Default to person for now
                    'conf': event.get('conf', 0.8),
                    'bearing_deg': event.get('bearing', 0.0),
                    'dist_m': event.get('distance', 2.0)
                }
                guidance_events.append(guidance_event)
                
            elif event_type == 'sign':
                sign_label = event.get('label', '').upper()
                if 'STOP' in sign_label:
                    guidance_event = {
                        'intent': 'STOP',
                        'conf': event.get('conf', 0.9),
                        'bearing_deg': event.get('bearing', 0.0)
                    }
                    guidance_events.append(guidance_event)
                elif 'EXIT' in sign_label:
                    direction = 'RIGHT' if 'RIGHT' in sign_label else 'LEFT'
                    guidance_event = {
                        'intent': f'EXIT_{direction}',
                        'conf': event.get('conf', 0.8),
                        'bearing_deg': event.get('bearing', 0.0)
                    }
                    guidance_events.append(guidance_event)
        
        return guidance_events

def ep_providers(ep_flag: str) -> List:
    """
    Get ONNX Runtime execution providers based on flag.
    
    Args:
        ep_flag: 'cpu' or 'qnn'
        
    Returns:
        List of execution providers for ONNX Runtime
    """
    if ep_flag == "qnn":
        try:
            # Try to import QNN execution provider
            import onnxruntime as ort
            providers = [
                ("QNNExecutionProvider", {
                    "backend_path": "",  # Default HTP backend
                    "qnn_htp_performance_mode": "burst"
                }),
                "CPUExecutionProvider"
            ]
            logger.info("QNN execution provider configured")
            return providers
        except Exception as e:
            logger.warning(f"QNN execution provider not available: {e}. Falling back to CPU.")
            return ["CPUExecutionProvider"]
    else:
        return ["CPUExecutionProvider"]

def create_video_capture(video_path: Optional[str] = None, camera_idx: int = 0):
    """Create video capture from file or camera."""
    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        logger.info(f"Opened video file: {video_path}")
    else:
        cap = cv2.VideoCapture(camera_idx)
        logger.info(f"Opened camera: {camera_idx}")
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")
    
    return cap

def main():
    """Main application loop."""
    parser = argparse.ArgumentParser(description="VisionMate - Blind Navigation Assistant")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--phone-camera", action="store_true", 
                       help="Use phone camera stream instead of local camera")
    parser.add_argument("--ep", choices=["cpu", "qnn"], default="cpu", 
                       help="Execution provider: cpu or qnn")
    parser.add_argument("--controller", choices=["on", "off"], default="off",
                       help="Enable phone controller (on/off)")
    parser.add_argument("--route", type=str, help="Route service URL (optional)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.video and args.camera is None and not args.phone_camera:
        logger.error("Must specify either --video, --camera, or --phone-camera")
        return 1
    
    logger.info(f"Starting VisionMate with EP: {args.ep}, Controller: {args.controller}")
    
    # Initialize execution providers
    providers = ep_providers(args.ep)
    actual_mode = "QNN" if args.ep == "qnn" and len(providers) > 1 else "CPU"
    
    # Initialize perception and guidance
    if PERCEPTION_AVAILABLE:
        perception = PerceptionPipeline()
        logger.info("Using real PerceptionPipeline")
    else:
        perception = MockPerceptionPipeline()
        logger.info("Using mock PerceptionPipeline")
    
    # Always use MockGuidanceEngine which now integrates Person 2's system
    guidance = MockGuidanceEngine()
    logger.info("Using MockGuidanceEngine with Person 2 integration")
    
    # Initialize Person 2's TTS system
    tts = None
    if TTS_AVAILABLE:
        try:
            tts = TTS()
            logger.info("Initialized Person 2's TTS system")
        except Exception as e:
            logger.warning(f"Failed to initialize TTS: {e}")
            tts = None
    
    # Initialize route client if specified
    route_client = None
    if args.route:
        route_client = RouteClient(args.route)
    
    # Start WebSocket server if controller enabled
    if args.controller == "on":
        ws_thread = start_websocket_server_thread()
        logger.info("WebSocket server started on port 8765")
    
    # Start phone camera stream server if using phone camera
    if args.phone_camera:
        phone_stream_thread = start_phone_stream_server_thread()
        logger.info("Phone camera stream server started on port 8766")
    
    try:
        # Create video capture or get phone stream
        if args.phone_camera:
            cap = None  # We'll get frames from phone stream
            fps = 30  # Phone stream FPS
            width, height = 640, 480  # Phone stream resolution
            logger.info(f"Using phone camera stream: {width}x{height} @ {fps} FPS")
            phone_stream_receiver = get_phone_stream_receiver()
        else:
            cap = create_video_capture(args.video, args.camera)
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Video: {width}x{height} @ {fps} FPS")
        
        # Main processing loop
        frame_count = 0
        start_time = time.time()
        
        while True:
            frame_start = time.time()
            
            # Grab frame
            with Timer("grab"):
                if args.phone_camera:
                    # Get frame from phone stream
                    frame = phone_stream_receiver.get_current_frame()
                    if frame is None:
                        # No frame available yet, wait a bit
                        if frame_count % 100 == 0:  # Log every 100 attempts
                            logger.info("⏳ Waiting for phone camera frames...")
                        time.sleep(0.033)  # ~30 FPS
                        continue
                    else:
                        if frame_count % 60 == 0:  # Log every 2 seconds at 30fps
                            logger.info(f"📱 Received phone frame: {frame.shape}")
                    ret = True
                else:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("End of video stream")
                        break
            
            frame_count += 1
            
            # Process with perception
            with Timer("perception"):
                events = perception.process_frame(frame, fps)
            
            # Get controller state
            controller_state = get_controller_state()
            
            # Check for stop command
            if controller_state.get("should_stop"):
                logger.info("Stop command received from controller")
                break
            
            # Process with guidance
            with Timer("guidance"):
                frame_with_hud, utterance = guidance.step(
                    frame, events, fps, mode=actual_mode
                )
            
            # Handle TTS (if not muted and utterance available)
            if utterance and not controller_state.get("is_muted"):
                if tts:
                    # Use Person 2's TTS system
                    try:
                        tts.say(utterance)
                        logger.info(f"TTS: {utterance}")
                    except Exception as e:
                        logger.error(f"TTS error: {e}")
                        logger.info(f"TTS (fallback): {utterance}")
                else:
                    # Fallback to logging
                    logger.info(f"TTS: {utterance}")
            
            # Handle repeat command
            if controller_state.get("should_repeat") and hasattr(guidance, 'last_utterance'):
                if guidance.last_utterance:
                    if tts:
                        try:
                            tts.say(guidance.last_utterance)
                            logger.info(f"TTS Repeat: {guidance.last_utterance}")
                        except Exception as e:
                            logger.error(f"TTS repeat error: {e}")
                            logger.info(f"TTS Repeat (fallback): {guidance.last_utterance}")
                    else:
                        logger.info(f"TTS Repeat: {guidance.last_utterance}")
                controller_state["should_repeat"] = False  # Reset flag
            
            # Render frame
            with Timer("render"):
                cv2.imshow("VisionMate", frame_with_hud)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    logger.info("ESC pressed, exiting")
                    break
                elif key == ord('m'):  # Manual mute toggle
                    controller_state["is_muted"] = not controller_state.get("is_muted")
                    logger.info(f"Manual mute toggle: {controller_state['is_muted']}")
            
            # Performance logging every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                logger.info(f"Frame {frame_count}: {avg_fps:.1f} FPS, {perf_line()}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        return 1
    finally:
        if not args.phone_camera and cap:
            cap.release()
        cv2.destroyAllWindows()
        
        # Log final statistics
        total_time = time.time() - start_time
        final_fps = frame_count / total_time if total_time > 0 else 0
        logger.info(f"Final stats: {frame_count} frames in {total_time:.1f}s ({final_fps:.1f} FPS)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
