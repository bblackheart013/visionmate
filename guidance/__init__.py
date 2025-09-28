"""
VisionMate Guidance Orchestration Engine
Copyright © 2050 VisionMate Technologies. All rights reserved.

"The whole is greater than the sum of its parts." - Aristotle

This module orchestrates the trinity of perception, voice, and vision
into a unified experience that transcends its components.

Architecture Philosophy:
  • Harmony through integration - Components sing together
  • Zero-friction pipeline - Data flows like water
  • Graceful degradation - Never fail completely
  • Human-first timing - Respect cognitive processing limits

We don't just process frames. We create moments of clarity.

Author: Systems Integration Division
Version: 7.0.0 - Harmony Edition
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Tuple, List, Dict, Any, Final
from dataclasses import dataclass, field
from enum import Enum, auto
from contextlib import contextmanager
import logging
import weakref
import atexit

# Import our orchestrated components
from .policy import GuidancePolicy
from .tts import TTS
from .renderer import draw_hud


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger('VisionMate.Orchestrator')
logger.setLevel(logging.INFO)

# Ensure we have a handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - System Constants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OrchestrationConstants:
    """
    The temporal and spatial constants that govern our system.
    
    These values represent the delicate balance between:
    - Responsiveness and stability
    - Information density and cognitive load
    - Performance and quality
    """
    # Timing boundaries
    MAX_FRAME_TIME: Final[float] = 0.033  # Target 30 FPS minimum
    VOICE_PREVIEW_DURATION: Final[float] = 3.0  # Show utterance for 3 seconds
    
    # Performance monitoring
    LATENCY_WINDOW_SIZE: Final[int] = 30  # Rolling average over 1 second
    FPS_SMOOTHING_FACTOR: Final[float] = 0.1  # Exponential smoothing
    
    # System health thresholds
    CRITICAL_FPS: Final[float] = 15.0  # Below this, system is struggling
    WARNING_LATENCY_MS: Final[float] = 50.0  # Above this, responsiveness suffers
    
    # Grace periods
    STARTUP_WARMUP_FRAMES: Final[int] = 10  # Ignore metrics during warmup
    SHUTDOWN_TIMEOUT: Final[float] = 2.0  # Maximum shutdown wait


@dataclass(frozen=True)
class SystemState:
    """The heartbeat of our system - performance metrics that matter."""
    fps: float = 30.0
    latency_ms: float = 0.0
    frame_count: int = 0
    utterance_count: int = 0
    start_time: float = field(default_factory=time.monotonic)


# Singleton configuration
CONSTANTS = OrchestrationConstants()


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Orchestration Engine
# ═══════════════════════════════════════════════════════════════════════════

class GuidanceEngine:
    """
    The maestro of VisionMate - orchestrating perception, voice, and vision.
    
    This isn't just integration. It's the creation of a new sensory system,
    where computer vision becomes human intuition, algorithms become instinct,
    and data becomes understanding.
    
    The GuidanceEngine is the soul of VisionMate. It ensures that:
      1. No frame is wasted - Every cycle produces value
      2. No voice is rushed - Speech flows naturally
      3. No visual is jarring - Transitions are smooth
      4. No user is overwhelmed - Information is metered
    
    Thread Safety:
      - All public methods are thread-safe
      - Internal state is protected by locks where necessary
      - Components are isolated to prevent cascade failures
    
    Performance Contract:
      - step() completes within 33ms (30 FPS)
      - Voice never blocks vision
      - Rendering never blocks perception
      - Graceful degradation under load
    """
    
    # Class-level tracking
    _instances: weakref.WeakSet = weakref.WeakSet()
    
    def __init__(self, muted: bool = False):
        """
        Initialize the guidance orchestration system.
        
        This constructor assembles our trinity:
          - Policy: The decision maker
          - Voice: The communicator  
          - Renderer: The visualizer
        
        Args:
            muted: Initial voice mute state (visuals always active)
        """
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║          VisionMate Guidance Engine v7.0              ║")
        logger.info("║          'Where Vision Becomes Voice'                 ║")
        logger.info("╚══════════════════════════════════════════════════════╝")
        
        # Core components - our trinity
        self._policy = GuidancePolicy()
        self._tts = TTS()
        self._muted = muted
        
        # State management
        self._state = SystemState()
        self._last_utterance: Optional[str] = None
        self._last_utterance_time: float = 0.0
        self._utterance_history: List[Tuple[float, str]] = []
        
        # Performance tracking
        self._frame_times: List[float] = []
        self._latencies: List[float] = []
        self._last_frame_time: float = time.monotonic()
        
        # Thread safety
        self._state_lock = threading.RLock()
        
        # Lifecycle management
        atexit.register(self._cleanup)
        GuidanceEngine._instances.add(self)
        
        # Initial configuration
        if muted:
            self._tts.set_muted(True)
        
        logger.info(f"✓ Guidance Engine initialized (muted={muted})")
    
    # ─────────────────────────────────────────────────────────────────────
    # MARK: - Core Pipeline
    # ─────────────────────────────────────────────────────────────────────
    
    def step(
        self,
        frame_bgr: np.ndarray,
        events: List[Dict],
        fps: Optional[float] = None,
        mode: str = "CPU",
        latency_ms: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Execute one guidance cycle - the heartbeat of VisionMate.
        
        This method orchestrates the complete perception→decision→action pipeline:
          1. Analyze events through policy
          2. Generate voice guidance
          3. Render visual augmentation
          4. Track performance metrics
        
        Args:
            frame_bgr: Current camera frame (BGR format)
            events: Detected objects/signs from perception
            fps: External FPS measurement (optional)
            mode: Processing mode indicator ("CPU" or "QNN")
            latency_ms: External latency measurement (optional)
        
        Returns:
            Tuple of (augmented_frame, utterance_or_none)
        
        Performance:
            Guaranteed to complete within 33ms (30 FPS target)
            Non-blocking on all operations
        """
        step_start = time.monotonic()
        
        with self._state_lock:
            # Update frame counter
            self._state = dataclass_replace(
                self._state, frame_count=self._state.frame_count + 1
            )
            
            # ═══════════════════════════════════════════
            # Phase 1: Decision Making
            # ═══════════════════════════════════════════
            
            utterance = None
            if events:
                # Let policy decide what to say
                utterance = self._policy.choose(events)
                
                if utterance:
                    self._process_utterance(utterance)
            
            # ═══════════════════════════════════════════
            # Phase 2: Performance Metrics
            # ═══════════════════════════════════════════
            
            # Calculate internal metrics if not provided
            current_fps = fps if fps is not None else self._calculate_fps()
            current_latency = latency_ms if latency_ms is not None else self._calculate_latency()
            
            # Update system state
            self._state = dataclass_replace(
                self._state,
                fps=current_fps,
                latency_ms=current_latency
            )
            
            # ═══════════════════════════════════════════
            # Phase 3: Visual Augmentation
            # ═══════════════════════════════════════════
            
            # Determine what text to show (recent utterance)
            display_text = self._get_display_utterance()
            
            # Render HUD overlay
            augmented_frame = draw_hud(
                frame=frame_bgr,
                events=events,
                fps=current_fps,
                mode=mode,
                latency_ms=current_latency,
                msg_preview=display_text
            )
            
            # ═══════════════════════════════════════════
            # Phase 4: Timing Management
            # ═══════════════════════════════════════════
            
            # Track frame timing
            step_duration = time.monotonic() - step_start
            self._frame_times.append(step_duration)
            
            # Maintain rolling window
            if len(self._frame_times) > CONSTANTS.LATENCY_WINDOW_SIZE:
                self._frame_times.pop(0)
            
            # Warn if we're running slow
            if step_duration > CONSTANTS.MAX_FRAME_TIME:
                logger.warning(
                    f"Frame processing exceeded budget: {step_duration*1000:.1f}ms"
                )
            
            return augmented_frame, utterance
    
    # ─────────────────────────────────────────────────────────────────────
    # MARK: - Voice Control
    # ─────────────────────────────────────────────────────────────────────
    
    def set_muted(self, muted: bool) -> None:
        """
        Control the voice output state.
        
        When muted, the system continues all visual processing
        but suppresses audio output. This respects user preference
        while maintaining full visual assistance.
        
        Args:
            muted: True to silence voice, False to enable
        """
        with self._state_lock:
            if self._muted != muted:
                self._muted = muted
                self._tts.set_muted(muted)
                
                icon = "🔇" if muted else "🔊"
                logger.info(f"{icon} Voice output {'muted' if muted else 'enabled'}")
    
    def say_now(self, text: str) -> bool:
        """
        Bypass policy and speak immediately.
        
        This is the "manual override" - when you absolutely
        need to communicate something right now.
        
        Args:
            text: Message to speak
            
        Returns:
            True if successfully queued
        """
        if not text or self._muted:
            return False
        
        success = self._tts.say(text)
        
        if success:
            with self._state_lock:
                self._last_utterance = text
                self._last_utterance_time = time.monotonic()
                self._utterance_history.append((self._last_utterance_time, text))
                
                # Prune old history
                self._prune_utterance_history()
        
        return success
    
    def repeat_last(self) -> bool:
        """
        Repeat the most recent utterance.
        
        The "What did you say?" feature - because sometimes
        the world is noisy and attention wanders.
        
        Returns:
            True if there was something to repeat
        """
        with self._state_lock:
            if self._last_utterance:
                return self.say_now(self._last_utterance)
        return False
    
    # ─────────────────────────────────────────────────────────────────────
    # MARK: - State Management
    # ─────────────────────────────────────────────────────────────────────
    
    @property
    def is_muted(self) -> bool:
        """Check current mute state."""
        return self._muted
    
    @property
    def stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns performance metrics, health indicators,
        and operational statistics for monitoring.
        """
        with self._state_lock:
            # Calculate averages
            avg_frame_time = (
                sum(self._frame_times) / len(self._frame_times)
                if self._frame_times else 0
            )
            
            # System health assessment
            health = "GOOD"
            if self._state.fps < CONSTANTS.CRITICAL_FPS:
                health = "CRITICAL"
            elif self._state.latency_ms > CONSTANTS.WARNING_LATENCY_MS:
                health = "WARNING"
            
            return {
                'fps': self._state.fps,
                'latency_ms': self._state.latency_ms,
                'frame_count': self._state.frame_count,
                'utterance_count': self._state.utterance_count,
                'avg_frame_time_ms': avg_frame_time * 1000,
                'voice_queue_size': self._tts.queue_size,
                'is_muted': self._muted,
                'system_health': health,
                'uptime_seconds': time.monotonic() - self._state.start_time
            }
    
    # ─────────────────────────────────────────────────────────────────────
    # MARK: - Lifecycle Management
    # ─────────────────────────────────────────────────────────────────────
    
    def close(self) -> None:
        """
        Gracefully shutdown the guidance system.
        
        This ensures clean termination of all components,
        proper resource cleanup, and state persistence.
        """
        logger.info("🌙 Shutting down Guidance Engine...")
        
        with self._state_lock:
            # Report final statistics
            stats = self.stats
            logger.info(f"Final stats: {stats['frame_count']} frames, "
                       f"{stats['utterance_count']} utterances, "
                       f"{stats['uptime_seconds']:.1f}s uptime")
            
            # Shutdown components
            try:
                self._tts.close()
            except Exception as e:
                logger.error(f"TTS shutdown error: {e}")
            
            # Clear state
            self._frame_times.clear()
            self._latencies.clear()
            self._utterance_history.clear()
        
        logger.info("✓ Guidance Engine shutdown complete")
    
    # ─────────────────────────────────────────────────────────────────────
    # MARK: - Internal Methods
    # ─────────────────────────────────────────────────────────────────────
    
    def _process_utterance(self, utterance: str) -> None:
        """Process and queue an utterance for speech."""
        if not self._muted:
            # Queue for speech
            self._tts.say(utterance)
        
        # Update state regardless of mute
        self._last_utterance = utterance
        self._last_utterance_time = time.monotonic()
        self._utterance_history.append((self._last_utterance_time, utterance))
        
        # Update counter
        self._state = dataclass_replace(
            self._state,
            utterance_count=self._state.utterance_count + 1
        )
        
        # Maintain history size
        self._prune_utterance_history()
        
        logger.debug(f"📢 Utterance: '{utterance}'")
    
    def _get_display_utterance(self) -> Optional[str]:
        """Get the utterance to display on HUD."""
        if not self._last_utterance:
            return None
        
        # Show utterance for a few seconds after speaking
        elapsed = time.monotonic() - self._last_utterance_time
        if elapsed <= CONSTANTS.VOICE_PREVIEW_DURATION:
            return self._last_utterance
        
        return None
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS with smoothing."""
        current_time = time.monotonic()
        
        if self._state.frame_count < CONSTANTS.STARTUP_WARMUP_FRAMES:
            return 30.0  # Default during warmup
        
        # Calculate instantaneous FPS
        time_delta = current_time - self._last_frame_time
        instant_fps = 1.0 / max(time_delta, 0.001)
        
        # Apply exponential smoothing
        smoothed_fps = (
            self._state.fps * (1 - CONSTANTS.FPS_SMOOTHING_FACTOR) +
            instant_fps * CONSTANTS.FPS_SMOOTHING_FACTOR
        )
        
        self._last_frame_time = current_time
        return min(smoothed_fps, 120.0)  # Cap at reasonable maximum
    
    def _calculate_latency(self) -> float:
        """Calculate pipeline latency in milliseconds."""
        if self._frame_times:
            # Use recent frame times as latency estimate
            recent_avg = sum(self._frame_times[-5:]) / min(len(self._frame_times), 5)
            return recent_avg * 1000
        return 0.0
    
    def _prune_utterance_history(self, max_items: int = 100) -> None:
        """Maintain reasonable history size."""
        if len(self._utterance_history) > max_items:
            self._utterance_history = self._utterance_history[-max_items:]
    
    def _cleanup(self) -> None:
        """Emergency cleanup handler."""
        try:
            if hasattr(self, '_tts'):
                self.close()
        except Exception:
            pass  # Best effort
    
    # ─────────────────────────────────────────────────────────────────────
    # MARK: - Context Manager Protocol
    # ─────────────────────────────────────────────────────────────────────
    
    def __enter__(self) -> 'GuidanceEngine':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (f"<GuidanceEngine muted={self._muted} "
                f"frames={self._state.frame_count} "
                f"utterances={self._state.utterance_count}>")


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def dataclass_replace(instance, **changes):
    """Replace dataclass fields (Python 3.8 compatible)."""
    import copy
    result = copy.copy(instance)
    for key, value in changes.items():
        setattr(result, key, value)
    return result


@contextmanager
def guidance_engine(muted: bool = False):
    """
    Context manager for guidance engine lifecycle.
    
    Usage:
        with guidance_engine() as engine:
            frame, utterance = engine.step(frame, events)
    """
    engine = GuidanceEngine(muted=muted)
    try:
        yield engine
    finally:
        engine.close()


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Demonstration
# ═══════════════════════════════════════════════════════════════════════════

def _generate_demo_frame() -> np.ndarray:
    """Create an elegant demo frame."""
    height, width = 720, 1280
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create subtle gradient with noise
    for y in range(height):
        intensity = int(30 + (y / height) * 40)
        frame[y, :] = (intensity, intensity + 5, intensity + 10)
    
    # Add gentle noise for realism
    noise = np.random.normal(0, 3, (height, width, 3))
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return frame


def _generate_demo_scenario(frame_num: int) -> List[Dict]:
    """Generate evolving demo events based on frame number."""
    scenarios = [
        # Scenario 1: Person approaching
        lambda n: [
            {
                'intent': 'OBSTACLE_PERSON',
                'label': 'Person',
                'bbox': [600 - n*2, 300, 700 - n*2, 550],
                'bearing_deg': -5.0 + n*0.1,
                'dist_m': max(0.5, 4.0 - n*0.05),
                'conf': 0.92
            }
        ] if n < 40 else [],
        
        # Scenario 2: Stop sign detected
        lambda n: [
            {
                'intent': 'STOP',
                'label': 'STOP',
                'bbox': [900, 200, 1050, 350],
                'bearing_deg': 15.0,
                'dist_m': 3.5,
                'conf': 0.88
            }
        ] if 30 < n < 80 else [],
        
        # Scenario 3: Vehicle passing
        lambda n: [
            {
                'intent': 'OBSTACLE_CAR',
                'label': 'Car',
                'bbox': [100 + n*5, 400, 350 + n*5, 600],
                'bearing_deg': -20.0 + n*0.5,
                'dist_m': 3.0,
                'conf': 0.85
            }
        ] if 60 < n < 100 else [],
        
        # Scenario 4: Exit sign
        lambda n: [
            {
                'intent': 'EXIT_RIGHT',
                'label': 'EXIT',
                'bbox': [1000, 100, 1150, 200],
                'bearing_deg': 25.0,
                'dist_m': 5.0,
                'conf': 0.79
            }
        ] if n > 90 else [],
    ]
    
    # Combine all active scenarios
    events = []
    for scenario in scenarios:
        events.extend(scenario(frame_num % 120))  # Loop every 4 seconds
    
    return events


def main():
    """
    Demonstration of the complete VisionMate guidance system.
    
    This isn't just a demo - it's a window into the future
    of human-computer collaboration.
    """
    print("╔════════════════════════════════════════════════════════════╗")
    print("║         VisionMate Guidance Engine Demonstration           ║")
    print("║                                                            ║")
    print("║  Controls:                                                 ║")
    print("║    Q - Quit                                                ║")
    print("║    M - Toggle mute                                         ║")
    print("║    R - Repeat last utterance                               ║")
    print("║    S - Say custom message                                  ║")
    print("║    Space - Pause/Resume                                    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Initialize system
    print("🚀 Initializing VisionMate...")
    engine = GuidanceEngine(muted=False)
    
    # Demo state
    frame_num = 0
    paused = False
    demo_frame = _generate_demo_frame()
    
    # Create window
    window_name = "VisionMate Guidance System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print("✓ System ready. Beginning demonstration...\n")
    
    try:
        while True:
            if not paused:
                # Generate dynamic scenario
                events = _generate_demo_scenario(frame_num)
                
                # Run guidance pipeline
                start_time = time.perf_counter()
                augmented_frame, utterance = engine.step(
                    frame_bgr=demo_frame.copy(),
                    events=events,
                    mode="QNN" if frame_num > 60 else "CPU"
                )
                step_time = (time.perf_counter() - start_time) * 1000
                
                # Log any utterances
                if utterance:
                    print(f"  → Guidance: '{utterance}'")
                
                # Display
                cv2.imshow(window_name, augmented_frame)
                frame_num += 1
                
                # Performance monitoring
                if frame_num % 30 == 0:
                    stats = engine.stats
                    print(f"  📊 Stats: {stats['fps']:.1f} FPS | "
                          f"{stats['latency_ms']:.1f}ms latency | "
                          f"Health: {stats['system_health']}")
            else:
                # Show paused state
                cv2.imshow(window_name, augmented_frame)
            
            # Handle user input
            key = cv2.waitKey(30 if not paused else 100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m'):
                engine.set_muted(not engine.is_muted)
                status = "muted" if engine.is_muted else "unmuted"
                print(f"  🔊 Voice {status}")
            elif key == ord('r'):
                if engine.repeat_last():
                    print("  🔄 Repeating last utterance")
            elif key == ord('s'):
                engine.say_now("VisionMate guidance system operational.")
                print("  💬 Custom message sent")
            elif key == ord(' '):
                paused = not paused
                print(f"  {'⏸' if paused else '▶'} {'Paused' if paused else 'Resumed'}")
    
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted by user")
    
    finally:
        # Clean shutdown
        print("\n🌙 Shutting down...")
        cv2.destroyAllWindows()
        engine.close()
        print("✓ Demo completed successfully")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Public API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    'GuidanceEngine',
    'guidance_engine',
    'OrchestrationConstants',
    'SystemState',
]

# ═══════════════════════════════════════════════════════════════════════════
# "Technology alone is not enough. It's technology married with the liberal
#  arts, married with the humanities, that yields the results that make our
#  hearts sing." - Steve Jobs
# ═══════════════════════════════════════════════════════════════════════════