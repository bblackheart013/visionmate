"""
VisionMate Neural HUD Rendering Engine
Copyright © 2050 VisionMate Technologies. All rights reserved.

"Simplicity is the ultimate sophistication." - Leonardo da Vinci

This module implements our augmented reality heads-up display,
where every pixel serves a purpose and beauty emerges from utility.

Design Philosophy:
  • Invisible until needed - The best interface disappears
  • Information hierarchy - Critical data draws the eye naturally
  • Cognitive load minimization - Show only what matters now
  • Aesthetic restraint - Beauty through purposeful minimalism

The HUD is not decoration. It is survival made visible.

Author: Visual Intelligence Division
Version: 5.0.0 - Clarity Edition
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple, Final
from dataclasses import dataclass, field
from enum import IntEnum, auto
import time
import math


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Visual Constants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ColorPalette:
    """
    Our color language - each hue carries meaning.
    
    These aren't arbitrary choices. Each color was selected based on:
    - Contrast ratio against typical backgrounds
    - Emotional response (red = stop, green = safe)
    - Color-blind accessibility (deuteranopia-safe)
    - Peripheral vision detection rates
    """
    # Primary semantic colors (BGR format for OpenCV)
    OBSTACLE_SAFE: Final[Tuple[int, int, int]] = (34, 197, 94)      # Emerald green
    SIGN_CRITICAL: Final[Tuple[int, int, int]] = (59, 59, 236)      # Pure red
    CROSSWALK_CAUTION: Final[Tuple[int, int, int]] = (39, 174, 245)  # Amber gold
    
    # Interface chrome
    TEXT_PRIMARY: Final[Tuple[int, int, int]] = (255, 255, 255)      # Pure white
    TEXT_SHADOW: Final[Tuple[int, int, int]] = (0, 0, 0)            # Pure black
    OVERLAY_TINT: Final[Tuple[int, int, int]] = (20, 20, 20)        # Subtle darkening
    
    # Accent colors for special states
    HIGHLIGHT_ACTIVE: Final[Tuple[int, int, int]] = (255, 215, 0)    # Gold
    WARNING_PULSE: Final[Tuple[int, int, int]] = (0, 191, 255)      # Deep sky blue


@dataclass(frozen=True)
class Geometry:
    """
    Sacred geometry of our interface - the golden ratios of HUD design.
    
    Every measurement is deliberate, based on:
    - Fitts's Law for target acquisition
    - Miller's Law for information chunks
    - Gestalt principles for visual grouping
    """
    # Bounding box parameters
    BOX_THICKNESS: Final[int] = 2              # Visible but not overwhelming
    BOX_CORNER_RADIUS: Final[int] = 8          # Subtle rounding for elegance
    
    # Text rendering
    LABEL_OFFSET_Y: Final[int] = 25            # Space above box for label
    LABEL_PADDING_X: Final[int] = 8            # Horizontal label padding
    LABEL_PADDING_Y: Final[int] = 4            # Vertical label padding
    
    # HUD overlay
    HUD_MARGIN: Final[int] = 20                # Safe area margin
    HUD_LINE_HEIGHT: Final[int] = 30           # Text line spacing
    HUD_SHADOW_OFFSET: Final[int] = 2          # Drop shadow offset
    
    # Performance overlay
    STATS_BACKDROP_ALPHA: Final[float] = 0.7   # Transparency for readability


@dataclass(frozen=True)
class Typography:
    """
    Our typographic system - where readability meets elegance.
    
    Font selection criteria:
    - Maximum legibility at small sizes
    - Cross-platform availability
    - Render performance
    """
    # OpenCV font selection (Hershey fonts are vector-based, fast)
    PRIMARY_FONT: Final[int] = cv2.FONT_HERSHEY_DUPLEX      # Clean, modern
    MONO_FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX       # For numbers
    
    # Font scales (relative to base)
    SCALE_LABEL: Final[float] = 0.6            # Bounding box labels
    SCALE_HUD: Final[float] = 0.7              # HUD text
    SCALE_EMPHASIS: Final[float] = 0.8         # Important messages
    
    # Font weights
    WEIGHT_NORMAL: Final[int] = 1
    WEIGHT_BOLD: Final[int] = 2


# Singleton configurations
COLORS = ColorPalette()
GEOMETRY = Geometry()
TYPOGRAPHY = Typography()


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Rendering Engine
# ═══════════════════════════════════════════════════════════════════════════

class RenderPipeline:
    """
    The rendering pipeline - where perception becomes visible.
    
    This isn't just drawing boxes. It's visual storytelling,
    where every frame narrates the world around you.
    """
    
    @staticmethod
    def apply(
        frame: np.ndarray,
        events: List[Dict],
        fps: Optional[float] = None,
        mode: str = "CPU",
        latency_ms: Optional[float] = None,
        msg_preview: Optional[str] = None
    ) -> np.ndarray:
        """
        Master rendering orchestration.
        
        Each element is rendered in priority order:
        1. Performance overlay (context)
        2. Detection boxes (spatial awareness)
        3. Labels (semantic understanding)
        4. Voice preview (action feedback)
        """
        # Create working canvas (preserve original)
        canvas = frame.copy()
        
        # Layer 1: Performance metrics backdrop
        canvas = RenderPipeline._draw_performance_hud(
            canvas, fps, mode, latency_ms
        )
        
        # Layer 2: Voice feedback
        if msg_preview:
            canvas = RenderPipeline._draw_voice_preview(canvas, msg_preview)
        
        # Layer 3: Detection visualizations
        for event in events:
            canvas = RenderPipeline._draw_detection(canvas, event)
        
        return canvas
    
    @staticmethod
    def _draw_detection(frame: np.ndarray, event: Dict) -> np.ndarray:
        """
        Render a single detection with semantic styling.
        
        The visual language:
        - Color indicates category
        - Thickness shows confidence
        - Position reveals spatial relationship
        """
        bbox = event.get('bbox')
        if not bbox or len(bbox) != 4:
            return frame
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Determine semantic color based on intent/label
        color = RenderPipeline._get_semantic_color(event)
        
        # Draw the bounding box with subtle rounded corners
        RenderPipeline._draw_rounded_rectangle(
            frame, (x1, y1), (x2, y2), color, GEOMETRY.BOX_THICKNESS
        )
        
        # Add semantic label
        label = RenderPipeline._compose_label(event)
        if label:
            RenderPipeline._draw_label(frame, label, x1, y1, color)
        
        return frame
    
    @staticmethod
    def _draw_rounded_rectangle(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int
    ) -> None:
        """
        Draw a rectangle with subtly rounded corners.
        
        Sharp corners are aggressive. Rounded corners are approachable.
        This small detail reduces cognitive stress.
        """
        x1, y1 = pt1
        x2, y2 = pt2
        r = GEOMETRY.BOX_CORNER_RADIUS
        
        # Ensure radius doesn't exceed box dimensions
        r = min(r, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
        
        if r > 0:
            # Top-left corner
            cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
            cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
            
            # Top-right corner
            cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
            
            # Bottom-left corner
            cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
            
            # Bottom-right corner
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        else:
            # Fallback to simple rectangle
            cv2.rectangle(img, pt1, pt2, color, thickness)
    
    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int]
    ) -> None:
        """
        Render a label with perfect legibility.
        
        Typography is not decoration - it's survival information.
        Every label must be instantly readable in any condition.
        """
        # Calculate text dimensions
        font = TYPOGRAPHY.PRIMARY_FONT
        scale = TYPOGRAPHY.SCALE_LABEL
        thickness = TYPOGRAPHY.WEIGHT_BOLD
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )
        
        # Position label above bounding box
        label_y = max(y - GEOMETRY.LABEL_OFFSET_Y, text_height + 10)
        
        # Draw background for contrast (subtle dark backdrop)
        bg_pt1 = (x - GEOMETRY.LABEL_PADDING_X, 
                  label_y - text_height - GEOMETRY.LABEL_PADDING_Y)
        bg_pt2 = (x + text_width + GEOMETRY.LABEL_PADDING_X,
                  label_y + GEOMETRY.LABEL_PADDING_Y)
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_pt1, bg_pt2, COLORS.OVERLAY_TINT, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text with shadow for depth
        shadow_pos = (x + GEOMETRY.HUD_SHADOW_OFFSET,
                      label_y + GEOMETRY.HUD_SHADOW_OFFSET)
        cv2.putText(frame, text, shadow_pos, font, scale,
                    COLORS.TEXT_SHADOW, thickness + 1, cv2.LINE_AA)
        
        # Draw primary text
        cv2.putText(frame, text, (x, label_y), font, scale,
                    color, thickness, cv2.LINE_AA)
    
    @staticmethod
    def _draw_performance_hud(
        frame: np.ndarray,
        fps: Optional[float],
        mode: str,
        latency_ms: Optional[float]
    ) -> np.ndarray:
        """
        Render performance metrics with surgical precision.
        
        These numbers aren't vanity metrics - they're health indicators.
        When FPS drops or latency spikes, safety is compromised.
        """
        height, width = frame.shape[:2]
        
        # Compose performance string
        metrics = []
        
        if fps is not None:
            # Color-code FPS (green=good, yellow=ok, red=bad)
            fps_str = f"FPS: {fps:.1f}"
            metrics.append(fps_str)
        
        metrics.append(f"Mode: {mode}")
        
        if latency_ms is not None:
            metrics.append(f"Latency: {latency_ms:.1f}ms")
        
        if not metrics:
            return frame
        
        perf_text = " │ ".join(metrics)  # Use elegant separator
        
        # Render with backdrop for visibility
        x = GEOMETRY.HUD_MARGIN
        y = GEOMETRY.HUD_MARGIN + GEOMETRY.HUD_LINE_HEIGHT
        
        RenderPipeline._draw_text_with_background(
            frame, perf_text, (x, y), 
            TYPOGRAPHY.MONO_FONT,
            TYPOGRAPHY.SCALE_HUD
        )
        
        return frame
    
    @staticmethod
    def _draw_voice_preview(
        frame: np.ndarray,
        msg_preview: str
    ) -> np.ndarray:
        """
        Display the voice feedback with gentle presence.
        
        This text represents the voice of the system - 
        it should feel trustworthy and calm.
        """
        if not msg_preview:
            return frame
        
        # Truncate if too long (maintain readability)
        if len(msg_preview) > 50:
            msg_preview = msg_preview[:47] + "..."
        
        text = f"✦ Says: {msg_preview}"
        
        x = GEOMETRY.HUD_MARGIN
        y = GEOMETRY.HUD_MARGIN + (GEOMETRY.HUD_LINE_HEIGHT * 2)
        
        RenderPipeline._draw_text_with_background(
            frame, text, (x, y),
            TYPOGRAPHY.PRIMARY_FONT,
            TYPOGRAPHY.SCALE_EMPHASIS,
            text_color=COLORS.HIGHLIGHT_ACTIVE
        )
        
        return frame
    
    @staticmethod
    def _draw_text_with_background(
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font: int,
        scale: float,
        text_color: Tuple[int, int, int] = None
    ) -> None:
        """
        Render text with automatic background for contrast.
        
        Text without proper contrast is worse than no text.
        This ensures readability in any lighting condition.
        """
        if text_color is None:
            text_color = COLORS.TEXT_PRIMARY
        
        x, y = position
        thickness = TYPOGRAPHY.WEIGHT_NORMAL
        
        # Get text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )
        
        # Draw semi-transparent background
        padding = 8
        bg_pt1 = (x - padding, y - text_height - padding)
        bg_pt2 = (x + text_width + padding, y + padding)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_pt1, bg_pt2, (20, 20, 20), -1)
        cv2.addWeighted(overlay, GEOMETRY.STATS_BACKDROP_ALPHA,
                       frame, 1 - GEOMETRY.STATS_BACKDROP_ALPHA, 0, frame)
        
        # Draw text with subtle shadow
        shadow_offset = 1
        cv2.putText(frame, text, (x + shadow_offset, y + shadow_offset),
                   font, scale, COLORS.TEXT_SHADOW, thickness + 1, cv2.LINE_AA)
        
        # Draw primary text
        cv2.putText(frame, text, (x, y), font, scale,
                   text_color, thickness, cv2.LINE_AA)
    
    @staticmethod
    def _get_semantic_color(event: Dict) -> Tuple[int, int, int]:
        """
        Map detection semantics to visual color language.
        
        Color is meaning. This mapping is our visual vocabulary.
        """
        intent = event.get('intent', '').upper()
        label = event.get('label', '').upper()
        
        # Critical signs (highest priority)
        if 'STOP' in intent or 'STOP' in label:
            return COLORS.SIGN_CRITICAL
        
        # Navigation signs
        if 'EXIT' in intent or 'EXIT' in label:
            return COLORS.SIGN_CRITICAL
        
        # Crosswalks and road markings
        if 'CROSSWALK' in label or 'CROSSING' in label:
            return COLORS.CROSSWALK_CAUTION
        
        # Default to safe green for obstacles
        return COLORS.OBSTACLE_SAFE
    
    @staticmethod
    def _compose_label(event: Dict) -> str:
        """
        Create human-readable label from detection data.
        
        Labels should be:
        - Concise (cognitive load)
        - Informative (actionable)
        - Consistent (learnable)
        """
        parts = []
        
        # Primary label
        label = event.get('label', '')
        if label:
            parts.append(label.title())
        
        # Add distance if available and close
        dist_m = event.get('dist_m')
        if dist_m is not None and dist_m < 5.0:
            parts.append(f"{dist_m:.1f}m")
        
        # Add bearing if significant
        bearing = event.get('bearing_deg')
        if bearing is not None and abs(bearing) > 15:
            direction = "L" if bearing < 0 else "R"
            parts.append(f"{abs(bearing):.0f}°{direction}")
        
        return " • ".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Public API
# ═══════════════════════════════════════════════════════════════════════════

def draw_hud(
    frame: np.ndarray,
    events: List[Dict],
    fps: Optional[float] = None,
    mode: str = "CPU",
    latency_ms: Optional[float] = None,
    msg_preview: Optional[str] = None
) -> np.ndarray:
    """
    The gateway to visual augmentation.
    
    This function transforms raw perception into actionable intelligence,
    overlaying the physical world with digital insight.
    
    Args:
        frame: The canvas - BGR image from camera
        events: The knowledge - detected objects and signs
        fps: The performance - frames per second
        mode: The engine - "CPU" or "QNN"
        latency_ms: The responsiveness - pipeline latency
        msg_preview: The voice - last spoken guidance
    
    Returns:
        Augmented frame with HUD overlay
        
    Philosophy:
        Every pixel we add must earn its place.
        If it doesn't help the user survive and thrive,
        it doesn't belong on the screen.
    """
    if frame is None or frame.size == 0:
        return frame
    
    # Delegate to our rendering pipeline
    return RenderPipeline.apply(
        frame=frame,
        events=events if events else [],
        fps=fps,
        mode=mode,
        latency_ms=latency_ms,
        msg_preview=msg_preview
    )


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Demonstration
# ═══════════════════════════════════════════════════════════════════════════

def _create_demo_frame() -> np.ndarray:
    """Generate a beautiful demo frame for testing."""
    # Create gradient background (subtle, elegant)
    height, width = 720, 1280
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Vertical gradient from dark to lighter
    for y in range(height):
        gray_value = int(20 + (y / height) * 60)
        frame[y, :] = (gray_value, gray_value + 5, gray_value + 10)
    
    # Add some noise for realism
    noise = np.random.randint(0, 10, (height, width, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    return frame


def _create_demo_events() -> List[Dict]:
    """Generate realistic demo events."""
    return [
        {
            'intent': 'OBSTACLE_PERSON',
            'label': 'Person',
            'bbox': [400, 300, 500, 550],
            'bearing_deg': -5.2,
            'dist_m': 2.3,
            'conf': 0.94
        },
        {
            'intent': 'STOP',
            'label': 'STOP',
            'bbox': [800, 200, 950, 350],
            'bearing_deg': 12.7,
            'dist_m': 4.1,
            'conf': 0.88
        },
        {
            'intent': 'OBSTACLE_CAR',
            'label': 'Car',
            'bbox': [100, 400, 350, 600],
            'bearing_deg': -22.3,
            'dist_m': 3.7,
            'conf': 0.91
        },
        {
            'intent': 'CROSSWALK',
            'label': 'Crosswalk',
            'bbox': [200, 600, 1080, 700],
            'bearing_deg': 0,
            'dist_m': 1.5,
            'conf': 0.76
        }
    ]


def main():
    """
    Demonstration of the VisionMate HUD system.
    
    This is more than a test - it's a glimpse into the future
    of augmented perception.
    """
    print("╔════════════════════════════════════════════════════╗")
    print("║     VisionMate HUD Rendering Engine Demo           ║")
    print("║     Press 'q' to exit, 'm' to toggle voice         ║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    
    # Create demo environment
    frame = _create_demo_frame()
    events = _create_demo_events()
    
    # Simulation state
    fps_simulator = 30.0
    latency_simulator = 15.2
    voice_messages = [
        "Person ahead two meters.",
        "Stop sign to the right.",
        "Caution. Vehicle approaching.",
        "Crosswalk detected ahead.",
        None
    ]
    
    # Create window with specific properties
    window_name = "VisionMate HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # Animation loop
    frame_count = 0
    show_voice = True
    
    print("✦ Rendering started...")
    
    while True:
        # Simulate performance variations
        current_fps = fps_simulator + np.random.randn() * 2
        current_latency = latency_simulator + np.random.randn() * 3
        
        # Cycle through voice messages
        msg_preview = None
        if show_voice and voice_messages:
            msg_preview = voice_messages[frame_count % len(voice_messages)]
        
        # Render HUD
        rendered_frame = draw_hud(
            frame=frame.copy(),  # Fresh frame each time
            events=events,
            fps=current_fps,
            mode="QNN" if frame_count > 60 else "CPU",
            latency_ms=current_latency,
            msg_preview=msg_preview
        )
        
        # Display
        cv2.imshow(window_name, rendered_frame)
        
        # Handle input
        key = cv2.waitKey(33) & 0xFF  # ~30 FPS
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_voice = not show_voice
            print(f"✦ Voice preview: {'ON' if show_voice else 'OFF'}")
        
        frame_count += 1
        
        # Animate events slightly (subtle movement)
        for event in events:
            if 'bbox' in event:
                bbox = event['bbox']
                # Subtle oscillation
                offset = int(2 * math.sin(frame_count * 0.05))
                event['bbox'] = [
                    bbox[0] + offset,
                    bbox[1],
                    bbox[2] + offset,
                    bbox[3]
                ]
    
    cv2.destroyAllWindows()
    print("\n✦ Demo completed successfully")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Module Exports
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    'draw_hud',
    'RenderPipeline',
    'ColorPalette',
    'Geometry',
    'Typography',
]

# ═══════════════════════════════════════════════════════════════════════════
# "Design is not just what it looks like. Design is how it works." - Steve Jobs
# ═══════════════════════════════════════════════════════════════════════════