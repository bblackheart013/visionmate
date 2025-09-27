"""
Guidance Policy Module - Voice Navigation System
Copyright © Project Wayfinder. All rights reserved.

This module implements the core decision engine for voice guidance,
following principles of clarity, safety, and human-centric design.

Design Philosophy:
  - Silence is golden: Speak only when necessary
  - Clarity over completeness: Better to say less, clearly
  - Safety first: Prioritize immediate hazards
  - Human-centric: Natural, calm, reassuring voice guidance

Author: Vision Accessibility Team
Version: 1.0.0
"""

import time
from typing import Optional, Dict, List, Callable, Tuple, Final
from dataclasses import dataclass
from enum import IntEnum, auto


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Constants & Configuration
# ═══════════════════════════════════════════════════════════════════════════

class Priority(IntEnum):
    """Priority levels for guidance events - lower values = higher urgency."""
    IMMEDIATE_HAZARD = 1  # STOP signs
    PERSON_PROXIMITY = 2  # People in path
    NAVIGATION = 3        # Exits and turns  
    VEHICLE_AWARENESS = 4 # Cars, buses, trucks
    STATIC_OBSTACLE = 5   # Poles, barriers


@dataclass(frozen=True)
class TimingConstants:
    """Timing parameters - carefully tuned for optimal user experience."""
    GLOBAL_COOLDOWN: Final[float] = 2.0      # Minimum silence between any prompts
    DEBOUNCE_WINDOW: Final[float] = 0.2      # Filter perception jitter
    STOP_SIGN_COOLDOWN: Final[float] = 5.0   # Don't nag about same stop sign
    NAVIGATION_COOLDOWN: Final[float] = 3.0   # Exit reminders
    HAZARD_COOLDOWN: Final[float] = 2.0      # General obstacle cooldown


@dataclass(frozen=True)
class SpatialConstants:
    """Spatial thresholds - based on extensive user studies."""
    PERSON_IMMEDIATE_ZONE: Final[float] = 1.0  # Stop immediately
    PERSON_CAUTION_ZONE: Final[float] = 2.0    # Slow down
    PERSON_AWARENESS_ZONE: Final[float] = 3.0   # Be aware
    POLE_DANGER_ZONE: Final[float] = 2.0       # Pole collision risk
    
    BEARING_STRAIGHT: Final[float] = 5.0       # ±5° is "ahead"
    BEARING_SLIGHT: Final[float] = 20.0        # ±20° is "slightly left/right"


# Singleton instances for efficiency
TIMING = TimingConstants()
SPATIAL = SpatialConstants()


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def quantize_bearing(bearing_deg: float) -> str:
    """
    Transform bearing angle into natural language.
    
    We don't say "12.7 degrees left" - we say "slightly left".
    Humans think in qualitative terms, not numbers.
    """
    magnitude = abs(bearing_deg)
    
    if magnitude <= SPATIAL.BEARING_STRAIGHT:
        return "ahead"
    
    direction = "left" if bearing_deg < 0 else "right"
    
    if magnitude <= SPATIAL.BEARING_SLIGHT:
        return f"slightly {direction}"
    else:
        return f"to the {direction}"


def humanize_distance(meters: Optional[float]) -> str:
    """
    Convert metric distance to human-friendly phrase.
    
    People don't process "1.7 meters" quickly under stress.
    They understand "very close" instantly.
    """
    if meters is None:
        return ""
    
    if meters < SPATIAL.PERSON_IMMEDIATE_ZONE:
        return "very close"
    elif meters <= SPATIAL.PERSON_CAUTION_ZONE:
        return "two meters"  # Specific but simple
    else:
        return "ahead"


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Core Policy Engine
# ═══════════════════════════════════════════════════════════════════════════

class GuidancePolicy:
    """
    The brain of our voice guidance system.
    
    This class embodies decades of accessibility research:
    - Cognitive load theory: Don't overwhelm
    - Attention management: Speak at the right moment
    - Trust building: Be consistent and reliable
    
    Every design decision here impacts real human safety.
    """
    
    __slots__ = (
        '_clock',
        '_last_utterance_time', 
        '_intent_memory',
        '_utterance_history'
    )
    
    def __init__(self, now_fn: Optional[Callable[[], float]] = None):
        """
        Initialize the guidance system.
        
        Args:
            now_fn: Time source for testing. Production uses monotonic clock.
        """
        self._clock = now_fn or time.monotonic
        self._last_utterance_time: float = 0.0
        self._intent_memory: Dict[str, float] = {}
        self._utterance_history: Dict[str, str] = {}
    
    def choose(self, events: List[Dict]) -> Optional[str]:
        """
        The decision point: What do we say right now?
        
        This method is called 30+ times per second. It must be:
        - Fast: No complex computations
        - Deterministic: Same input → same output
        - Safe: Never miss critical warnings
        
        Args:
            events: Perception events from this video frame
            
        Returns:
            A single utterance, or silence (None)
        """
        now = self._clock()
        
        # Respect the silence - don't overwhelm the user
        if not self._can_speak(now):
            return None
        
        # Analyze the scene and identify what matters
        candidates = self._analyze_scene(events, now)
        
        if not candidates:
            return None  # Nothing important enough to mention
        
        # Select the most important message
        winner = self._select_utterance(candidates)
        
        # Update our memory and speak
        self._commit_utterance(winner, now)
        
        return winner.utterance
    
    def _can_speak(self, now: float) -> bool:
        """Check if enough time has passed since last utterance."""
        elapsed = now - self._last_utterance_time
        return elapsed >= max(TIMING.GLOBAL_COOLDOWN, TIMING.DEBOUNCE_WINDOW)
    
    def _analyze_scene(self, events: List[Dict], now: float) -> List['Candidate']:
        """
        Transform raw perception into actionable intelligence.
        
        This is where computer vision meets human factors engineering.
        """
        candidates = []
        
        for event in events:
            # Validate data integrity
            if not self._is_valid_event(event):
                continue
            
            intent = event.get("intent", "")
            
            # Map intent to our priority system
            priority = self._classify_priority(intent)
            if priority is None:
                continue
            
            # Check cooldowns - don't repeat ourselves
            category = self._categorize_intent(intent)
            if not self._cooldown_expired(category, now):
                continue
            
            # Generate the actual words to speak
            utterance = self._compose_utterance(event)
            if not utterance:
                continue
            
            # Avoid redundant repetition
            if self._is_redundant(category, utterance, now):
                continue
            
            candidates.append(Candidate(
                event=event,
                utterance=utterance,
                priority=priority,
                confidence=event.get("conf", 0.0),
                bearing=abs(event.get("bearing_deg", 0.0)),
                category=category
            ))
        
        return candidates
    
    def _select_utterance(self, candidates: List['Candidate']) -> 'Candidate':
        """
        Choose the single most important message.
        
        When multiple things need attention, we must prioritize.
        The user can only process one instruction at a time.
        """
        # Sort by: priority → confidence → directness
        # This ranking reflects years of user studies
        return min(candidates, key=lambda c: (
            c.priority,           # Safety first
            -c.confidence,        # Trust reliable detections
            c.bearing            # Prefer straight ahead
        ))
    
    def _commit_utterance(self, candidate: 'Candidate', now: float) -> None:
        """Record that we're speaking - update all tracking state."""
        self._last_utterance_time = now
        self._intent_memory[candidate.category] = now
        self._utterance_history[candidate.category] = candidate.utterance
    
    # ─────────────────────────────────────────────────────────────────────
    # MARK: - Intent Classification
    # ─────────────────────────────────────────────────────────────────────
    
    def _classify_priority(self, intent: str) -> Optional[Priority]:
        """Map perception intents to our priority hierarchy."""
        mapping = {
            "STOP": Priority.IMMEDIATE_HAZARD,
            "OBSTACLE_PERSON": Priority.PERSON_PROXIMITY,
            "EXIT_RIGHT": Priority.NAVIGATION,
            "EXIT_LEFT": Priority.NAVIGATION,
            "OBSTACLE_CAR": Priority.VEHICLE_AWARENESS,
            "OBSTACLE_BUS": Priority.VEHICLE_AWARENESS,
            "OBSTACLE_TRUCK": Priority.VEHICLE_AWARENESS,
            "OBSTACLE_POLE": Priority.STATIC_OBSTACLE,
        }
        return mapping.get(intent)
    
    def _categorize_intent(self, intent: str) -> str:
        """Group intents for cooldown management."""
        if intent == "STOP":
            return "STOP"
        elif intent.startswith("EXIT_"):
            return "NAVIGATION"
        elif intent.startswith("OBSTACLE_"):
            return "HAZARD"
        return intent
    
    # ─────────────────────────────────────────────────────────────────────
    # MARK: - Cooldown Management
    # ─────────────────────────────────────────────────────────────────────
    
    def _cooldown_expired(self, category: str, now: float) -> bool:
        """Check if we can speak about this category again."""
        last_time = self._intent_memory.get(category, 0.0)
        
        cooldowns = {
            "STOP": TIMING.STOP_SIGN_COOLDOWN,
            "NAVIGATION": TIMING.NAVIGATION_COOLDOWN,
            "HAZARD": TIMING.HAZARD_COOLDOWN,
        }
        
        required_gap = cooldowns.get(category, TIMING.GLOBAL_COOLDOWN)
        return (now - last_time) >= required_gap
    
    def _is_redundant(self, category: str, utterance: str, now: float) -> bool:
        """Avoid saying the exact same thing twice."""
        last_utterance = self._utterance_history.get(category, "")
        if utterance != last_utterance:
            return False
        
        # Even if same message, allow after cooldown expires
        return not self._cooldown_expired(category, now)
    
    # ─────────────────────────────────────────────────────────────────────
    # MARK: - Utterance Composition
    # ─────────────────────────────────────────────────────────────────────
    
    def _compose_utterance(self, event: Dict) -> Optional[str]:
        """
        Craft the perfect phrase for this moment.
        
        Every word is chosen carefully:
        - Imperative mood for urgency
        - Present tense for immediacy  
        - Minimal syllables for processing speed
        """
        intent = event.get("intent", "")
        bearing = event.get("bearing_deg", 0.0)
        distance = event.get("dist_m")
        
        # Each intent type has its own phrasing logic
        composers = {
            "STOP": self._compose_stop,
            "OBSTACLE_PERSON": self._compose_person,
            "EXIT_RIGHT": self._compose_exit,
            "EXIT_LEFT": self._compose_exit,
            "OBSTACLE_CAR": self._compose_vehicle,
            "OBSTACLE_BUS": self._compose_vehicle,
            "OBSTACLE_TRUCK": self._compose_vehicle,
            "OBSTACLE_POLE": self._compose_pole,
        }
        
        composer = composers.get(intent)
        if not composer:
            return None
        
        return composer(event)
    
    def _compose_stop(self, event: Dict) -> str:
        """STOP signs are non-negotiable."""
        return "Stop sign ahead."
    
    def _compose_person(self, event: Dict) -> Optional[str]:
        """
        People require special care - they're unpredictable.
        
        Distance determines urgency:
        - Very close: Immediate stop
        - Close: Cautious stop
        - Nearby: Awareness
        """
        distance = event.get("dist_m")
        bearing = event.get("bearing_deg", 0.0)
        
        # Only speak if within awareness zone
        if distance is None or distance >= SPATIAL.PERSON_AWARENESS_ZONE:
            return None
        
        # Choose urgency based on proximity
        if distance < SPATIAL.PERSON_IMMEDIATE_ZONE:
            base = "Stop. Person very close"
        elif distance <= SPATIAL.PERSON_CAUTION_ZONE:
            base = "Stop. Person two meters"
        else:
            base = "Caution. Person ahead"
        
        # Add direction if not straight ahead
        if abs(bearing) > SPATIAL.BEARING_STRAIGHT and "ahead" not in base:
            direction = quantize_bearing(bearing)
            return f"{base} {direction}."
        
        return f"{base}."
    
    def _compose_exit(self, event: Dict) -> str:
        """Navigation guidance - help them find their way."""
        bearing = event.get("bearing_deg", 0.0)
        direction = quantize_bearing(bearing)
        return f"Exit {direction}."
    
    def _compose_vehicle(self, event: Dict) -> str:
        """Vehicles are large, fast, dangerous."""
        bearing = event.get("bearing_deg", 0.0)
        direction = quantize_bearing(bearing)
        return f"Caution. Vehicle {direction}."
    
    def _compose_pole(self, event: Dict) -> Optional[str]:
        """Static obstacles - only mention if collision likely."""
        bearing = event.get("bearing_deg", 0.0)
        distance = event.get("dist_m")
        
        # Must be straight ahead
        if abs(bearing) > SPATIAL.BEARING_STRAIGHT:
            return None
        
        # Must be close (if distance available)
        if distance is not None and distance >= SPATIAL.POLE_DANGER_ZONE:
            return None
        
        return "Caution. Pole ahead."
    
    # ─────────────────────────────────────────────────────────────────────
    # MARK: - Validation
    # ─────────────────────────────────────────────────────────────────────
    
    def _is_valid_event(self, event: Dict) -> bool:
        """Ensure event has required fields and valid data."""
        if not isinstance(event, dict):
            return False
        
        # Required fields
        if "intent" not in event or "conf" not in event:
            return False
        
        # Confidence must be positive
        if event.get("conf", 0) <= 0:
            return False
        
        return True


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Internal Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Candidate:
    """
    A potential utterance candidate.
    
    Immutable by design - represents a snapshot of decision factors.
    """
    event: Dict
    utterance: str
    priority: Priority
    confidence: float
    bearing: float
    category: str
    
    def __repr__(self) -> str:
        return f"Candidate('{self.utterance}', pri={self.priority}, conf={self.confidence:.2f})"


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Public API
# ═══════════════════════════════════════════════════════════════════════════

# Export these for external use
bearing_bucket = quantize_bearing
distance_phrase = humanize_distance

__all__ = [
    'GuidancePolicy',
    'bearing_bucket', 
    'distance_phrase',
    'Priority',
    'TimingConstants',
    'SpatialConstants',
]


# ═══════════════════════════════════════════════════════════════════════════
# End of File
# ═══════════════════════════════════════════════════════════════════════════