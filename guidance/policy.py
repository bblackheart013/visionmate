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
Version: 8.0.0 - Production Edition
"""

import time
from typing import Optional, Dict, List, Callable, Tuple, Final
from dataclasses import dataclass
from enum import IntEnum


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Constants & Configuration
# ═══════════════════════════════════════════════════════════════════════════

class Priority(IntEnum):
    """Priority levels for guidance events - lower values = higher urgency."""
    IMMEDIATE_HAZARD = 1  # STOP signs
    PERSON_PROXIMITY = 2  # People in path (< 3m)
    NAVIGATION = 3        # Exits and turns  
    VEHICLE_AWARENESS = 4 # Cars, buses, trucks
    STATIC_OBSTACLE = 5   # Poles, barriers


# Timing constants
GLOBAL_COOLDOWN: Final[float] = 2.0      # Minimum silence between any prompts
DEBOUNCE_WINDOW: Final[float] = 0.2      # Filter perception jitter
STOP_COOLDOWN: Final[float] = 5.0        # Don't nag about same stop sign
NAVIGATION_COOLDOWN: Final[float] = 3.0   # Exit reminders
HAZARD_COOLDOWN: Final[float] = 2.0      # General obstacle cooldown
UTTERANCE_DEDUPE_COOLDOWN: Final[float] = 3.0  # Prevent exact text repetition

# Spatial constants
PERSON_IMMEDIATE_ZONE: Final[float] = 1.0   # Stop immediately
PERSON_CAUTION_ZONE: Final[float] = 2.0     # Stop and be careful
PERSON_AWARENESS_ZONE: Final[float] = 4.0   # Be aware
PERSON_PRIORITY_ZONE: Final[float] = 3.0    # High priority threshold
POLE_DANGER_ZONE: Final[float] = 2.0        # Pole collision risk
BEARING_STRAIGHT: Final[float] = 5.0        # ±5° is "ahead"
BEARING_SLIGHT: Final[float] = 20.0         # ±20° is "slightly left/right"


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def bearing_bucket(bearing_deg: float) -> str:
    """
    Transform bearing angle into natural language.
    
    Returns:
        "ahead" if |bearing| <= 5°
        "slightly left/right" if 5° < |bearing| <= 20°
        "to the left/right" if |bearing| > 20°
    """
    magnitude = abs(bearing_deg)
    
    if magnitude <= BEARING_STRAIGHT:
        return "ahead"
    
    direction = "left" if bearing_deg < 0 else "right"
    
    if magnitude <= BEARING_SLIGHT:
        return f"slightly {direction}"
    else:
        return f"to the {direction}"


def bucket_rank(bearing_text: str) -> int:
    """
    Rank bearing bucket informativeness.
    
    Returns:
        0 for "ahead" (least informative)
        1 for "slightly left/right" 
        2 for "to the left/right" (most informative)
    """
    if "to the" in bearing_text:
        return 2
    elif "slightly" in bearing_text:
        return 1
    else:  # "ahead"
        return 0


def distance_phrase(meters: Optional[float]) -> str:
    """
    Convert metric distance to human-friendly phrase.
    
    Returns:
        "very close" if < 1.0m
        "two meters" if 1.0-2.0m
        "a few meters" if 2.0-4.0m
        "" otherwise
    """
    if meters is None:
        return ""
    
    if meters < PERSON_IMMEDIATE_ZONE:
        return "very close"
    elif meters <= PERSON_CAUTION_ZONE:
        return "two meters"
    elif meters <= PERSON_AWARENESS_ZONE:
        return "a few meters"
    else:
        return ""


def intent_group(intent: str) -> str:
    """
    Map intent to its cooldown group.
    
    Returns:
        "STOP" for STOP
        "EXIT" for EXIT_*
        "HAZARD" for OBSTACLE_*
        "INFO" otherwise
    """
    if intent == "STOP":
        return "STOP"
    elif intent.startswith("EXIT_"):
        return "EXIT"
    elif intent.startswith("OBSTACLE_"):
        return "HAZARD"
    else:
        return "INFO"


def cooldown_for(group: str) -> float:
    """
    Get the cooldown duration for an intent group.
    
    Returns:
        Cooldown duration in seconds
    """
    cooldowns = {
        'STOP': STOP_COOLDOWN,
        'EXIT': NAVIGATION_COOLDOWN,
        'HAZARD': HAZARD_COOLDOWN,
    }
    return cooldowns.get(group, GLOBAL_COOLDOWN)


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Core Policy Engine
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Candidate:
    """Represents a potential utterance candidate with all decision factors."""
    utterance: str
    priority: Priority
    confidence: float
    bearing: float
    bearing_text: str
    group: str
    intent: str
    
    def bearing_sign_pref(self) -> int:
        """Return 0 for right (>=0), 1 for left (<0)."""
        return 0 if self.bearing >= 0 else 1


class GuidancePolicy:
    """
    The brain of our voice guidance system.
    
    This class embodies decades of accessibility research:
    - Cognitive load theory: Don't overwhelm
    - Attention management: Speak at the right moment
    - Trust building: Be consistent and reliable
    
    Every design decision here impacts real human safety.
    """
    
    def __init__(self, now_fn: Optional[Callable[[], float]] = None):
        """
        Initialize the guidance system.
        
        Args:
            now_fn: Time source for testing. Production uses monotonic clock.
        """
        self._clock = now_fn or time.monotonic
        self._last_utterance_time: float = 0.0
        self._last_utterance_text: Optional[str] = None
        self._last_by_group: Dict[str, float] = {}  # Group -> last time spoken
        self._last_by_text: Dict[str, float] = {}   # Text -> last time spoken
        self._last_stop_time: Optional[float] = None  # Dedicated STOP tracking
    
    def _is_urgent_person(self, event: Dict) -> bool:
        """Check if this is a safety-critical near-range person."""
        return event.get('intent') == 'OBSTACLE_PERSON' and \
               (event.get('dist_m') is not None and event['dist_m'] <= PERSON_CAUTION_ZONE)
    
    def choose(self, events: List[Dict]) -> Optional[str]:
        """
        The decision point: What do we say right now?
        
        Process:
        1) Check debounce and global cooldown (only after first utterance)
        2) Build candidates from events
        3) Filter by group and text cooldowns
        4) Sort by priority rules and select best
        
        Args:
            events: Perception events from this video frame
            
        Returns:
            A single utterance, or silence (None)
        """
        now = self._clock()
        
        # Check if we've spoken at least once
        has_spoken = (self._last_utterance_text is not None)
        
        # Check if any event is an urgent person (safety-critical)
        has_urgent_person = any(self._is_urgent_person(event) for event in events)
        
        # Only apply debounce and global cooldown after first utterance AND if no urgent person
        if not has_urgent_person:
            if has_spoken and now - self._last_utterance_time < DEBOUNCE_WINDOW:
                return None
            if has_spoken and now - self._last_utterance_time < GLOBAL_COOLDOWN:
                return None
        
        # Build candidates from events
        candidates: List[Candidate] = []
        
        for event in events:
            if not self._is_valid_event(event):
                continue
            
            intent = event.get('intent', '')
            priority = self._get_priority(event)
            if priority is None:
                continue
            
            utterance = self._generate_utterance(event)
            if not utterance:
                continue
            
            group = intent_group(intent)
            
            candidates.append(Candidate(
                utterance=utterance,
                priority=priority,
                confidence=event.get('conf', 0.0),
                bearing=event.get('bearing_deg', 0.0),
                bearing_text=bearing_bucket(event.get('bearing_deg', 0.0)),
                group=group,
                intent=intent
            ))
        
        # Filter candidates by cooldowns
        filtered_candidates: List[Candidate] = []
        
        for candidate in candidates:
            group = candidate.group
            text = candidate.utterance
            group_cd = cooldown_for(group)
            
            # Special handling for STOP signs - enforce extended cooldown
            if candidate.intent == 'STOP':
                # Check if STOP was recently announced using dedicated tracker
                if self._last_stop_time is not None:
                    if now - self._last_stop_time < STOP_COOLDOWN:
                        continue  # Skip this STOP, it's still in cooldown
            
            # Standard per-group cooldown
            if group in self._last_by_group:
                if now - self._last_by_group[group] < group_cd:
                    continue
            
            # Per-text cooldown: only enforce if there is a recorded time for this exact text
            if text in self._last_by_text:
                text_cooldown = max(group_cd, UTTERANCE_DEDUPE_COOLDOWN)
                if now - self._last_by_text[text] < text_cooldown:
                    continue
            
            filtered_candidates.append(candidate)
        
        if not filtered_candidates:
            return None
        
        # Sort candidates by priority rules
        # Order: priority ASC, -confidence DESC, sign_pref ASC, -bucket_rank DESC, abs(bearing) ASC
        def sort_key(c: Candidate) -> Tuple[int, float, int, int, float]:
            return (
                c.priority,                         # Lower priority value = higher priority
                -c.confidence,                      # Negative for descending sort
                c.bearing_sign_pref(),              # 0 for right, 1 for left (prefer right)
                -bucket_rank(c.bearing_text),       # Negative for descending (prefer informative)
                abs(c.bearing)                      # Smaller absolute bearing is better
            )
        
        filtered_candidates.sort(key=sort_key)
        
        # Select best candidate and update state
        best = filtered_candidates[0]
        
        # Update all tracking state
        self._last_utterance_time = now
        self._last_utterance_text = best.utterance
        self._last_by_group[best.group] = now
        self._last_by_text[best.utterance] = now
        
        # Track STOP specifically
        if best.intent == 'STOP':
            self._last_stop_time = now
        
        return best.utterance
    
    def _get_priority(self, event: Dict) -> Optional[Priority]:
        """
        Determine priority for an event.
        
        Special case: OBSTACLE_PERSON with dist < 3m gets high priority.
        """
        intent = event.get('intent', '')
        
        if intent == 'STOP':
            return Priority.IMMEDIATE_HAZARD
        
        if intent == 'OBSTACLE_PERSON':
            dist = event.get('dist_m')
            if dist is not None and dist < PERSON_PRIORITY_ZONE:
                return Priority.PERSON_PROXIMITY
            return None  # Person too far or no distance
        
        if intent in ['EXIT_RIGHT', 'EXIT_LEFT']:
            return Priority.NAVIGATION
        
        if intent in ['OBSTACLE_CAR', 'OBSTACLE_BUS', 'OBSTACLE_TRUCK']:
            return Priority.VEHICLE_AWARENESS
        
        if intent == 'OBSTACLE_POLE':
            bearing = abs(event.get('bearing_deg', 0.0))
            dist = event.get('dist_m')
            
            # Must be straight ahead
            if bearing > BEARING_STRAIGHT:
                return None
            
            # Must be close if distance known
            if dist is not None and dist >= POLE_DANGER_ZONE:
                return None
            
            return Priority.STATIC_OBSTACLE
        
        return None
    
    def _generate_utterance(self, event: Dict) -> Optional[str]:
        """Generate appropriate utterance for an event."""
        intent = event.get('intent', '')
        
        if intent == 'STOP':
            return "Stop sign ahead."
        
        elif intent == 'OBSTACLE_PERSON':
            return self._generate_person_utterance(event)
        
        elif intent in ['EXIT_RIGHT', 'EXIT_LEFT']:
            return self._generate_exit_utterance(event)
        
        elif intent in ['OBSTACLE_CAR', 'OBSTACLE_BUS', 'OBSTACLE_TRUCK']:
            return self._generate_vehicle_utterance(event)
        
        elif intent == 'OBSTACLE_POLE':
            return "Caution. Pole ahead."
        
        return None
    
    def _generate_person_utterance(self, event: Dict) -> Optional[str]:
        """
        Generate utterance for person obstacle.
        
        Format:
        - < 1m: "Stop. Person very close."
        - 1-2m (inclusive): "Stop. Person two meters." (with optional bearing)
        - 2-4m: "Caution. Person ahead." (with optional bearing)
        """
        dist = event.get('dist_m')
        bearing = event.get('bearing_deg', 0.0)
        
        if dist is None:
            return None
        
        # Determine base phrase by distance
        if dist < PERSON_IMMEDIATE_ZONE:  # < 1.0
            # Very close - no bearing needed
            return "Stop. Person very close."
        elif dist <= PERSON_CAUTION_ZONE:  # 1.0 <= dist <= 2.0 (inclusive)
            base = "Stop. Person two meters"
            # Add bearing if significant
            if abs(bearing) > BEARING_STRAIGHT:
                bearing_text = bearing_bucket(bearing)
                if bearing_text != "ahead":
                    return f"{base} {bearing_text}."
            return f"{base}."
        elif dist <= PERSON_AWARENESS_ZONE:  # 2.0 < dist <= 4.0
            # Always include "ahead" in base
            base = "Caution. Person ahead"
            # Add bearing modifier if significant
            if abs(bearing) > BEARING_STRAIGHT:
                bearing_text = bearing_bucket(bearing)
                if bearing_text != "ahead":
                    # Append bearing after "ahead"
                    return f"{base} {bearing_text}."
            return f"{base}."
        else:
            return None  # Too far
    
    def _generate_exit_utterance(self, event: Dict) -> str:
        """Generate utterance for exit signs."""
        bearing = event.get('bearing_deg', 0.0)
        bearing_text = bearing_bucket(bearing)
        return f"Exit {bearing_text}."
    
    def _generate_vehicle_utterance(self, event: Dict) -> str:
        """Generate utterance for vehicles."""
        bearing = event.get('bearing_deg', 0.0)
        bearing_text = bearing_bucket(bearing)
        return f"Caution. Vehicle {bearing_text}."
    
    def _is_valid_event(self, event: Dict) -> bool:
        """Validate event has required fields and valid data."""
        if not isinstance(event, dict):
            return False
        
        if 'intent' not in event:
            return False
        
        conf = event.get('conf', 0)
        if conf <= 0:
            return False
        
        return True


# ═══════════════════════════════════════════════════════════════════════════
# MARK: - Public API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    'GuidancePolicy',
    'bearing_bucket',
    'bucket_rank',
    'distance_phrase',
    'intent_group',
    'cooldown_for',
    'Priority',
]


# ═══════════════════════════════════════════════════════════════════════════
# End of File
# ═══════════════════════════════════════════════════════════════════════════