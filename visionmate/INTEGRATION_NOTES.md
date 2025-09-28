# GuidedSight Perception Integration Notes

## For Person 2 (Guidance Policy)

### Event Consumption
You'll receive events from the perception pipeline with this stable schema:

```json
{
  "schema": "v1",
  "ts": 12.345,
  "type": "obstacle|sign",
  "label": "person|car|bus|truck|pole|STOP|EXIT",
  "intent": "OBSTACLE_PERSON|STOP|EXIT_RIGHT|EXIT_LEFT|etc",
  "conf": 0.91,
  "bbox": [200, 100, 280, 250],
  "bearing_deg": -5.2,
  "dist_m": 2.1,
  "sources": ["yolo", "ocr"]
}
```

### Stable Fields for Guidance Logic
- **intent**: Use this for guidance decisions (OBSTACLE_PERSON, STOP, EXIT_RIGHT, etc.)
- **bearing_deg**: Horizontal angle (-30 to +30 typical, negative=left, positive=right)
- **dist_m**: Distance in meters (only for people currently, null for others)
- **conf**: Confidence 0-1 (already filtered to high confidence)
- **ts**: Timestamp for temporal reasoning

### Priority Recommendations
1. **STOP** signs: Highest priority, immediate audio alert
2. **OBSTACLE_PERSON** with dist_m < 3.0: High priority collision warning
3. **EXIT_RIGHT/LEFT**: Medium priority navigation assistance
4. **OBSTACLE_CAR/BUS/TRUCK**: Medium priority based on bearing_deg
5. **OBSTACLE_POLE**: Lower priority unless bearing_deg near 0

### Testing Without Models
```bash
cd visionmate/tools
python mock_perception.py --duration 30 --out test_events.json
```

## For Person 3 (Orchestrator)

### Main API Usage
```python
from visionmate.perception import PerceptionPipeline, PerceptionConfig

# Initialize once
config = PerceptionConfig()  # Or customize thresholds
pipeline = PerceptionPipeline(config)

# Per frame (main loop)
events = pipeline.process_frame(frame_bgr, fps=30.0)

# events is List[Event] - only promoted events that passed persistence
for event in events:
    print(f"{event.intent} at {event.bearing_deg}° conf={event.conf}")
```

### Backend Switching (Future)
```python
# Placeholder for Snapdragon optimization
pipeline.set_backend("qnn")  # "cpu", "gpu", "qnn"
```

### Performance Monitoring
```python
stats = pipeline.get_stats()
print(f"Processing at {stats['frames_processed']/elapsed:.1f} FPS")
print(f"Events promoted: {stats['events_promoted']}")
```

### Phone Integration Points
1. **Camera frames**: Convert phone camera to BGR format for process_frame()
2. **WebSocket/HTTP**: Send events as JSON using `event_to_dict(event)`
3. **Frame rate**: Pass actual fps to process_frame() for temporal consistency

### Configuration Tuning
Adjust these in PerceptionConfig without code changes:
```python
config = PerceptionConfig(
    yolo_conf_obstacle=0.6,  # Higher = fewer false positives
    ocr_conf_text=0.8,       # Higher = fewer false text detections
    persist_frames=2,        # Lower = more responsive, higher = more stable
    ocr_stride=5             # Higher = less CPU usage, lower = more text detection
)
```

## Dependencies Installation

```bash
pip install ultralytics easyocr opencv-python numpy
```

## File Structure
```
visionmate/
├── perception/
│   ├── __init__.py          # Main PerceptionPipeline class
│   ├── config.py           # All tunable parameters
│   ├── det_seg.py          # YOLOv8 detection
│   ├── ocr.py              # EasyOCR text detection
│   ├── events.py           # Event building & persistence
│   └── utils.py            # Spatial calculations
├── tools/
│   ├── replay.py           # Process videos: python replay.py --video test.mp4 --out events.json
│   └── mock_perception.py  # Synthetic events: python mock_perception.py --duration 60
└── models/                 # Empty (uses pretrained)
```

## Event Intent Mapping
```
Label → Intent:
- "person" → "OBSTACLE_PERSON"
- "car" → "OBSTACLE_CAR"
- "STOP" → "STOP"
- "EXIT" + bearing_deg > 10 → "EXIT_RIGHT"
- "EXIT" + bearing_deg < -10 → "EXIT_LEFT"
- "EXIT" + OCR "EXIT →" → "EXIT_RIGHT"
```

## Testing Strategy
1. **Unit test**: Use mock_perception.py for synthetic events
2. **Integration test**: Use replay.py with test videos
3. **Real-time test**: Phone camera → laptop processing

## Performance Notes
- Target: ~15 FPS on CPU (720p input)
- YOLO runs every frame (~10ms)
- OCR runs every 3rd frame (~30ms when active)
- Memory usage: ~500MB for models
- Startup time: ~3-5 seconds for model loading