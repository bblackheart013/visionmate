# GuidedSight Perception Pipeline

Real-time perception for blind/low-vision navigation using YOLOv8-Seg + OCR.

## Quick Start

```bash
# Install dependencies
pip install ultralytics easyocr opencv-python numpy

# Test with synthetic events
cd tools
python mock_perception.py --duration 30

# Process a video
python replay.py --video input.mp4 --out events.json --overlay output.mp4

# Use in your code
from visionmate.perception import PerceptionPipeline
pipeline = PerceptionPipeline()
events = pipeline.process_frame(frame_bgr)
```

## Architecture

- **YOLOv8-Seg**: Detects obstacles (person, car, bus, truck, pole)
- **EasyOCR**: Detects signs (STOP, EXIT with directional arrows)
- **Event Builder**: Merges sources, adds persistence, canonical intents
- **Output**: Structured JSON events with bearing, distance, confidence

## Integration

See `INTEGRATION_NOTES.md` for Person 2 (Guidance) and Person 3 (Orchestrator) APIs.

## Configuration

All parameters tunable in `PerceptionConfig`:
- Detection thresholds
- Persistence frames
- OCR frequency
- Spatial calculations

Built for Snapdragon Multiverse Hackathon 2024.