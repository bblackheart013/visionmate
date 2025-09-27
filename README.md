```markdown
# VisionMate

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen?style=flat-square)

**Real-Time Voice Guidance System for Visual Accessibility**

## Overview

SignSense is a real-time traffic sign detection and voice guidance system built at Qualcomm's Multiverse Hackathon. The system provides natural, context-aware navigation assistance for visually impaired users through intelligent prioritization and natural language generation.

### Key Features

- **Real-time STOP sign detection** using YOLOv8n model optimized for edge devices
- **Intelligent voice guidance** with priority-based announcement system
- **Natural language generation** for human-friendly navigation instructions
- **Multi-device edge deployment** supporting Snapdragon X Elite laptops
- **Camera/UI demonstrations** with Flask web interface

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Virtual environment support
- Camera access for real-time detection

### Installation

```bash
# Clone the repository
git clone https://github.com/bblackheart013/signsense.git
cd signsense

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Access the web interface at `http://localhost:5000`

## Project Structure

```
signsense/
├── guidance/           # Core guidance system
│   ├── policy.py      # Decision engine for voice guidance
│   ├── tts.py         # Text-to-speech synthesis
│   └── tests/         # Unit tests
├── models/            # Machine learning models
├── static/            # Web interface assets
├── templates/         # HTML templates
├── app.py            # Flask application
└── requirements.txt   # Python dependencies
```

## Core Components

### Guidance Policy Engine

The `GuidancePolicy` class implements intelligent decision-making for voice announcements:

```python
from guidance.policy import GuidancePolicy

# Initialize policy engine
policy = GuidancePolicy()

# Process detection events
events = [
    {'intent': 'STOP', 'conf': 0.95},
    {'intent': 'OBSTACLE_PERSON', 'dist_m': 2.0, 'conf': 0.88}
]

# Get prioritized guidance
utterance = policy.choose(events)
# Output: "Stop sign ahead."
```

### Priority System

| Priority Level | Category | Example |
|---------------|----------|---------|
| 1 - IMMEDIATE_HAZARD | Safety-critical | STOP signs |
| 2 - PERSON_PROXIMITY | People < 3m | "Stop. Person two meters." |
| 3 - NAVIGATION | Wayfinding | Exit signs |
| 4 - VEHICLE_AWARENESS | Vehicles | Cars, buses |
| 5 - STATIC_OBSTACLE | Fixed hazards | Poles, barriers |

## Testing

```bash
# Run all tests
python -m unittest discover guidance/tests -v

# Run specific test module
python -m unittest guidance.tests.test_policy -v

# Check coverage
coverage run -m unittest discover
coverage report
```

## Hackathon Achievements

- **1st Place** - Qualcomm Multiverse Hackathon
- **YOLOv8n Integration** - Trained model on ONNX runtime
- **Snapdragon X Elite Deployment** - Optimized for edge AI
- **Multi-device Support** - Camera/UI edge deployment
- **Flask Web Interface** - Real-time visualization

## Technologies Used

- **Computer Vision**: YOLOv8n, ONNX Runtime
- **Voice Synthesis**: pyttsx3
- **Web Framework**: Flask
- **Edge Computing**: Snapdragon X Elite
- **Testing**: unittest, coverage

## Team

Built at Qualcomm Multiverse Hackathon by:
- Mohd Sarfaraz Faiyaz ([@bblackheart013](https://github.com/bblackheart013))

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Qualcomm for hosting the Multiverse Hackathon
- The accessibility community for inspiration
- Open source contributors and dependencies

---

**Note**: This project was developed as a hackathon prototype. For production deployment, additional testing and optimization may be required.
```

This README is specifically tailored for your repository with:
- Correct GitHub username and repo URL
- Focus on the hackathon context and achievements  
- Emphasis on the real-time STOP sign detection aspect
- Practical installation and usage instructions
- Appropriate scope for a hackathon project

Save this content directly as `README.md` in your repository root.
