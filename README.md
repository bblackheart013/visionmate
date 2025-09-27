# VisionMate - Blind Navigation Assistant

**Snapdragon Multiverse Hackathon Project**  
*Person 3: Integration & Snapdragon Lead*

VisionMate is a multi-device blind navigation assistant that uses computer vision and AI to help visually impaired users navigate indoor environments safely. The system combines real-time obstacle detection, sign recognition, and audio guidance with multi-device control capabilities.

## 🎯 Multiverse Architecture

VisionMate demonstrates true multi-device integration:

- **📱 Phone**: Web-based controller for navigation commands and settings
- **💻 Snapdragon Laptop**: Real-time vision processing with QNN acceleration  
- **☁️ Cloud** (Optional): Route planning and waypoint guidance
- **🔊 Audio**: Text-to-speech guidance and alerts

## 🚀 Quick Start

### Prerequisites

- **Snapdragon X Elite Laptop**: For QNN acceleration testing
- **x64 Python 3.11+**: Required for ONNX Runtime QNN support
- **Phone/Tablet**: For controller interface (any modern web browser)

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd visionmate
   python setup.py --dev
   ```

2. **Install QNN support** (Snapdragon laptop only):
   ```bash
   python setup.py --qnn
   ```

3. **Verify installation**:
   ```bash
   python app/main.py --video samples/city.mp4 --ep cpu --controller on
   ```

## 🎮 Usage

### Basic Usage

```bash
# Run with video file (CPU mode)
python app/main.py --video samples/city.mp4 --ep cpu

# Run with webcam (QNN mode)
python app/main.py --camera 0 --ep qnn

# Run with phone controller
python app/main.py --video samples/city.mp4 --ep cpu --controller on
```

### Phone Controller

1. **Start the application** with `--controller on`
2. **Open your phone browser** and navigate to:
   ```
   http://<laptop-ip>:8765
   ```
3. **Use the controller** to:
   - ▶️ Start/Stop navigation
   - 🔄 Repeat last instruction
   - 🔇 Mute/Unmute audio
   - 🎯 Set destination

### Command Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--video PATH` | Path to video file | None |
| `--camera INDEX` | Camera index (0, 1, 2...) | 0 |
| `--ep {cpu,qnn}` | Execution provider | cpu |
| `--controller {on,off}` | Enable phone controller | off |
| `--route URL` | Cloud route service URL | None |

## 🏗️ Architecture

### Core Components

```
visionmate/
├── app/
│   ├── main.py           # Main orchestrator
│   ├── perf.py           # Performance monitoring
│   ├── ws_server.py      # WebSocket server
│   └── route_client.py   # Cloud route integration
├── webui/
│   └── controller.html   # Phone controller interface
├── tools/
│   └── bench.py          # Performance benchmarking
└── samples/
    ├── city.mp4          # Sample video
    └── route.json        # Prebaked routes
```

### Integration Points

- **Person 1 (Perception)**: `from perception import PerceptionPipeline`
- **Person 2 (Guidance)**: `from guidance import GuidanceEngine`
- **Mock Fallbacks**: Automatically used if modules unavailable

### Performance Monitoring

The system provides real-time performance metrics:

```
FPS: 23.1 | Grab 2.1ms | Perception 28.4ms | Guidance 3.6ms | Render 4.9ms
```

## 🔧 Snapdragon Integration

### QNN Execution Provider

VisionMate supports Qualcomm's Neural Network (QNN) execution provider for hardware acceleration:

```python
# QNN Configuration
providers = [
    ("QNNExecutionProvider", {
        "backend_path": "",              # Default HTP
        "qnn_htp_performance_mode": "burst"
    }),
    "CPUExecutionProvider"              # Fallback
]
```

### Performance Comparison

Use the benchmarking tool to compare CPU vs QNN performance:

```bash
# Compare execution providers
python tools/bench.py --video samples/city.mp4 --ep both

# CPU only
python tools/bench.py --video samples/city.mp4 --ep cpu

# QNN only  
python tools/bench.py --video samples/city.mp4 --ep qnn
```

**Expected Results**:
- **CPU**: Baseline performance on any hardware
- **QNN**: 2-5x speedup for neural network inference on Snapdragon

## 🌐 Multi-Device Setup

### Network Configuration

1. **Find laptop IP**:
   ```bash
   ipconfig  # Windows
   ifconfig  # macOS/Linux
   ```

2. **Update controller URL**:
   - Edit `webui/controller.html`
   - Replace `localhost` with laptop IP
   - Or use: `ws://<laptop-ip>:8765`

3. **Test connection**:
   ```bash
   # Start application with controller
   python app/main.py --controller on
   
   # Open phone browser to laptop IP
   ```

### Controller Commands

The phone controller sends WebSocket commands:

```json
{
  "cmd": "start|stop|repeat|mute|unmute|set_goal",
  "arg": "destination_name"
}
```

## ☁️ Cloud Integration (Optional)

### Route Service

VisionMate can integrate with cloud route services:

```bash
# Use cloud route service
python app/main.py --route http://route-service.com/api/navigate

# Use local prebaked routes
python app/main.py --video samples/city.mp4
```

### Route Format

Routes are provided as waypoint lists:

```json
{
  "waypoints": [
    {
      "id": "start",
      "lat": 37.7749,
      "lon": -122.4194,
      "name": "Main Lobby",
      "instructions": "Begin navigation from the main lobby",
      "distance_m": 0.0,
      "bearing_deg": 0.0
    }
  ]
}
```

## 🛠️ Development

### Adding New Features

1. **Extend mock modules** for testing without Person 1/2
2. **Add new controller commands** in `ws_server.py`
3. **Enhance performance monitoring** in `perf.py`
4. **Update route integration** in `route_client.py`

### Testing

```bash
# Test WebSocket server
python app/ws_server.py

# Test route client
python app/route_client.py --start lobby --goal cafeteria

# Test performance monitoring
python app/perf.py
```

### Building Executable

Create a standalone Windows executable:

```bash
python setup.py --build
```

This creates `dist/VisionMate.exe` with all dependencies bundled.

## 📊 Performance Benchmarks

### Expected Performance (Snapdragon X Elite)

| Component | CPU (ms) | QNN (ms) | Speedup |
|-----------|----------|----------|---------|
| Frame Grab | 2.1 | 2.1 | 1.0x |
| Perception | 28.4 | 8.2 | 3.5x |
| Guidance | 3.6 | 3.6 | 1.0x |
| Render | 4.9 | 4.9 | 1.0x |
| **Total** | **39.0** | **18.8** | **2.1x** |

### Demo Script for Judges

```bash
# 1. Show CPU baseline
python app/main.py --video samples/city.mp4 --ep cpu --controller on

# 2. Show QNN acceleration  
python app/main.py --video samples/city.mp4 --ep qnn --controller on

# 3. Show phone controller
# Open http://<laptop-ip>:8765 on phone

# 4. Show performance comparison
python tools/bench.py --video samples/city.mp4 --ep both
```

## 🔍 Troubleshooting

### Common Issues

**QNN not available**:
```
Warning: QNN execution provider not available. Falling back to CPU.
```
- ✅ Normal on non-Snapdragon hardware
- ✅ App continues with CPU execution

**WebSocket connection failed**:
```
Connection error: [Errno 61] Connection refused
```
- Check laptop IP address
- Ensure firewall allows port 8765
- Verify `--controller on` is enabled

**Video file not found**:
```
Error: Video file not found: samples/city.mp4
```
- Run `python setup.py --dev` to create sample video
- Or use `--camera 0` for webcam input

**Import errors**:
```
Warning: perception module not available. Using mock.
```
- ✅ Normal during development
- Person 1/2 modules will replace mocks

### Performance Issues

**Low FPS**:
- Check if QNN is actually being used
- Reduce video resolution
- Close other applications

**High CPU usage**:
- Ensure QNN execution provider is loaded
- Check for memory leaks in perception/guidance

## 📝 Integration Notes

### For Person 1 (Perception)

Your module should provide:

```python
class PerceptionPipeline:
    def process_frame(self, frame_bgr, fps=30) -> List[Event]:
        # Return list of events with schema v1
        pass
```

**Event Schema**:
```python
{
    "schema": "v1",
    "ts": 12.345,
    "type": "obstacle|sign|crosswalk", 
    "label": "person|STOP|EXIT",
    "intent": "OBSTACLE_PERSON|STOP|EXIT_RIGHT",
    "conf": 0.91,
    "bbox": [x1, y1, x2, y2],
    "bearing_deg": -5,
    "dist_m": 2.1,
    "sources": ["yolo", "ocr"]
}
```

### For Person 2 (Guidance)

Your module should provide:

```python
class GuidanceEngine:
    def step(self, frame_bgr, events, fps=None, mode="CPU", latency_ms=None):
        # Return (frame_with_hud, utterance_str_or_None)
        pass
```

**Integration Points**:
- TTS queuing is handled internally (non-blocking)
- HUD overlay should show execution mode
- Performance timing is automatic

## 🏆 Demo Checklist

### For Judges

- [ ] **Multi-device**: Phone controls laptop app
- [ ] **Snapdragon**: QNN acceleration demonstrated  
- [ ] **Performance**: Clear CPU vs QNN comparison
- [ ] **Functionality**: Obstacle detection + navigation
- [ ] **Robustness**: Graceful fallbacks and error handling

### Demo Script

1. **Start application** with phone controller
2. **Show phone interface** controlling laptop
3. **Switch execution providers** (CPU ↔ QNN)
4. **Show performance metrics** in real-time
5. **Demonstrate navigation** with sample video
6. **Run benchmark comparison** for judges

## 📞 Support

- **Person 3**: Integration & Snapdragon lead
- **Issues**: Check troubleshooting section
- **Development**: Use mock modules for testing

---

**VisionMate** - Empowering independence through AI-powered navigation 🎯
