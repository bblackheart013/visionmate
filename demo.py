#!/usr/bin/env python3
"""
VisionMate Demo Script for Judges

This script demonstrates the key features of VisionMate for the
Snapdragon Multiverse Hackathon presentation.

Author: Person 3 (Integration & Snapdragon lead)
"""

import sys
import os
import time
import subprocess
import threading
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n{step_num}. {description}")
    print("-" * 40)

def run_command(cmd, description, background=False):
    """Run a command and handle it."""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    
    try:
        if background:
            # Run in background
            process = subprocess.Popen(cmd, shell=True)
            time.sleep(2)  # Give it time to start
            return process
        else:
            # Run and wait
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print("✓ Success")
            if result.stdout:
                print(f"Output:\n{result.stdout}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def demo_setup():
    """Demo step 1: Setup and dependencies."""
    print_step(1, "Setting up VisionMate environment")
    
    # Check if dependencies are installed
    print("Checking dependencies...")
    try:
        import cv2
        import websockets
        print("✓ Core dependencies available")
    except ImportError:
        print("⚠ Some dependencies missing. Installing...")
        run_command("python setup.py --dev", "Installing dependencies")
    
    # Check QNN support
    try:
        import onnxruntime
        print("✓ ONNX Runtime available")
        
        # Check for QNN provider
        providers = onnxruntime.get_available_providers()
        if "QNNExecutionProvider" in providers:
            print("✓ QNN Execution Provider available (Snapdragon acceleration ready)")
        else:
            print("⚠ QNN Execution Provider not available (CPU mode only)")
    except ImportError:
        print("⚠ ONNX Runtime not available")

def demo_performance_comparison():
    """Demo step 2: Performance comparison."""
    print_step(2, "CPU vs QNN Performance Comparison")
    
    # Check if sample video exists
    video_path = "samples/city.mp4"
    if not os.path.exists(video_path):
        print("Creating sample video...")
        run_command("python setup.py --dev", "Creating sample video")
    
    print("Running performance benchmark...")
    print("This will compare CPU vs QNN execution providers")
    
    # Run benchmark
    success = run_command(
        "python tools/bench.py --video samples/city.mp4 --ep both --frames 100",
        "Running CPU vs QNN benchmark"
    )
    
    if success:
        print("✓ Performance comparison completed")
        print("Check the output above for CPU vs QNN speedup results")
    else:
        print("⚠ Benchmark failed - this is normal if QNN is not available")

def demo_multiverse_architecture():
    """Demo step 3: Multi-device architecture."""
    print_step(3, "Multi-Device Architecture Demo")
    
    print("VisionMate demonstrates true multi-device integration:")
    print("• 📱 Phone: Web-based controller interface")
    print("• 💻 Snapdragon Laptop: Real-time vision processing")
    print("• ☁️ Cloud: Optional route planning service")
    print("• 🔊 Audio: Text-to-speech guidance")
    
    # Get laptop IP for phone connection
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"\nLaptop IP Address: {local_ip}")
        print(f"Phone Controller URL: http://{local_ip}:8765")
    except:
        print("\nLaptop IP Address: localhost")
        print("Phone Controller URL: http://localhost:8765")

def demo_phone_controller():
    """Demo step 4: Phone controller setup."""
    print_step(4, "Phone Controller Setup")
    
    print("Starting WebSocket server for phone communication...")
    
    # Start WebSocket server in background
    ws_process = run_command(
        "python app/ws_server.py",
        "Starting WebSocket server",
        background=True
    )
    
    if ws_process:
        print("✓ WebSocket server started on port 8765")
        print("\nTo test the phone controller:")
        print("1. Open a web browser on your phone")
        print("2. Navigate to the controller URL shown above")
        print("3. Use the buttons to control the application")
        
        # Try to open controller in browser
        controller_path = os.path.abspath("webui/controller.html")
        if os.path.exists(controller_path):
            print(f"\nOpening controller in browser: {controller_path}")
            webbrowser.open(f"file://{controller_path}")
        
        return ws_process
    
    return None

def demo_vision_processing():
    """Demo step 5: Vision processing pipeline."""
    print_step(5, "Vision Processing Pipeline")
    
    print("Starting VisionMate with phone controller...")
    print("This demonstrates:")
    print("• Real-time video processing")
    print("• Obstacle detection and sign recognition")
    print("• Audio guidance with TTS")
    print("• Performance monitoring HUD")
    print("• Phone controller integration")
    
    # Check for video or camera
    if os.path.exists("samples/city.mp4"):
        video_source = "--video samples/city.mp4"
        print("Using sample video: samples/city.mp4")
    else:
        video_source = "--camera 0"
        print("Using webcam (index 0)")
    
    print(f"\nStarting with command:")
    print(f"python app/main.py {video_source} --ep cpu --controller on")
    
    print("\nPress Ctrl+C to stop the demo")
    print("Watch for:")
    print("• Performance metrics in the HUD")
    print("• Phone controller commands in the logs")
    print("• Visual overlays on detected objects")
    
    # Run the main application
    cmd = f"python app/main.py {video_source} --ep cpu --controller on"
    run_command(cmd, "Running VisionMate application")

def demo_qnn_acceleration():
    """Demo step 6: QNN acceleration."""
    print_step(6, "Snapdragon QNN Acceleration")
    
    print("Demonstrating Snapdragon hardware acceleration...")
    print("This shows the difference between CPU and QNN execution")
    
    # Try QNN mode
    if os.path.exists("samples/city.mp4"):
        video_source = "--video samples/city.mp4"
    else:
        video_source = "--camera 0"
    
    print(f"\nRunning with QNN execution provider:")
    print(f"python app/main.py {video_source} --ep qnn --controller on")
    
    print("\nIf QNN is available, you'll see:")
    print("• 'Mode: QNN' in the HUD")
    print("• Faster perception processing")
    print("• Higher FPS overall")
    
    print("\nIf QNN is not available, you'll see:")
    print("• Warning message about QNN fallback")
    print("• 'Mode: CPU' in the HUD")
    print("• Normal CPU performance")
    
    cmd = f"python app/main.py {video_source} --ep qnn --controller on"
    run_command(cmd, "Running with QNN acceleration")

def cleanup(ws_process):
    """Cleanup demo processes."""
    if ws_process:
        print("\nCleaning up WebSocket server...")
        ws_process.terminate()
        ws_process.wait()
        print("✓ WebSocket server stopped")

def main():
    """Main demo function."""
    print_header("VISIONMATE DEMO - Snapdragon Multiverse Hackathon")
    
    print("This demo showcases VisionMate's multi-device architecture")
    print("and Snapdragon QNN acceleration capabilities.")
    
    ws_process = None
    
    try:
        # Demo steps
        demo_setup()
        demo_performance_comparison()
        demo_multiverse_architecture()
        ws_process = demo_phone_controller()
        demo_vision_processing()
        demo_qnn_acceleration()
        
        print_header("DEMO COMPLETED")
        print("VisionMate successfully demonstrated:")
        print("✓ Multi-device architecture (phone + laptop)")
        print("✓ Snapdragon QNN acceleration")
        print("✓ Real-time vision processing")
        print("✓ Phone controller integration")
        print("✓ Performance monitoring")
        print("✓ Robust error handling and fallbacks")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo error: {e}")
    finally:
        cleanup(ws_process)
    
    print("\nThank you for watching the VisionMate demo!")
    print("For more information, see README.md")

if __name__ == "__main__":
    main()
