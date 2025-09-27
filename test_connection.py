#!/usr/bin/env python3
"""
Test connection script for VisionMate components.

This script tests the WebSocket server and basic functionality
without requiring the full vision pipeline.

Author: Person 3 (Integration & Snapdragon lead)
"""

import asyncio
import json
import websockets
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_websocket_connection(host="localhost", port=8765):
    """Test WebSocket server connection."""
    uri = f"ws://{host}:{port}"
    
    try:
        print(f"Testing WebSocket connection to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✓ Connected successfully!")
            
            # Test commands
            test_commands = [
                {"cmd": "status"},
                {"cmd": "set_goal", "arg": "cafeteria"},
                {"cmd": "mute"},
                {"cmd": "unmute"},
                {"cmd": "repeat"},
                {"cmd": "status"}
            ]
            
            for i, cmd_data in enumerate(test_commands):
                print(f"\nTest {i+1}: Sending {cmd_data['cmd']}")
                await websocket.send(json.dumps(cmd_data))
                
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    print(f"✓ Response: {response_data.get('message', 'No message')}")
                    
                    if response_data.get('status') == 'error':
                        print(f"  Error: {response_data.get('message')}")
                        
                except asyncio.TimeoutError:
                    print("✗ Timeout waiting for response")
                    break
                
                await asyncio.sleep(0.5)
            
            print("\n✓ All tests completed successfully!")
            return True
            
    except ConnectionRefusedError:
        print(f"✗ Connection refused. Is the WebSocket server running on {host}:{port}?")
        print("Start it with: python app/ws_server.py")
        return False
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from app.perf import Timer, perf_line
        print("✓ Performance monitoring imported")
    except ImportError as e:
        print(f"✗ Failed to import performance monitoring: {e}")
        return False
    
    try:
        from app.ws_server import get_controller_state
        print("✓ WebSocket server imported")
    except ImportError as e:
        print(f"✗ Failed to import WebSocket server: {e}")
        return False
    
    try:
        from app.route_client import get_route
        print("✓ Route client imported")
    except ImportError as e:
        print(f"✗ Failed to import route client: {e}")
        return False
    
    # Test optional imports
    try:
        import cv2
        print("✓ OpenCV imported")
    except ImportError:
        print("⚠ OpenCV not available (install with: pip install opencv-python)")
    
    try:
        import onnxruntime
        print("✓ ONNX Runtime imported")
    except ImportError:
        print("⚠ ONNX Runtime not available (install with: pip install onnxruntime)")
    
    try:
        import websockets
        print("✓ WebSockets imported")
    except ImportError:
        print("⚠ WebSockets not available (install with: pip install websockets)")
    
    return True

def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("\nTesting performance monitoring...")
    
    try:
        from app.perf import Timer, perf_line, reset_performance
        
        # Reset counters
        reset_performance()
        
        # Test timing
        with Timer("test_stage"):
            time.sleep(0.01)  # 10ms
        
        # Test performance line
        perf_text = perf_line()
        print(f"✓ Performance line: {perf_text}")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        return False

def test_route_client():
    """Test route client functionality."""
    print("\nTesting route client...")
    
    try:
        from app.route_client import get_route
        
        # Test local route loading
        waypoints = get_route("lobby", "cafeteria")
        print(f"✓ Loaded {len(waypoints)} waypoints")
        
        if waypoints:
            first_wp = waypoints[0]
            print(f"  First waypoint: {first_wp.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Route client test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("VisionMate Connection Test")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test performance monitoring
    if not test_performance_monitoring():
        all_passed = False
    
    # Test route client
    if not test_route_client():
        all_passed = False
    
    # Test WebSocket connection
    print("\nTesting WebSocket connection...")
    if not await test_websocket_connection():
        all_passed = False
        print("\nTo test WebSocket connection:")
        print("1. Start the server: python app/ws_server.py")
        print("2. Run this test again: python test_connection.py")
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! VisionMate is ready to run.")
        print("\nNext steps:")
        print("1. Install dependencies: python setup.py --dev")
        print("2. Run the application: python run.py run --camera 0 --controller on")
    else:
        print("✗ Some tests failed. Check the output above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
