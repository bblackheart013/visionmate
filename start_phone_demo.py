#!/usr/bin/env python3
"""
Phone Camera Demo Startup Script

This script starts VisionMate with phone camera streaming for the multiverse demo.
It sets up both the phone camera stream server and the main VisionMate application.

Author: Vaibhav Chandgir (Integration & Snapdragon lead)
"""

import subprocess
import sys
import time
import webbrowser
import socket
import os

def get_laptop_ip():
    """Get the laptop's IP address."""
    try:
        # Connect to a remote address to get local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "localhost"

def start_http_server():
    """Start HTTP server for phone camera interface."""
    print("Starting HTTP server for phone camera...")
    try:
        # Start HTTP server in background
        process = subprocess.Popen([
            sys.executable, "-m", "http.server", "8000", 
            "--directory", "webui"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✓ HTTP server started on port 8000")
        return process
    except Exception as e:
        print(f"✗ Failed to start HTTP server: {e}")
        return None

def main():
    """Main demo startup function."""
    print("🎯 VisionMate Phone Camera Demo Setup")
    print("=" * 50)
    
    # Get laptop IP
    laptop_ip = get_laptop_ip()
    print(f"Laptop IP: {laptop_ip}")
    
    # Start HTTP server
    http_process = start_http_server()
    if not http_process:
        return 1
    
    print("\n📱 Phone Setup Instructions:")
    print("-" * 30)
    print("1. Make sure your phone is on the same WiFi network")
    print("2. Open your phone's web browser")
    print(f"3. Go to: http://{laptop_ip}:8000/phone_camera.html")
    print("4. Allow camera access when prompted")
    print("5. Click 'Start Camera Stream'")
    
    print(f"\n💻 Laptop Setup:")
    print("-" * 30)
    print("Once phone is streaming, run this command:")
    print(f"python app/main.py --phone-camera --ep cpu --controller on")
    
    print(f"\n🎮 Phone Controller:")
    print("-" * 30)
    print(f"Also available at: http://{laptop_ip}:8000/controller.html")
    
    print(f"\n🚀 Ready for Demo!")
    print("=" * 50)
    print("The phone will stream its camera to the laptop,")
    print("and the laptop will process it for navigation guidance!")
    
    # Open browser to phone camera page
    try:
        phone_camera_url = f"http://{laptop_ip}:8000/phone_camera.html"
        print(f"\nOpening phone camera page: {phone_camera_url}")
        webbrowser.open(phone_camera_url)
    except:
        print(f"\nPlease manually open: http://{laptop_ip}:8000/phone_camera.html")
    
    print("\nPress Ctrl+C to stop the HTTP server...")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping HTTP server...")
        if http_process:
            http_process.terminate()
            http_process.wait()
        print("✓ Demo setup stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
