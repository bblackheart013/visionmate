#!/usr/bin/env python3
"""
VisionMate Quick Start Script

This script provides easy access to common VisionMate operations
without needing to remember complex command-line arguments.

Author: Person 3 (Integration & Snapdragon lead)
"""

import sys
import os
import argparse
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_main_app(args):
    """Run the main VisionMate application."""
    cmd = ["python", "app/main.py"]
    
    if args.video:
        cmd.extend(["--video", args.video])
    elif args.camera is not None:
        cmd.extend(["--camera", str(args.camera)])
    
    if args.ep:
        cmd.extend(["--ep", args.ep])
    
    if args.controller:
        cmd.extend(["--controller", args.controller])
    
    if args.route:
        cmd.extend(["--route", args.route])
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)

def run_benchmark(args):
    """Run performance benchmark."""
    cmd = ["python", "tools/bench.py", "--video", args.video]
    
    if args.ep:
        cmd.extend(["--ep", args.ep])
    
    if args.frames:
        cmd.extend(["--frames", str(args.frames)])
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)

def run_websocket_server():
    """Run standalone WebSocket server."""
    cmd = ["python", "app/ws_server.py"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)

def run_setup():
    """Run setup script."""
    cmd = ["python", "setup.py", "--dev"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)

def main():
    """Main function with subcommands."""
    parser = argparse.ArgumentParser(description="VisionMate Quick Start")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Main app command
    app_parser = subparsers.add_parser("run", help="Run VisionMate application")
    app_parser.add_argument("--video", help="Video file path")
    app_parser.add_argument("--camera", type=int, help="Camera index")
    app_parser.add_argument("--ep", choices=["cpu", "qnn"], default="cpu", help="Execution provider")
    app_parser.add_argument("--controller", choices=["on", "off"], default="off", help="Enable controller")
    app_parser.add_argument("--route", help="Route service URL")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("bench", help="Run performance benchmark")
    bench_parser.add_argument("--video", required=True, help="Video file path")
    bench_parser.add_argument("--ep", choices=["cpu", "qnn", "both"], default="both", help="Execution provider")
    bench_parser.add_argument("--frames", type=int, default=300, help="Number of frames")
    
    # WebSocket server command
    subparsers.add_parser("server", help="Run standalone WebSocket server")
    
    # Setup command
    subparsers.add_parser("setup", help="Run setup and install dependencies")
    
    args = parser.parse_args()
    
    if args.command == "run":
        return run_main_app(args)
    elif args.command == "bench":
        return run_benchmark(args)
    elif args.command == "server":
        return run_websocket_server()
    elif args.command == "setup":
        return run_setup()
    else:
        print("VisionMate Quick Start")
        print("====================")
        print("\nAvailable commands:")
        print("  run     - Run VisionMate application")
        print("  bench   - Run performance benchmark")
        print("  server  - Run WebSocket server")
        print("  setup   - Install dependencies")
        print("\nExamples:")
        print("  python run.py run --video samples/city.mp4 --ep qnn --controller on")
        print("  python run.py bench --video samples/city.mp4 --ep both")
        print("  python run.py server")
        print("  python run.py setup")
        return 0

if __name__ == "__main__":
    sys.exit(main())
