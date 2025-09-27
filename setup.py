#!/usr/bin/env python3
"""
Setup script for VisionMate - Blind Navigation Assistant.

This script provides installation and build utilities for the VisionMate
application, including PyInstaller builds for Windows deployment.

Author: Person 3 (Integration & Snapdragon lead)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    return run_command("pip install -r requirements.txt", "Installing requirements")

def install_qnn_support():
    """Install QNN support for Snapdragon (if available)."""
    print("Installing Snapdragon QNN support...")
    return run_command("pip install onnxruntime-qnn", "Installing QNN execution provider")

def create_sample_video():
    """Create a sample video for testing."""
    print("Creating sample video for testing...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('samples/city.mp4', fourcc, 30.0, (640, 480))
        
        for i in range(300):  # 10 seconds at 30 FPS
            # Create a simple moving pattern
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add some moving rectangles to simulate objects
            x = int(100 + 50 * np.sin(i * 0.1))
            y = int(200 + 30 * np.cos(i * 0.15))
            cv2.rectangle(frame, (x, y), (x+50, y+50), (0, 255, 0), -1)
            
            # Add some text
            cv2.putText(frame, f"Frame {i}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print("✓ Sample video created: samples/city.mp4")
        return True
        
    except ImportError:
        print("✗ OpenCV not available. Skipping sample video creation.")
        return False
    except Exception as e:
        print(f"✗ Failed to create sample video: {e}")
        return False

def build_windows_executable():
    """Build Windows executable using PyInstaller."""
    print("Building Windows executable...")
    
    # Check if PyInstaller is available
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        if not run_command("pip install pyinstaller", "Installing PyInstaller"):
            return False
    
    # Create PyInstaller spec file
    spec_content = '''
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app/main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('webui', 'webui'),
        ('samples', 'samples'),
        ('models', 'models'),
    ],
    hiddenimports=[
        'perception',
        'guidance',
        'cv2',
        'ultralytics',
        'easyocr',
        'onnxruntime',
        'pyttsx3',
        'websockets',
        'fastapi',
        'uvicorn',
        'requests',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VisionMate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open('visionmate.spec', 'w') as f:
        f.write(spec_content)
    
    # Build the executable
    return run_command("pyinstaller visionmate.spec", "Building executable")

def setup_development_environment():
    """Set up development environment."""
    print("Setting up development environment...")
    
    # Create necessary directories
    os.makedirs("samples", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("build", exist_ok=True)
    os.makedirs("dist", exist_ok=True)
    
    print("✓ Development directories created")
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create sample video
    create_sample_video()
    
    print("✓ Development environment ready")
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="VisionMate Setup")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--qnn", action="store_true", help="Install QNN support")
    parser.add_argument("--dev", action="store_true", help="Set up development environment")
    parser.add_argument("--build", action="store_true", help="Build Windows executable")
    parser.add_argument("--all", action="store_true", help="Do everything")
    
    args = parser.parse_args()
    
    if args.all or args.dev:
        if not setup_development_environment():
            return 1
    
    if args.all or args.install:
        if not install_dependencies():
            return 1
    
    if args.all or args.qnn:
        install_qnn_support()
    
    if args.all or args.build:
        if not build_windows_executable():
            return 1
    
    if not any([args.install, args.qnn, args.dev, args.build, args.all]):
        print("VisionMate Setup")
        print("===============")
        print("Use --help to see available options")
        print("\nQuick start:")
        print("  python setup.py --dev    # Set up development environment")
        print("  python setup.py --qnn    # Install QNN support (Snapdragon)")
        print("  python setup.py --build  # Build Windows executable")
        print("  python setup.py --all    # Do everything")
    
    print("\n✓ Setup completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
