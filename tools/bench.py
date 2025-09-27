#!/usr/bin/env python3
"""
Performance benchmarking tool for VisionMate.

This tool measures and compares performance between CPU and QNN execution
providers for the perception and guidance pipelines.

Author: Person 3 (Integration & Snapdragon lead)
"""

import argparse
import cv2
import time
import sys
import os
from typing import Dict, List, Tuple
import statistics

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from perception import PerceptionPipeline
    from guidance import GuidanceEngine
    PERCEPTION_AVAILABLE = True
    GUIDANCE_AVAILABLE = True
except ImportError:
    print("Warning: Perception/Guidance modules not available. Using mock implementations.")
    PERCEPTION_AVAILABLE = False
    GUIDANCE_AVAILABLE = False

from app.perf import Timer, PerformanceMonitor, benchmark_stage
from app.main import MockPerceptionPipeline, MockGuidanceEngine, ep_providers

class BenchmarkRunner:
    """Runs performance benchmarks on the vision pipeline."""
    
    def __init__(self, video_path: str, ep_flag: str = "cpu", num_frames: int = 300):
        """
        Initialize benchmark runner.
        
        Args:
            video_path: Path to video file for benchmarking
            ep_flag: Execution provider flag ('cpu' or 'qnn')
            num_frames: Number of frames to process
        """
        self.video_path = video_path
        self.ep_flag = ep_flag
        self.num_frames = num_frames
        self.results = {}
        
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize perception and guidance components."""
        # Initialize execution providers
        providers = ep_providers(self.ep_flag)
        self.actual_mode = "QNN" if self.ep_flag == "qnn" and len(providers) > 1 else "CPU"
        
        # Initialize perception
        if PERCEPTION_AVAILABLE:
            self.perception = PerceptionPipeline()
        else:
            self.perception = MockPerceptionPipeline()
        
        # Initialize guidance
        if GUIDANCE_AVAILABLE:
            self.guidance = GuidanceEngine()
        else:
            self.guidance = MockGuidanceEngine()
        
        print(f"Initialized components with {self.actual_mode} execution provider")
    
    def run_benchmark(self) -> Dict[str, float]:
        """
        Run the complete benchmark.
        
        Returns:
            Dictionary with performance metrics
        """
        print(f"Running benchmark on {self.video_path}")
        print(f"Execution provider: {self.actual_mode}")
        print(f"Processing {self.num_frames} frames...")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = min(self.num_frames, total_frames)
        
        print(f"Video: {fps} FPS, {total_frames} total frames")
        print(f"Processing {frames_to_process} frames...")
        
        # Timing containers
        grab_times = []
        perception_times = []
        guidance_times = []
        render_times = []
        total_times = []
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while frame_count < frames_to_process:
                frame_start = time.time()
                
                # Grab frame
                grab_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break
                grab_end = time.perf_counter()
                grab_times.append((grab_end - grab_start) * 1000)
                
                # Process with perception
                perception_start = time.perf_counter()
                events = self.perception.process_frame(frame, fps)
                perception_end = time.perf_counter()
                perception_times.append((perception_end - perception_start) * 1000)
                
                # Process with guidance
                guidance_start = time.perf_counter()
                frame_with_hud, utterance = self.guidance.step(
                    frame, events, fps, mode=self.actual_mode
                )
                guidance_end = time.perf_counter()
                guidance_times.append((guidance_end - guidance_start) * 1000)
                
                # Render (simulate)
                render_start = time.perf_counter()
                # Simulate rendering work
                _ = cv2.resize(frame_with_hud, (640, 480))
                render_end = time.perf_counter()
                render_times.append((render_end - render_start) * 1000)
                
                frame_end = time.time()
                total_times.append((frame_end - frame_start) * 1000)
                
                frame_count += 1
                
                # Progress update
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed
                    print(f"  Processed {frame_count}/{frames_to_process} frames ({current_fps:.1f} FPS)")
        
        finally:
            cap.release()
        
        # Calculate statistics
        total_elapsed = time.time() - start_time
        final_fps = frame_count / total_elapsed
        
        results = {
            "execution_provider": self.actual_mode,
            "total_frames": frame_count,
            "total_time_s": total_elapsed,
            "average_fps": final_fps,
            "grab_avg_ms": statistics.mean(grab_times),
            "grab_std_ms": statistics.stdev(grab_times) if len(grab_times) > 1 else 0,
            "perception_avg_ms": statistics.mean(perception_times),
            "perception_std_ms": statistics.stdev(perception_times) if len(perception_times) > 1 else 0,
            "guidance_avg_ms": statistics.mean(guidance_times),
            "guidance_std_ms": statistics.stdev(guidance_times) if len(guidance_times) > 1 else 0,
            "render_avg_ms": statistics.mean(render_times),
            "render_std_ms": statistics.stdev(render_times) if len(render_times) > 1 else 0,
            "total_avg_ms": statistics.mean(total_times),
            "total_std_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0,
        }
        
        self.results = results
        return results
    
    def print_results(self):
        """Print formatted benchmark results."""
        if not self.results:
            print("No results to display. Run benchmark first.")
            return
        
        print("\n" + "="*60)
        print("VISIONMATE PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        print(f"Execution Provider: {self.results['execution_provider']}")
        print(f"Total Frames: {self.results['total_frames']}")
        print(f"Total Time: {self.results['total_time_s']:.2f}s")
        print(f"Average FPS: {self.results['average_fps']:.2f}")
        print()
        
        print("PER-STAGE TIMINGS (milliseconds):")
        print("-" * 40)
        stages = [
            ("Frame Grab", "grab_avg_ms", "grab_std_ms"),
            ("Perception", "perception_avg_ms", "perception_std_ms"),
            ("Guidance", "guidance_avg_ms", "guidance_std_ms"),
            ("Render", "render_avg_ms", "render_std_ms"),
            ("Total Pipeline", "total_avg_ms", "total_std_ms"),
        ]
        
        for stage_name, avg_key, std_key in stages:
            avg = self.results[avg_key]
            std = self.results[std_key]
            print(f"{stage_name:15}: {avg:6.2f} ± {std:5.2f} ms")
        
        print("\nPERFORMANCE BREAKDOWN:")
        print("-" * 40)
        total_pipeline = self.results['total_avg_ms']
        for stage_name, avg_key, _ in stages[:-1]:  # Exclude total pipeline
            avg = self.results[avg_key]
            percentage = (avg / total_pipeline) * 100
            print(f"{stage_name:15}: {percentage:5.1f}% of pipeline")
        
        print("="*60)

def compare_providers(video_path: str, num_frames: int = 300) -> Tuple[Dict, Dict]:
    """
    Compare CPU vs QNN execution providers.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to process
        
    Returns:
        Tuple of (cpu_results, qnn_results)
    """
    print("Running CPU vs QNN comparison...")
    
    # Run CPU benchmark
    print("\n1. CPU Benchmark:")
    cpu_runner = BenchmarkRunner(video_path, "cpu", num_frames)
    cpu_results = cpu_runner.run_benchmark()
    cpu_runner.print_results()
    
    # Run QNN benchmark
    print("\n2. QNN Benchmark:")
    qnn_runner = BenchmarkRunner(video_path, "qnn", num_frames)
    qnn_results = qnn_runner.run_benchmark()
    qnn_runner.print_results()
    
    # Print comparison
    print("\n" + "="*60)
    print("CPU vs QNN COMPARISON")
    print("="*60)
    
    if qnn_results['execution_provider'] == 'QNN':
        print(f"Average FPS - CPU: {cpu_results['average_fps']:.2f}, QNN: {qnn_results['average_fps']:.2f}")
        fps_improvement = ((qnn_results['average_fps'] - cpu_results['average_fps']) / cpu_results['average_fps']) * 100
        print(f"FPS Improvement: {fps_improvement:+.1f}%")
        
        print("\nPerception Stage:")
        cpu_perception = cpu_results['perception_avg_ms']
        qnn_perception = qnn_results['perception_avg_ms']
        perception_improvement = ((cpu_perception - qnn_perception) / cpu_perception) * 100
        print(f"  CPU: {cpu_perception:.2f}ms, QNN: {qnn_perception:.2f}ms ({perception_improvement:+.1f}%)")
        
        print("\nTotal Pipeline:")
        cpu_total = cpu_results['total_avg_ms']
        qnn_total = qnn_results['total_avg_ms']
        total_improvement = ((cpu_total - qnn_total) / cpu_total) * 100
        print(f"  CPU: {cpu_total:.2f}ms, QNN: {qnn_total:.2f}ms ({total_improvement:+.1f}%)")
    else:
        print("QNN execution provider not available - running CPU baseline only")
    
    print("="*60)
    
    return cpu_results, qnn_results

def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="VisionMate Performance Benchmark")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--ep", choices=["cpu", "qnn", "both"], default="both",
                       help="Execution provider to test")
    parser.add_argument("--frames", type=int, default=300,
                       help="Number of frames to process")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    try:
        if args.ep == "both":
            cpu_results, qnn_results = compare_providers(args.video, args.frames)
        else:
            runner = BenchmarkRunner(args.video, args.ep, args.frames)
            results = runner.run_benchmark()
            runner.print_results()
        
        print("\nBenchmark completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
