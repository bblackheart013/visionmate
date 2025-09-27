#!/usr/bin/env python3
"""
Performance monitoring utilities for VisionMate.

This module provides timing context managers and performance metrics
for monitoring the vision processing pipeline stages.

Author: Person 3 (Integration & Snapdragon lead)
"""

import time
import threading
from typing import Dict, List
from contextlib import contextmanager
from collections import deque

class PerformanceMonitor:
    """Thread-safe performance monitor with moving averages."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.timers: Dict[str, deque] = {}
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.frame_count = 0
    
    def add_timing(self, name: str, duration_ms: float):
        """Add a timing measurement."""
        with self.lock:
            if name not in self.timers:
                self.timers[name] = deque(maxlen=self.window_size)
            self.timers[name].append(duration_ms)
    
    def get_average(self, name: str) -> float:
        """Get average timing for a stage."""
        with self.lock:
            if name not in self.timers or not self.timers[name]:
                return 0.0
            return sum(self.timers[name]) / len(self.timers[name])
    
    def get_fps(self) -> float:
        """Get current FPS."""
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def increment_frame(self):
        """Increment frame counter."""
        with self.lock:
            self.frame_count += 1
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get all timing averages."""
        with self.lock:
            return {name: sum(times) / len(times) if times else 0.0 
                   for name, times in self.timers.items()}

# Global performance monitor instance
_perf_monitor = PerformanceMonitor()

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            _perf_monitor.add_timing(self.name, duration_ms)

@contextmanager
def timer(name: str):
    """Alternative context manager for timing."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        _perf_monitor.add_timing(name, duration_ms)

def perf_line() -> str:
    """
    Generate a performance summary line.
    
    Returns:
        String with FPS and per-stage timings
    """
    _perf_monitor.increment_frame()
    
    fps = _perf_monitor.get_fps()
    averages = _perf_monitor.get_all_averages()
    
    # Build performance line
    parts = [f"FPS: {fps:.1f}"]
    
    # Common stages in order
    stage_order = ["grab", "perception", "guidance", "render"]
    for stage in stage_order:
        if stage in averages:
            parts.append(f"{stage.capitalize()} {averages[stage]:.1f}ms")
    
    return " | ".join(parts)

def reset_performance():
    """Reset performance counters."""
    global _perf_monitor
    _perf_monitor = PerformanceMonitor()

def get_performance_stats() -> Dict[str, float]:
    """Get detailed performance statistics."""
    stats = _perf_monitor.get_all_averages()
    stats["fps"] = _perf_monitor.get_fps()
    stats["frame_count"] = _perf_monitor.frame_count
    return stats

def benchmark_stage(func, name: str, iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark a function multiple times.
    
    Args:
        func: Function to benchmark
        name: Name for the benchmark
        iterations: Number of iterations
        
    Returns:
        Dictionary with timing statistics
    """
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        duration_ms = (time.perf_counter() - start) * 1000
        times.append(duration_ms)
    
    return {
        "name": name,
        "iterations": iterations,
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
    }

# Example usage and testing
if __name__ == "__main__":
    import random
    
    # Test the performance monitoring
    print("Testing performance monitoring...")
    
    # Simulate some work
    for i in range(50):
        with Timer("test_stage_1"):
            time.sleep(random.uniform(0.001, 0.005))  # 1-5ms
        
        with Timer("test_stage_2"):
            time.sleep(random.uniform(0.002, 0.008))  # 2-8ms
        
        if i % 10 == 0:
            print(f"Frame {i}: {perf_line()}")
    
    # Final stats
    stats = get_performance_stats()
    print(f"\nFinal performance stats:")
    for name, value in stats.items():
        print(f"  {name}: {value:.2f}")
