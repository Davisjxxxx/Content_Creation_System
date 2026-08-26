#!/usr/bin/env python3
"""
GPU Memory Auto-Cleaner
Automatically monitors and cleans GPU memory when it gets too full

LEGACY REFERENCE ONLY.

Avatar V2 does not import or auto-run this file. Its aggressive mode can
terminate unrelated Python GPU processes. The production-safe implementation
is avatar_v2/gpu_memory.py.
"""

import subprocess
import time
import gc
import os
import signal
import psutil
from typing import List, Tuple

class GPUMemoryCleaner:
    def __init__(self, memory_threshold: float = 0.85, check_interval: int = 5):
        """
        Initialize GPU memory cleaner
        
        Args:
            memory_threshold: Clean memory when usage exceeds this percentage (0.0-1.0)
            check_interval: How often to check memory usage in seconds
        """
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.running = False
    
    def get_gpu_memory_info(self) -> Tuple[int, int, float]:
        """Get current GPU memory usage"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                                   '--format=csv,nounits,noheader'], 
                                  capture_output=True, text=True, check=True)
            
            used, total = map(int, result.stdout.strip().split(', '))
            usage_percent = used / total
            return used, total, usage_percent
            
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            print("Error: Could not get GPU memory info. Is nvidia-smi available?")
            return 0, 0, 0.0
    
    def get_gpu_processes(self) -> List[Tuple[int, str, int]]:
        """Get list of processes using GPU memory"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
                                   '--format=csv,nounits,noheader'], 
                                  capture_output=True, text=True, check=True)
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        pid, name, memory = parts[0], parts[1], parts[2]
                        try:
                            processes.append((int(pid), name, int(memory)))
                        except ValueError:
                            continue
            return processes
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
    
    def cleanup_python_cache(self):
        """Clean Python/PyTorch/TensorFlow caches"""
        print("🧹 Cleaning Python caches...")
        
        # PyTorch cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("  ✓ PyTorch cache cleared")
        except ImportError:
            pass
        
        # TensorFlow cleanup
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            print("  ✓ TensorFlow session cleared")
        except ImportError:
            pass
        
        # Force garbage collection
        gc.collect()
        print("  ✓ Garbage collection completed")
    
    def kill_memory_hog_processes(self, min_memory_mb: int = 1000):
        """Kill Python processes using significant GPU memory"""
        processes = self.get_gpu_processes()
        killed_count = 0
        
        for pid, name, memory_mb in processes:
            if 'python' in name.lower() and memory_mb >= min_memory_mb:
                try:
                    # Check if process exists and is not this script
                    if pid != os.getpid() and psutil.pid_exists(pid):
                        process = psutil.Process(pid)
                        print(f"  🔪 Killing {name} (PID: {pid}) using {memory_mb}MB")
                        process.terminate()
                        
                        # Wait a bit, then force kill if still alive
                        time.sleep(2)
                        if process.is_running():
                            process.kill()
                        
                        killed_count += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    continue
        
        if killed_count > 0:
            print(f"  ✓ Killed {killed_count} memory-hogging processes")
            time.sleep(3)  # Wait for processes to fully terminate
        
        return killed_count
    
    def cleanup_memory(self, aggressive: bool = False):
        """Perform memory cleanup"""
        print(f"\n🚀 Starting {'aggressive' if aggressive else 'gentle'} GPU memory cleanup...")
        
        # First try gentle cleanup
        self.cleanup_python_cache()
        
        # Check if we need more aggressive cleanup
        used, total, usage = self.get_gpu_memory_info()
        print(f"📊 Memory after cache cleanup: {used}MB / {total}MB ({usage:.1%})")
        
        if usage > self.memory_threshold or aggressive:
            print("🔥 Memory still high, killing memory-hogging processes...")
            killed = self.kill_memory_hog_processes()
            
            if killed > 0:
                # Check memory again after killing processes
                time.sleep(2)
                used, total, usage = self.get_gpu_memory_info()
                print(f"📊 Memory after cleanup: {used}MB / {total}MB ({usage:.1%})")
        
        print("✅ Cleanup completed!\n")
    
    def monitor_and_cleanup(self):
        """Continuously monitor and cleanup GPU memory"""
        print(f"🔍 Starting GPU memory monitor (threshold: {self.memory_threshold:.1%})")
        print("Press Ctrl+C to stop")
        
        self.running = True
        
        try:
            while self.running:
                used, total, usage = self.get_gpu_memory_info()
                
                if total > 0:  # Valid GPU info
                    print(f"📊 GPU Memory: {used}MB / {total}MB ({usage:.1%})", end="")
                    
                    if usage > self.memory_threshold:
                        print(" - CLEANING! 🧹")
                        self.cleanup_memory()
                    else:
                        print(" - OK ✅")
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Stopping GPU memory monitor")
            self.running = False

def emergency_cleanup():
    """Emergency cleanup function"""
    cleaner = GPUMemoryCleaner()
    cleaner.cleanup_memory(aggressive=True)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Memory Auto-Cleaner")
    parser.add_argument("--threshold", type=float, default=0.85, 
                       help="Memory threshold for cleanup (0.0-1.0, default: 0.85)")
    parser.add_argument("--interval", type=int, default=5,
                       help="Check interval in seconds (default: 5)")
    parser.add_argument("--cleanup", action="store_true",
                       help="Run one-time cleanup and exit")
    parser.add_argument("--emergency", action="store_true",
                       help="Emergency cleanup - kill all Python GPU processes")
    parser.add_argument("--monitor", action="store_true",
                       help="Continuously monitor and auto-cleanup")
    
    args = parser.parse_args()
    
    if args.emergency:
        print("🚨 EMERGENCY CLEANUP MODE")
        emergency_cleanup()
    elif args.cleanup:
        cleaner = GPUMemoryCleaner(args.threshold, args.interval)
        cleaner.cleanup_memory()
    elif args.monitor:
        cleaner = GPUMemoryCleaner(args.threshold, args.interval)
        cleaner.monitor_and_cleanup()
    else:
        # Default: show current status and cleanup if needed
        cleaner = GPUMemoryCleaner(args.threshold, args.interval)
        used, total, usage = cleaner.get_gpu_memory_info()
        
        print(f"📊 Current GPU Memory: {used}MB / {total}MB ({usage:.1%})")
        
        if usage > args.threshold:
            print("Memory usage high - running cleanup...")
            cleaner.cleanup_memory()
        else:
            print("Memory usage OK - no cleanup needed")

if __name__ == "__main__":
    main()
