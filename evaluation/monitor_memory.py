#!/usr/bin/env python3
"""
Memory monitoring script for YOLO+SAM2 pipeline
"""

import psutil
import GPUtil
import time
import os

def get_memory_info():
    """Get current memory usage"""
    # CPU Memory
    cpu_memory = psutil.virtual_memory()
    
    # GPU Memory
    gpu_memory = None
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use first GPU
            gpu_memory = {
                'used': gpu.memoryUsed,
                'total': gpu.memoryTotal,
                'free': gpu.memoryFree,
                'utilization': gpu.memoryUtil * 100
            }
    except:
        pass
    
    return cpu_memory, gpu_memory

def monitor_memory(interval=5, log_file=None):
    """Monitor memory usage continuously"""
    if log_file:
        f = open(log_file, 'w')
        f.write("timestamp,cpu_used_gb,cpu_total_gb,cpu_percent,gpu_used_mb,gpu_total_mb,gpu_percent\n")
    
    try:
        while True:
            cpu_mem, gpu_mem = get_memory_info()
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cpu_used_gb = cpu_mem.used / (1024**3)
            cpu_total_gb = cpu_mem.total / (1024**3)
            cpu_percent = cpu_mem.percent
            
            if gpu_mem:
                gpu_used_mb = gpu_mem['used']
                gpu_total_mb = gpu_mem['total']
                gpu_percent = gpu_mem['utilization']
                print(f"{timestamp} | CPU: {cpu_used_gb:.1f}/{cpu_total_gb:.1f}GB ({cpu_percent:.1f}%) | GPU: {gpu_used_mb}/{gpu_total_mb}MB ({gpu_percent:.1f}%)")
            else:
                gpu_used_mb = gpu_total_mb = gpu_percent = 0
                print(f"{timestamp} | CPU: {cpu_used_gb:.1f}/{cpu_total_gb:.1f}GB ({cpu_percent:.1f}%) | GPU: Not available")
            
            if log_file:
                f.write(f"{timestamp},{cpu_used_gb:.2f},{cpu_total_gb:.2f},{cpu_percent:.1f},{gpu_used_mb},{gpu_total_mb},{gpu_percent:.1f}\n")
                f.flush()
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
        if log_file:
            f.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monitor memory usage')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in seconds')
    parser.add_argument('--log', type=str, help='Log file path')
    args = parser.parse_args()
    
    monitor_memory(args.interval, args.log)









