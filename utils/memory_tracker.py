"""
Memory Tracker Utility for HF-LLM-RUNNER
Monitors GPU and system memory usage during model loading and inference
"""

import os
import time
import subprocess
from typing import Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class MemoryStats:
    """Memory statistics data class"""
    vram_used_gb: float
    vram_total_gb: float
    vram_free_gb: float
    vram_utilization_percent: float
    
    ram_used_gb: float
    ram_total_gb: float
    ram_free_gb: float
    ram_utilization_percent: float
    
    swap_used_gb: float
    swap_total_gb: float
    
    timestamp: float


class MemoryTracker:
    """
    Memory tracker for monitoring GPU VRAM, system RAM, and swap usage.
    Can be used as a context manager or standalone tracker.
    """
    
    def __init__(
        self,
        log_interval: float = 1.0,
        warning_threshold_vram: float = 90.0,
        warning_threshold_ram: float = 90.0,
        callback: Optional[Callable[[MemoryStats], None]] = None
    ):
        """
        Initialize memory tracker.
        
        Args:
            log_interval: Interval in seconds between memory checks
            warning_threshold_vram: VRAM utilization % to trigger warning
            warning_threshold_ram: RAM utilization % to trigger warning
            callback: Optional callback function for memory stats
        """
        self.log_interval = log_interval
        self.warning_threshold_vram = warning_threshold_vram
        self.warning_threshold_ram = warning_threshold_ram
        self.callback = callback
        
        self._tracking = False
        self._stats_history: list = []
    
    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # GPU VRAM stats
        vram_stats = self._get_vram_stats()
        
        # System RAM stats
        ram_stats = self._get_ram_stats()
        
        # Swap stats
        swap_stats = self._get_swap_stats()
        
        return MemoryStats(
            vram_used_gb=vram_stats['used_gb'],
            vram_total_gb=vram_stats['total_gb'],
            vram_free_gb=vram_stats['free_gb'],
            vram_utilization_percent=vram_stats['utilization_percent'],
            
            ram_used_gb=ram_stats['used_gb'],
            ram_total_gb=ram_stats['total_gb'],
            ram_free_gb=ram_stats['free_gb'],
            ram_utilization_percent=ram_stats['utilization_percent'],
            
            swap_used_gb=swap_stats['used_gb'],
            swap_total_gb=swap_stats['total_gb'],
            
            timestamp=time.time()
        )
    
    def _get_vram_stats(self) -> Dict:
        """Get GPU VRAM statistics using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse first GPU
            parts = result.stdout.strip().split('\n')[0].split(',')
            used_mb = float(parts[0].strip())
            total_mb = float(parts[1].strip())
            free_mb = float(parts[2].strip())
            
            return {
                'used_gb': used_mb / 1024,
                'total_gb': total_mb / 1024,
                'free_gb': free_mb / 1024,
                'utilization_percent': (used_mb / total_mb * 100) if total_mb > 0 else 0
            }
        except Exception:
            return {
                'used_gb': 0,
                'total_gb': 0,
                'free_gb': 0,
                'utilization_percent': 0
            }
    
    def _get_ram_stats(self) -> Dict:
        """Get system RAM statistics"""
        try:
            # Linux: read from /proc/meminfo
            meminfo = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    parts = line.split()
                    key = parts[0].rstrip(':')
                    value = float(parts[1])  # in KB
                    meminfo[key] = value
            
            total_kb = meminfo.get('MemTotal', 0)
            available_kb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
            used_kb = total_kb - available_kb
            
            return {
                'used_gb': used_kb / (1024 * 1024),
                'total_gb': total_kb / (1024 * 1024),
                'free_gb': available_kb / (1024 * 1024),
                'utilization_percent': (used_kb / total_kb * 100) if total_kb > 0 else 0
            }
        except Exception:
            # Fallback: use psutil if available
            try:
                import psutil
                mem = psutil.virtual_memory()
                return {
                    'used_gb': mem.used / (1024**3),
                    'total_gb': mem.total / (1024**3),
                    'free_gb': mem.available / (1024**3),
                    'utilization_percent': mem.percent
                }
            except ImportError:
                return {
                    'used_gb': 0,
                    'total_gb': 0,
                    'free_gb': 0,
                    'utilization_percent': 0
                }
    
    def _get_swap_stats(self) -> Dict:
        """Get swap memory statistics"""
        try:
            # Linux: read from /proc/meminfo
            meminfo = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    parts = line.split()
                    key = parts[0].rstrip(':')
                    value = float(parts[1])  # in KB
                    meminfo[key] = value
            
            total_kb = meminfo.get('SwapTotal', 0)
            free_kb = meminfo.get('SwapFree', 0)
            used_kb = total_kb - free_kb
            
            return {
                'used_gb': used_kb / (1024 * 1024),
                'total_gb': total_kb / (1024 * 1024)
            }
        except Exception:
            try:
                import psutil
                swap = psutil.swap_memory()
                return {
                    'used_gb': swap.used / (1024**3),
                    'total_gb': swap.total / (1024**3)
                }
            except ImportError:
                return {
                    'used_gb': 0,
                    'total_gb': 0
                }
    
    def format_stats(self, stats: MemoryStats) -> str:
        """Format memory stats for display"""
        return (
            f"VRAM: {stats.vram_used_gb:.2f}/{stats.vram_total_gb:.2f} GB "
            f"({stats.vram_utilization_percent:.1f}%) | "
            f"RAM: {stats.ram_used_gb:.2f}/{stats.ram_total_gb:.2f} GB "
            f"({stats.ram_utilization_percent:.1f}%) | "
            f"Swap: {stats.swap_used_gb:.2f}/{stats.swap_total_gb:.2f} GB"
        )
    
    def start_tracking(self):
        """Start continuous memory tracking"""
        self._tracking = True
        self._stats_history = []
    
    def stop_tracking(self) -> list:
        """Stop tracking and return history"""
        self._tracking = False
        return self._stats_history
    
    def record_snapshot(self, label: str = "") -> MemoryStats:
        """Record a memory snapshot with optional label"""
        stats = self.get_current_stats()
        stats.label = label  # type: ignore
        self._stats_history.append((label, stats))
        
        # Check thresholds
        if stats.vram_utilization_percent > self.warning_threshold_vram:
            print(f"⚠️  WARNING: VRAM utilization at {stats.vram_utilization_percent:.1f}%")
        if stats.ram_utilization_percent > self.warning_threshold_ram:
            print(f"⚠️  WARNING: RAM utilization at {stats.ram_utilization_percent:.1f}%")
        
        if self.callback:
            self.callback(stats)
        
        return stats
    
    def __enter__(self):
        """Context manager entry"""
        self.start_tracking()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_tracking()
        return False
    
    def print_summary(self):
        """Print memory usage summary"""
        if not self._stats_history:
            print("No memory snapshots recorded")
            return
        
        print("\n" + "=" * 60)
        print("MEMORY USAGE SUMMARY")
        print("=" * 60)
        
        for label, stats in self._stats_history:
            print(f"\n[{label}]")
            print(f"  VRAM: {stats.vram_used_gb:.2f}/{stats.vram_total_gb:.2f} GB")
            print(f"  RAM:  {stats.ram_used_gb:.2f}/{stats.ram_total_gb:.2f} GB")
            print(f"  Swap: {stats.swap_used_gb:.2f}/{stats.swap_total_gb:.2f} GB")
        
        print("=" * 60)
