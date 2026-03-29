"""
Hardware Detector Module for HF-LLM-RUNNER
Parses nvidia-smi output to detect GPU capabilities, system RAM, and disk space
"""

import os
import re
import subprocess
import shutil
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

# Import logger
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class GPUInfo:
    """GPU information data class"""
    index: int
    name: str
    total_vram_mb: int
    free_vram_mb: int
    used_vram_mb: int
    cuda_version: str
    driver_version: str
    compute_capability: str
    temperature: int
    power_draw: int
    power_limit: int
    
    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_mb / 1024
    
    @property
    def free_vram_gb(self) -> float:
        return self.free_vram_mb / 1024
    
    @property
    def used_vram_gb(self) -> float:
        return self.used_vram_mb / 1024
    
    @property
    def usable_vram_gb(self, safety_margin_gb: int = 3) -> float:
        """Calculate usable VRAM with safety margin"""
        return max(0, self.total_vram_gb - safety_margin_gb)


@dataclass
class HardwareInfo:
    """Complete hardware information"""
    gpus: List[GPUInfo]
    system_ram_gb: float
    available_ram_gb: float
    cuda_version: str
    driver_version: str
    
    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        """Get primary (first) GPU"""
        return self.gpus[0] if self.gpus else None
    
    @property
    def total_vram_gb(self) -> float:
        """Total VRAM across all GPUs"""
        return sum(gpu.total_vram_gb for gpu in self.gpus)
    
    @property
    def total_free_vram_gb(self) -> float:
        """Total free VRAM across all GPUs"""
        return sum(gpu.free_vram_gb for gpu in self.gpus)


def run_nvidia_smi(args: List[str]) -> Optional[str]:
    """
    Run nvidia-smi command and return output.
    
    Args:
        args: List of arguments for nvidia-smi
        
    Returns:
        Command output as string, or None if failed
    """
    try:
        result = subprocess.run(
            ['nvidia-smi'] + args,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi command failed: {e}")
        return None
    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi command timed out")
        return None
    except FileNotFoundError:
        logger.error("nvidia-smi not found. Please ensure NVIDIA drivers are installed.")
        return None


def get_gpu_info(safety_margin_gb: int = 3) -> Dict:
    """
    Get GPU information using nvidia-smi.
    
    This is the main function for VRAM detection.
    Parses nvidia-smi output to extract:
    - GPU name
    - Total VRAM
    - Free VRAM
    - Usable VRAM (Total - safety margin)
    - CUDA version
    - Driver version
    
    Args:
        safety_margin_gb: GB to reserve for system overhead (default: 3)
        
    Returns:
        Dictionary with GPU information
    """
    # Check if nvidia-smi is available
    if shutil.which('nvidia-smi') is None:
        logger.error("nvidia-smi not found in PATH!")
        return {
            'gpu_name': 'Unknown',
            'total_vram_gb': 0.0,
            'free_vram_gb': 0.0,
            'usable_vram_gb': 0.0,
            'safety_margin_gb': safety_margin_gb,
            'cuda_version': 'Unknown',
            'driver_version': 'Unknown',
            'gpus': []
        }
    
    # Get GPU info using nvidia-smi
    output = run_nvidia_smi([
        '--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,power.draw,power.limit',
        '--format=csv,noheader,nounits'
    ])
    
    if not output:
        logger.error("Failed to get GPU info from nvidia-smi")
        return {
            'gpu_name': 'Unknown',
            'total_vram_gb': 0.0,
            'free_vram_gb': 0.0,
            'usable_vram_gb': 0.0,
            'safety_margin_gb': safety_margin_gb,
            'cuda_version': 'Unknown',
            'driver_version': 'Unknown',
            'gpus': []
        }
    
    # Parse GPU info
    gpus = []
    for line in output.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 8:
            try:
                gpu = GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_vram_mb=int(float(parts[2])),
                    free_vram_mb=int(float(parts[3])),
                    used_vram_mb=int(float(parts[4])),
                    temperature=int(float(parts[5])),
                    power_draw=int(float(parts[6].split('.')[0])) if '.' in parts[6] else int(float(parts[6])),
                    power_limit=int(float(parts[7].split('.')[0])) if '.' in parts[7] else int(float(parts[7])),
                    cuda_version='',  # Will be filled below
                    driver_version='',  # Will be filled below
                    compute_capability=''  # Will be filled below
                )
                gpus.append(gpu)
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse GPU line: {line} - {e}")
    
    # Get CUDA and driver version
    version_output = run_nvidia_smi(['--query-gpu=driver_version', '--format=csv,noheader,nounits'])
    driver_version = version_output.strip().split('\n')[0] if version_output else 'Unknown'
    
    cuda_output = run_nvidia_smi(['--query=compute_cap', '--format=csv,noheader'])
    compute_cap = cuda_output.strip() if cuda_output else 'Unknown'
    
    # Get CUDA version from nvidia-smi output
    cuda_version = 'Unknown'
    smi_output = run_nvidia_smi([])
    if smi_output:
        cuda_match = re.search(r'CUDA Version:\s*(\d+\.\d+)', smi_output)
        if cuda_match:
            cuda_version = cuda_match.group(1)
    
    # Update GPU info with versions
    for gpu in gpus:
        gpu.driver_version = driver_version
        gpu.cuda_version = cuda_version
        gpu.compute_capability = compute_cap
    
    if not gpus:
        return {
            'gpu_name': 'Unknown',
            'total_vram_gb': 0.0,
            'free_vram_gb': 0.0,
            'usable_vram_gb': 0.0,
            'safety_margin_gb': safety_margin_gb,
            'cuda_version': cuda_version,
            'driver_version': driver_version,
            'gpus': []
        }
    
    primary_gpu = gpus[0]
    
    return {
        'gpu_name': primary_gpu.name,
        'total_vram_gb': primary_gpu.total_vram_gb,
        'free_vram_gb': primary_gpu.free_vram_gb,
        'usable_vram_gb': max(0, primary_gpu.total_vram_gb - safety_margin_gb),
        'safety_margin_gb': safety_margin_gb,
        'cuda_version': cuda_version,
        'driver_version': driver_version,
        'gpus': gpus
    }


def get_system_ram() -> Dict:
    """
    Get system RAM information.
    
    Returns:
        Dictionary with RAM information
    """
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
            'total_ram_gb': total_kb / (1024 * 1024),
            'available_ram_gb': available_kb / (1024 * 1024),
            'used_ram_gb': used_kb / (1024 * 1024),
            'utilization_percent': (used_kb / total_kb * 100) if total_kb > 0 else 0
        }
    except FileNotFoundError:
        # Fallback: use psutil if available
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'total_ram_gb': mem.total / (1024**3),
                'available_ram_gb': mem.available / (1024**3),
                'used_ram_gb': mem.used / (1024**3),
                'utilization_percent': mem.percent
            }
        except ImportError:
            logger.warning("Could not detect system RAM - /proc/meminfo not found and psutil not available")
            return {
                'total_ram_gb': 0,
                'available_ram_gb': 0,
                'used_ram_gb': 0,
                'utilization_percent': 0
            }


def get_disk_space(path: str = ".") -> Dict:
    """
    Get available disk space for a given path.
    
    Args:
        path: Path to check disk space for
        
    Returns:
        Dictionary with disk space information
    """
    try:
        # Create path if it doesn't exist
        Path(path).mkdir(parents=True, exist_ok=True)
        
        stat = os.statvfs(path)
        total_bytes = stat.f_blocks * stat.f_frsize
        free_bytes = stat.f_bavail * stat.f_frsize
        used_bytes = total_bytes - free_bytes
        
        return {
            'total_gb': total_bytes / (1024**3),
            'free_gb': free_bytes / (1024**3),
            'used_gb': used_bytes / (1024**3),
            'utilization_percent': (used_bytes / total_bytes * 100) if total_bytes > 0 else 0,
            'path': os.path.abspath(path)
        }
    except Exception as e:
        logger.error(f"Failed to get disk space for {path}: {e}")
        return {
            'total_gb': 0,
            'free_gb': 0,
            'used_gb': 0,
            'utilization_percent': 0,
            'path': path
        }


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available via PyTorch.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch not installed, cannot check CUDA availability")
        return False


def get_cuda_device_count() -> int:
    """
    Get number of CUDA devices via PyTorch.
    
    Returns:
        Number of CUDA devices
    """
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def get_full_hardware_info(safety_margin_gb: int = 3) -> HardwareInfo:
    """
    Get complete hardware information.
    
    Args:
        safety_margin_gb: VRAM safety margin in GB
        
    Returns:
        HardwareInfo object with all hardware details
    """
    gpu_info = get_gpu_info(safety_margin_gb)
    ram_info = get_system_ram()
    
    gpus = gpu_info.get('gpus', [])
    
    return HardwareInfo(
        gpus=gpus,
        system_ram_gb=ram_info['total_ram_gb'],
        available_ram_gb=ram_info['available_ram_gb'],
        cuda_version=gpu_info.get('cuda_version', 'Unknown'),
        driver_version=gpu_info.get('driver_version', 'Unknown')
    )


def print_hardware_summary(info: HardwareInfo = None):
    """Print a formatted summary of hardware information"""
    if info is None:
        info = get_full_hardware_info()
    
    print("\n" + "=" * 60)
    print("HARDWARE SUMMARY")
    print("=" * 60)
    
    print(f"\nCUDA Version: {info.cuda_version}")
    print(f"Driver Version: {info.driver_version}")
    print(f"System RAM: {info.system_ram_gb:.2f} GB (Available: {info.available_ram_gb:.2f} GB)")
    
    if info.gpus:
        print(f"\nGPUs Detected: {len(info.gpus)}")
        for gpu in info.gpus:
            print(f"\n  [{gpu.index}] {gpu.name}")
            print(f"      VRAM: {gpu.total_vram_gb:.2f} GB total, {gpu.free_vram_gb:.2f} GB free")
            print(f"      Temperature: {gpu.temperature}°C")
            print(f"      Power: {gpu.power_draw}W / {gpu.power_limit}W")
    else:
        print("\n⚠️  No GPUs detected!")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test the hardware detector
    print_hardware_summary()
