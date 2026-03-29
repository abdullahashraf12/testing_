"""
Swap Manager Module for HF-LLM-RUNNER
Creates and manages swap files for additional memory during large model loading
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SwapManager:
    """
    Manages swap file creation and cleanup for additional memory.
    
    This is essential for running large models (70B+) where:
    - Model parameters exceed available VRAM
    - CPU RAM is insufficient for full offloading
    - SSD/NVMe storage is used as extended memory
    
    Usage:
        swap_manager = SwapManager(swap_path="./swap", swap_size_gb=100)
        swap_manager.create_swap()
        # ... load and run model ...
        swap_manager.remove_swap()
    """
    
    def __init__(
        self,
        swap_path: str = "./model_swap",
        swap_size_gb: Optional[int] = None
    ):
        """
        Initialize swap manager.
        
        Args:
            swap_path: Directory path for swap file (default: ./model_swap)
            swap_size_gb: Size in GB (auto-calculated if None)
        """
        self.swap_path = Path(swap_path).resolve()
        self.swap_file = self.swap_path / "swapfile"
        self.swap_size_gb = swap_size_gb
        self._swap_created = False
        self._original_swap_info = None
    
    def calculate_recommended_swap(
        self,
        model_params_billion: float,
        precision: str = "fp16",
        safety_factor: float = 1.5
    ) -> int:
        """
        Calculate recommended swap size based on model size.
        
        Formula:
        - FP16: model_size_gb = params_billion * 2
        - BF16: model_size_gb = params_billion * 2
        - FP32: model_size_gb = params_billion * 4
        - recommended_swap = model_size_gb * safety_factor
        
        Args:
            model_params_billion: Model parameters in billions (e.g., 120 for 120B)
            precision: Precision type (fp16, bf16, fp32)
            safety_factor: Multiplier for safety buffer (default: 1.5)
            
        Returns:
            Recommended swap size in GB
        """
        # Bytes per parameter based on precision
        bytes_per_param = {
            'fp16': 2,
            'bf16': 2,
            'fp32': 4
        }.get(precision.lower(), 2)
        
        # Calculate model size in GB
        model_size_gb = model_params_billion * bytes_per_param
        
        # Calculate recommended swap with safety factor
        recommended_swap = int(model_size_gb * safety_factor)
        
        logger.info(
            f"Calculated swap size: {recommended_swap} GB "
            f"(Model: {model_params_billion}B params, {precision.upper()}, "
            f"Size: {model_size_gb:.1f} GB, Factor: {safety_factor}x)"
        )
        
        return recommended_swap
    
    def get_current_swap_info(self) -> Dict:
        """
        Get current system swap information.
        
        Returns:
            Dictionary with swap info
        """
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    key = parts[0].rstrip(':')
                    value = int(parts[1])  # in KB
                    meminfo[key] = value
                
                total_swap_kb = meminfo.get('SwapTotal', 0)
                free_swap_kb = meminfo.get('SwapFree', 0)
                
                return {
                    'total_swap_gb': total_swap_kb / (1024 * 1024),
                    'free_swap_gb': free_swap_kb / (1024 * 1024),
                    'used_swap_gb': (total_swap_kb - free_swap_kb) / (1024 * 1024)
                }
        except Exception as e:
            logger.warning(f"Could not get swap info: {e}")
            return {'total_swap_gb': 0, 'free_swap_gb': 0, 'used_swap_gb': 0}
    
    def check_disk_space(self, required_gb: int) -> bool:
        """
        Check if there's enough disk space for swap file.
        
        Args:
            required_gb: Required space in GB
            
        Returns:
            True if enough space available
        """
        # Ensure parent directory exists
        parent_dir = self.swap_path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        
        stat = os.statvfs(parent_dir)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        
        if free_gb < required_gb:
            logger.error(
                f"Insufficient disk space: {free_gb:.2f} GB available, "
                f"{required_gb} GB required"
            )
            return False
        
        logger.info(f"Disk space check passed: {free_gb:.2f} GB available")
        return True
    
    def create_swap(self) -> bool:
        """
        Create swap file using system commands.
        
        Steps:
        1. Create swap directory
        2. Create empty file with fallocate
        3. Set permissions to 600
        4. Format as swap with mkswap
        5. Enable with swapon
        
        Returns:
            True if successful, False otherwise
        """
        if self._swap_created:
            logger.warning("Swap already created")
            return True
        
        if self.swap_size_gb is None:
            logger.error("Swap size not specified. Set swap_size_gb or call calculate_recommended_swap()")
            return False
        
        # Store original swap info
        self._original_swap_info = self.get_current_swap_info()
        
        # Check disk space
        if not self.check_disk_space(self.swap_size_gb):
            return False
        
        try:
            # Create swap directory
            self.swap_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created swap directory: {self.swap_path}")
            
            # Create swap file using fallocate
            swap_size_bytes = self.swap_size_gb * 1024**3
            logger.info(f"Creating swap file: {self.swap_file} ({self.swap_size_gb} GB)")
            
            # Use fallocate for fast file creation
            result = subprocess.run(
                ['fallocate', '-l', f'{self.swap_size_gb}G', str(self.swap_file)],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                # Fallback to dd if fallocate fails
                logger.warning(f"fallocate failed, falling back to dd: {result.stderr}")
                result = subprocess.run(
                    ['dd', 'if=/dev/zero', f'of={self.swap_file}',
                     f'bs=1G', f'count={self.swap_size_gb}'],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode != 0:
                    logger.error(f"Failed to create swap file: {result.stderr}")
                    return False
            
            # Set permissions
            os.chmod(self.swap_file, 0o600)
            logger.info("Set swap file permissions to 600")
            
            # Format as swap
            result = subprocess.run(
                ['mkswap', str(self.swap_file)],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"mkswap failed: {result.stderr}")
                self._cleanup_swap_file()
                return False
            
            logger.info("Formatted swap file")
            
            # Enable swap
            result = subprocess.run(
                ['swapon', str(self.swap_file)],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"swapon failed: {result.stderr}")
                self._cleanup_swap_file()
                return False
            
            self._swap_created = True
            logger.info(f"✓ Swap file successfully created and enabled: {self.swap_file}")
            
            # Verify swap is active
            new_swap_info = self.get_current_swap_info()
            logger.info(
                f"Swap status: {new_swap_info['total_swap_gb']:.2f} GB total, "
                f"{new_swap_info['free_swap_gb']:.2f} GB free"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create swap: {e}")
            self._cleanup_swap_file()
            return False
    
    def _cleanup_swap_file(self):
        """Remove swap file if it exists"""
        try:
            if self.swap_file.exists():
                self.swap_file.unlink()
                logger.info(f"Removed swap file: {self.swap_file}")
        except Exception as e:
            logger.warning(f"Could not remove swap file: {e}")
    
    def remove_swap(self) -> bool:
        """
        Remove swap file and clean up.
        
        Steps:
        1. Disable swap with swapoff
        2. Remove swap file
        
        Returns:
            True if successful
        """
        if not self._swap_created:
            logger.info("No swap to remove")
            return True
        
        try:
            # Disable swap
            result = subprocess.run(
                ['swapoff', str(self.swap_file)],
                capture_output=True,
                text=True,
                check=False,
                timeout=300  # 5 minute timeout for swapoff
            )
            
            if result.returncode != 0:
                logger.warning(f"swapoff returned non-zero: {result.stderr}")
                # Try to continue with cleanup anyway
            
            logger.info(f"Disabled swap: {self.swap_file}")
            
            # Remove swap file
            self._cleanup_swap_file()
            
            # Remove directory if empty
            try:
                if self.swap_path.exists() and not any(self.swap_path.iterdir()):
                    self.swap_path.rmdir()
                    logger.info(f"Removed empty swap directory: {self.swap_path}")
            except Exception:
                pass
            
            self._swap_created = False
            logger.info("✓ Swap cleanup complete")
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("swapoff timed out - swap may still be in heavy use")
            return False
        except Exception as e:
            logger.error(f"Failed to remove swap: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry - creates swap"""
        self.create_swap()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - removes swap"""
        self.remove_swap()
        return False
    
    @property
    def is_active(self) -> bool:
        """Check if swap is currently active"""
        return self._swap_created


def estimate_model_memory_requirements(
    model_params_billion: float,
    precision: str = "fp16",
    include_kv_cache: bool = True,
    kv_cache_tokens: int = 4096
) -> Dict:
    """
    Estimate total memory requirements for a model.
    
    Args:
        model_params_billion: Model parameters in billions
        precision: Precision type (fp16, bf16, fp32)
        include_kv_cache: Whether to include KV cache estimate
        kv_cache_tokens: Number of tokens for KV cache estimation
        
    Returns:
        Dictionary with memory requirements breakdown
    """
    bytes_per_param = {
        'fp16': 2,
        'bf16': 2,
        'fp32': 4
    }.get(precision.lower(), 2)
    
    # Model weights
    model_weights_gb = model_params_billion * bytes_per_param
    
    # KV cache (approximately 2 bytes per parameter per token for attention)
    kv_cache_gb = 0
    if include_kv_cache:
        # Rough estimate: 2 * num_layers * hidden_size * seq_len * 2 bytes
        # For a 120B model with ~80 layers and ~8k hidden size:
        kv_cache_gb = (model_params_billion / 100) * (kv_cache_tokens / 1000) * 0.5
    
    # Activation checkpoints (approximately 10-20% of model size)
    activations_gb = model_weights_gb * 0.15
    
    total_gb = model_weights_gb + kv_cache_gb + activations_gb
    
    return {
        'model_weights_gb': model_weights_gb,
        'kv_cache_gb': kv_cache_gb,
        'activations_gb': activations_gb,
        'total_gb': total_gb,
        'recommended_min_ram_gb': total_gb * 0.25,  # At least 25% in RAM
        'recommended_swap_gb': total_gb * 1.5,  # 1.5x for safety
    }


if __name__ == "__main__":
    # Test swap manager
    print("Testing SwapManager...")
    
    # Calculate recommended swap for 120B model
    manager = SwapManager(swap_size_gb=10)  # Small test size
    recommended = manager.calculate_recommended_swap(
        model_params_billion=120,
        precision="fp16"
    )
    print(f"Recommended swap for 120B FP16 model: {recommended} GB")
    
    # Estimate memory requirements
    mem_req = estimate_model_memory_requirements(120, "fp16")
    print(f"\nMemory requirements for 120B FP16 model:")
    for key, value in mem_req.items():
        print(f"  {key}: {value:.2f} GB")
