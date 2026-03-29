"""
DeepSpeed Configuration Generator for HF-LLM-RUNNER
Generates ZeRO-3 configuration dynamically based on available hardware
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class OffloadConfig:
    """Offloading configuration"""
    device: str = "cpu"
    pin_memory: bool = True
    nvme_path: Optional[str] = None
    buffer_size: float = 4e9  # 4 GB buffer
    buffer_count: int = 5
    fast_init: bool = False


@dataclass
class ZeROConfig:
    """ZeRO optimization configuration"""
    stage: int = 3
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int = 16777216
    stage3_prefetch_bucket_size: int = 1549639
    stage3_param_persistence_threshold: int = 102400
    stage3_max_live_parameters: int = 100000000
    stage3_max_reuse_distance: int = 100000000
    stage3_gather_16bit_weights_on_model_save: bool = True
    offload_optimizer: Optional[Dict] = None
    offload_param: Optional[Dict] = None


@dataclass
class FP16Config:
    """FP16 mixed precision configuration"""
    enabled: bool = True
    loss_scale: int = 0
    loss_scale_window: int = 1000
    hysteresis: int = 2
    min_loss_scale: int = 1


@dataclass
class BF16Config:
    """BF16 mixed precision configuration"""
    enabled: bool = False


@dataclass
class ActivationCheckpointingConfig:
    """Activation checkpointing configuration"""
    partition_activations: bool = True
    cpu_checkpointing: bool = True
    contiguous_memory_optimization: bool = True
    number_checkpoints: int = 16
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False


class DeepSpeedConfigGenerator:
    """
    Generates DeepSpeed ZeRO-3 configuration for large model inference.
    
    The configuration is dynamically generated based on:
    - Available VRAM
    - System RAM
    - SSD/NVMe offload path
    - Precision (FP16/BF16/FP32)
    
    Key Features:
    - ZeRO Stage 3: Shards parameters, gradients, and optimizer states
    - CPU Offloading: Optimizer states on CPU RAM
    - NVMe Offloading: Parameters on SSD for large models
    - No Quantization: Full precision (FP16/BF16/FP32)
    """
    
    def __init__(
        self,
        usable_vram_gb: float,
        system_ram_gb: float,
        offload_path: str = "./offload_dir",
        precision: str = "fp16"
    ):
        """
        Initialize DeepSpeed config generator.
        
        Args:
            usable_vram_gb: Available VRAM in GB (after safety margin)
            system_ram_gb: Total system RAM in GB
            offload_path: Path for NVMe/SSD offloading
            precision: Precision type ("fp16", "bf16", "fp32")
        """
        self.usable_vram_gb = usable_vram_gb
        self.system_ram_gb = system_ram_gb
        self.offload_path = Path(offload_path).resolve()
        self.precision = precision.lower()
        
        # Validate precision
        if self.precision not in ["fp16", "bf16", "fp32"]:
            logger.warning(f"Unknown precision '{precision}', defaulting to fp16")
            self.precision = "fp16"
    
    def _calculate_buffer_sizes(self) -> Dict[str, int]:
        """
        Calculate optimal buffer sizes based on available memory.
        
        Returns:
            Dictionary with buffer size configurations
        """
        # Larger systems can use larger buffers for better throughput
        if self.system_ram_gb >= 128:
            buffer_size = int(8e9)  # 8 GB
            buffer_count = 10
        elif self.system_ram_gb >= 64:
            buffer_size = int(4e9)  # 4 GB
            buffer_count = 5
        elif self.system_ram_gb >= 32:
            buffer_size = int(2e9)  # 2 GB
            buffer_count = 5
        else:
            buffer_size = int(1e9)  # 1 GB
            buffer_count = 3
        
        return {
            'buffer_size': buffer_size,
            'buffer_count': buffer_count
        }
    
    def _calculate_bucket_sizes(self) -> Dict[str, int]:
        """
        Calculate optimal bucket sizes for gradient reduction.
        
        Larger buckets improve throughput but use more memory.
        
        Returns:
            Dictionary with bucket size configurations
        """
        # Scale bucket sizes based on VRAM
        if self.usable_vram_gb >= 40:
            reduce_bucket_size = 16777216 * 2  # 2x for large VRAM
            prefetch_bucket_size = 1549639 * 2
        elif self.usable_vram_gb >= 20:
            reduce_bucket_size = 16777216
            prefetch_bucket_size = 1549639
        else:
            reduce_bucket_size = 16777216 // 2  # Smaller for limited VRAM
            prefetch_bucket_size = 1549639 // 2
        
        return {
            'reduce_bucket_size': reduce_bucket_size,
            'prefetch_bucket_size': prefetch_bucket_size
        }
    
    def generate_offload_optimizer_config(self) -> Dict:
        """
        Generate optimizer offloading configuration.
        
        Optimizer states are offloaded to CPU for memory efficiency.
        """
        return {
            "device": "cpu",
            "pin_memory": True
        }
    
    def generate_offload_param_config(self, use_nvme: bool = True) -> Dict:
        """
        Generate parameter offloading configuration.
        
        For large models, parameters are offloaded to NVMe/SSD.
        The data flow is: SSD → CPU → GPU
        
        Args:
            use_nvme: Whether to use NVMe/SSD offloading
        """
        buffer_config = self._calculate_buffer_sizes()
        
        if use_nvme:
            return {
                "device": "nvme",
                "nvme_path": str(self.offload_path),
                "pin_memory": True,
                "buffer_size": buffer_config['buffer_size'],
                "buffer_count": buffer_config['buffer_count'],
                "fast_init": False
            }
        else:
            return {
                "device": "cpu",
                "pin_memory": True
            }
    
    def generate_zero_config(self, use_nvme_offload: bool = True) -> Dict:
        """
        Generate ZeRO-3 optimization configuration.
        
        This is the core of memory-efficient large model loading:
        - Stage 3: Shards parameters across devices
        - Offloading: Moves data to CPU/SSD when not actively computing
        - Overlapping: Hides communication latency
        
        Args:
            use_nvme_offload: Whether to enable NVMe offloading
        """
        bucket_config = self._calculate_bucket_sizes()
        
        config = {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": bucket_config['reduce_bucket_size'],
            "stage3_prefetch_bucket_size": bucket_config['prefetch_bucket_size'],
            "stage3_param_persistence_threshold": 102400,
            "stage3_max_live_parameters": 100000000,
            "stage3_max_reuse_distance": 100000000,
            "stage3_gather_16bit_weights_on_model_save": True,
            
            # Optimizer offloading to CPU
            "offload_optimizer": self.generate_offload_optimizer_config(),
            
            # Parameter offloading to NVMe/SSD or CPU
            "offload_param": self.generate_offload_param_config(use_nvme_offload)
        }
        
        return config
    
    def generate_fp16_config(self) -> Dict:
        """Generate FP16 mixed precision configuration"""
        if self.precision == "fp16":
            return {
                "enabled": True,
                "loss_scale": 0,  # Dynamic loss scaling
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
        return {"enabled": False}
    
    def generate_bf16_config(self) -> Dict:
        """Generate BF16 mixed precision configuration"""
        if self.precision == "bf16":
            return {"enabled": True}
        return {"enabled": False}
    
    def generate_activation_checkpointing_config(self) -> Dict:
        """
        Generate activation checkpointing configuration.
        
        This trades compute for memory by recomputing activations
        during backward pass instead of storing them.
        """
        # Enable more aggressive checkpointing for limited memory
        if self.usable_vram_gb < 24:
            number_checkpoints = 32  # More checkpoints for less VRAM
        elif self.usable_vram_gb < 48:
            number_checkpoints = 16
        else:
            number_checkpoints = 8
        
        return {
            "partition_activations": True,
            "cpu_checkpointing": True,  # Checkpoint to CPU
            "contiguous_memory_optimization": True,
            "number_checkpoints": number_checkpoints,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }
    
    def generate_config(
        self,
        use_nvme_offload: bool = True,
        enable_activation_checkpointing: bool = True,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1
    ) -> Dict:
        """
        Generate complete DeepSpeed configuration.
        
        Args:
            use_nvme_offload: Enable NVMe/SSD parameter offloading
            enable_activation_checkpointing: Enable activation checkpointing
            train_batch_size: Training batch size (1 for inference)
            gradient_accumulation_steps: Gradient accumulation steps
            
        Returns:
            Complete DeepSpeed configuration dictionary
        """
        config = {
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            
            # Optimizer (AdamW is standard for LLMs)
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.0
                }
            },
            
            # Precision settings
            "fp16": self.generate_fp16_config(),
            "bf16": self.generate_bf16_config(),
            
            # ZeRO-3 optimization
            "zero_optimization": self.generate_zero_config(use_nvme_offload),
            
            # Activation checkpointing
            "activation_checkpointing": (
                self.generate_activation_checkpointing_config() 
                if enable_activation_checkpointing 
                else None
            ),
            
            # Other settings
            "gradient_clipping": 1.0,
            "steps_per_print": 10,
            "wall_clock_breakdown": False
        }
        
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        return config
    
    def save_config(self, output_path: str, config: Optional[Dict] = None) -> str:
        """
        Save configuration to JSON file.
        
        Args:
            output_path: Path to save configuration
            config: Configuration to save (generates if None)
            
        Returns:
            Path to saved configuration
        """
        if config is None:
            config = self.generate_config()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved DeepSpeed config to: {output_path}")
        return str(output_path)


def generate_deepspeed_config(
    usable_vram_gb: float,
    system_ram_gb: float,
    swap_path: str,
    precision: str = "fp16",
    use_nvme_offload: bool = True,
    output_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to generate DeepSpeed configuration.
    
    Args:
        usable_vram_gb: Available VRAM in GB (after safety margin)
        system_ram_gb: Total system RAM in GB
        swap_path: Path for NVMe/SSD offloading
        precision: Precision type ("fp16", "bf16", "fp32")
        use_nvme_offload: Enable NVMe offloading
        output_path: Optional path to save configuration
        
    Returns:
        DeepSpeed configuration dictionary
    """
    generator = DeepSpeedConfigGenerator(
        usable_vram_gb=usable_vram_gb,
        system_ram_gb=system_ram_gb,
        offload_path=swap_path,
        precision=precision
    )
    
    config = generator.generate_config(use_nvme_offload=use_nvme_offload)
    
    if output_path:
        generator.save_config(output_path, config)
    
    return config


def print_config_summary(config: Dict):
    """Print a human-readable summary of the DeepSpeed configuration"""
    print("\n" + "=" * 60)
    print("DEEPSPEED CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"\nBatch Size: {config.get('train_batch_size', 1)}")
    
    # Precision
    fp16 = config.get('fp16', {})
    bf16 = config.get('bf16', {})
    if fp16.get('enabled'):
        print("Precision: FP16 (mixed precision)")
    elif bf16.get('enabled'):
        print("Precision: BF16 (bfloat16)")
    else:
        print("Precision: FP32 (full precision)")
    
    # ZeRO
    zero = config.get('zero_optimization', {})
    print(f"\nZeRO Stage: {zero.get('stage', 'N/A')}")
    
    # Offloading
    offload_opt = zero.get('offload_optimizer', {})
    offload_param = zero.get('offload_param', {})
    
    print(f"\nOptimizer Offload: {offload_opt.get('device', 'N/A')}")
    print(f"Parameter Offload: {offload_param.get('device', 'N/A')}")
    
    if offload_param.get('device') == 'nvme':
        print(f"  NVMe Path: {offload_param.get('nvme_path', 'N/A')}")
        buffer_size_gb = offload_param.get('buffer_size', 0) / 1e9
        print(f"  Buffer Size: {buffer_size_gb:.1f} GB")
        print(f"  Buffer Count: {offload_param.get('buffer_count', 'N/A')}")
    
    # Activation checkpointing
    act_ckpt = config.get('activation_checkpointing', {})
    if act_ckpt:
        print(f"\nActivation Checkpointing: Enabled")
        print(f"  CPU Checkpointing: {act_ckpt.get('cpu_checkpointing', False)}")
        print(f"  Number of Checkpoints: {act_ckpt.get('number_checkpoints', 'N/A')}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test configuration generation
    print("Generating test DeepSpeed configuration...")
    
    config = generate_deepspeed_config(
        usable_vram_gb=13,  # 16GB GPU - 3GB margin
        system_ram_gb=64,
        swap_path="./offload_dir",
        precision="fp16",
        output_path="./test_deepspeed_config.json"
    )
    
    print_config_summary(config)
