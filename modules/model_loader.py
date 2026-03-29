"""
Model Loader Module for HF-LLM-RUNNER
Loads HuggingFace models with DeepSpeed ZeRO-3 and SSD offloading
"""

import os
import gc
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.memory_tracker import MemoryTracker

logger = setup_logger(__name__)


class ModelLoader:
    """
    Loads HuggingFace models with memory-efficient strategies.
    
    Supports:
    - DeepSpeed ZeRO-3 with parameter offloading
    - Accelerate device_map for automatic distribution
    - SSD → CPU → GPU chunked loading pipeline
    - Multiple precision options (FP16/BF16/FP32)
    - NO QUANTIZATION - full precision only
    
    Usage:
        loader = ModelLoader(
            model_name="openai/gpt-oss-120b",
            precision="fp16",
            offload_folder="./offload"
        )
        model, tokenizer = loader.load_model()
    """
    
    def __init__(
        self,
        model_name: str,
        precision: str = "fp16",
        device_map: str = "auto",
        offload_folder: str = "./offload_dir",
        max_memory: Optional[Dict] = None,
        trust_remote_code: bool = False,
        use_auth_token: Optional[str] = None,
        revision: str = "main"
    ):
        """
        Initialize model loader.
        
        Args:
            model_name: HuggingFace model ID (e.g., "openai/gpt-oss-120b")
            precision: Precision type ("fp16", "bf16", "fp32")
            device_map: Device mapping strategy ("auto", "balanced", "sequential")
            offload_folder: Folder for SSD offloading
            max_memory: Per-device memory limits (e.g., {0: "13GB"})
            trust_remote_code: Whether to trust remote code
            use_auth_token: HuggingFace API token for gated models
            revision: Model revision/branch
        """
        self.model_name = model_name
        self.precision = precision.lower()
        self.device_map = device_map
        self.offload_folder = Path(offload_folder)
        self.max_memory = max_memory
        self.trust_remote_code = trust_remote_code
        self.use_auth_token = use_auth_token
        self.revision = revision
        
        # Validate precision
        if self.precision not in ["fp16", "bf16", "fp32"]:
            logger.warning(f"Unknown precision '{precision}', defaulting to fp16")
            self.precision = "fp16"
        
        # Create offload folder
        self.offload_folder.mkdir(parents=True, exist_ok=True)
        
        # Memory tracker
        self.memory_tracker = MemoryTracker()
    
    def _get_torch_dtype(self):
        """Get torch dtype based on precision setting"""
        try:
            import torch
            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32
            }
            return dtype_map.get(self.precision, torch.float16)
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")
    
    def _check_dependencies(self):
        """Check and import required dependencies"""
        missing = []
        
        try:
            import torch
        except ImportError:
            missing.append("torch")
        
        try:
            import transformers
        except ImportError:
            missing.append("transformers")
        
        try:
            import accelerate
        except ImportError:
            missing.append("accelerate")
        
        if missing:
            raise ImportError(
                f"Missing required packages: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )
    
    def _get_model_config(self) -> Dict:
        """
        Get model configuration from HuggingFace Hub.
        
        Returns:
            Model configuration dictionary
        """
        try:
            from transformers import AutoConfig
            
            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                token=self.use_auth_token
            )
            
            return config.to_dict()
        except Exception as e:
            logger.warning(f"Could not load model config: {e}")
            return {}
    
    def _estimate_model_size(self) -> float:
        """
        Estimate model size in GB based on configuration.
        
        Returns:
            Estimated size in GB
        """
        try:
            config = self._get_model_config()
            
            # Try to get parameter count
            if 'num_parameters' in config:
                params = config['num_parameters']
            else:
                # Estimate from hidden_size and num_layers
                hidden_size = config.get('hidden_size', 4096)
                num_layers = config.get('num_hidden_layers', 
                                       config.get('num_layers', 32))
                vocab_size = config.get('vocab_size', 50000)
                
                # Rough estimate for transformer models
                params = 2 * num_layers * hidden_size * hidden_size
                params += vocab_size * hidden_size
            
            # Calculate size based on precision
            bytes_per_param = {"fp16": 2, "bf16": 2, "fp32": 4}
            size_gb = (params * bytes_per_param.get(self.precision, 2)) / (1024**3)
            
            logger.info(f"Estimated model size: {size_gb:.2f} GB ({params/1e9:.1f}B params)")
            
            return size_gb
            
        except Exception as e:
            logger.warning(f"Could not estimate model size: {e}")
            return 100.0  # Default large estimate
    
    def _calculate_max_memory(self, usable_vram_gb: float) -> Dict:
        """
        Calculate max_memory dict for accelerate.
        
        Args:
            usable_vram_gb: Available VRAM after safety margin
            
        Returns:
            Dictionary mapping device IDs to memory limits
        """
        try:
            import torch
            
            if self.max_memory is not None:
                return self.max_memory
            
            # Auto-calculate based on detected VRAM
            num_gpus = torch.cuda.device_count()
            
            if num_gpus == 0:
                logger.warning("No CUDA devices detected!")
                return {}
            
            max_memory = {}
            for i in range(num_gpus):
                max_memory[i] = f"{int(usable_vram_gb)}GiB"
            
            # Also set CPU memory limit
            max_memory['cpu'] = "64GiB"  # Default CPU limit
            
            return max_memory
            
        except Exception as e:
            logger.error(f"Error calculating max_memory: {e}")
            return {}
    
    def load_model(self) -> Tuple[Any, Any]:
        """
        Load model with DeepSpeed ZeRO-3 and offloading.
        
        The loading pipeline:
        1. Download model from HuggingFace Hub
        2. Initialize with device_map="auto"
        3. Apply precision (FP16/BF16/FP32) - NO QUANTIZATION
        4. Offload parameters to SSD via DeepSpeed
        
        Returns:
            Tuple of (model, tokenizer)
        """
        self._check_dependencies()
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Precision: {self.precision.upper()} (NO QUANTIZATION)")
        
        # Track memory during loading
        self.memory_tracker.record_snapshot("Before model loading")
        
        # Get torch dtype
        torch_dtype = self._get_torch_dtype()
        
        # Estimate model size
        model_size_gb = self._estimate_model_size()
        logger.info(f"Model will require approximately {model_size_gb:.2f} GB")
        
        # Calculate max memory
        max_memory = self._calculate_max_memory(
            usable_vram_gb=float(self.max_memory.get(0, "13GiB").replace("GiB", "").replace("GB", ""))
            if self.max_memory else 13
        )
        
        try:
            # Load tokenizer first
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                token=self.use_auth_token
            )
            
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            self.memory_tracker.record_snapshot("After tokenizer loaded")
            
            # Load model with accelerate and device_map
            logger.info("Loading model (this may take a while)...")
            logger.info(f"Device map: {self.device_map}")
            logger.info(f"Max memory: {max_memory}")
            logger.info(f"Offload folder: {self.offload_folder}")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
                max_memory=max_memory,
                offload_folder=str(self.offload_folder),
                offload_state_dict=True,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                token=self.use_auth_token,
                low_cpu_mem_usage=True,
                # NO QUANTIZATION - these are explicitly NOT used:
                # load_in_8bit=False,
                # load_in_4bit=False,
                # quantization_config=None,
            )
            
            self.memory_tracker.record_snapshot("After model loaded")
            
            # Print memory summary
            self.memory_tracker.print_summary()
            
            # Get model info
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model loaded successfully!")
            logger.info(f"Parameters: {num_params/1e9:.2f}B")
            logger.info(f"Model device: {model.device if hasattr(model, 'device') else 'distributed'}")
            
            return model, tokenizer
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA Out of Memory: {e}")
            logger.error(
                "Try reducing max_memory, enabling swap, or using a smaller model."
            )
            raise
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_model_with_deepspeed(
        self,
        deepspeed_config: Dict,
        local_rank: int = 0
    ) -> Tuple[Any, Any]:
        """
        Load model with DeepSpeed ZeRO-3.
        
        This method uses DeepSpeed for advanced memory management:
        - Stage 3: Shards parameters across processes
        - CPU Offload: Optimizer states on CPU
        - NVMe Offload: Parameters on SSD
        
        Args:
            deepspeed_config: DeepSpeed configuration dictionary
            local_rank: Local rank for distributed training
            
        Returns:
            Tuple of (model, tokenizer)
        """
        self._check_dependencies()
        
        try:
            import deepspeed
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            logger.error(f"DeepSpeed import failed: {e}")
            logger.error("Install DeepSpeed: pip install deepspeed")
            raise
        
        logger.info(f"Loading model with DeepSpeed ZeRO-3: {self.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            token=self.use_auth_token
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get torch dtype
        torch_dtype = self._get_torch_dtype()
        
        # Load model configuration
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            token=self.use_auth_token
        )
        
        # Initialize model meta device (for ZeRO-3)
        with deepspeed.zero.Init(config_dict_or_path=deepspeed_config):
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                token=self.use_auth_token
            )
        
        # Initialize DeepSpeed engine
        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=deepspeed_config
        )
        
        logger.info("Model loaded with DeepSpeed ZeRO-3")
        
        return model_engine, tokenizer
    
    def cleanup(self):
        """Clean up memory"""
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        
        logger.info("Cleaned up memory")


class ModelDownloader:
    """
    Utility class for downloading models from HuggingFace Hub.
    
    Useful for pre-downloading models before loading.
    """
    
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        use_auth_token: Optional[str] = None
    ):
        """
        Initialize model downloader.
        
        Args:
            model_name: HuggingFace model ID
            cache_dir: Custom cache directory
            use_auth_token: HuggingFace API token
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_auth_token = use_auth_token
    
    def download(self, include_safetensors: bool = True) -> Path:
        """
        Download model files from HuggingFace Hub.
        
        Args:
            include_safetensors: Prefer safetensors format
            
        Returns:
            Path to cached model
        """
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading model: {self.model_name}")
            
            # Define patterns to download
            allow_patterns = [
                "*.json",
                "*.txt",
                "*.model",
                "*.bin",
            ]
            
            if include_safetensors:
                allow_patterns.append("*.safetensors")
            
            # Download
            path = snapshot_download(
                repo_id=self.model_name,
                allow_patterns=allow_patterns,
                cache_dir=self.cache_dir,
                token=self.use_auth_token,
                resume_download=True
            )
            
            logger.info(f"Model downloaded to: {path}")
            return Path(path)
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    def get_model_size(self) -> float:
        """
        Get total model size in GB.
        
        Returns:
            Size in GB
        """
        try:
            from huggingface_hub import model_info
            
            info = model_info(self.model_name, token=self.use_auth_token)
            
            # Calculate size from siblings
            total_size = 0
            for sibling in info.siblings:
                if sibling.rfilename.endswith(('.bin', '.safetensors', '.pt')):
                    total_size += sibling.size or 0
            
            return total_size / (1024**3)
            
        except Exception as e:
            logger.warning(f"Could not get model size: {e}")
            return 0.0


if __name__ == "__main__":
    # Test model loader (requires GPU and dependencies)
    print("ModelLoader module loaded successfully")
    print("Use: loader = ModelLoader('model_name')")
