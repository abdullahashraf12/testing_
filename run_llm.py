#!/usr/bin/env python3
"""
HF-LLM-RUNNER: Run any HuggingFace LLM with DeepSpeed ZeRO-3 SSD Offloading

This script enables running ANY HuggingFace language model on ANY NVIDIA GPU,
regardless of VRAM limitations, using DeepSpeed ZeRO-3 with CPU/SSD offloading.

Key Features:
- NO QUANTIZATION - Full FP16/BF16/FP32 precision
- Dynamic VRAM detection via nvidia-smi
- Optional SSD swap for extended memory
- SSD → CPU → GPU chunked loading pipeline
- Works with ANY HuggingFace model

Usage:
    python run_llm.py --model openai/gpt-oss-120b --use-swap --swap-size 200
    
    python run_llm.py --model meta-llama/Llama-2-70b-hf --precision bf16
    
    python run_llm.py --config config.json --prompt "Hello, world!"

Author: HF-LLM-RUNNER
License: MIT
"""

import os
import sys
import json
import argparse
import signal
import time
from pathlib import Path
from typing import Dict, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import modules
from modules.hardware_detector import (
    get_gpu_info, 
    get_system_ram, 
    get_disk_space,
    get_full_hardware_info,
    print_hardware_summary,
    HardwareInfo
)
from modules.swap_manager import SwapManager, estimate_model_memory_requirements
from modules.deepspeed_generator import (
    generate_deepspeed_config, 
    DeepSpeedConfigGenerator,
    print_config_summary
)
from modules.model_loader import ModelLoader
from modules.inference_engine import InferenceEngine, LongTextGenerator
from utils.logger import setup_logger
from utils.memory_tracker import MemoryTracker

# Setup logger
logger = setup_logger(__name__)


# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run any HuggingFace LLM with DeepSpeed ZeRO-3 SSD Offloading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GPT-120B with swap
  python run_llm.py --model openai/gpt-oss-120b --use-swap --swap-size 200
  
  # Run Llama-2-70B with BF16
  python run_llm.py --model meta-llama/Llama-2-70b-hf --precision bf16
  
  # Run with custom prompt
  python run_llm.py --model openai/gpt-oss-120b --prompt "Generate a poem"
  
  # Use config file
  python run_llm.py --config config.json
  
  # Interactive mode (chat)
  python run_llm.py --model mistralai/Mistral-7B-v0.1 --interactive

IMPORTANT:
  - NO QUANTIZATION is used - full precision (FP16/BF16/FP32) only
  - VRAM safety margin: 3GB reserved by default
  - SSD offloading requires fast storage (NVMe recommended)
  - Works on ANY NVIDIA GPU with nvidia-smi installed
        """
    )
    
    # ===================
    # Model arguments
    # ===================
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="openai/gpt-oss-120b",
        help="HuggingFace model ID (default: openai/gpt-oss-120b)"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision/branch (default: main)"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for models requiring custom code"
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default=None,
        help="HuggingFace API token for gated models"
    )
    
    # ===================
    # Precision arguments
    # ===================
    parser.add_argument(
        "--precision", "-p",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Precision: fp16 (default), bf16 (Ampere+), fp32 (max quality, 2x memory)"
    )
    
    # ===================
    # Swap arguments
    # ===================
    parser.add_argument(
        "--use-swap",
        action="store_true",
        help="Enable swap file creation for additional memory"
    )
    parser.add_argument(
        "--swap-size",
        type=int,
        default=None,
        help="Swap size in GB (auto-calculated if not specified)"
    )
    parser.add_argument(
        "--swap-path",
        type=str,
        default="./model_swap",
        help="Path for swap file (default: ./model_swap)"
    )
    
    # ===================
    # Offload arguments
    # ===================
    parser.add_argument(
        "--offload-path",
        type=str,
        default="./offload_dir",
        help="Path for SSD offloading (default: ./offload_dir)"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=4,
        help="Buffer size in GB for NVMe offloading (default: 4)"
    )
    
    # ===================
    # VRAM arguments
    # ===================
    parser.add_argument(
        "--vram-safety-margin",
        type=int,
        default=3,
        help="VRAM safety margin in GB (default: 3)"
    )
    parser.add_argument(
        "--max-vram",
        type=int,
        default=None,
        help="Override max VRAM in GB (auto-detected if not specified)"
    )
    
    # ===================
    # Inference arguments
    # ===================
    parser.add_argument(
        "--prompt",
        type=str,
        default="HI generate 500 lines of random words poem as each word is 50 letters as line consists of 15 words",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens to generate (default: 500)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1)"
    )
    
    # ===================
    # Mode arguments
    # ===================
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Stream output (default: True)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output"
    )
    
    # ===================
    # Config file
    # ===================
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config.json file (overrides CLI arguments)"
    )
    
    # ===================
    # Utility arguments
    # ===================
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print hardware info and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running model"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> Dict:
    """Merge config file with CLI arguments (CLI takes precedence)"""
    merged = {}
    
    # Model settings
    merged['model_name'] = args.model or config.get('model', {}).get('name', 'openai/gpt-oss-120b')
    merged['revision'] = args.revision or config.get('model', {}).get('revision', 'main')
    merged['trust_remote_code'] = args.trust_remote_code or config.get('model', {}).get('trust_remote_code', False)
    merged['auth_token'] = args.auth_token or config.get('model', {}).get('use_auth_token')
    
    # Precision
    merged['precision'] = args.precision or config.get('precision', {}).get('type', 'fp16')
    
    # Memory
    merged['vram_safety_margin'] = args.vram_safety_margin or config.get('memory', {}).get('vram_safety_margin_gb', 3)
    merged['cpu_offload'] = config.get('memory', {}).get('cpu_offload', True)
    merged['nvme_offload'] = config.get('memory', {}).get('nvme_offload', True)
    
    # Offload
    merged['offload_path'] = args.offload_path or config.get('offload', {}).get('offload_path', './offload_dir')
    merged['buffer_size'] = args.buffer_size or config.get('offload', {}).get('buffer_size_gb', 4)
    
    # Swap
    merged['use_swap'] = args.use_swap or config.get('swap', {}).get('enabled', False)
    merged['swap_path'] = args.swap_path or config.get('swap', {}).get('path', './model_swap')
    merged['swap_size'] = args.swap_size
    if merged['swap_size'] is None and config.get('swap', {}).get('size_gb') != 'auto':
        merged['swap_size'] = config.get('swap', {}).get('size_gb')
    
    # Inference
    merged['prompt'] = args.prompt
    merged['max_tokens'] = args.max_tokens or config.get('inference', {}).get('max_new_tokens', 500)
    merged['temperature'] = args.temperature or config.get('inference', {}).get('temperature', 0.7)
    merged['top_p'] = args.top_p or config.get('inference', {}).get('top_p', 0.9)
    merged['top_k'] = args.top_k or config.get('inference', {}).get('top_k', 50)
    merged['repetition_penalty'] = args.repetition_penalty or config.get('inference', {}).get('repetition_penalty', 1.1)
    
    # Stream
    merged['stream'] = not args.no_stream if args.no_stream else True
    
    return merged


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Print banner
    print_banner()
    
    # ===================
    # STEP 0: Info mode
    # ===================
    if args.info:
        print_hardware_summary()
        return 0
    
    # ===================
    # Load configuration
    # ===================
    config = {}
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        merged = merge_config_with_args(config, args)
    else:
        merged = merge_config_with_args({}, args)
    
    # ===================
    # STEP 1: Hardware Detection
    # ===================
    print_step(1, "Hardware Detection")
    
    gpu_info = get_gpu_info(safety_margin_gb=merged['vram_safety_margin'])
    ram_info = get_system_ram()
    disk_info = get_disk_space(merged['offload_path'])
    
    # Print hardware info
    print_hardware_info(gpu_info, ram_info, disk_info)
    
    # Verify CUDA availability
    if gpu_info['total_vram_gb'] == 0:
        logger.error("No NVIDIA GPU detected or nvidia-smi not available!")
        logger.error("This tool requires an NVIDIA GPU with CUDA drivers installed.")
        return 1
    
    # Override max VRAM if specified
    if args.max_vram:
        usable_vram = args.max_vram
        logger.info(f"Override max VRAM: {usable_vram} GB")
    else:
        usable_vram = gpu_info['usable_vram_gb']
    
    # ===================
    # STEP 2: Estimate Model Requirements
    # ===================
    print_step(2, "Estimating Model Requirements")
    
    # Estimate model size (rough estimate based on model name)
    model_params_billion = estimate_params_from_name(merged['model_name'])
    mem_requirements = estimate_model_memory_requirements(
        model_params_billion,
        merged['precision']
    )
    
    print_memory_requirements(mem_requirements, gpu_info, ram_info)
    
    # Check if we need swap
    total_available = usable_vram + ram_info['available_ram_gb']
    if mem_requirements['total_gb'] > total_available * 0.5:
        logger.warning(f"Model may require more memory than available!")
        if not merged['use_swap']:
            logger.warning("Consider using --use-swap for additional memory")
    
    # ===================
    # STEP 3: Swap Management (Optional)
    # ===================
    swap_manager = None
    if merged['use_swap']:
        print_step(3, "Swap Management")
        
        swap_manager = SwapManager(
            swap_path=merged['swap_path'],
            swap_size_gb=merged['swap_size']
        )
        
        # Calculate swap size if not specified
        if merged['swap_size'] is None:
            merged['swap_size'] = swap_manager.calculate_recommended_swap(
                model_params_billion,
                merged['precision']
            )
            logger.info(f"Auto-calculated swap size: {merged['swap_size']} GB")
        
        # Create swap
        if swap_manager.create_swap():
            print(f"✓ Swap file created: {merged['swap_path']} ({merged['swap_size']} GB)")
        else:
            logger.warning("Failed to create swap file, continuing without swap")
            swap_manager = None
    
    # Setup cleanup on exit
    def cleanup_handler(signum=None, frame=None):
        logger.info("Cleaning up...")
        if swap_manager:
            swap_manager.remove_swap()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    # ===================
    # Dry run mode
    # ===================
    if args.dry_run:
        logger.info("Dry run mode - exiting without loading model")
        if swap_manager:
            swap_manager.remove_swap()
        return 0
    
    # ===================
    # STEP 4: Generate DeepSpeed Configuration
    # ===================
    print_step(4, "Generating DeepSpeed Configuration")
    
    ds_config = generate_deepspeed_config(
        usable_vram_gb=usable_vram,
        system_ram_gb=ram_info['total_ram_gb'],
        swap_path=merged['offload_path'],
        precision=merged['precision'],
        use_nvme_offload=merged['nvme_offload']
    )
    
    # Save DeepSpeed config
    ds_config_path = str(PROJECT_ROOT / "deepspeed_config.json")
    with open(ds_config_path, 'w') as f:
        json.dump(ds_config, f, indent=2)
    
    print(f"✓ DeepSpeed config saved to: {ds_config_path}")
    print_config_summary(ds_config)
    
    # ===================
    # STEP 5: Load Model
    # ===================
    print_step(5, "Loading Model")
    print(f"Model: {merged['model_name']}")
    print(f"Precision: {merged['precision'].upper()} (NO QUANTIZATION)")
    print("This may take a while for large models...")
    
    # Calculate max memory
    max_memory = {0: f"{int(usable_vram)}GB"}
    
    try:
        loader = ModelLoader(
            model_name=merged['model_name'],
            precision=merged['precision'],
            device_map="auto",
            offload_folder=merged['offload_path'],
            max_memory=max_memory,
            trust_remote_code=merged['trust_remote_code'],
            use_auth_token=merged['auth_token'],
            revision=merged['revision']
        )
        
        model, tokenizer = loader.load_model()
        
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        if swap_manager:
            swap_manager.remove_swap()
        return 1
    
    # ===================
    # STEP 6: Inference
    # ===================
    print_step(6, "Running Inference")
    
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        deepspeed_config=ds_config if merged['nvme_offload'] else None
    )
    
    # Interactive mode
    if args.interactive:
        run_interactive_mode(engine, merged)
    else:
        # Single prompt mode
        print(f"\nPrompt: {merged['prompt']}")
        print(f"Max tokens: {merged['max_tokens']}")
        print("\nGenerating...")
        print("-" * 60)
        
        start_time = time.time()
        
        if merged['stream']:
            # Streaming output
            output = ""
            for token in engine.generate(
                prompt=merged['prompt'],
                max_new_tokens=merged['max_tokens'],
                temperature=merged['temperature'],
                top_p=merged['top_p'],
                top_k=merged['top_k'],
                repetition_penalty=merged['repetition_penalty'],
                stream=True
            ):
                print(token, end='', flush=True)
                output += token
        else:
            # Batch output
            output = engine.generate(
                prompt=merged['prompt'],
                max_new_tokens=merged['max_tokens'],
                temperature=merged['temperature'],
                top_p=merged['top_p'],
                top_k=merged['top_k'],
                repetition_penalty=merged['repetition_penalty'],
                stream=False
            )
            print(output)
        
        elapsed = time.time() - start_time
        print("\n" + "-" * 60)
        print(f"Generation completed in {elapsed:.2f}s")
    
    # ===================
    # STEP 7: Cleanup
    # ===================
    print_step(7, "Cleanup")
    
    if swap_manager:
        logger.info("Removing swap file...")
        swap_manager.remove_swap()
        print("✓ Swap file removed")
    
    print("\n✓ Done!")
    return 0


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_banner():
    """Print application banner"""
    print("\n" + "=" * 70)
    print("  HF-LLM-RUNNER")
    print("  Run any HuggingFace LLM with DeepSpeed ZeRO-3 SSD Offloading")
    print("=" * 70)
    print()
    print("  Features:")
    print("  • NO QUANTIZATION - Full FP16/BF16/FP32 precision")
    print("  • Dynamic VRAM detection via nvidia-smi")
    print("  • SSD → CPU → GPU chunked loading pipeline")
    print("  • Works with ANY HuggingFace model")
    print("=" * 70)
    print()


def print_step(step: int, title: str):
    """Print step header"""
    print("\n" + "=" * 70)
    print(f"  STEP {step}: {title}")
    print("=" * 70)


def print_hardware_info(gpu_info: Dict, ram_info: Dict, disk_info: Dict):
    """Print formatted hardware information"""
    print(f"\n  GPU: {gpu_info['gpu_name']}")
    print(f"  Total VRAM: {gpu_info['total_vram_gb']:.2f} GB")
    print(f"  Free VRAM: {gpu_info['free_vram_gb']:.2f} GB")
    print(f"  Usable VRAM (Total - {gpu_info.get('safety_margin', 3)}GB margin): {gpu_info['usable_vram_gb']:.2f} GB")
    print(f"  CUDA Version: {gpu_info['cuda_version']}")
    print(f"  Driver Version: {gpu_info['driver_version']}")
    
    print(f"\n  System RAM: {ram_info['total_ram_gb']:.2f} GB total")
    print(f"  Available RAM: {ram_info['available_ram_gb']:.2f} GB")
    
    print(f"\n  Offload Path: {disk_info['path']}")
    print(f"  Available Disk: {disk_info['free_gb']:.2f} GB")


def print_memory_requirements(mem_req: Dict, gpu_info: Dict, ram_info: Dict):
    """Print memory requirements analysis"""
    print(f"\n  Model Weights: {mem_req['model_weights_gb']:.2f} GB")
    print(f"  KV Cache (est): {mem_req['kv_cache_gb']:.2f} GB")
    print(f"  Activations: {mem_req['activations_gb']:.2f} GB")
    print(f"  Total Estimate: {mem_req['total_gb']:.2f} GB")
    
    print(f"\n  Available Resources:")
    print(f"  Usable VRAM: {gpu_info['usable_vram_gb']:.2f} GB")
    print(f"  Available RAM: {ram_info['available_ram_gb']:.2f} GB")
    print(f"  Combined: {gpu_info['usable_vram_gb'] + ram_info['available_ram_gb']:.2f} GB")
    
    # Warning if memory is tight
    total_available = gpu_info['usable_vram_gb'] + ram_info['available_ram_gb']
    if mem_req['total_gb'] > total_available:
        print(f"\n  ⚠️  WARNING: Model requires more memory than available!")
        print(f"     Recommended swap: {mem_req['recommended_swap_gb']:.0f} GB")


def estimate_params_from_name(model_name: str) -> float:
    """
    Estimate model parameters from model name.
    
    This is a rough estimate for planning purposes.
    """
    name_lower = model_name.lower()
    
    # Try to extract from name
    import re
    
    # Look for patterns like "7b", "13b", "70b", "120b"
    match = re.search(r'(\d+)b', name_lower)
    if match:
        return float(match.group(1))
    
    # Look for patterns like "7-b", "70-b"
    match = re.search(r'(\d+)-?b', name_lower)
    if match:
        return float(match.group(1))
    
    # Known models
    known_models = {
        'gpt-oss-120b': 120,
        'llama-2-70b': 70,
        'llama-2-13b': 13,
        'llama-2-7b': 7,
        'llama-3-70b': 70,
        'llama-3-8b': 8,
        'mistral-7b': 7,
        'falcon-40b': 40,
        'falcon-7b': 7,
    }
    
    for key, params in known_models.items():
        if key in name_lower:
            return float(params)
    
    # Default assumption
    logger.warning(f"Could not estimate parameters for {model_name}, assuming 70B")
    return 70.0


def run_interactive_mode(engine: InferenceEngine, config: Dict):
    """Run interactive chat mode"""
    print("\n" + "=" * 70)
    print("  INTERACTIVE MODE")
    print("  Type your message and press Enter to generate.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 70 + "\n")
    
    messages = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            # Add to messages
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            print("\nAssistant: ", end='', flush=True)
            
            response = engine.generate(
                prompt=user_input,
                max_new_tokens=config['max_tokens'],
                temperature=config['temperature'],
                stream=True
            )
            
            full_response = ""
            for token in response:
                print(token, end='', flush=True)
                full_response += token
            
            print()  # Newline after response
            
            messages.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except EOFError:
            print("\nGoodbye!")
            break


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
