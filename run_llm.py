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
from dataclasses import dataclass, asdict

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


@dataclass
class RuntimePlan:
    """Resolved runtime execution plan"""
    selected_backend: str
    streaming_mode: bool
    offload_mode: str
    swap_mode: str
    precision_mode: str
    extreme_slow_mode: bool
    strict_compat: bool


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
        default=None,
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision/branch"
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
        default=None,
        help="Precision: fp16, bf16 (Ampere+), fp32 (max quality, 2x memory)"
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
        "--swap-policy",
        type=str,
        choices=["required", "preferred", "disabled"],
        default=None,
        help="Swap policy: required (fail if unavailable), preferred (warn), disabled (never create)"
    )
    parser.add_argument(
        "--swap-path",
        type=str,
        default=None,
        help="Path for swap file"
    )
    
    # ===================
    # Offload arguments
    # ===================
    parser.add_argument(
        "--offload-path",
        type=str,
        default=None,
        help="Path for SSD offloading"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Buffer size in GB for NVMe offloading"
    )
    
    # ===================
    # VRAM arguments
    # ===================
    parser.add_argument(
        "--vram-safety-margin",
        type=int,
        default=None,
        help="VRAM safety margin in GB"
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
        default=None,
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty"
    )
    
    # ===================
    # Mode arguments
    # ===================
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument("--stream", dest="stream", action="store_true", default=None, help="Enable streaming output")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming output")
    parser.add_argument(
        "--disable-deepspeed",
        action="store_true",
        help="Force Accelerate loading path instead of DeepSpeed runtime"
    )
    parser.add_argument(
        "--strict-compat",
        action="store_true",
        help="Fail fast when compatibility checks detect unsupported hardware/runtime"
    )
    parser.add_argument(
        "--extreme-slow-mode",
        action="store_true",
        help="Prioritize eventual completion over speed with conservative generation settings"
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
    parser.add_argument(
        "--startup-report-path",
        type=str,
        default="./startup_report.json",
        help="Path to save resolved startup/runtime report"
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
    
    def _pick(cli_value, config_value, default_value):
        return cli_value if cli_value is not None else (
            config_value if config_value is not None else default_value
        )
    
    # Model settings
    model_cfg = config.get('model', {})
    merged['model_name'] = _pick(args.model, model_cfg.get('name'), 'openai/gpt-oss-120b')
    merged['revision'] = _pick(args.revision, model_cfg.get('revision'), 'main')
    merged['trust_remote_code'] = args.trust_remote_code or model_cfg.get('trust_remote_code', False)
    merged['auth_token'] = _pick(args.auth_token, model_cfg.get('use_auth_token'), None)
    
    # Precision
    merged['precision'] = _pick(args.precision, config.get('precision', {}).get('type'), 'fp16')
    
    # Memory
    memory_cfg = config.get('memory', {})
    merged['vram_safety_margin'] = _pick(args.vram_safety_margin, memory_cfg.get('vram_safety_margin_gb'), 3)
    merged['cpu_offload'] = memory_cfg.get('cpu_offload', True)
    merged['nvme_offload'] = memory_cfg.get('nvme_offload', True)
    
    # Offload
    offload_cfg = config.get('offload', {})
    merged['offload_path'] = _pick(args.offload_path, offload_cfg.get('offload_path'), './offload_dir')
    merged['buffer_size'] = _pick(args.buffer_size, offload_cfg.get('buffer_size_gb'), 4)
    
    # Swap
    swap_cfg = config.get('swap', {})
    merged['use_swap'] = args.use_swap or swap_cfg.get('enabled', False)
    merged['swap_path'] = _pick(args.swap_path, swap_cfg.get('path'), './model_swap')
    merged['swap_size'] = args.swap_size
    merged['swap_policy'] = _pick(args.swap_policy, swap_cfg.get('policy'), 'preferred')
    if merged['swap_size'] is None and swap_cfg.get('size_gb') != 'auto':
        merged['swap_size'] = swap_cfg.get('size_gb')
    
    # Inference
    inference_cfg = config.get('inference', {})
    merged['prompt'] = _pick(
        args.prompt,
        config.get('generation_examples', {}).get('poem_example', {}).get('prompt'),
        "HI generate 500 lines of random words poem as each word is 50 letters as line consists of 15 words"
    )
    merged['max_tokens'] = _pick(args.max_tokens, inference_cfg.get('max_new_tokens'), 500)
    merged['temperature'] = _pick(args.temperature, inference_cfg.get('temperature'), 0.7)
    merged['top_p'] = _pick(args.top_p, inference_cfg.get('top_p'), 0.9)
    merged['top_k'] = _pick(args.top_k, inference_cfg.get('top_k'), 50)
    merged['repetition_penalty'] = _pick(args.repetition_penalty, inference_cfg.get('repetition_penalty'), 1.1)
    
    # Stream
    merged['stream'] = _pick(args.stream, config.get('performance', {}).get('stream_output'), True)
    
    # Runtime backend
    merged['use_deepspeed'] = (not args.disable_deepspeed) and merged['nvme_offload']
    merged['strict_compat'] = args.strict_compat
    merged['extreme_slow_mode'] = args.extreme_slow_mode

    return merged


def build_runtime_plan(merged: Dict) -> RuntimePlan:
    """Build resolved runtime plan from merged config."""
    return RuntimePlan(
        selected_backend="deepspeed" if merged['use_deepspeed'] else "accelerate",
        streaming_mode=not merged.get('extreme_slow_mode', False) and merged['stream'],
        offload_mode="nvme" if merged['nvme_offload'] else ("cpu" if merged['cpu_offload'] else "none"),
        swap_mode=merged.get('swap_policy', 'preferred'),
        precision_mode=merged['precision'],
        extreme_slow_mode=merged.get('extreme_slow_mode', False),
        strict_compat=merged.get('strict_compat', False)
    )


def detect_model_family(model_name: str) -> str:
    """Best-effort model family detection for compatibility hints."""
    name = model_name.lower()
    known = [
        "llama", "mistral", "mixtral", "falcon", "qwen", "gpt", "phi", "gemma", "opt"
    ]
    for fam in known:
        if fam in name:
            return fam
    return "unknown"


def classify_performance(usable_vram_gb: float, ram_gb: float) -> str:
    """Classify expected runtime performance tier."""
    combined = usable_vram_gb + ram_gb
    if usable_vram_gb >= 40 and ram_gb >= 64:
        return "interactive"
    if combined >= 48:
        return "slow"
    if combined >= 24:
        return "very_slow"
    return "extreme_slow"


def run_compatibility_checks(
    merged: Dict,
    runtime_plan: RuntimePlan,
    gpu_info: Dict,
    ram_info: Dict,
    disk_info: Dict
) -> Dict:
    """
    Run compatibility checks. Returns report with warnings/errors.
    """
    report = {"warnings": [], "errors": [], "model_family": detect_model_family(merged['model_name'])}
    
    if gpu_info['total_vram_gb'] <= 0:
        report["errors"].append("No NVIDIA GPU detected via nvidia-smi.")
    
    if merged['precision'] == 'bf16':
        cuda_version = gpu_info.get('cuda_version', 'Unknown')
        if cuda_version == 'Unknown':
            report["warnings"].append("BF16 selected but CUDA version is unknown.")
        elif float(cuda_version) < 11.0:
            report["errors"].append(f"BF16 selected but CUDA version {cuda_version} is likely incompatible.")
    
    if merged['nvme_offload'] and disk_info.get('free_gb', 0) < 50:
        report["warnings"].append(
            f"Low offload disk space ({disk_info.get('free_gb', 0):.2f} GB). Large models may fail."
        )
    
    if ram_info.get('available_ram_gb', 0) < 8:
        report["warnings"].append("Available system RAM is very low (<8GB). Expect severe slowdown or failure.")
    
    tested_families = {"llama", "mistral", "mixtral", "falcon", "qwen", "phi", "gemma", "opt", "gpt"}
    if report["model_family"] == "unknown":
        report["warnings"].append(
            "Model family could not be identified as a commonly tested architecture."
        )
    elif report["model_family"] not in tested_families:
        report["warnings"].append(
            f"Model family '{report['model_family']}' is outside currently tested baseline."
        )
    
    # Backend strict profile checks
    if runtime_plan.selected_backend == "deepspeed" and merged['nvme_offload']:
        if disk_info.get('free_gb', 0) < 100:
            report["warnings"].append("DeepSpeed+NVMe selected with <100GB free disk; large models may fail.")
    
    if runtime_plan.selected_backend == "accelerate" and merged['model_name'].lower().endswith("120b"):
        report["warnings"].append("120B-class model on Accelerate backend may be impractically slow or unstable.")
    
    return report


def save_startup_report(
    path: str,
    merged: Dict,
    runtime_plan: RuntimePlan,
    compat_report: Dict,
    performance_class: Optional[str] = None
):
    """Persist startup report for operator diagnostics."""
    payload = {
        "timestamp": int(time.time()),
        "merged_config": merged,
        "runtime_plan": asdict(runtime_plan),
        "compatibility": compat_report,
        "performance_class": performance_class
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


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
    
    runtime_plan = build_runtime_plan(merged)
    performance_class = classify_performance(
        usable_vram_gb=gpu_info.get('usable_vram_gb', 0),
        ram_gb=ram_info.get('available_ram_gb', 0)
    )
    compat_report = run_compatibility_checks(merged, runtime_plan, gpu_info, ram_info, disk_info)
    logger.info(
        f"Runtime plan: backend={runtime_plan.selected_backend}, "
        f"offload={runtime_plan.offload_mode}, stream={runtime_plan.streaming_mode}, "
        f"performance_class={performance_class}"
    )
    for warning in compat_report["warnings"]:
        logger.warning(f"[compat] {warning}")
    for error in compat_report["errors"]:
        logger.error(f"[compat] {error}")
    
    if runtime_plan.strict_compat and (compat_report["warnings"] or compat_report["errors"]):
        logger.error("Strict compatibility mode enabled and issues were detected. Aborting.")
        save_startup_report(
            args.startup_report_path,
            merged,
            runtime_plan,
            compat_report,
            performance_class=performance_class
        )
        return 1
    
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
    swap_policy = merged.get('swap_policy', 'preferred')
    if merged['use_swap'] and swap_policy != 'disabled':
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
            if swap_policy == 'required':
                logger.error("Swap policy is 'required' and swap creation failed. Aborting.")
                save_startup_report(
                    args.startup_report_path,
                    merged,
                    runtime_plan,
                    compat_report,
                    performance_class=performance_class
                )
                return 1
            logger.warning("Failed to create swap file, continuing without swap")
            swap_manager = None
    elif swap_policy == 'disabled':
        logger.info("Swap policy is disabled; skipping swap setup.")
    
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
    
    save_startup_report(
        args.startup_report_path,
        merged,
        runtime_plan,
        compat_report,
        performance_class=performance_class
    )
    logger.info(f"Startup report saved: {args.startup_report_path}")
    
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
        
        if merged['use_deepspeed']:
            logger.info("Loading runtime backend: DeepSpeed ZeRO-3")
            model, tokenizer = loader.load_model_with_deepspeed(
                deepspeed_config=ds_config
            )
        else:
            logger.info("Loading runtime backend: Transformers + Accelerate")
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
        deepspeed_config=ds_config if merged['use_deepspeed'] else None
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
        
        if runtime_plan.extreme_slow_mode:
            logger.warning("Extreme slow mode enabled: using chunked generation for completion-first behavior.")
            long_engine = LongTextGenerator(
                model=model,
                tokenizer=tokenizer,
                deepspeed_config=ds_config if merged['use_deepspeed'] else None
            )
            output = long_engine.generate_in_chunks(
                prompt=merged['prompt'],
                total_tokens=merged['max_tokens'],
                chunk_size=max(64, min(256, merged['max_tokens']))
            )
            print(output)
        elif merged['stream']:
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
    print(f"  Usable VRAM (Total - {gpu_info.get('safety_margin_gb', 3)}GB margin): {gpu_info['usable_vram_gb']:.2f} GB")
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
