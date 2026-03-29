# 🚀 HF-LLM-RUNNER

**Run ANY HuggingFace LLM on ANY NVIDIA GPU with DeepSpeed ZeRO-3 SSD Offloading**

This tool enables running large language models (including 120B+ parameter models) on consumer hardware with limited VRAM, using DeepSpeed ZeRO-3 with CPU and SSD offloading.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **NO QUANTIZATION** | Full precision (FP16/BF16/FP32) - no quality loss |
| **Dynamic VRAM Detection** | Automatically detects and adapts to your GPU |
| **SSD Offloading** | Load models larger than VRAM + RAM combined |
| **Universal Compatibility** | Works with ANY HuggingFace model |
| **Swap Support** | Optional swap file for extended memory |

---

## 📋 Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| NVIDIA GPU | Any CUDA-capable GPU | RTX 3090 / RTX 4090 / A100 |
| VRAM | 8 GB | 24 GB+ |
| System RAM | 32 GB | 64-128 GB |
| SSD Storage | 100 GB free | 500 GB+ NVMe SSD |

### Software Requirements

- **Python**: 3.8+
- **CUDA**: 11.6+ (for BF16 support)
- **nvidia-smi**: Must be installed and accessible
- **OS**: Linux (recommended) / Windows with WSL2

---

## 🔧 Installation

### Quick Start

```bash
# Clone or download the project
cd hf_llm_runner

# Install dependencies
pip install -r requirements.txt
```

### Detailed Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 2. Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 3. Install other dependencies
pip install transformers accelerate safetensors huggingface-hub

# 4. Install DeepSpeed
pip install deepspeed
# For NVMe offloading support:
pip install deepspeed[nvme]
```

---

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default model (GPT-120B)
python run_llm.py

# Run specific model
python run_llm.py --model meta-llama/Llama-2-70b-hf

# Run with swap enabled
python run_llm.py --model openai/gpt-oss-120b --use-swap --swap-size 200
```

### Using Config File

```bash
# Edit config.json to set your preferences
python run_llm.py --config config.json
```

### Custom Prompt

```bash
python run_llm.py --model mistralai/Mistral-7B-v0.1 \
    --prompt "Write a poem about artificial intelligence" \
    --max-tokens 1000
```

### Interactive Mode

```bash
python run_llm.py --model meta-llama/Llama-2-7b-hf --interactive
```

---

## 📖 Usage Examples

### Example 1: Running a 120B Model on 24GB VRAM

```bash
# This will use SSD offloading to run a 240GB model on limited VRAM
python run_llm.py \
    --model openai/gpt-oss-120b \
    --precision fp16 \
    --use-swap \
    --swap-size 300 \
    --offload-path ./offload_dir
```

### Example 2: Running Llama-2-70B with BF16

```bash
# BF16 provides better numerical stability on Ampere+ GPUs
python run_llm.py \
    --model meta-llama/Llama-2-70b-hf \
    --precision bf16 \
    --auth-token YOUR_HF_TOKEN
```

### Example 3: The Poem Generation Example

```bash
# Generate the long poem as specified by the user
python run_llm.py \
    --model openai/gpt-oss-120b \
    --prompt "HI generate 500 lines of random words poem as each word is 50 letters as line consists of 15 words" \
    --max-tokens 4000 \
    --use-swap
```

### Example 4: Check Hardware Info

```bash
# Print detected hardware information
python run_llm.py --info
```

### Example 5: Dry Run (Test Configuration)

```bash
# Test without actually loading the model
python run_llm.py --model openai/gpt-oss-120b --dry-run
```

---

## ⚙️ Configuration

### config.json

The `config.json` file provides comprehensive configuration options:

```json
{
    "model": {
        "name": "openai/gpt-oss-120b",
        "revision": "main",
        "trust_remote_code": false,
        "use_auth_token": null
    },
    "precision": {
        "type": "fp16"
    },
    "memory": {
        "vram_safety_margin_gb": 3,
        "cpu_offload": true,
        "nvme_offload": true
    },
    "offload": {
        "offload_path": "./offload_dir",
        "buffer_size_gb": 4
    },
    "inference": {
        "max_new_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9
    }
}
```

### Precision Options

| Precision | Memory Usage | Quality | GPU Support |
|-----------|--------------|---------|-------------|
| `fp16` | 2 bytes/param | High | All CUDA GPUs |
| `bf16` | 2 bytes/param | High (better stability) | Ampere+ (RTX 30xx+) |
| `fp32` | 4 bytes/param | Maximum | All CUDA GPUs |

**Note**: This tool does NOT use quantization (8-bit/4-bit). All precision options are full precision.

---

## 📊 Memory Requirements

### Model Size Calculation

| Model | Parameters | FP16 Size | FP32 Size | Recommended Swap |
|-------|------------|-----------|-----------|------------------|
| Mistral-7B | 7B | 14 GB | 28 GB | 21 GB |
| Llama-2-13B | 13B | 26 GB | 52 GB | 39 GB |
| Falcon-40B | 40B | 80 GB | 160 GB | 120 GB |
| Llama-2-70B | 70B | 140 GB | 280 GB | 210 GB |
| GPT-120B | 120B | 240 GB | 480 GB | 360 GB |

### VRAM Safety Margin

The tool automatically reserves 3GB of VRAM for system overhead:

```
Usable VRAM = Total VRAM - Safety Margin
Example: 16GB GPU → 13GB usable
```

---

## 🏗️ How It Works

### DeepSpeed ZeRO-3 Offloading

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │   GPU (VRAM)    │  ← Active layer parameters (minimal)       │
│  │   ~24 GB        │  ← Current activations                     │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   CPU (RAM)     │  ← Optimizer states                        │
│  │   ~64-128 GB    │  ← Gradient buffers                        │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   SSD/NVMe      │  ← All model parameters (sharded)          │
│  │   ~360+ GB      │  ← Checkpoint data                         │
│  └─────────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Loading Pipeline

```
SSD (Model Shards) → CPU (Buffer) → GPU (Active Parameters)
         ▲                                    │
         └────────────────────────────────────┘
                    (Next Chunk Cycle)
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce max VRAM or enable swap
python run_llm.py --max-vram 10 --use-swap --swap-size 100
```

#### 2. DeepSpeed Installation Fails

```bash
# Install build dependencies
pip install ninja

# Try specific CUDA version
DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 pip install deepspeed
```

#### 3. Model Download Fails

```bash
# Login to HuggingFace
huggingface-cli login

# Or use token
python run_llm.py --auth-token YOUR_TOKEN
```

#### 4. nvidia-smi Not Found

Ensure NVIDIA drivers are installed:
```bash
# Ubuntu
sudo apt install nvidia-utils-535  # or your driver version

# Verify
nvidia-smi
```

---

## 📁 Project Structure

```
hf_llm_runner/
├── run_llm.py              # Main entry point
├── config.json             # User configuration
├── deepspeed_config.json   # Auto-generated DeepSpeed config
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── modules/
│   ├── __init__.py
│   ├── hardware_detector.py    # nvidia-smi parsing
│   ├── swap_manager.py         # Swap file management
│   ├── deepspeed_generator.py  # DeepSpeed config generator
│   ├── model_loader.py         # HuggingFace model loading
│   └── inference_engine.py     # Text generation
│
└── utils/
    ├── __init__.py
    ├── logger.py               # Logging utilities
    └── memory_tracker.py       # Memory monitoring
```

---

## 🤝 Running Any HuggingFace Model

This tool works with **ANY** HuggingFace text generation model:

### Popular Models

```bash
# Mistral
python run_llm.py --model mistralai/Mistral-7B-v0.1

# Llama 2
python run_llm.py --model meta-llama/Llama-2-7b-hf
python run_llm.py --model meta-llama/Llama-2-13b-hf
python run_llm.py --model meta-llama/Llama-2-70b-hf --use-swap

# Llama 3
python run_llm.py --model meta-llama/Meta-Llama-3-8B
python run_llm.py --model meta-llama/Meta-Llama-3-70B --use-swap

# Falcon
python run_llm.py --model tiiuae/falcon-7b
python run_llm.py --model tiiuae/falcon-40b --use-swap

# GPT-J / GPT-NeoX
python run_llm.py --model EleutherAI/gpt-j-6b
python run_llm.py --model EleutherAI/gpt-neox-20b

# Any other model from HuggingFace Hub!
python run_llm.py --model YOUR_MODEL_ID
```

---

## ⚠️ Important Notes

1. **NO QUANTIZATION**: This tool uses full precision (FP16/BF16/FP32). No 8-bit or 4-bit quantization.

2. **SSD Required**: For large models, ensure you have enough SSD space (3x model size recommended).

3. **Performance**: This is a proof-of-concept. Performance is secondary to feasibility. Expect slow token generation for very large models on limited hardware.

4. **Gated Models**: For models like Llama-2, you need:
   - HuggingFace account
   - Accepted model license
   - Authentication token

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for the transformers library
- [Microsoft DeepSpeed](https://www.deepspeed.ai/) for ZeRO optimization
- [PyTorch](https://pytorch.org/) for the deep learning framework
