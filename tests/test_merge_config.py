import argparse

from run_llm import merge_config_with_args, apply_runtime_policy


def _args(**overrides):
    base = {
        "model": None,
        "revision": None,
        "trust_remote_code": False,
        "auth_token": None,
        "precision": None,
        "use_swap": False,
        "swap_size": None,
        "swap_path": None,
        "swap_policy": None,
        "offload_path": None,
        "buffer_size": None,
        "vram_safety_margin": None,
        "max_vram": None,
        "prompt": None,
        "max_tokens": None,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": None,
        "interactive": False,
        "stream": None,
        "disable_deepspeed": False,
        "strict_compat": False,
        "extreme_slow_mode": False,
        "runtime_policy": None,
        "config": None,
        "info": False,
        "dry_run": False,
        "verbose": False,
        "startup_report_path": "./startup_report.json",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_config_values_used_when_cli_omitted():
    config = {
        "model": {"name": "mistralai/Mistral-7B-v0.1", "revision": "main"},
        "precision": {"type": "bf16"},
        "memory": {"vram_safety_margin_gb": 2, "cpu_offload": True, "nvme_offload": True},
        "swap": {"enabled": True, "path": "./swap", "policy": "preferred", "size_gb": 100},
        "inference": {"max_new_tokens": 256, "temperature": 0.5, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05},
    }
    merged = merge_config_with_args(config, _args())
    assert merged["model_name"] == "mistralai/Mistral-7B-v0.1"
    assert merged["precision"] == "bf16"
    assert merged["vram_safety_margin"] == 2
    assert merged["swap_policy"] == "preferred"
    assert merged["max_tokens"] == 256


def test_cli_overrides_config_values():
    config = {
        "model": {"name": "foo/model"},
        "precision": {"type": "fp16"},
        "swap": {"policy": "preferred"},
        "memory": {"nvme_offload": True},
    }
    merged = merge_config_with_args(
        config,
        _args(model="bar/model", precision="fp32", disable_deepspeed=True, swap_policy="disabled")
    )
    assert merged["model_name"] == "bar/model"
    assert merged["precision"] == "fp32"
    assert merged["swap_policy"] == "disabled"
    assert merged["use_deepspeed"] is False


def test_runtime_policy_pack_applies_overrides():
    merged = {
        "use_deepspeed": False,
        "nvme_offload": True,
        "stream": True,
        "strict_compat": False,
        "extreme_slow_mode": False,
        "runtime_policy": "deepspeed_strict",
        "policy_packs": {
            "deepspeed_strict": {
                "use_deepspeed": True,
                "strict_compat": True,
                "stream": False,
            }
        },
    }
    updated = apply_runtime_policy(merged)
    assert updated["use_deepspeed"] is True
    assert updated["strict_compat"] is True
    assert updated["stream"] is False
