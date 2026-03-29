import json

from run_llm import (
    RuntimePlan,
    detect_model_family,
    classify_performance,
    run_compatibility_checks,
    save_startup_report,
    build_failure,
)


def test_detect_model_family():
    assert detect_model_family("meta-llama/Llama-2-7b-hf") == "llama"
    assert detect_model_family("unknown-org/custom-model") == "unknown"


def test_classify_performance():
    assert classify_performance(48, 96) == "interactive"
    assert classify_performance(8, 8) == "extreme_slow"


def test_compatibility_warns_on_unknown_family_and_low_disk():
    merged = {
        "model_name": "myorg/custom-big-model",
        "precision": "fp16",
        "nvme_offload": True,
    }
    plan = RuntimePlan(
        selected_backend="deepspeed",
        streaming_mode=True,
        offload_mode="nvme",
        swap_mode="preferred",
        precision_mode="fp16",
        extreme_slow_mode=False,
        strict_compat=False,
    )
    gpu = {"total_vram_gb": 16, "cuda_version": "12.2"}
    ram = {"available_ram_gb": 16}
    disk = {"free_gb": 20}
    report = run_compatibility_checks(merged, plan, gpu, ram, disk)
    assert any("Model family could not be identified" in w for w in report["warnings"])
    assert any("Low offload disk space" in w for w in report["warnings"])


def test_family_profile_enforces_min_cuda_and_trust_remote_code_warning():
    merged = {
        "model_name": "qwen/Qwen2.5-7B",
        "precision": "fp16",
        "nvme_offload": False,
        "trust_remote_code": False,
        "compatibility_profiles": {
            "qwen": {"min_cuda": "12.0", "require_trust_remote_code": True}
        },
        "tested_families": ["qwen"],
    }
    plan = RuntimePlan(
        selected_backend="accelerate",
        streaming_mode=True,
        offload_mode="cpu",
        swap_mode="preferred",
        precision_mode="fp16",
        extreme_slow_mode=False,
        strict_compat=False,
    )
    gpu = {"total_vram_gb": 24, "cuda_version": "11.8"}
    ram = {"available_ram_gb": 32}
    disk = {"free_gb": 200}
    report = run_compatibility_checks(merged, plan, gpu, ram, disk)
    assert any("requires CUDA >=" in e for e in report["errors"])
    assert any("trust_remote_code=true" in w for w in report["warnings"])


def test_startup_report_persists_normalization_actions(tmp_path):
    path = tmp_path / "report.json"
    merged = {"runtime_policy": "deepspeed_strict"}
    plan = RuntimePlan(
        selected_backend="deepspeed",
        streaming_mode=False,
        offload_mode="nvme",
        swap_mode="preferred",
        precision_mode="fp16",
        extreme_slow_mode=False,
        strict_compat=True,
    )
    save_startup_report(
        str(path),
        merged,
        plan,
        compat_report={"warnings": [], "errors": []},
        performance_class="slow",
        normalization_actions=[{"setting": "stream", "from": True, "to": False, "reason": "test"}],
        failures=[build_failure("TEST_CODE", "test message", "test remediation")],
    )
    with open(path, "r") as f:
        payload = json.load(f)
    assert payload["runtime_policy"] == "deepspeed_strict"
    assert len(payload["normalization_actions"]) == 1
    assert len(payload["failures"]) == 1
    assert payload["failures"][0]["code"] == "TEST_CODE"
