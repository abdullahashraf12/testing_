from run_llm import RuntimePlan, detect_model_family, classify_performance, run_compatibility_checks


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

