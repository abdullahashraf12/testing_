from tools.scenario_matrix import build_scenarios


def test_scenario_matrix_contains_expected_profiles():
    scenarios = build_scenarios()
    names = {s["name"] for s in scenarios}
    assert "accelerate_safe_fp16" in names
    assert "deepspeed_strict_fp16" in names
    assert "extreme_slow_universal_fp16" in names
    assert "accelerate_safe_bf16" in names

