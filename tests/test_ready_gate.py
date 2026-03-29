from pathlib import Path

from tools.ready_gate import evaluate_repo


def test_ready_gate_core_files_present():
    root = Path(__file__).resolve().parent.parent
    result = evaluate_repo(root)
    assert isinstance(result["all_passed"], bool)
    assert len(result["checks"]) >= 1
    assert any(item["check"] == "exists:run_llm.py" for item in result["checks"])

