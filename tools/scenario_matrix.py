#!/usr/bin/env python3
"""
Scenario matrix runner for HF-LLM-RUNNER.

Generates and optionally executes dry-run startup checks across multiple
runtime policy / precision / swap combinations to improve operational confidence.
"""

import argparse
import json
import subprocess
from pathlib import Path


def build_scenarios():
    return [
        {"name": "accelerate_safe_fp16", "args": ["--runtime-policy", "accelerate_safe", "--precision", "fp16"]},
        {"name": "deepspeed_strict_fp16", "args": ["--runtime-policy", "deepspeed_strict", "--precision", "fp16"]},
        {"name": "extreme_slow_universal_fp16", "args": ["--runtime-policy", "extreme_slow_universal", "--precision", "fp16"]},
        {"name": "accelerate_safe_bf16", "args": ["--runtime-policy", "accelerate_safe", "--precision", "bf16"]},
    ]


def run_dry_run(project_root: Path, scenario: dict, config_path: str = "config.json"):
    cmd = ["python", "run_llm.py", "--config", config_path, "--dry-run"] + scenario["args"]
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "name": scenario["name"],
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    }


def main():
    parser = argparse.ArgumentParser(description="Run dry-run scenario matrix for startup verification.")
    parser.add_argument("--execute-dry-run", action="store_true", help="Execute scenarios in dry-run mode")
    parser.add_argument("--emit-json", type=str, default=None, help="Path to write scenario results JSON")
    parser.add_argument("--config", type=str, default="config.json", help="Config path passed to run_llm.py")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    scenarios = build_scenarios()
    results = []

    if args.execute_dry_run:
        for scenario in scenarios:
            results.append(run_dry_run(project_root, scenario, config_path=args.config))
    else:
        results = [{"name": s["name"], "cmd_preview": "python run_llm.py --config config.json --dry-run " + " ".join(s["args"])} for s in scenarios]

    if args.emit_json:
        out_path = Path(args.emit_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

