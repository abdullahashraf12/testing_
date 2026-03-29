#!/usr/bin/env python3
"""
Ready gate utility for production readiness quick checks.
"""

import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
    "run_llm.py",
    "config.json",
    "README.md",
    "PRODUCTION_CHECKLIST.md",
    ".github/workflows/ci.yml",
    "tools/scenario_matrix.py",
]


def evaluate_repo(root: Path) -> dict:
    checks = []
    for rel in REQUIRED_FILES:
        path = root / rel
        checks.append({"check": f"exists:{rel}", "passed": path.exists()})
    
    all_passed = all(item["passed"] for item in checks)
    return {"all_passed": all_passed, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description="Production ready gate checker.")
    parser.add_argument("--emit-json", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parent.parent
    result = evaluate_repo(root)
    
    if args.emit_json:
        out = Path(args.emit_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))
    
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()

