"""Read-only framework preflight; actual trials use the paired deployment runner."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["framework-check"])
    parser.add_argument("--tool-set", choices=["pyrua", "strong-grasp"], default="pyrua")
    args = parser.parse_args()
    modules = ("harnesslab", "pyrualean", "mcp")
    if args.tool_set == "strong-grasp":
        modules += ("strong_grasp",)
    dependencies = {name: bool(importlib.util.find_spec(name)) for name in modules}
    print(json.dumps({"framework": "harness", "tool_set": args.tool_set, "dependencies": dependencies,
        "execution_performed": False, "model_calls": 0,
        "protocol": "fresh native RGB Test scene -> Agent -> persistent Python policy "
                    "-> bounded tools -> feedback -> Agent -> native final verdict",
        "source": str(Path(__file__).resolve().parent.parent)}, ensure_ascii=False, indent=2))
    return 0 if all(dependencies.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
