#!/usr/bin/env python3
"""
harness.py — Local dev runner for the LLM pipeline.
Mirrors the GitHub Actions workflow exactly, runs on your machine.

Usage:
    # Full pipeline (Gemini scaffold → Qwen + GLM implement → test → report)
    python harness.py

    # Skip scaffold (reuse existing scaffold/scaffold.json)
    python harness.py --skip-scaffold

    # Skip scaffold + skip implement (reuse existing src/ / src_glm/)
    python harness.py --skip-scaffold --skip-impl
    python harness.py --skip-scaffold --skip-impl --only qwen

    # Test + iterate only (alias for --skip-scaffold --skip-impl)
    python harness.py --test-only
    python harness.py --test-only --only qwen
    python harness.py --test-only --only glm

    # Only run one model (implement + test)
    python harness.py --only qwen
    python harness.py --only glm

    # Override iteration cap
    python harness.py --max-iter 5

Typical debug loop after a failed test:
    python harness.py --test-only --only qwen --max-iter 3

Requirements:
    pip install httpx
    GEMINI_API_KEY, OPENROUTER_API_KEY must be set as env vars
    (or in a .env file — this script loads .env automatically)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def load_dotenv():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def run_step(label: str, script: str, extra_args: list[str] = None) -> bool:
    cmd = [sys.executable, str(ROOT / "pipeline" / script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    status = "✓ PASS" if result.returncode == 0 else "✗ FAIL"
    print(f"  {status}  ({elapsed:.1f}s)")
    return result.returncode == 0


def check_env(keys: list[str]) -> bool:
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        print(f"[harness] Missing env vars: {', '.join(missing)}")
        print("          Set them in .env or export them before running.")
        return False
    return True


def check_impl_exists(model: str) -> bool:
    """Verify that implementation files exist before skipping implement step."""
    src_dir = ROOT / ("src_glm" if model == "glm" else "src")
    if not src_dir.exists() or not any(src_dir.rglob("*.ts")):
        print(f"[harness] --skip-impl set but {src_dir} is empty or missing.")
        print(f"          Run without --skip-impl first to generate implementation.")
        return False
    return True


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Local LLM pipeline runner")
    parser.add_argument("--skip-scaffold", action="store_true",
                        help="Reuse existing scaffold/scaffold.json")
    parser.add_argument("--skip-impl", action="store_true",
                        help="Reuse existing src/ / src_glm/ — skip implement step")
    parser.add_argument("--test-only", action="store_true",
                        help="Alias for --skip-scaffold --skip-impl (jump straight to test)")
    parser.add_argument("--only", choices=["qwen", "glm"],
                        help="Run only one model")
    parser.add_argument("--max-iter", type=int, default=3,
                        help="Max fix iterations per model")
    args = parser.parse_args()

    # --test-only is sugar for --skip-scaffold --skip-impl
    if args.test_only:
        args.skip_scaffold = True
        args.skip_impl = True

    results: dict[str, bool] = {}

    # ── Step 2: Gemini scaffold ──────────────────────────────────────────────
    if not args.skip_scaffold:
        if not check_env(["GEMINI_API_KEY"]):
            sys.exit(1)
        ok = run_step("Step 2 — Gemini scaffold", "02_scaffold_gemini.py")
        results["scaffold"] = ok
        if not ok:
            print("\n[harness] Scaffold failed. Stopping.")
            sys.exit(1)
    else:
        scaffold_json = ROOT / "scaffold" / "scaffold.json"
        if not scaffold_json.exists():
            print("[harness] --skip-scaffold set but scaffold/scaffold.json not found.")
            sys.exit(1)
        print("[harness] Skipping scaffold (reusing existing scaffold.json)")

    # ── Step 3+4+5: Implement + test ─────────────────────────────────────────
    models = ["qwen", "glm"] if args.only is None else [args.only]

    for model in models:
        if not check_env(["OPENROUTER_API_KEY"]):
            results[f"impl_{model}"] = False
            print(f"[harness] Skipping {model.upper()} — missing OPENROUTER_API_KEY.")
            continue

        script_map = {"qwen": "03a_implement_qwen.py", "glm": "03b_implement_glm.py"}

        # ── Step 3: Implement ────────────────────────────────────────────────
        if args.skip_impl:
            if not check_impl_exists(model):
                sys.exit(1)
            print(f"[harness] Skipping {model.upper()} implement (reusing existing src files)")
            results[f"impl_{model}"] = True
        else:
            ok = run_step(f"Step 3 — {model.upper()} implement", script_map[model])
            results[f"impl_{model}"] = ok

        # ── Step 4+5: Test + iterate ─────────────────────────────────────────
        ok = run_step(
            f"Step 4+5 — {model.upper()} test + iterate",
            "04_test_and_iterate.py",
            ["--impl", model, "--max-iter", str(args.max_iter)],
        )
        results[f"test_{model}"] = ok

    # ── Step 6: Report ───────────────────────────────────────────────────────
    run_step("Step 6 — Report", "05_report.py")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for key, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {key}")

    all_ok = all(results.values())
    print(f"\n  Overall: {'✅ PASS' if all_ok else '❌ FAIL'}")
    print(f"\n  Full report → reports/summary.md")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
