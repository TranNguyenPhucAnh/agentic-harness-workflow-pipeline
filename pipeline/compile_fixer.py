"""
pipeline/9c_compile_fixer_loop.py
==================================
Interactive compile-error fix loop.

Mục đích: Sau khi executor gen code, chạy script này để được hướng dẫn
từng bước fix lỗi compile (TypeScript). Không có LLM tự sửa file —
tất cả action do human thực hiện. Script chỉ:

  1. Chạy tsc, classify lỗi
  2. In ra những việc cần làm (commands, error list)
  3. Chờ human làm xong, nhận feedback text
  4. Loop lại cho đến khi clean hoặc human gõ "done"

Flow:
  Round N:
    ├── chạy tsc → parse + classify errors
    ├── nếu clean → ghi artifacts → exit ✓
    ├── in ACTION REQUIRED box (commands + errors)
    ├── prompt: "Bạn đã xử lý chưa? [feedback / Enter / done]: "
    ├── nếu "done" / "q" → exit với lỗi còn lại
    └── loop với feedback inject vào context round tiếp

Artifacts:
  compile_fixer/loop.json    — short-term overwrite: state session hiện tại
  compile_fixer/fixer_log.md — long-term append: history mọi session

Reads:
  executor/manifest.json     — để filter lỗi trong executor files

Usage:
  python 9c_compile_fixer_loop.py --project my-app
  python 9c_compile_fixer_loop.py --project my-app --dir path/to/project
  python 9c_compile_fixer_loop.py --project my-app --max-rounds 8
  python 9c_compile_fixer_loop.py --project my-app --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Bootstrap sys.path ───────────────────────────────────────────────────────
_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE))

from artifacts.paths import (          # noqa: E402
    EXECUTOR_OVERWRITE_MANIFEST,
    SRC_DIR,
    artifact_root,
    ensure_dirs,
    _LazyPath,
)
from modules.artifact_tracking import (  # noqa: E402
    track_read,
    track_write,
    print_summary as print_artifact_summary,
)
from modules.cost import print_summary as print_cost_summary  # noqa: E402
from modules.post_interactive import prompt_next_step          # noqa: E402

# ── Artifact paths (compile_fixer/ folder) ───────────────────────────────────
COMPILE_FIXER_LOOP    = _LazyPath("compile_fixer/loop.json")      # short-term overwrite
COMPILE_FIXER_LOG     = _LazyPath("compile_fixer/fixer_log.md")   # long-term append

ROLE = "compile_fixer"


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="9c_compile_fixer_loop.py",
        description="Interactive compile-error fix loop (human-driven, no LLM patching).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--project",
        default=os.environ.get("PIPELINE_PROJECT"),
        help="Project slug (sets PIPELINE_PROJECT).",
    )
    p.add_argument(
        "--dir",
        default=None,
        help="Project root directory (overrides --project lookup).",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="Path to executor manifest.json (overrides auto-detect).",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=6,
        help="Max feedback rounds before giving up (default: 6).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run tsc + classify but do not write artifacts.",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable TTY prompts (non-interactive mode).",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return
    if os.environ.get("PIPELINE_PROJECT"):
        return
    parser.error("PIPELINE_PROJECT is not set. Use --project <name>.")


# ════════════════════════════════════════════════════════════════════════════
# compile_fixer internals (import từ module gốc)
# ════════════════════════════════════════════════════════════════════════════

def _import_compile_fixer():
    """
    Import các hàm cần thiết từ compile_fixer.py (module gốc).
    Tách thành function để lỗi import rõ ràng hơn.
    """
    try:
        from modules.compile_fixer import (  # type: ignore
            load_manifest,
            parse_manifest,
            validate_manifest_assumptions,
            cross_check_plan,
            _run_tsc,
            parse_tsc_output,
            _filter_errors_to_manifest,
            classify_errors,
            _detect_package_manager,
            _pm_name,
            _format_escalation_box,
            ErrorKind,
            CompileError,
        )
        return {
            "load_manifest":                  load_manifest,
            "parse_manifest":                 parse_manifest,
            "validate_manifest_assumptions":  validate_manifest_assumptions,
            "cross_check_plan":               cross_check_plan,
            "_run_tsc":                       _run_tsc,
            "parse_tsc_output":               parse_tsc_output,
            "_filter_errors_to_manifest":     _filter_errors_to_manifest,
            "classify_errors":                classify_errors,
            "_detect_package_manager":        _detect_package_manager,
            "_pm_name":                       _pm_name,
            "_format_escalation_box":         _format_escalation_box,
            "ErrorKind":                      ErrorKind,
            "CompileError":                   CompileError,
        }
    except ImportError:
        # Thử tìm compile_fixer.py trong pipeline/ folder
        pipeline_dir = Path(__file__).parent
        cf_path = pipeline_dir / "compile_fixer.py"
        if not cf_path.exists():
            raise ImportError(
                "Cannot find compile_fixer.py. "
                "Expected at modules/compile_fixer.py or pipeline/compile_fixer.py."
            )
        import importlib.util
        spec = importlib.util.spec_from_file_location("compile_fixer", cf_path)
        mod = importlib.util.module_from_spec(spec)   # type: ignore
        spec.loader.exec_module(mod)                  # type: ignore
        return {
            "load_manifest":                  mod.load_manifest,
            "parse_manifest":                 mod.parse_manifest,
            "validate_manifest_assumptions":  mod.validate_manifest_assumptions,
            "cross_check_plan":               mod.cross_check_plan,
            "_run_tsc":                       mod._run_tsc,
            "parse_tsc_output":               mod.parse_tsc_output,
            "_filter_errors_to_manifest":     mod._filter_errors_to_manifest,
            "classify_errors":                mod.classify_errors,
            "_detect_package_manager":        mod._detect_package_manager,
            "_pm_name":                       mod._pm_name,
            "_format_escalation_box":         mod._format_escalation_box,
            "ErrorKind":                      mod.ErrorKind,
            "CompileError":                   mod.CompileError,
        }


# ════════════════════════════════════════════════════════════════════════════
# Resolve project_dir
# ════════════════════════════════════════════════════════════════════════════

def _resolve_project_dir(args: argparse.Namespace) -> Path:
    if args.dir:
        return Path(args.dir).resolve()
    try:
        return Path(str(SRC_DIR)).parent   # artifacts_<slug>/output/
    except Exception:
        return Path.cwd()


# ════════════════════════════════════════════════════════════════════════════
# Action box formatter
# ════════════════════════════════════════════════════════════════════════════

def _format_action_box(
    round_num:       int,
    max_rounds:      int,
    commands:        list[str],
    errors:          list[Any],
    project_dir:     Path,
    prev_feedback:   str,
    warnings:        list[str],
    cf:              dict,
) -> str:
    """
    Render the full ACTION REQUIRED box shown each round.

    Sections:
      1. Warnings (env issues)
      2. Commands to run (install, shadcn add, etc.)
      3. Remaining errors grouped by kind
      4. Previous feedback echo (if any)
    """
    W = 68
    lines: list[str] = []

    def rule(char="─"):
        lines.append(char * W)

    def center(text: str, char="═"):
        pad = max(0, W - len(text) - 2)
        left = pad // 2
        right = pad - left
        lines.append(f"{char * left} {text} {char * right}")

    def wrap_line(text: str, indent: int = 2):
        prefix = " " * indent
        wrapped = textwrap.wrap(text, width=W - indent - 1) or [""]
        for w in wrapped:
            lines.append(f"{prefix}{w}")

    lines.append("")
    center(f"COMPILE FIXER — Round {round_num}/{max_rounds}")
    rule()

    # Warnings
    if warnings:
        lines.append("  ⚠ Environment issues:")
        for w in warnings:
            wrap_line(f"• {w}", indent=4)
        rule()

    # Previous feedback
    if prev_feedback:
        lines.append("  Last feedback:")
        wrap_line(f'"{prev_feedback}"', indent=4)
        rule()

    # Commands to run
    if commands:
        lines.append(f"  📦 Run these commands in:")
        wrap_line(str(project_dir), indent=4)
        lines.append("")
        for cmd in commands:
            lines.append(f"    $ {cmd}")
        rule()

    # Errors grouped by kind
    ErrorKind = cf["ErrorKind"]
    kind_order = [
        ErrorKind.SHADCN_NOT_INITIALIZED,
        ErrorKind.SHADCN_NOT_INSTALLED,
        ErrorKind.LIB_NOT_INSTALLED,
        ErrorKind.MISSING_DEPENDS_ON,
        ErrorKind.IMPORT_PATH_WRONG,
        ErrorKind.IMPORT_PATH_ALIAS,
        ErrorKind.MISSING_EXPORT,
        ErrorKind.TYPE_ERROR,
        ErrorKind.OTHER,
    ]

    # Friendly descriptions per kind
    kind_labels = {
        ErrorKind.SHADCN_NOT_INITIALIZED: "shadcn not initialized (run: npx shadcn@latest init)",
        ErrorKind.SHADCN_NOT_INSTALLED:   "shadcn components not installed",
        ErrorKind.LIB_NOT_INSTALLED:      "npm packages not installed",
        ErrorKind.MISSING_DEPENDS_ON:     "files referenced in plan but missing on disk",
        ErrorKind.IMPORT_PATH_WRONG:      "wrong import paths (auto-fixable next round)",
        ErrorKind.IMPORT_PATH_ALIAS:      "@/ alias not resolving",
        ErrorKind.MISSING_EXPORT:         "missing exports",
        ErrorKind.TYPE_ERROR:             "TypeScript type errors",
        ErrorKind.OTHER:                  "other errors",
    }

    by_kind: dict[Any, list[Any]] = {}
    for e in errors:
        by_kind.setdefault(e.kind, []).append(e)

    if errors:
        lines.append(f"  🔴 {len(errors)} error(s) remaining:")
        lines.append("")
        for kind in kind_order:
            errs = by_kind.get(kind, [])
            if not errs:
                continue
            label = kind_labels.get(kind, kind.value)
            lines.append(f"  [{len(errs)}] {label}")

            # Show up to 5 errors per kind
            shown = errs[:5]
            for e in shown:
                # Trim file path to relative-looking form
                fpath = e.file
                if str(project_dir) in fpath:
                    fpath = fpath.replace(str(project_dir) + "/", "")
                wrap_line(f"• {fpath}:{e.line} — {e.message[:80]}", indent=6)
            if len(errs) > 5:
                lines.append(f"      … and {len(errs) - 5} more")
            lines.append("")
        rule()

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Artifact writers
# ════════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _error_to_dict(e: Any) -> dict:
    return {
        "file":    e.file,
        "line":    e.line,
        "col":     e.col,
        "code":    e.code,
        "message": e.message,
        "kind":    e.kind.value,
    }


def _write_loop_json(
    *,
    project_dir:    Path,
    status:         str,
    round_num:      int,
    max_rounds:     int,
    rounds_history: list[dict],
    remaining:      list[Any],
    commands:       list[str],
    dry_run:        bool,
) -> None:
    """Overwrite compile_fixer/loop.json — machine-readable current state."""
    payload = {
        "generated_at":  _now_iso(),
        "project_dir":   str(project_dir),
        "status":        status,           # "clean" | "in_progress" | "abandoned"
        "round":         round_num,
        "max_rounds":    max_rounds,
        "remaining_errors": [_error_to_dict(e) for e in remaining],
        "pending_commands": commands,
        "rounds":        rounds_history,
    }
    if not dry_run:
        COMPILE_FIXER_LOOP.parent.mkdir(parents=True, exist_ok=True)
        COMPILE_FIXER_LOOP.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        track_write(COMPILE_FIXER_LOOP)


def _append_fixer_log(
    *,
    project_dir:    Path,
    status:         str,
    total_rounds:   int,
    rounds_history: list[dict],
    remaining:      list[Any],
    dry_run:        bool,
) -> None:
    """Append one session entry to compile_fixer/fixer_log.md."""
    if dry_run:
        return

    ts = _now_iso()
    counts = Counter(e.kind.value for e in remaining)
    count_str = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) if counts else "none"

    section_lines = [
        f"## Session {ts}",
        f"",
        f"- **Project**: `{project_dir}`",
        f"- **Status**: `{status}`",
        f"- **Rounds used**: {total_rounds}",
        f"- **Errors remaining**: {len(remaining)} ({count_str})",
        f"",
        f"### Round history",
        f"",
    ]
    for r in rounds_history:
        rn = r["round"]
        err_count = r["error_count"]
        cmds = r.get("commands", [])
        fb = r.get("feedback", "")
        section_lines.append(f"**Round {rn}** — {err_count} error(s)")
        if cmds:
            section_lines.append(f"  Commands suggested: {', '.join(f'`{c}`' for c in cmds)}")
        if fb:
            section_lines.append(f"  Feedback: _{fb}_")
        section_lines.append("")

    if remaining:
        section_lines += ["### Remaining errors", ""]
        for e in remaining[:20]:
            section_lines.append(f"- `{e.file}:{e.line}` [{e.code}] {e.kind.value}: {e.message[:80]}")
        if len(remaining) > 20:
            section_lines.append(f"- … and {len(remaining) - 20} more")
        section_lines.append("")

    section_lines.append("---\n")
    block = "\n".join(section_lines)

    log_path = Path(str(COMPILE_FIXER_LOG))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(block)
    track_write(COMPILE_FIXER_LOG)


# ════════════════════════════════════════════════════════════════════════════
# Flush stdin helper (same pattern as 09_debugger.py)
# ════════════════════════════════════════════════════════════════════════════

def _flush_stdin() -> None:
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# One round: tsc + classify + escalate
# ════════════════════════════════════════════════════════════════════════════

def _run_one_round(
    project_dir:     Path,
    manifest_path:   Path | None,
    verbose:         bool,
    cf:              dict,
) -> tuple[list[Any], list[str], list[str]]:
    """
    Chạy tsc, classify, trả về:
      (errors_in_manifest, human_commands, warnings)

    errors_in_manifest: CompileError objects còn lại sau auto-classify.
    human_commands:     list[str] commands user cần chạy.
    warnings:           list[str] env warnings.
    """
    # Load manifest
    manifest = cf["load_manifest"](
        manifest_path=manifest_path,
        project_dir=project_dir,
    )
    ctx = cf["parse_manifest"](manifest, project_dir)
    warnings = cf["validate_manifest_assumptions"](ctx, project_dir)

    # Pre-tsc: cross-check depends_on
    plan_errors = cf["cross_check_plan"](ctx, project_dir)

    # Run tsc
    print(f"[9c] Running tsc …")
    try:
        raw = cf["_run_tsc"](project_dir)
    except Exception as exc:
        print(f"[9c] tsc failed: {exc}")
        return [], [], [str(exc)]

    all_errors = cf["parse_tsc_output"](raw, project_dir)

    if not all_errors:
        print("[9c] ✓ No TypeScript errors.")
        if plan_errors:
            return plan_errors, [], warnings
        return [], [], warnings

    # Filter to manifest files
    errors, outside = cf["_filter_errors_to_manifest"](all_errors, ctx.executor_files)

    if outside and verbose:
        print(f"[9c] {len(outside)} error(s) outside manifest files (skipped)")

    if not errors:
        print("[9c] ✓ No errors in manifest files.")
        if plan_errors:
            return plan_errors, [], warnings
        return [], [], warnings

    # Classify
    errors = cf["classify_errors"](errors, project_dir)
    errors = errors + plan_errors

    # Build human commands from escalated kinds
    ErrorKind = cf["ErrorKind"]
    pm = cf["_detect_package_manager"](project_dir)
    pm_name = cf["_pm_name"](pm)

    commands: list[str] = []
    seen_cmds: set[str] = set()

    def _add_cmd(cmd: str) -> None:
        if cmd not in seen_cmds:
            seen_cmds.add(cmd)
            commands.append(cmd)

    for e in errors:
        k = e.kind
        if k == ErrorKind.LIB_NOT_INSTALLED:
            pkg = e.meta.get("package", "")
            if pkg:
                _add_cmd(f"{pm_name} install {pkg}")
            else:
                _add_cmd(f"{pm_name} install")

        elif k == ErrorKind.SHADCN_NOT_INITIALIZED:
            _add_cmd(f"npx shadcn@latest init")

        elif k == ErrorKind.SHADCN_NOT_INSTALLED:
            comp = e.meta.get("component", "")
            if comp:
                if pm_name == "npm":
                    _add_cmd(f"npx shadcn@latest add {comp} --yes")
                else:
                    _add_cmd(f"{pm_name} dlx shadcn@latest add {comp} --yes")

    return errors, commands, warnings


# ════════════════════════════════════════════════════════════════════════════
# Interactive loop
# ════════════════════════════════════════════════════════════════════════════

def run_interactive_loop(
    project_dir:      Path,
    manifest_path:    Path | None,
    max_rounds:       int,
    dry_run:          bool,
    no_interactive:   bool,
    verbose:          bool,
    cf:               dict,
) -> str:
    """
    Main loop. Returns final status: "clean" | "in_progress" | "abandoned".
    """
    rounds_history: list[dict] = []
    prev_feedback = ""
    status = "in_progress"

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'═' * 68}")
        print(f"[9c] Round {round_num}/{max_rounds}")
        print(f"{'═' * 68}")

        errors, commands, warnings = _run_one_round(
            project_dir, manifest_path, verbose, cf
        )

        if not errors and not commands:
            status = "clean"
            print("\n[9c] ✓ All compile errors resolved!")
            # Write final clean state
            _write_loop_json(
                project_dir=project_dir,
                status=status,
                round_num=round_num,
                max_rounds=max_rounds,
                rounds_history=rounds_history,
                remaining=[],
                commands=[],
                dry_run=dry_run,
            )
            break

        # Print action box
        box = _format_action_box(
            round_num=round_num,
            max_rounds=max_rounds,
            commands=commands,
            errors=errors,
            project_dir=project_dir,
            prev_feedback=prev_feedback,
            warnings=warnings,
            cf=cf,
        )
        print(box)

        # Record this round
        rounds_history.append({
            "round":       round_num,
            "error_count": len(errors),
            "error_kinds": dict(Counter(e.kind.value for e in errors)),
            "commands":    commands,
            "feedback":    prev_feedback,
        })

        # Write state after each round
        _write_loop_json(
            project_dir=project_dir,
            status="in_progress",
            round_num=round_num,
            max_rounds=max_rounds,
            rounds_history=rounds_history,
            remaining=errors,
            commands=commands,
            dry_run=dry_run,
        )

        if round_num >= max_rounds:
            print(f"[9c] Reached max rounds ({max_rounds}). Stopping.")
            status = "in_progress"
            break

        # ── Prompt ───────────────────────────────────────────────────────────
        if no_interactive or not sys.stdin.isatty():
            print("[9c] Non-interactive mode — stopping after first round.")
            status = "in_progress"
            break

        _flush_stdin()
        print("\n  Sau khi thực hiện các action trên, nhấn Enter để chạy lại.")
        print("  Hoặc gõ mô tả thêm (feedback) rồi Enter.")
        print('  Gõ "done" hoặc "q" để thoát.\n')

        try:
            raw_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[9c] Interrupted.")
            status = "abandoned"
            break

        if raw_input.lower() in ("done", "q", "quit", "exit"):
            print("[9c] Exiting loop at user request.")
            status = "abandoned"
            break

        prev_feedback = raw_input
        if prev_feedback:
            print(f"[9c] Feedback noted: {prev_feedback!r}")

    return status


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, parser)
    ensure_dirs()

    # Ensure compile_fixer/ dir
    cf_dir = artifact_root() / "compile_fixer"
    cf_dir.mkdir(parents=True, exist_ok=True)

    project_dir = _resolve_project_dir(args)
    manifest_path = Path(args.manifest).resolve() if args.manifest else None

    if not project_dir.exists():
        print(f"[9c] ERROR: project_dir not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 68)
    print("  COMPILE FIXER LOOP")
    print("=" * 68)
    print(f"  project_dir : {project_dir}")
    print(f"  manifest    : {manifest_path or '(auto-detect)'}")
    print(f"  max_rounds  : {args.max_rounds}")
    print(f"  dry_run     : {args.dry_run}")
    print()

    # Import compile_fixer internals
    try:
        cf = _import_compile_fixer()
    except ImportError as exc:
        print(f"[9c] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    exit_code = 0

    try:
        status = run_interactive_loop(
            project_dir=project_dir,
            manifest_path=manifest_path,
            max_rounds=args.max_rounds,
            dry_run=args.dry_run,
            no_interactive=args.no_interactive,
            verbose=args.verbose,
            cf=cf,
        )

        # Final artifacts
        # Re-run one last time to get current state for log
        final_errors, final_commands, _ = _run_one_round(
            project_dir, manifest_path, args.verbose, cf
        ) if status != "clean" else ([], [], [])

        _append_fixer_log(
            project_dir=project_dir,
            status=status,
            total_rounds=args.max_rounds,
            rounds_history=[],  # already written per-round in loop.json
            remaining=final_errors,
            dry_run=args.dry_run,
        )

        if status == "clean":
            print("\n[9c] ✓ Compile clean. Ready for next step.")
            exit_code = 0
        elif status == "abandoned":
            print(f"\n[9c] Session ended by user.")
            exit_code = 0  # user chose to exit, not an error
        else:
            remaining_count = len(final_errors)
            print(f"\n[9c] Session ended with {remaining_count} error(s) remaining.")
            print(f"[9c] Re-run when ready to continue.")
            exit_code = 1

    except Exception as exc:
        print(f"[9c] ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1

    finally:
        print_cost_summary("[9c]")
        print_artifact_summary("[9c]")
        prompt_next_step(ROLE, prefix="[9c]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()