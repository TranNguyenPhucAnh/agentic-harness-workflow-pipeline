"""
compile_fixer.py
========================
Phase 1 + Phase 2 của compile-error auto-fix pipeline.

Consumes executor manifest.json để:
  - Filter tsc errors chỉ trong files executor đã generate
  - Validate stack assumptions (shadcn, Tailwind v4, …) trước khi chạy
  - Cross-check depends_on từ full_plan.json để detect missing imports sớm
  - Skip shadcn install nếu ui_components không phải shadcn/ui

Chạy độc lập hoặc được gọi từ 09_debugger.py trước LLM repair loop.

Phase 1 — Static Collector
───────────────────────────
  Chạy `tsc --noEmit`, parse stderr, classify từng error:

    IMPORT_PATH_WRONG     : import path relative sai, file tồn tại ở chỗ khác
    IMPORT_PATH_ALIAS     : dùng '@/' alias nhưng path thực không tồn tại
    SHADCN_NOT_INSTALLED  : @/components/ui/<name> chưa được shadcn add
    LIB_NOT_INSTALLED     : module nằm trong node_modules nhưng chưa install
    MISSING_EXPORT        : file tồn tại nhưng không export tên được import
    MISSING_DEPENDS_ON    : depends_on trong plan chưa được tạo (pre-tsc check)
    TYPE_ERROR            : property/type mismatch — cần LLM
    OTHER                 : không classify được — cần LLM

Phase 2 — Auto Patcher (in-process, no side-effects ngoài source files)
───────────────────────────────────────────────────────────────────────
  Xử lý: IMPORT_PATH_WRONG, IMPORT_PATH_ALIAS
  Escalate: SHADCN_NOT_INSTALLED, LIB_NOT_INSTALLED, node_modules thiếu
  Pass-through → LLM: MISSING_EXPORT, TYPE_ERROR, MISSING_DEPENDS_ON, OTHER

Return type:
    FixerResult(status, human_actions, llm_errors)
      status = "clean" | "needs_human" | "needs_llm"

Exit codes (standalone):
    0 = clean
    1 = needs LLM repair
    2 = needs human action (install commands printed)

Usage từ debugger
─────────────────
    from modules.compile_fixer import run_compile_fixer, CompileError, FixerResult

    result: FixerResult = run_compile_fixer(
        project_dir   = SRC_DIR.parent,
        manifest_path = Path("artifacts_myapp/executor/manifest.json"),
        verbose       = args.verbose,
    )

    if result.status == "needs_human":
        # print result.human_actions, exit / prompt user
    elif result.status == "needs_llm":
        # feed result.llm_errors into LLM repair loop

Usage standalone
────────────────
    python 9b_compile_fixer.py --project my-app
    python 9b_compile_fixer.py --manifest path/to/manifest.json --dir path/to/project
    python 9b_compile_fixer.py --project my-app --dry-run
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


# ─── Error taxonomy ───────────────────────────────────────────────────────────


class ErrorKind(str, Enum):
    IMPORT_PATH_WRONG    = "IMPORT_PATH_WRONG"
    IMPORT_PATH_ALIAS    = "IMPORT_PATH_ALIAS"
    SHADCN_NOT_INSTALLED   = "SHADCN_NOT_INSTALLED"
    SHADCN_NOT_INITIALIZED = "SHADCN_NOT_INITIALIZED"
    LIB_NOT_INSTALLED    = "LIB_NOT_INSTALLED"
    MISSING_EXPORT       = "MISSING_EXPORT"
    MISSING_DEPENDS_ON   = "MISSING_DEPENDS_ON"
    TYPE_ERROR           = "TYPE_ERROR"
    OTHER                = "OTHER"


@dataclass
class CompileError:
    file:    str
    line:    int
    col:     int
    code:    str
    message: str
    kind:    ErrorKind
    meta:    dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.file}:{self.line}:{self.col} "
            f"[{self.code}] {self.kind.value}: {self.message[:80]}"
        )


@dataclass
class FixerResult:
    status:        Literal["clean", "needs_human", "needs_llm"]
    human_actions: list[str]
    llm_errors:    list[CompileError]
    warnings:      list[str] = field(default_factory=list)  # env warnings


# ─── Known shadcn components ──────────────────────────────────────────────────

_SHADCN_COMPONENTS: frozenset[str] = frozenset({
    "accordion", "alert", "alert-dialog", "aspect-ratio", "avatar",
    "badge", "breadcrumb", "button", "calendar", "card", "carousel",
    "chart", "checkbox", "collapsible", "command", "context-menu",
    "dialog", "drawer", "dropdown-menu", "form", "hover-card", "input",
    "input-otp", "label", "menubar", "navigation-menu", "pagination",
    "popover", "progress", "radio-group", "resizable", "scroll-area",
    "select", "separator", "sheet", "sidebar", "skeleton", "slider",
    "sonner", "switch", "table", "tabs", "textarea", "toast", "toggle",
    "toggle-group", "tooltip",
})


# ─── Utilities ────────────────────────────────────────────────────────────────

def _detect_pm_from_lockfile(project_dir: Path) -> str | None:
    """Detect which PM the lockfile implies, regardless of PATH availability.
    Also checks packageManager field in package.json (used by corepack/shadcn).
    """
    # Check packageManager field first — shadcn respects this
    pkg_json = project_dir / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text())
            pm_field = pkg.get("packageManager", "")
            if pm_field.startswith("pnpm"):
                return "pnpm"
            elif pm_field.startswith("yarn"):
                return "yarn"
            elif pm_field.startswith("npm"):
                return "npm"
        except Exception:
            pass

    # Fallback to lockfile detection
    if (project_dir / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (project_dir / "yarn.lock").exists():
        return "yarn"
    if (project_dir / "package-lock.json").exists():
        return "npm"
    return None

def _lockfile_name(pm: str) -> str:
    return {
        "pnpm": "pnpm-lock.yaml",
        "yarn": "yarn.lock",
        "npm": "package-lock.json",
    }.get(pm, "lockfile")


def _detect_package_manager(project_dir: Path) -> str:
    """Detect pnpm/yarn/npm via lockfile then PATH."""
    if (project_dir / "pnpm-lock.yaml").exists() and shutil.which("pnpm"):
        return "pnpm"
    if (project_dir / "yarn.lock").exists() and shutil.which("yarn"):
        return "yarn"
    if (project_dir / "package-lock.json").exists() and shutil.which("npm"):
        return "npm"
    for pm in ("pnpm", "yarn", "npm"):
        if shutil.which(pm):
            return pm
    return "npm"


def _pm_name(pm: str) -> str:
    """Short name from full path: /opt/homebrew/bin/pnpm → pnpm."""
    return Path(pm).name


def _shadcn_cmd(pm: str, component: str) -> list[str]:
    """Build shadcn add command. npm uses npx, pnpm/yarn use dlx."""
    if _pm_name(pm) == "npm":
        npx = shutil.which("npx") or "npx"
        return [npx, "shadcn@latest", "add", component, "--yes"]
    return [pm, "dlx", "shadcn@latest", "add", component, "--yes"]


def _ensure_components_json(project_dir: Path) -> None:
    """Create minimal components.json if missing — shadcn CLI requires it."""
    cfg_path = project_dir / "components.json"
    if cfg_path.exists():
        return
    import json as _json
    default = {
        "$schema": "https://ui.shadcn.com/schema.json",
        "style": "default", "rsc": False, "tsx": True,
        "tailwind": {"config": "tailwind.config.js", "css": "src/index.css",
                      "baseColor": "slate", "cssVariables": True},
        "aliases": {"components": "@/components", "utils": "@/lib/utils"},
    }
    try:
        cfg_path.write_text(_json.dumps(default, indent=2))
        print("[compile_fixer] Created minimal components.json for shadcn CLI")
    except Exception as exc:
        print(f"[compile_fixer][warn] Could not create components.json: {exc}")


def _prompt_confirm(prompt: str) -> bool:
    """Prompt y/n. Default Y. Returns False on n/no."""
    try:
        ans = input(f"  {prompt} [Y/n]: ").strip().lower()
        return ans not in ("n", "no")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _check_node_modules(project_dir: Path) -> bool:
    """
    Returns True nếu node_modules đã install đủ.
    Kiểm tra: node_modules tồn tại + react có trong đó.
    """
    nm = project_dir / "node_modules"
    if not nm.exists():
        return False
    return (nm / "react").exists() or (nm / "@types" / "react").exists()


# ─── Escalation formatting ────────────────────────────────────────────────────


def _format_escalation_box(commands: list[str], project_dir: Path, warnings: list[str] | None = None) -> str:
    lines: list[str] = []
    lines.append("")
    
    if warnings:
        lines.append("╭─ ENVIRONMENT ISSUES ─────────────────────────────────────╮")
        for w in warnings:
            for wline in w.splitlines():
                lines.append(f"│ {wline:<57}│")
        lines.append("╰───────────────────────────────────────────────────────────╯")
        lines.append("")
    
    lines.append("╭─ MANUAL ACTION REQUIRED ─────────────────────────────────╮")
    lines.append("│ Run the following commands in:                            │")
    lines.append(f"│   {str(project_dir):<55}│")
    lines.append("│                                                           │")
    for cmd in commands:
        padded = f"  $ {cmd}"
        lines.append(f"│{padded:<59}│")
    lines.append("│                                                           │")
    lines.append("│ Then re-run the pipeline.                                 │")
    lines.append("╰───────────────────────────────────────────────────────────╯")
    lines.append("")
    return "\n".join(lines)


# ─── Manifest loader + validator ──────────────────────────────────────────────


def load_manifest(manifest_path: Path | None = None, project_dir: Path | None = None) -> dict[str, Any]:
    candidates: list[Path] = []

    if manifest_path is not None:
        candidates.append(Path(manifest_path))

    try:
        _here = Path(__file__).parent.parent
        sys.path.insert(0, str(_here))
        from artifacts.paths import EXECUTOR_OVERWRITE_MANIFEST
        candidates.append(Path(str(EXECUTOR_OVERWRITE_MANIFEST)))
    except Exception:
        pass

    if project_dir is not None:
        artifact_root = project_dir.parent
        candidates.append(artifact_root / "executor" / "manifest.json")
        candidates.append(project_dir / "manifest.json")

    candidates.append(Path.cwd() / "executor" / "manifest.json")
    candidates.append(Path.cwd() / "manifest.json")

    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                print(f"[compile_fixer] Manifest loaded from: {path}")
                return data
            except Exception as exc:
                print(f"[compile_fixer][warn] Failed to parse manifest {path}: {exc}")

    return {}


@dataclass
class ManifestContext:
    """Parsed, validated context từ manifest.json."""
    executor_files: set[str]
    uses_shadcn:    bool
    uses_tailwind4: bool
    scope:          str
    mode:           str
    plan_path:      Path | None
    raw:            dict[str, Any]


def parse_manifest(manifest: dict[str, Any], project_dir: Path) -> ManifestContext:
    stack = manifest.get("stack", {})
    ui_components = (stack.get("ui_components") or "").lower()
    styling       = (stack.get("styling") or "").lower()

    plan_ref  = manifest.get("plan")
    plan_path = None
    if plan_ref:
        artifact_root = project_dir.parent
        for base in (artifact_root, project_dir):
            candidate = base / plan_ref
            if candidate.exists():
                plan_path = candidate
                break

    return ManifestContext(
        executor_files = set(manifest.get("files", [])),
        uses_shadcn    = "shadcn" in ui_components,
        uses_tailwind4 = "v4" in styling or "tailwind css v4" in styling,
        scope          = manifest.get("scope", ""),
        mode           = manifest.get("mode", ""),
        plan_path      = plan_path,
        raw            = manifest,
    )


def validate_manifest_assumptions(ctx: ManifestContext, project_dir: Path) -> list[str]:
    """Validate assumptions từ manifest. Returns list[str] warnings."""
    warnings: list[str] = []

    if ctx.uses_shadcn:
        ui_dir = project_dir / "src" / "components" / "ui"
        if not ui_dir.exists():
            warnings.append(
                f"shadcn/ui declared in stack but {ui_dir} does not exist — "
                "shadcn components may not be installed yet"
            )

    if ctx.uses_tailwind4:
        pkg_json = project_dir / "package.json"
        if pkg_json.exists():
            try:
                pkg = json.loads(pkg_json.read_text())
                all_deps = {
                    **pkg.get("dependencies", {}),
                    **pkg.get("devDependencies", {}),
                }
                has_tw4 = (
                    "@tailwindcss/vite" in all_deps
                    or any(
                        k == "tailwindcss" and str(v).lstrip("^~>=").startswith("4")
                        for k, v in all_deps.items()
                    )
                )
                if not has_tw4:
                    warnings.append(
                        "Tailwind CSS v4 declared in stack but @tailwindcss/vite "
                        "not found in package.json"
                    )
            except Exception:
                pass

    missing_files = [
        f for f in ctx.executor_files
        if not (project_dir / f).exists()
    ]
    if missing_files:
        warnings.append(
            f"{len(missing_files)} executor file(s) missing on disk: "
            + ", ".join(sorted(missing_files)[:5])
            + (" …" if len(missing_files) > 5 else "")
        )

    if ctx.plan_path is None and ctx.raw.get("plan"):
        warnings.append(
            f"plan pointer '{ctx.raw['plan']}' declared but file not found — "
            "cross-check of depends_on will be skipped"
        )

    return warnings


# ─── Pre-tsc: cross-check depends_on từ full_plan.json ───────────────────────


def cross_check_plan(ctx: ManifestContext, project_dir: Path) -> list[CompileError]:
    """
    Đọc full_plan.json, kiểm tra depends_on có tồn tại trên disk không.
    Returns list[CompileError] với kind=MISSING_DEPENDS_ON.
    """
    if ctx.plan_path is None:
        return []

    try:
        plan_data = json.loads(ctx.plan_path.read_text())
    except Exception as exc:
        print(f"[compile_fixer][warn] Cannot read plan {ctx.plan_path}: {exc}")
        return []

    file_entries: list[dict[str, Any]] = []
    if isinstance(plan_data, list):
        file_entries = plan_data
    elif isinstance(plan_data, dict):
        file_entries = plan_data.get("files", [])

    errors: list[CompileError] = []

    for entry in file_entries:
        file_path_str = entry.get("path", "")
        if file_path_str not in ctx.executor_files:
            continue

        depends_on: list[str] = entry.get("depends_on", [])
        for dep in depends_on:
            dep_resolved = _resolve_dep_path(dep, project_dir)
            if dep_resolved is None:
                continue
            if not _find_file_with_extensions(dep_resolved):
                errors.append(CompileError(
                    file    = file_path_str,
                    line    = 0,
                    col     = 0,
                    code    = "PLAN",
                    message = f"depends_on '{dep}' not found on disk",
                    kind    = ErrorKind.MISSING_DEPENDS_ON,
                    meta    = {"file": file_path_str, "missing_dep": dep},
                ))

    if errors:
        print(f"[compile_fixer] Pre-tsc plan check: {len(errors)} missing depends_on detected")

    return errors


def _resolve_dep_path(dep: str, project_dir: Path) -> Path | None:
    """Resolve một depends_on entry thành absolute Path."""
    if dep.startswith("@/"):
        return project_dir / "src" / dep[2:]
    if dep.startswith((".", "/")):
        return project_dir / dep
    return None


# ─── TSC runner + parser ──────────────────────────────────────────────────────

_RE_TSC_LINE = re.compile(
    r"^(?P<file>[^(]+)\((?P<line>\d+),(?P<col>\d+)\):\s+error\s+(?P<code>TS\d+):\s+(?P<msg>.+)$"
)
_RE_IMPORT_FROM = re.compile(r"""from\s+['"](?P<path>[^'"]+)['"]""")
_RE_IMPORT_BARE = re.compile(r"""import\s+['"](?P<path>[^'"]+)['"]""")


def _run_tsc(project_dir: Path, tsconfig: str = "tsconfig.app.json") -> str:
    """
    Tìm tsconfig và chạy tsc --noEmit.
    Handles misplaced tsconfig in src/, project references, missing paths alias.
    """
    candidates: list[tuple[Path, Path]] = [
        (project_dir / tsconfig,          project_dir),
        (project_dir / "src" / tsconfig,  project_dir),
        (project_dir / "tsconfig.json",   project_dir),
        (project_dir / "src" / "tsconfig.json", project_dir),
    ]

    cfg: Path | None = None
    cwd: Path        = project_dir

    for cfg_candidate, cwd_candidate in candidates:
        if cfg_candidate.exists():
            cfg = cfg_candidate
            cwd = cwd_candidate
            break

    if cfg is None:
        return (
            f"error: no tsconfig found. Checked:\n"
            + "\n".join(f"  {c}" for c, _ in candidates)
        )

    # Detect project references root — follow to child with include/files
    try:
        cfg_data = json.loads(cfg.read_text())
        if (
            cfg_data.get("references")
            and not cfg_data.get("include")
            and not cfg_data.get("files")
            and not cfg_data.get("compilerOptions", {}).get("paths")
        ):
            for ref in cfg_data["references"]:
                ref_path = cfg.parent / ref.get("path", "")
                if ref_path.is_dir():
                    ref_path = ref_path / "tsconfig.json"
                if ref_path.exists():
                    print(f"[compile_fixer][debug] project references root, using child: {ref_path}")
                    cfg = ref_path
                    break
    except Exception:
        pass

    # ── Patch tsconfig if misplaced in src/ ──────────────────────────────────
    if cfg is not None:
        try:
            cfg_data = json.loads(cfg.read_text())
            changed = False

            if cfg.parent == project_dir / "src":
                includes = cfg_data.get("include", [])
                new_includes: list[str] = []
                for inc in includes:
                    if inc in ("**/*", ".", "./") or inc.startswith("src/"):
                        new_includes.append("src/**/*")
                    elif not inc.startswith("src"):
                        new_includes.append(f"src/{inc.lstrip('./')}")
                    else:
                        new_includes.append(inc)
                seen: set[str] = set()
                new_includes = [x for x in new_includes if not (x in seen or seen.add(x))]  # type: ignore
                if not new_includes:
                    new_includes = ["src/**/*"]
                cfg_data["include"] = new_includes

                compiler_opts = cfg_data.get("compilerOptions", {})
                paths = compiler_opts.get("paths", {})
                if "@/*" not in paths:
                    paths["@/*"] = ["./src/*"]
                    compiler_opts["paths"] = paths
                    cfg_data["compilerOptions"] = compiler_opts

                new_cfg = project_dir / cfg.name
                new_cfg.write_text(json.dumps(cfg_data, indent=2))
                cfg.unlink()
                cfg = new_cfg
                cwd = project_dir
                changed = True
                print(
                    f"[compile_fixer][warn] Moved misplaced tsconfig from src/ to project root. "
                    f"include={new_includes}"
                )

            excludes = list(cfg_data.get("exclude", []))
            if "node_modules" not in excludes:
                excludes.append("node_modules")
                cfg_data["exclude"] = excludes
                changed = True

            compiler_opts = cfg_data.get("compilerOptions", {})
            paths = compiler_opts.get("paths", {})
            if "@/*" not in paths:
                paths["@/*"] = ["./src/*"]
                compiler_opts["paths"] = paths
                cfg_data["compilerOptions"] = compiler_opts
                changed = True
                print("[compile_fixer] Added @/* paths alias to tsconfig")

            if "baseUrl" not in compiler_opts and "paths" in compiler_opts:
                compiler_opts["baseUrl"] = "."
                cfg_data["compilerOptions"] = compiler_opts
                changed = True

            if changed:
                cfg.write_text(json.dumps(cfg_data, indent=2))
        except Exception as exc:
            print(f"[compile_fixer][warn] Could not patch tsconfig: {exc}")

    print(f"[compile_fixer][debug] tsc config: {cfg} (cwd: {cwd})")

    result = subprocess.run(
        ["npx", "tsc", "--noEmit", "--skipLibCheck", "--project", str(cfg)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )

    print(f"[compile_fixer][debug] tsc returncode: {result.returncode}")
    output = result.stdout + result.stderr
    if output.strip():
        print(f"[compile_fixer][debug] tsc output ({len(output)} chars): {output[:500]}")

    return output


def parse_tsc_output(raw: str, project_dir: Path) -> list[CompileError]:
    """Parse raw tsc output thành list[CompileError] chưa classified."""
    errors: list[CompileError] = []
    seen:   set[tuple[str, int, str]] = set()

    for line in raw.splitlines():
        m = _RE_TSC_LINE.match(line.strip())
        if not m:
            continue

        file_str = m.group("file").strip()
        lineno   = int(m.group("line"))
        col      = int(m.group("col"))
        code     = m.group("code")
        message  = m.group("msg").strip()

        key = (file_str, lineno, code)
        if key in seen:
            continue
        seen.add(key)

        errors.append(CompileError(
            file    = file_str,
            line    = lineno,
            col     = col,
            code    = code,
            message = message,
            kind    = ErrorKind.OTHER,
        ))

    return errors


def _filter_errors_to_manifest(
    errors:         list[CompileError],
    executor_files: set[str],
    tsc_cwd:        Path | None = None,
) -> tuple[list[CompileError], list[CompileError]]:
    if not executor_files:
        return errors, []

    def _normalize(path: str) -> str:
        p = path.replace("\\", "/")
        if tsc_cwd:
            root = str(tsc_cwd).replace("\\", "/").rstrip("/") + "/"
            if p.startswith(root):
                p = p[len(root):]
        return p

    normalized_manifest = {f.replace("\\", "/") for f in executor_files}

    in_manifest:      list[CompileError] = []
    outside_manifest: list[CompileError] = []

    for e in errors:
        norm = _normalize(e.file)
        if norm in normalized_manifest or ("src/" + norm) in normalized_manifest:
            if norm not in normalized_manifest:
                e.file = "src/" + norm
            else:
                e.file = norm
            in_manifest.append(e)
        else:
            outside_manifest.append(e)

    return in_manifest, outside_manifest


# ─── Classifier ───────────────────────────────────────────────────────────────


def _read_source_line(project_dir: Path, rel_file: str, lineno: int) -> str:
    try:
        path = project_dir / rel_file
        if not path.exists():
            return ""
        lines = path.read_text(errors="replace").splitlines()
        if 1 <= lineno <= len(lines):
            return lines[lineno - 1]
    except Exception:
        pass
    return ""


def _extract_import_path(source_line: str) -> str | None:
    for pattern in (_RE_IMPORT_FROM, _RE_IMPORT_BARE):
        m = pattern.search(source_line)
        if m:
            return m.group("path")
    return None


def _resolve_alias(import_path: str, project_dir: Path) -> Path | None:
    """Resolve '@/' alias → path thực."""
    if not import_path.startswith("@/"):
        return None

    for tsconfig_name in ("tsconfig.app.json", "tsconfig.json"):
        tsconfig_path = project_dir / tsconfig_name
        if not tsconfig_path.exists():
            continue
        try:
            data  = json.loads(tsconfig_path.read_text())
            co    = data.get("compilerOptions", {})
            paths = co.get("paths", {})
            base  = co.get("baseUrl", ".")
            for alias_key, alias_vals in paths.items():
                if alias_key in ("@/*", "@/"):
                    for val in alias_vals:
                        root      = val.rstrip("/*").rstrip("/")
                        rel       = import_path[2:]
                        candidate = project_dir / base / root / rel
                        return candidate
        except Exception:
            pass

    # Fallback: @/ → src/
    return project_dir / "src" / import_path[2:]


def _find_file_with_extensions(base: Path) -> Path | None:
    if base.exists():
        return base
    for ext in (".tsx", ".ts", ".jsx", ".js", "/index.tsx", "/index.ts"):
        candidate = Path(str(base) + ext)
        if candidate.exists():
            return candidate
    return None


def _is_shadcn_component(import_path: str) -> tuple[bool, str]:
    """Returns (is_shadcn, component_name)."""
    patterns = [
        re.compile(r"@/components/ui/([a-z-]+)"),
        re.compile(r"\.\.?/(?:components/)?ui/([a-z-]+)"),
    ]
    for pat in patterns:
        m = pat.search(import_path)
        if m:
            component = m.group(1)
            if component in _SHADCN_COMPONENTS:
                return True, component
    return False, ""


def _is_npm_package(import_path: str) -> bool:
    return not import_path.startswith((".", "/", "@/"))


def classify_errors(
    errors:      list[CompileError],
    project_dir: Path,
) -> list[CompileError]:
    """Classify từng error vào ErrorKind. Mutates in-place."""
    extra_errors: list[CompileError] = []

    for err in errors:
        if err.kind == ErrorKind.MISSING_DEPENDS_ON:
            continue

        source_line = _read_source_line(project_dir, err.file, err.line)
        err.meta["source_line"] = source_line          # always store for LLM context
        import_path = _extract_import_path(source_line)

        # ── TS2307: Cannot find module ────────────────────────────────────────
        if err.code == "TS2307":
            if import_path is None:
                err.kind = ErrorKind.OTHER
                continue

            is_shadcn, component = _is_shadcn_component(import_path)
            if is_shadcn:
                err.kind = ErrorKind.SHADCN_NOT_INSTALLED
                err.meta = {"component": component, "import_str": import_path}
                continue

            if _is_npm_package(import_path):
                pkg = import_path.split("/")[0]
                if import_path.startswith("@"):
                    pkg = "/".join(import_path.split("/")[:2])
                # shadcn init deps → classify as SHADCN_NOT_INITIALIZED
                if pkg in _SHADCN_INIT_DEPS:
                    err.kind = ErrorKind.SHADCN_NOT_INITIALIZED
                    err.meta = {"package": pkg, "reason": "shadcn init not run"}
                else:
                    err.kind = ErrorKind.LIB_NOT_INSTALLED
                    err.meta = {"package": pkg}
                continue

            if import_path.startswith("."):
                current_file = project_dir / err.file
                resolved     = (current_file.parent / import_path).resolve()
                actual       = _find_file_with_extensions(resolved)

                if actual is not None:
                    try:
                        rel = actual.relative_to(project_dir / "src")
                        candidate = f"@/{rel}".replace("\\", "/")
                    except ValueError:
                        candidate = None
                    err.kind = ErrorKind.IMPORT_PATH_WRONG
                    err.meta = {"import_str": import_path, "candidate": candidate}
                else:
                    is_shadcn2, component2 = _is_shadcn_component(import_path)
                    if is_shadcn2:
                        err.kind = ErrorKind.SHADCN_NOT_INSTALLED
                        err.meta = {"component": component2, "import_str": import_path}
                    else:
                        err.kind = ErrorKind.IMPORT_PATH_WRONG
                        err.meta = {"import_str": import_path, "candidate": None}
                continue

            if import_path.startswith("@/"):
                # Special case: @/lib/utils → shadcn init artifact
                if import_path in ("@/lib/utils", "@/lib/utils.ts"):
                    utils_path = project_dir / _SHADCN_UTILS_PATH
                    if not utils_path.exists():
                        err.kind = ErrorKind.SHADCN_NOT_INITIALIZED
                        err.meta = {
                            "package": "@/lib/utils",
                            "reason":  "shadcn init not run — src/lib/utils.ts missing",
                        }
                        continue

                resolved = _resolve_alias(import_path, project_dir)
                actual   = _find_file_with_extensions(resolved) if resolved else None
                err.kind = ErrorKind.IMPORT_PATH_ALIAS
                err.meta = {
                    "import_str": import_path,
                    "resolved":   str(actual) if actual else None,
                }
                continue

            err.kind = ErrorKind.OTHER

        # ── TS2305: Module has no exported member ─────────────────────────────
        elif err.code == "TS2305":
            m = re.search(
                r"Module '([^']+)' has no exported member '([^']+)'",
                err.message,
            )
            if m:
                err.kind = ErrorKind.MISSING_EXPORT
                err.meta = {"from_module": m.group(1), "name": m.group(2)}
            else:
                err.kind = ErrorKind.OTHER

        # ── TS2875: jsx-runtime not found → missing @types/react ──────────────
        elif err.code == "TS2875":
            err.kind = ErrorKind.LIB_NOT_INSTALLED
            err.meta = {"package": "@types/react"}
            from dataclasses import replace as _dc_replace
            dom_err = _dc_replace(err, meta={"package": "@types/react-dom"})
            extra_errors.append(dom_err)

        # ── Type errors → LLM ─────────────────────────────────────────────────
        elif err.code in ("TS2339", "TS2345", "TS2322", "TS2741", "TS2554"):
            err.kind = ErrorKind.TYPE_ERROR

        else:
            err.kind = ErrorKind.OTHER

    errors.extend(extra_errors)
    return errors


# ─── Phase 2: Auto Patcher (in-process only) ─────────────────────────────────


def _rewrite_import(file_path: Path, old_import: str, new_import: str) -> bool:
    """Thay thế import path trong file. Handles single và double quotes."""
    try:
        src = file_path.read_text(errors="replace")
        pattern = re.compile(
            r"""(from\s+|import\s+)(['"])""" + re.escape(old_import) + r"""(['"])"""
        )
        new_src, count = pattern.subn(
            lambda m: f"{m.group(1)}{m.group(2)}{new_import}{m.group(3)}",
            src,
        )
        if count > 0:
            file_path.write_text(new_src)
            return True
    except Exception as exc:
        print(f"  [compile_fixer][warn] rewrite_import failed: {exc}")
    return False


def _fix_import_path_wrong(
    err:         CompileError,
    project_dir: Path,
    dry_run:     bool,
) -> bool:
    """Fix IMPORT_PATH_WRONG: rewrite relative path to correct candidate."""
    import_str = err.meta.get("import_str", "")
    candidate  = err.meta.get("candidate")
    if not import_str or not candidate:
        return False

    file_path = project_dir / err.file
    if not file_path.exists():
        return False

    # Strip extension — TypeScript imports không dùng extension
    new_import = candidate
    for ext in (".tsx", ".ts", ".jsx", ".js"):
        if new_import.endswith(ext):
            new_import = new_import[: -len(ext)]
            break

    print(f"  [fix] IMPORT_PATH_WRONG: {err.file}:{err.line}")
    print(f"        {import_str!r} → {new_import!r}")

    if dry_run:
        return True
    return _rewrite_import(file_path, import_str, new_import)


def _fix_import_path_alias(
    err:         CompileError,
    project_dir: Path,
    dry_run:     bool,
) -> bool:
    """Fix IMPORT_PATH_ALIAS: rewrite alias path nếu resolved file tồn tại."""
    import_str = err.meta.get("import_str", "")
    resolved   = err.meta.get("resolved")
    if not import_str or not resolved:
        return False

    file_path = project_dir / err.file
    if not file_path.exists():
        return False

    # resolved là absolute path → convert lại thành @/ alias
    resolved_path = Path(resolved)
    try:
        rel = resolved_path.relative_to(project_dir / "src")
        new_import = f"@/{rel}".replace("\\", "/")
    except ValueError:
        return False

    # Strip extension
    for ext in (".tsx", ".ts", ".jsx", ".js"):
        if new_import.endswith(ext):
            new_import = new_import[: -len(ext)]
            break

    if new_import == import_str:
        return False

    print(f"  [fix] IMPORT_PATH_ALIAS: {err.file}:{err.line}")
    print(f"        {import_str!r} → {new_import!r}")

    if dry_run:
        return True
    return _rewrite_import(file_path, import_str, new_import)



# ─── Shadcn init check ────────────────────────────────────────────────────────

# Packages mà `shadcn init` install — cần thiết cho TẤT CẢ shadcn components
_SHADCN_INIT_DEPS: frozenset[str] = frozenset({
    "class-variance-authority",
    "clsx",
    "tailwind-merge",
    "lucide-react",
})

# File mà `shadcn init` tạo ra
_SHADCN_UTILS_PATH = "src/lib/utils.ts"


def _shadcn_is_initialized(project_dir: Path) -> tuple[bool, list[str]]:
    """
    Detect xem `shadcn init` đã chạy chưa.

    Kiểm tra 2 dấu hiệu:
      1. src/lib/utils.ts tồn tại (file shadcn init tạo ra)
      2. class-variance-authority có trong node_modules

    Returns:
        (initialized, missing_items)
    """
    missing: list[str] = []

    utils_path = project_dir / _SHADCN_UTILS_PATH
    if not utils_path.exists():
        missing.append(f"File `{_SHADCN_UTILS_PATH}` not found")

    cva_path = project_dir / "node_modules" / "class-variance-authority"
    if not cva_path.exists():
        missing.append("Package `class-variance-authority` not in node_modules")

    return len(missing) == 0, missing


def _run_shadcn_init(project_dir: Path, pm: str, dry_run: bool) -> bool:
    """
    Run `shadcn init` to create src/lib/utils.ts and install init deps.
    Must be called BEFORE `shadcn add <component>`.

    Returns True if init succeeded (or was skipped in dry_run).
    """
    initialized, missing = _shadcn_is_initialized(project_dir)
    if initialized:
        return True

    print(f"[compile_fixer] shadcn not initialized — missing: {missing}")
    print(f"[compile_fixer] Running shadcn init ...")

    if dry_run:
        print(f"[compile_fixer][dry-run] Would run: shadcn init")
        return True

    _ensure_components_json(project_dir)

    # Build init command
    if _pm_name(pm) == "npm":
        npx = shutil.which("npx") or "npx"
        cmd = [npx, "shadcn@latest", "init", "--yes", "--defaults"]
    else:
        cmd = [pm, "dlx", "shadcn@latest", "init", "--yes", "--defaults"]

    print(f"[compile_fixer] Command: {' '.join(cmd)}")
    if not _prompt_confirm("Run shadcn init?"):
        print("[compile_fixer] Skipped shadcn init — components may fail to install.")
        return False

    try:
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            print(f"[compile_fixer][warn] shadcn init failed (rc={result.returncode}):")
            print(f"  stdout: {result.stdout[:400]}")
            print(f"  stderr: {result.stderr[:400]}")
            return False
        print("[compile_fixer] ✓ shadcn init completed")
        return True
    except Exception as exc:
        print(f"[compile_fixer][warn] shadcn init exception: {exc}")
        return False


# ─── Phase 2: orchestrator ────────────────────────────────────────────────────

def _validate_package_manager(pm: str, project_dir: Path) -> tuple[str, list[str]]:
    """
    Validate PM is available in PATH. If not, return warnings and fallback.
    
    Returns:
        (effective_pm, warnings)
        - effective_pm: PM to use for commands (may differ from detected)
        - warnings: list of warning strings to prepend to escalation
    """
    warnings: list[str] = []
    pm_short = _pm_name(pm)
    
    # Check if detected PM is actually in PATH
    if not shutil.which(pm_short):
        warnings.append(
            f"⚠ Detected '{pm_short}' from lockfile but it is NOT in PATH.\n"
            f"  Install it first:  npm install -g {pm_short}\n"
            f"  Or activate via:   corepack enable"
        )
        # Fallback: find any available PM
        for fallback in ("pnpm", "yarn", "npm"):
            if shutil.which(fallback):
                warnings.append(f"  Falling back to '{fallback}' for commands below.")
                return fallback, warnings
        # Nothing available at all
        warnings.append("  ✗ No package manager (pnpm/yarn/npm) found in PATH.")
        warnings.append("    Install Node.js first: https://nodejs.org/")
        return "npm", warnings
    
    # PM exists — but shadcn uses spawn internally, so also check npx/dlx availability
    if pm_short == "npm" and not shutil.which("npx"):
        warnings.append("⚠ 'npx' not found in PATH — shadcn CLI commands will fail.")
    
    return pm_short, warnings


def _build_escalation_commands(
    errors:      list[CompileError],
    project_dir: Path,
    pm:          str,
    uses_shadcn: bool,
) -> tuple[list[str], list[str]]:
    """
    Gom SHADCN_NOT_INSTALLED và LIB_NOT_INSTALLED thành commands cho human.
    Returns (commands, warnings).
    """
    effective_pm, warnings = _validate_package_manager(pm, project_dir)
    commands: list[str] = []

    # Check if shadcn will fail due to PM mismatch
    # shadcn detects PM from lockfile, not from how you invoke it
    lockfile_pm = _detect_pm_from_lockfile(project_dir)
    shadcn_pm_available = shutil.which(lockfile_pm) is not None if lockfile_pm else True

    # shadcn init → must run BEFORE add (prepend to commands)
    init_needed = any(e.kind == ErrorKind.SHADCN_NOT_INITIALIZED for e in errors)
    if init_needed and uses_shadcn:
        if effective_pm == "npm":
            commands.append("npx shadcn@latest init --yes --defaults")
        else:
            commands.append(f"{effective_pm} dlx shadcn@latest init --yes --defaults")

    # Shadcn components → 1 batch command
    if uses_shadcn:
        shadcn_components = list(dict.fromkeys(
            e.meta.get("component", "")
            for e in errors
            if e.kind == ErrorKind.SHADCN_NOT_INSTALLED and e.meta.get("component")
        ))
        if shadcn_components:
            if not shadcn_pm_available:
                warnings.append(
                    f"⚠ shadcn CLI will internally spawn '{lockfile_pm}' (detected from lockfile)\n"
                    f"  but '{lockfile_pm}' is NOT in PATH → shadcn add will fail with ENOENT.\n"
                    f"  Fix: install {lockfile_pm} first, OR delete {_lockfile_name(lockfile_pm)} and use npm."
                )
                # Still suggest the command but prepend the PM install
                commands.append(f"npm install -g {lockfile_pm}")

            if effective_pm == "npm":
                cmd = f"npx shadcn@latest add {' '.join(shadcn_components)} --yes"
            else:
                cmd = f"{effective_pm} dlx shadcn@latest add {' '.join(shadcn_components)} --yes"
            commands.append(cmd)

    # Lib packages → 1 batch command
    lib_packages = list(dict.fromkeys(
        e.meta.get("package", "")
        for e in errors
        if e.kind == ErrorKind.LIB_NOT_INSTALLED and e.meta.get("package")
    ))
    if lib_packages:
        if effective_pm in ("pnpm", "yarn"):
            cmd = f"{effective_pm} add {' '.join(lib_packages)}"
        else:
            cmd = f"npm install {' '.join(lib_packages)}"
        commands.append(cmd)

    return commands, warnings


def _auto_patch(
    errors:      list[CompileError],
    project_dir: Path,
    dry_run:     bool,
    verbose:     bool,
    pm:          str,
    uses_shadcn: bool,
) -> tuple[list[CompileError], list[CompileError], list[str], list[str]]:
    """
    Phase 2 orchestrator.

    Returns:
        (fixed, llm_remaining, human_commands, warnings)
        - fixed: errors đã fix in-process
        - llm_remaining: errors cần LLM
        - human_commands: commands cần human chạy (escalation)
        - warnings: environment warnings from PM validation
    """
    fixed:         list[CompileError] = []
    llm_remaining: list[CompileError] = []
    escalated:     list[CompileError] = []

    for err in errors:
        if err.kind == ErrorKind.IMPORT_PATH_WRONG:
            ok = _fix_import_path_wrong(err, project_dir, dry_run)
            (fixed if ok else llm_remaining).append(err)

        elif err.kind == ErrorKind.IMPORT_PATH_ALIAS:
            ok = _fix_import_path_alias(err, project_dir, dry_run)
            (fixed if ok else llm_remaining).append(err)

        elif err.kind in (
            ErrorKind.SHADCN_NOT_INITIALIZED,
            ErrorKind.SHADCN_NOT_INSTALLED,
            ErrorKind.LIB_NOT_INSTALLED,
        ):
            escalated.append(err)

        else:
            llm_remaining.append(err)

    # Build escalation commands from escalated errors
    human_commands, esc_warnings = _build_escalation_commands(escalated, project_dir, pm, uses_shadcn)

    return fixed, llm_remaining, human_commands, esc_warnings


# ─── Ensure tsconfig paths ────────────────────────────────────────────────────


def _ensure_tsconfig_paths(project_dir: Path) -> None:
    _JUNK_INCLUDES: frozenset[str] = frozenset({
        "tsconfig.json", "tsconfig.node.json", "tsconfig.app.json",
        "vite.config.ts", "vite.config.js",
    })
    _BARE_SRC_FILES = re.compile(r"^[a-zA-Z0-9_.-]+\.(ts|tsx|js|jsx)$")

    for cfg_name in ("tsconfig.app.json", "tsconfig.json"):
        cfg_path = project_dir / cfg_name
        if not cfg_path.exists():
            continue
        try:
            cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
            opts     = cfg_data.setdefault("compilerOptions", {})
            changed  = False

            paths = opts.get("paths", {})
            if "@/*" not in paths:
                paths["@/*"] = ["./src/*"]
                opts["paths"] = paths
                changed = True
                print(f"[compile_fixer] Added @/* paths alias to {cfg_name}")

            if "baseUrl" not in opts:
                opts["baseUrl"] = "."
                changed = True
                print(f"[compile_fixer] Added baseUrl='.' to {cfg_name}")

            if not opts.get("skipLibCheck"):
                opts["skipLibCheck"] = True
                changed = True
                print(f"[compile_fixer] Added skipLibCheck=true to {cfg_name}")

            if not opts.get("noEmit"):
                opts["noEmit"] = True
                changed = True

            excludes = list(cfg_data.get("exclude", []))
            if "node_modules" not in excludes:
                excludes.append("node_modules")
                cfg_data["exclude"] = excludes
                changed = True
                print(f"[compile_fixer] Added node_modules to exclude in {cfg_name}")

            if "references" in cfg_data:
                del cfg_data["references"]
                changed = True
                print(f"[compile_fixer] Removed project references from {cfg_name}")

            raw_includes: list[str] = cfg_data.get("include", [])
            if raw_includes:
                clean: list[str] = []
                for entry in raw_includes:
                    if entry in _JUNK_INCLUDES:
                        continue
                    if _BARE_SRC_FILES.match(entry):
                        continue
                    clean.append(entry)

                if "src/**/*" not in clean:
                    clean.insert(0, "src/**/*")

                seen: set[str] = set()
                deduped = [x for x in clean if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]

                if deduped != raw_includes:
                    removed = set(raw_includes) - set(deduped)
                    print(
                        f"[compile_fixer] Cleaned include in {cfg_name}: "
                        f"removed {sorted(removed)}, kept {deduped}"
                    )
                    cfg_data["include"] = deduped
                    changed = True
            else:
                cfg_data["include"] = ["src/**/*"]
                changed = True

            if changed:
                cfg_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[compile_fixer][warn] Could not patch {cfg_name}: {exc}")
        break


# ─── Main entry point ─────────────────────────────────────────────────────────


def run_compile_fixer(
    project_dir:   Path,
    *,
    manifest_path: Path | None            = None,
    manifest:      dict[str, Any] | None  = None,
    dry_run:       bool = False,
    verbose:       bool = False,
    max_rounds:    int  = 3,
    token_budget:  int  = 16000,
) -> FixerResult:
    """
    Chạy pre-tsc plan check + Phase 1 + Phase 2, loop tối đa max_rounds lần.

    Args:
        project_dir:   Root chứa package.json / tsconfig.json.
        manifest_path: Path đến manifest.json (optional).
        manifest:      Dict đã load sẵn (optional).
        dry_run:       Classify và plan nhưng không write files.
        verbose:       In thêm detail.
        max_rounds:    Số vòng auto-fix + tsc re-check tối đa.

    Returns:
        FixerResult with status, human_actions, llm_errors.
    """
    print(f"[compile_fixer] Project: {project_dir}")

    # ── Load + parse manifest ─────────────────────────────────────────────────
    if manifest is None:
        manifest = load_manifest(manifest_path=manifest_path, project_dir=project_dir)

    if manifest:
        print(f"[compile_fixer] Manifest loaded: {len(manifest.get('files', []))} files, "
              f"scope={manifest.get('scope', '?')}, mode={manifest.get('mode', '?')}")
    else:
        print("[compile_fixer] No manifest found — running without filtering")

    ctx = parse_manifest(manifest, project_dir)

    # ── Validate assumptions ──────────────────────────────────────────────────
    warnings = validate_manifest_assumptions(ctx, project_dir)
    for w in warnings:
        print(f"[compile_fixer][warn] {w}")

    # ── Pre-flight: ensure @/* paths alias in tsconfig ──────────────────────
    if not dry_run:
        _ensure_tsconfig_paths(project_dir)

    pm = _detect_package_manager(project_dir)
    print(f"[compile_fixer] Package manager: {_pm_name(pm)}")

    # ── Pre-flight: check node_modules ────────────────────────────────────────
    if not dry_run and not _check_node_modules(project_dir):
        install_cmd = f"{_pm_name(pm)} install"
        print(f"[compile_fixer] node_modules missing or incomplete — escalating.")
        return FixerResult(
            status="needs_human",
            human_actions=[install_cmd],
            llm_errors=[],
        )

    # ── Pre-tsc: cross-check depends_on từ full_plan.json ────────────────────
    plan_errors = cross_check_plan(ctx, project_dir)

    # ── Auto-fix loop ─────────────────────────────────────────────────────────
    all_human_commands: list[str] = []
    all_llm_errors:     list[CompileError] = []
    all_esc_warnings:   list[str] = []

    for round_num in range(1, max_rounds + 1):
        print(f"\n[compile_fixer] Round {round_num}/{max_rounds} — running tsc …")

        try:
            raw = _run_tsc(project_dir)
        except subprocess.TimeoutExpired:
            print("[compile_fixer][error] tsc timed out after 60s")
            break
        except FileNotFoundError:
            print("[compile_fixer][error] npx not found — is Node.js installed?")
            break

        all_errors = parse_tsc_output(raw, project_dir)

        if not all_errors:
            print("[compile_fixer] ✓ No TypeScript errors found.")
            if plan_errors:
                return FixerResult(
                    status="needs_llm",
                    human_actions=[],
                    llm_errors=plan_errors,
                )
            return FixerResult(status="clean", human_actions=[], llm_errors=[])

        # Filter to manifest files only
        errors, errors_outside = _filter_errors_to_manifest(all_errors, ctx.executor_files)
        if errors_outside and verbose:
            print(f"[compile_fixer] Skipping {len(errors_outside)} error(s) outside manifest files")

        if not errors:
            print("[compile_fixer] ✓ No errors in manifest files.")
            if plan_errors:
                return FixerResult(status="needs_llm", human_actions=[], llm_errors=plan_errors)
            return FixerResult(status="clean", human_actions=[], llm_errors=[])

        print(f"[compile_fixer] Found {len(errors)} error(s) in manifest files")

        # Phase 1: classify
        errors = classify_errors(errors, project_dir)

        # Summary
        counts = Counter(e.kind.value for e in errors)
        for kind, n in sorted(counts.items()):
            print(f"  {kind}: {n}")

        # Phase 2: auto-patch
        fixed, llm_remaining, human_commands, esc_warnings = _auto_patch(
            errors, project_dir, dry_run, verbose, pm, ctx.uses_shadcn,
        )

        all_human_commands.extend(human_commands)
        all_esc_warnings.extend(esc_warnings)
        all_llm_errors = llm_remaining  # overwrite — latest round is authoritative

        print(f"[compile_fixer] Fixed: {len(fixed)} | LLM: {len(llm_remaining)} | Human: {len(human_commands)} cmd(s)")

        if not fixed:
            print("[compile_fixer] No progress — stopping auto-fix loop.")
            break

        if dry_run:
            print("[compile_fixer] Dry run — not looping.")
            break

    # ── Deduplicate human commands ────────────────────────────────────────────
    seen_cmds: set[str] = set()
    unique_human: list[str] = []
    for cmd in all_human_commands:
        if cmd not in seen_cmds:
            seen_cmds.add(cmd)
            unique_human.append(cmd)

    # ── Merge plan_errors into llm_errors ─────────────────────────────────────
    final_llm = all_llm_errors + plan_errors

    # ── Determine status ──────────────────────────────────────────────────────
    if unique_human and final_llm:
        status: Literal["clean", "needs_human", "needs_llm"] = "needs_human"
    elif unique_human:
        status = "needs_human"
    elif final_llm:
        status = "needs_llm"
    else:
        status = "clean"

    # ── Phase 3: LLM fix for TYPE_ERROR / MISSING_EXPORT / OTHER ─────────────
    # Only runs when there are no blocking human actions (install commands),
    # because those must be resolved first before type errors make sense.
    if status == "needs_llm" and not dry_run:
        applied = run_phase3(final_llm, project_dir, dry_run=dry_run, token_budget=token_budget)
        if applied:
            # Re-run tsc to check if Phase 3 resolved everything
            print("\n[compile_fixer] Re-checking after Phase 3 patches …")
            try:
                raw_recheck = _run_tsc(project_dir)
                recheck_errors = parse_tsc_output(raw_recheck, project_dir)
                recheck_in_manifest, _ = _filter_errors_to_manifest(
                    recheck_errors, ctx.executor_files
                )
                if not recheck_in_manifest:
                    print("[compile_fixer] ✓ All errors resolved after Phase 3.")
                    return FixerResult(status="clean", human_actions=[], llm_errors=[])
                else:
                    print(
                        f"[compile_fixer] {len(recheck_in_manifest)} error(s) remain "
                        "after Phase 3 — may need another run."
                    )
                    return FixerResult(
                        status="needs_llm",
                        human_actions=unique_human,
                        llm_errors=recheck_in_manifest,
                        warnings=all_esc_warnings,
                    )
            except Exception as exc:
                print(f"[compile_fixer][warn] Re-check tsc failed: {exc}")

    return FixerResult(
        status=status,
        human_actions=unique_human,
        llm_errors=final_llm,
        warnings=all_esc_warnings,
    )


# ─── Phase 3: LLM fix for TYPE_ERROR / MISSING_EXPORT / OTHER ────────────────


# ── Function scope extractor (heuristic brace-counting) ──────────────────────

def _extract_function_scope(lines: list[str], error_line: int) -> tuple[int, int, str]:
    """
    Scan backwards from error_line to find enclosing function boundary.
    Returns (start_line, end_line, function_name) — 1-indexed, inclusive.
    Falls back to (error_line, error_line, "") if no function found.
    """
    # Patterns that mark a function start
    _FN_START = re.compile(
        r"""^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"""
        r"""(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(|"""
        r"""(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\w+\s*)?\(|"""
        r"""(\w+)\s*\([^)]*\)\s*[:{])""",
        re.VERBOSE,
    )

    n = len(lines)
    # Clamp to valid index (1-indexed → 0-indexed)
    err_idx = min(max(error_line - 1, 0), n - 1)

    # Scan backwards for function boundary
    fn_start_idx = None
    fn_name = ""
    for i in range(err_idx, max(err_idx - 120, -1), -1):
        line = lines[i]
        m = _FN_START.match(line)
        if m:
            fn_name = next((g for g in m.groups() if g), "")
            fn_start_idx = i
            break

    if fn_start_idx is None:
        # Fallback: ±10 lines
        s = max(error_line - 10, 1)
        e = min(error_line + 10, n)
        return s, e, ""

    # Scan forwards from fn_start to find matching closing brace
    depth = 0
    fn_end_idx = fn_start_idx
    started = False
    for i in range(fn_start_idx, min(fn_start_idx + 300, n)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}" and started:
                depth -= 1
        if started and depth == 0:
            fn_end_idx = i
            break

    return fn_start_idx + 1, fn_end_idx + 1, fn_name  # back to 1-indexed


def _get_function_source(
    file_content: str,
    error_line: int,
) -> tuple[int, int, str, str]:
    """
    Extract function source containing error_line.
    Returns (start, end, fn_name, source_text).
    """
    lines = file_content.splitlines()
    start, end, fn_name = _extract_function_scope(lines, error_line)
    snippet = "\n".join(lines[start - 1 : end])
    return start, end, fn_name, snippet


# ── Dependency collector ──────────────────────────────────────────────────────

# Callsite: word followed by ( not preceded by function/const keywords
_RE_CALL = re.compile(r"\b(?<!function\s)(?<!const\s)(?<!let\s)(?<!var\s)(\w{2,})\s*\(")
_RE_IMPORT_STMT = re.compile(
    r"""from\s+['"]([^'"]+)['"]""",
    re.MULTILINE,
)
_TS_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "typeof", "instanceof",
    "new", "delete", "void", "throw", "await", "import", "require", "describe",
    "it", "test", "expect", "vi", "console", "Object", "Array", "Promise",
    "Math", "JSON", "parseInt", "parseFloat", "setTimeout", "clearTimeout",
    "fetch", "Error", "Map", "Set", "Boolean", "String", "Number",
})


def _extract_callees(fn_source: str) -> list[str]:
    """Extract function names called inside fn_source."""
    names = _RE_CALL.findall(fn_source)
    return [n for n in set(names) if n not in _TS_KEYWORDS]


def _find_function_in_file(
    file_content: str,
    fn_name: str,
) -> tuple[int, int, str] | None:
    """Find a named function definition in file_content.
    Returns (start_line, end_line, source) or None."""
    lines = file_content.splitlines()
    pattern = re.compile(
        r"""^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"""
        r"""(?:function\s+""" + re.escape(fn_name) + r"""\b|"""
        r"""(?:const|let|var)\s+""" + re.escape(fn_name) + r"""\s*=)""",
    )
    for i, line in enumerate(lines):
        if pattern.match(line):
            start, end, _, src = _get_function_source(file_content, i + 1)
            return start, end, src
    return None


def _find_callers_in_src(
    src_dir: Path,
    fn_name: str,
    origin_file: str,
) -> list[tuple[str, int, int, str]]:
    """
    Scan all .ts/.tsx files in src_dir for calls to fn_name.
    Returns list of (rel_path, start_line, end_line, function_source).
    """
    results: list[tuple[str, int, int, str]] = []
    call_pattern = re.compile(r"\b" + re.escape(fn_name) + r"\s*\(")

    for ts_file in src_dir.rglob("*.ts*"):
        if ts_file.suffix not in (".ts", ".tsx"):
            continue
        try:
            content = ts_file.read_text(errors="replace")
        except Exception:
            continue

        rel = str(ts_file.relative_to(src_dir.parent)).replace("\\", "/")
        if rel == origin_file:
            continue

        for i, line in enumerate(content.splitlines()):
            if call_pattern.search(line):
                start, end, _, src = _get_function_source(content, i + 1)
                results.append((rel, start, end, src))
                break  # one match per file is enough

    return results


@dataclass
class FunctionContext:
    file:       str
    fn_name:    str
    start_line: int
    end_line:   int
    source:     str
    role:       str   # "error_site" | "callee" | "caller"


def collect_function_contexts(
    llm_errors: list[CompileError],
    project_dir: Path,
) -> list[FunctionContext]:
    """
    For each llm_error:
      1. Extract the enclosing function (error_site)
      2. Extract callees defined in same file
      3. Find callers across src/
    Returns deduplicated list of FunctionContext.
    """
    src_dir = project_dir / "src"
    seen: set[tuple[str, int]] = set()
    contexts: list[FunctionContext] = []

    # Cache file contents to avoid redundant reads
    file_cache: dict[str, str] = {}

    def _get_content(rel: str) -> str:
        if rel not in file_cache:
            p = project_dir / rel
            if p.exists():
                file_cache[rel] = p.read_text(errors="replace")
            else:
                file_cache[rel] = ""
        return file_cache[rel]

    def _add(fc: FunctionContext) -> None:
        key = (fc.file, fc.start_line)
        if key not in seen:
            seen.add(key)
            contexts.append(fc)

    for err in llm_errors:
        content = _get_content(err.file)
        if not content:
            continue

        start, end, fn_name, src = _get_function_source(content, err.line)
        _add(FunctionContext(
            file=err.file, fn_name=fn_name,
            start_line=start, end_line=end,
            source=src, role="error_site",
        ))

        if not fn_name:
            continue

        # Callees defined in same file
        callees = _extract_callees(src)
        for callee in callees[:6]:  # cap to avoid explosion
            result = _find_function_in_file(content, callee)
            if result:
                cs, ce, csrc = result
                _add(FunctionContext(
                    file=err.file, fn_name=callee,
                    start_line=cs, end_line=ce,
                    source=csrc, role="callee",
                ))

        # Callers across src/
        if src_dir.exists():
            callers = _find_callers_in_src(src_dir, fn_name, err.file)
            for cfile, cs, ce, csrc in callers[:4]:  # cap callers
                _add(FunctionContext(
                    file=cfile, fn_name="",
                    start_line=cs, end_line=ce,
                    source=csrc, role="caller",
                ))

    return contexts


# ── Token budget estimator ────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


# ── LLM batch call ────────────────────────────────────────────────────────────

_PHASE3_SYSTEM = """\
You are a senior TypeScript engineer fixing compile errors.

Given a list of TypeScript compile errors and the source context of affected
functions (with their callees and callers), produce full corrected file rewrites.

Guidelines:
- Prefer adding optional parameters (param?: T) over removing or reordering required ones.
- Prefer null-safe access (obj?.prop, obj ?? fallback) over non-null assertions (!).
- If a function signature changes, include ALL callers shown in the context as patches.
- Do not change code unrelated to the errors.
- Return COMPLETE file content for each patched file — not snippets or diffs.

Respond ONLY with valid JSON, no markdown fences:
{
  "analysis": "<concise explanation of root causes and changes>",
  "patches": [
    {"path": "src/components/Foo.tsx", "content": "<complete corrected file>"}
  ]
}
"""


def _build_phase3_prompt(
    llm_errors:  list[CompileError],
    fn_contexts: list[FunctionContext],
    project_dir: Path,
    token_budget: int = 16000,
) -> str:
    parts: list[str] = []

    # Error list
    error_lines: list[str] = ["## Compile Errors\n"]
    for e in llm_errors:
        sl = e.meta.get("source_line", "")
        sl_str = f"\n    source: `{sl.strip()}`" if sl else ""
        error_lines.append(
            f"- [{e.code}] {e.file}:{e.line}:{e.col} — {e.message}{sl_str}"
        )
    parts.append("\n".join(error_lines))

    # Function contexts
    ctx_lines: list[str] = ["\n## Affected Functions and Dependencies\n"]
    for fc in fn_contexts:
        role_label = {"error_site": "⚠ error here", "callee": "callee", "caller": "caller"}[fc.role]
        header = f"### [{role_label}] {fc.file}"
        if fc.fn_name:
            header += f" — `{fc.fn_name}()` (lines {fc.start_line}–{fc.end_line})"
        ctx_lines.append(header)
        ctx_lines.append(f"```typescript\n{fc.source}\n```")
    parts.append("\n".join(ctx_lines))

    prompt = "\n\n".join(parts)

    # If within budget, append full file contents for files with errors
    used = _estimate_tokens(prompt) + _estimate_tokens(_PHASE3_SYSTEM)
    remaining = token_budget - used - 1000  # 1k reserve for response

    if remaining > 2000:
        files_seen: set[str] = set()
        file_parts: list[str] = ["\n## Full File Contents (files with errors)\n"]
        for e in llm_errors:
            if e.file in files_seen:
                continue
            files_seen.add(e.file)
            p = project_dir / e.file
            if not p.exists():
                continue
            content = p.read_text(errors="replace")
            cost = _estimate_tokens(content)
            if cost > remaining:
                file_parts.append(f"### {e.file}\n(omitted — too large)")
                continue
            remaining -= cost
            file_parts.append(f"### {e.file}\n```typescript\n{content}\n```")
        prompt += "\n\n" + "\n\n".join(file_parts)

    return prompt


def _call_llm_phase3(prompt: str) -> dict:
    """Single LLM call for Phase 3. Returns parsed JSON or empty dict."""
    try:
        _here = Path(__file__).parent.parent
        sys.path.insert(0, str(_here))
        from modules.call_llm import call_llm_messages  # type: ignore
        from artifacts.models import get_model           # type: ignore
        from modules.cost import record_usage            # type: ignore
    except ImportError as exc:
        print(f"[compile_fixer][phase3] Cannot import LLM modules: {exc}")
        return {}

    role = "debugger"
    try:
        model = get_model(role)
    except Exception:
        model = "unknown"

    print(f"[compile_fixer][phase3] Calling LLM ({model}) …")

    messages = [{"role": "user", "content": prompt}]
    try:
        raw, usage = call_llm_messages(
            messages=messages,
            system=_PHASE3_SYSTEM,
            role=role,
            label="compile_fixer_phase3",
        )
        record_usage(usage)
    except Exception as exc:
        print(f"[compile_fixer][phase3] LLM call failed: {exc}")
        return {}

    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.splitlines()[1:])
        if "```" in clean:
            clean = clean[: clean.rfind("```")]

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        print(f"[compile_fixer][phase3] Could not parse LLM response as JSON")
        print(f"  raw[:300]: {raw[:300]}")
        return {}


# ── Diff preview + Y/N review ─────────────────────────────────────────────────

def _unified_diff(old: str, new: str, path: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{path}", tofile=f"b/{path}", n=3,
    ))
    return "".join(diff) if diff else "(no changes)"


def _review_and_apply_phase3(
    patches:     list[dict],
    analysis:    str,
    project_dir: Path,
    dry_run:     bool,
) -> bool:
    """Print analysis + diff preview, prompt Y/N, apply if confirmed."""
    if not patches:
        print("[compile_fixer][phase3] No patches generated.")
        return False

    print("\n" + "═" * 60)
    print("  COMPILE FIXER — Phase 3: LLM Type/Export Fix")
    print("═" * 60)
    print(f"\n📋 Analysis:\n  {analysis}")
    print(f"\n📁 Files to patch ({len(patches)}):")
    for p in patches:
        print(f"  • {p['path']}")

    print("\n" + "─" * 60)
    print("  Diff preview")
    print("─" * 60)

    for p in patches:
        path   = p["path"]
        new    = p.get("content", "")
        old_p  = project_dir / path
        old    = old_p.read_text(errors="replace") if old_p.exists() else ""

        print(f"\n── {path}")
        diff_text = _unified_diff(old, new, path)
        lines = diff_text.splitlines()
        if len(lines) > 80:
            print("\n".join(lines[:80]))
            print(f"  … ({len(lines) - 80} more lines)")
        else:
            print(diff_text)

    print("\n" + "═" * 60)

    if dry_run:
        print("[compile_fixer][phase3] Dry run — not applying.")
        return False

    while True:
        try:
            ans = input("\nApply all patches? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return False
        if ans in ("", "y", "yes"):
            break
        if ans in ("n", "no"):
            print("[compile_fixer][phase3] Patches not applied.")
            return False

    applied = 0
    for p in patches:
        path    = p["path"]
        content = p.get("content", "")
        if not content:
            print(f"  [skip] {path} — empty content")
            continue
        dest = project_dir / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        print(f"  ✓ {path}")
        applied += 1

    print(f"[compile_fixer][phase3] Applied {applied} file(s)")
    return applied > 0


def run_phase3(
    llm_errors:  list[CompileError],
    project_dir: Path,
    dry_run:     bool = False,
    token_budget: int = 16000,
) -> bool:
    """
    Phase 3: LLM fix for TYPE_ERROR / MISSING_EXPORT / OTHER.
    Returns True if patches were applied.
    """
    if not llm_errors:
        return False

    print(f"\n[compile_fixer] Phase 3 — LLM fix ({len(llm_errors)} error(s)) …")

    # Collect function contexts
    fn_contexts = collect_function_contexts(llm_errors, project_dir)
    print(f"[compile_fixer][phase3] Function contexts collected: {len(fn_contexts)}")

    # Build prompt
    prompt = _build_phase3_prompt(llm_errors, fn_contexts, project_dir, token_budget)
    token_est = _estimate_tokens(prompt)
    print(f"[compile_fixer][phase3] Estimated prompt tokens: {token_est}")

    # LLM call
    result = _call_llm_phase3(prompt)
    if not result:
        return False

    patches  = result.get("patches", [])
    analysis = result.get("analysis", "")

    return _review_and_apply_phase3(patches, analysis, project_dir, dry_run)


# ─── Standalone CLI ───────────────────────────────────────────────────────────


def _print_error_table(errors: list[CompileError]) -> None:
    if not errors:
        print("  (none)")
        return
    for e in errors:
        meta_str = ""
        if e.meta:
            meta_str = " | " + " ".join(f"{k}={v!r}" for k, v in e.meta.items())
        print(f"  [{e.kind.value}] {e.file}:{e.line} [{e.code}]{meta_str}")
        print(f"    {e.message[:100]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile-error auto-fixer (Phase 1 + Phase 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("PIPELINE_PROJECT"),
        help="Project slug (sets PIPELINE_PROJECT env var)",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Project root directory (overrides --project lookup)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to executor manifest.json (overrides auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and plan fixes but do not write files",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Max auto-fix + tsc re-check rounds (default: 3)",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=16000,
        help="Max tokens for Phase 3 LLM prompt (default: 16000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()

    # ── Resolve project_dir ───────────────────────────────────────────────────
    if args.dir:
        project_dir = Path(args.dir).resolve()
    elif args.project:
        os.environ["PIPELINE_PROJECT"] = args.project
        try:
            _here = Path(__file__).parent.parent
            sys.path.insert(0, str(_here))
            from artifacts.paths import SRC_DIR  # type: ignore
            project_dir = Path(str(SRC_DIR)).parent
        except Exception:
            project_dir = Path.cwd()
    else:
        project_dir = Path.cwd()

    if not project_dir.exists():
        print(f"[error] project_dir not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    manifest_path = Path(args.manifest).resolve() if args.manifest else None

    print("=" * 60)
    print("  COMPILE FIXER")
    print("=" * 60)
    print(f"  project_dir   : {project_dir}")
    print(f"  manifest      : {manifest_path or '(auto-detect)'}")
    print(f"  dry_run       : {args.dry_run}")
    print(f"  max_rounds    : {args.max_rounds}")
    print()

    result = run_compile_fixer(
        project_dir,
        manifest_path = manifest_path,
        dry_run       = args.dry_run,
        verbose       = args.verbose,
        max_rounds    = args.max_rounds,
        token_budget  = args.token_budget,
    )

    # ── Output based on status ────────────────────────────────────────────────
    print("\n" + "─" * 60)

    if result.status == "clean":
        print("✓ All compile errors resolved by auto-fixer.")
        sys.exit(0)

    elif result.status == "needs_human":
        print(_format_escalation_box(result.human_actions, project_dir, warnings=result.warnings))
        if result.llm_errors:
            print(f"\nAdditionally, {len(result.llm_errors)} error(s) need LLM repair after install:")
            _print_error_table(result.llm_errors)
        sys.exit(2)

    elif result.status == "needs_llm":
        print(f"Errors requiring LLM repair ({len(result.llm_errors)}):")
        _print_error_table(result.llm_errors)
        sys.exit(1)


if __name__ == "__main__":
    main()