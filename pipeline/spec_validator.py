"""
pipeline/04b_spec_validator.py
==============================
Spec Fact-Checker — validates package names, plugin names, and version
constraints declared in spec against the live npm registry.

Chạy ngay sau 04_specwright.py, trước 05_spectracker.py.

Flow:
    04_specwright → [04b_spec_validator] → 05_spectracker

Why no LLM for verification:
    Dùng LLM để verify package existence là circular — LLM có thể hallucinate
    cả kết quả verify. npm registry là ground truth duy nhất đáng tin.
    urllib calls tới registry.npmjs.org:
      - Deterministic: 200 = tồn tại, 404 = không tồn tại
      - Nhanh: ~200ms/package
      - Free: không tốn token
      - Accurate: không thể hallucinate

What is validated:
    1. Package existence    — GET /registry.npmjs.org/<package> → 200 or 404
    2. Plugin existence     — cùng mechanism, scoped packages (@wavesurfer/regions)
    3. Version constraint   — so sánh version declared trong spec vs
                              `dist-tags.latest` + versions[] từ registry
                              Detect: nonexistent major (v8 khi chỉ có v7),
                              deprecated, yanked

LLM chỉ được dùng ở bước extract (Phase 1) — parse spec text thành
structured list of { name, declared_version, context }.

Inputs:
    spec/<slug>.md                    — output của 04_specwright

Outputs:
    spec/spec_validation_report.md   — human-readable validation results
    spec/<slug>.md                   — patched in-place nếu có auto-corrections

Usage:
    python 04b_spec_validator.py --project layered-listen
    python 04b_spec_validator.py --project layered-listen --dry-run
    python 04b_spec_validator.py --project layered-listen --no-patch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import ssl
import urllib.error
import urllib.request

# ── SSL context: use certifi bundle if available, fallback to system ──────────
def _make_ssl_context() -> ssl.SSLContext:
    """
    Build SSL context using certifi bundle when available.
    Python 3.13 from python.org does not use system keychain by default —
    certifi provides the trusted CA bundle needed for HTTPS connections.
    """
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    except ImportError:
        pass
    # Fallback: try system default
    try:
        ctx = ssl.create_default_context()
        return ctx
    except Exception:
        # Last resort: unverified (for dev only)
        ctx = ssl._create_unverified_context()
        return ctx


_SSL_CONTEXT = _make_ssl_context()
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import ensure_dirs, get_spec_path          # noqa: E402
from artifacts.models import get_model                           # noqa: E402
from modules.call_llm import call_llm, call_llm_json                           # noqa: E402
from modules.cost import print_summary                          # noqa: E402

ROLE = "clarificator"   # reuse — rẻ, chỉ cần extract JSON, không cần reasoning

# Strict JSON Schema cho bước extract — guarantees parse, không cần strip fences
_EXTRACT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "package_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "packages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":             {"type": "string"},
                            "declared_version": {"type": ["string", "null"]},
                            "kind":             {"type": "string",
                                                  "enum": ["package", "plugin", "devDependency"]},
                            "context":          {"type": "string"},
                        },
                        "required": ["name", "declared_version", "kind", "context"],
                        "additionalProperties": False,
                    }
                }
            },
            "required": ["packages"],
            "additionalProperties": False,
        }
    }
}

# ─── Constants ────────────────────────────────────────────────────────────────

_NPM_REGISTRY   = "https://registry.npmjs.org"
_REQUEST_TIMEOUT = 8       # seconds per registry call
_RATE_LIMIT_MS  = 150      # ms between registry calls (polite rate limiting)
_MAX_TOKENS_EXTRACT = 4096


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PackageRef:
    """A package/plugin reference extracted from spec."""
    name:              str
    declared_version:  str | None    # e.g. "v7", "4.x", "latest", None
    kind:              str           # "package" | "plugin" | "devDependency"
    context:           str           # which section/line references it


@dataclass
class CheckResult:
    """Result of checking one PackageRef against npm registry."""
    ref:               PackageRef

    # Existence
    exists:            bool | None   # None = check failed (network/timeout)
    registry_name:     str | None    # canonical name from registry (may differ in casing)

    # Version
    latest_version:    str | None    # dist-tags.latest from registry
    available_majors:  list[int]     # e.g. [4, 5, 6, 7]
    declared_major:    int | None    # parsed from declared_version, e.g. 7 from "v7"
    version_ok:        bool | None   # None if can't determine
    version_note:      str           # human note about version status

    # Issues
    deprecated:        bool          # package is globally deprecated
    deprecation_msg:   str           # deprecation message if any

    error:             str | None    # error message if check failed

    @property
    def has_issue(self) -> bool:
        return (
            self.exists is False
            or self.exists is None
            or self.version_ok is False
            or self.deprecated
        )


# ─── Phase 1: Extract package references from spec via LLM ───────────────────

_SYSTEM_EXTRACT = """\
You are a dependency extraction tool. Given a technical specification document,
extract ALL npm package references — packages, plugins, build tools, and dev
dependencies declared or implied in the spec.

For each reference, extract:
- "name": exact npm package name (e.g. "wavesurfer.js", "@wavesurfer/regions",
  "vite-plugin-pwa", "dexie", "zustand")
- "declared_version": version string as written in spec (e.g. "v7", "4.x",
  "latest", "^5.0.0") or null if no version mentioned
- "kind": one of "package" | "plugin" | "devDependency"
  - plugin = explicitly called a plugin (e.g. Vite plugins, Wavesurfer plugins)
  - devDependency = build tools, type packages, linters (TypeScript, Vite, etc.)
  - package = runtime library
- "context": brief location reference (e.g. "Tech Stack table", "Architecture section")

Rules:
- Use EXACT npm package names, not display names
  (e.g. "wavesurfer.js" not "Wavesurfer.js", "dexie" not "Dexie.js")
- For scoped packages include scope (e.g. "@wavesurfer/regions")
- Skip browser-native APIs explicitly marked as "no npm dependency"
  (e.g. OPFS, IndexedDB, crypto.randomUUID)
- Skip deployment platforms (Vercel, Netlify)
- Include TypeScript, React, Vite, Tailwind, shadcn, and all explicit plugins

Output ONLY a JSON object, no markdown fences:
{"packages": [{"name": "...", "declared_version": "...", "kind": "...", "context": "..."}, ...]}
""".strip()


def extract_packages_from_spec(spec_content: str) -> list[PackageRef]:
    """
    Use LLM to extract structured package references from spec text.
    LLM is ONLY used here — for text parsing, not for fact-checking.
    """
    print("[spec_validator] Phase 1 — extracting package references via LLM …")

    # call_llm_json dùng strict JSON Schema — guaranteed parse, không cần strip fences
    # Fallback: nếu model không support response_format, call_llm_json vẫn retry
    # với _parse_json_object fallback parser.
    try:
        data, _ = call_llm_json(
            ROLE,
            _SYSTEM_EXTRACT,
            f"Extract all npm package references from this spec:\n\n{spec_content}",
            max_tokens=_MAX_TOKENS_EXTRACT,
            caller_file=__file__,
            label="[04b] extract",
            # Strict JSON Schema → provider routes to model that supports it
            extra_kwargs={"response_format": _EXTRACT_SCHEMA},
        )
    except Exception as exc:
        print(f"[spec_validator][warn] Extraction call failed: {exc}")
        return []

    packages: list[PackageRef] = []
    seen: set[str] = set()

    for p in data.get("packages", []):
        name = (p.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        packages.append(PackageRef(
            name             = name,
            declared_version = p.get("declared_version") or None,
            kind             = p.get("kind", "package"),
            context          = p.get("context", ""),
        ))

    return packages


# ─── Phase 2: Validate against npm registry ───────────────────────────────────

def _npm_get(path: str) -> dict[str, Any] | None:
    """
    GET https://registry.npmjs.org/<path>.
    Returns parsed JSON or None on any error.
    404 → returns sentinel {"_not_found": True}.
    """
    url = f"{_NPM_REGISTRY}/{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept":     "application/json",
            "User-Agent": "spec-validator/1.0 (pipeline fact-checker)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT, context=_SSL_CONTEXT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"_not_found": True}
        return {"_error": f"HTTP {e.code}"}
    except urllib.error.URLError as e:
        return {"_error": f"URLError: {e.reason}"}
    except Exception as exc:
        return {"_error": str(exc)}


def _parse_major(version_str: str | None) -> int | None:
    """
    Parse major version number from a version string.
    "v7" → 7, "4.x" → 4, "^5.0.0" → 5, "latest" → None, None → None.
    """
    if not version_str:
        return None
    version_str = version_str.strip().lower()
    if version_str in ("latest", "next", "*", ""):
        return None
    m = re.search(r"(\d+)", version_str)
    return int(m.group(1)) if m else None


def _extract_majors(versions_dict: dict[str, Any]) -> list[int]:
    """Extract unique major versions from registry versions dict."""
    majors: set[int] = set()
    for v in versions_dict.keys():
        m = re.match(r"^(\d+)\.", v)
        if m:
            majors.add(int(m.group(1)))
    return sorted(majors)


def check_package(ref: PackageRef) -> CheckResult:
    """
    Check one PackageRef against npm registry.
    Validates: existence, version constraint, deprecation.
    """
    time.sleep(_RATE_LIMIT_MS / 1000)

    # Encode scoped package names: @wavesurfer/regions → %40wavesurfer%2Fregions
    encoded = ref.name.replace("@", "%40").replace("/", "%2F")
    data = _npm_get(encoded)

    # ── Existence check ───────────────────────────────────────────────────────
    if data is None:
        return CheckResult(
            ref=ref, exists=None, registry_name=None,
            latest_version=None, available_majors=[], declared_major=None,
            version_ok=None, version_note="Registry call returned None",
            deprecated=False, deprecation_msg="", error="unknown error",
        )

    if data.get("_not_found"):
        return CheckResult(
            ref=ref, exists=False, registry_name=None,
            latest_version=None, available_majors=[], declared_major=None,
            version_ok=None, version_note="Package not found on npm",
            deprecated=False, deprecation_msg="", error=None,
        )

    if "_error" in data:
        return CheckResult(
            ref=ref, exists=None, registry_name=None,
            latest_version=None, available_majors=[], declared_major=None,
            version_ok=None, version_note="Registry check failed",
            deprecated=False, deprecation_msg="", error=data["_error"],
        )

    # ── Package exists ────────────────────────────────────────────────────────
    registry_name  = data.get("name", ref.name)
    dist_tags      = data.get("dist-tags", {})
    latest_version = dist_tags.get("latest")
    versions_dict  = data.get("versions", {})
    available_majors = _extract_majors(versions_dict)
    declared_major   = _parse_major(ref.declared_version)

    # ── Deprecation check ─────────────────────────────────────────────────────
    # Package-level deprecation: check latest version's deprecated field
    deprecated     = False
    deprecation_msg = ""
    if latest_version and latest_version in versions_dict:
        dep_msg = versions_dict[latest_version].get("deprecated")
        if dep_msg:
            deprecated      = True
            deprecation_msg = dep_msg

    # ── Version constraint check ──────────────────────────────────────────────
    version_ok   = None
    version_note = ""

    if declared_major is not None and available_majors:
        if declared_major in available_majors:
            version_ok   = True
            version_note = f"v{declared_major} exists (available: {available_majors})"
        else:
            version_ok   = False
            max_major    = max(available_majors)
            version_note = (
                f"v{declared_major} does NOT exist on npm. "
                f"Available majors: {available_majors}. "
                f"Latest: v{max_major} ({latest_version})"
            )
    elif declared_major is None and latest_version:
        version_ok   = True
        version_note = f"No version constraint declared — latest is {latest_version}"
    elif not available_majors:
        version_note = "No versions published yet"

    return CheckResult(
        ref=ref,
        exists=True,
        registry_name=registry_name,
        latest_version=latest_version,
        available_majors=available_majors,
        declared_major=declared_major,
        version_ok=version_ok,
        version_note=version_note,
        deprecated=deprecated,
        deprecation_msg=deprecation_msg,
        error=None,
    )


def validate_packages(packages: list[PackageRef]) -> list[CheckResult]:
    """
    Run registry check for all packages.
    Prints progress per package.
    """
    print(f"[spec_validator] Phase 2 — checking {len(packages)} package(s) against npm registry …")

    results: list[CheckResult] = []
    for i, ref in enumerate(packages, 1):
        print(f"  [{i:02d}/{len(packages):02d}] {ref.name} ", end="", flush=True)
        result = check_package(ref)

        if result.exists is False:
            print(f"✗  NOT FOUND")
        elif result.exists is None:
            print(f"?  ERROR: {result.error}")
        elif result.deprecated:
            print(f"⚠  DEPRECATED — {result.deprecation_msg[:60]}")
        elif result.version_ok is False:
            print(f"⚠  VERSION MISMATCH — {result.version_note[:80]}")
        else:
            ver = f" ({result.latest_version})" if result.latest_version else ""
            print(f"✓{ver}")

        results.append(result)

    return results


# ─── Phase 3: Auto-patch spec ─────────────────────────────────────────────────

def patch_spec(
    spec_content: str,
    results:      list[CheckResult],
) -> tuple[str, list[str]]:
    """
    Auto-correct spec content for issues that have a clear fix:
    - Package not found AND a name-casing correction is obvious
      (e.g. "Wavesurfer.js" → "wavesurfer.js")
    - Version declared as nonexistent major where the correct major is clear

    Conservative: only patches when the correction is unambiguous.
    Returns (patched_content, list_of_human_readable_changes).
    """
    patched = spec_content
    changes: list[str] = []

    for r in results:
        if not r.has_issue:
            continue

        # ── Case 1: Package not found — try lowercase / common variant ────────
        if r.exists is False:
            name     = r.ref.name
            lower    = name.lower()
            # Try common suffixes: "wavesurfer.js" → "wavesurfer" etc.
            variants = [lower, lower.rstrip(".js"), lower + ".js"]
            for variant in variants:
                if variant == name:
                    continue
                # Quick check if variant exists (silent, no print)
                time.sleep(_RATE_LIMIT_MS / 1000)
                encoded  = variant.replace("@", "%40").replace("/", "%2F")
                check    = _npm_get(encoded)
                if check and not check.get("_not_found") and not check.get("_error"):
                    # Auto-patch name in spec
                    new_content, n = _replace_package_name_in_spec(patched, name, variant)
                    if n > 0:
                        patched = new_content
                        changes.append(f"Package name: `{name}` → `{variant}` ({n} occurrence(s))")
                    break

        # ── Case 2: Version mismatch — declared major doesn't exist ───────────
        if r.version_ok is False and r.declared_major is not None and r.available_majors:
            # Only auto-patch if there's exactly one reasonable correction
            # e.g. declared v8 but only v7 exists and latest is 7.x
            latest_major = max(r.available_majors)
            if r.declared_major > latest_major:
                # Spec declares a future version that doesn't exist yet
                old_ver = f"v{r.declared_major}"
                new_ver = f"v{latest_major}"
                # Also try variants like "v7.x", "7.x", "^7"
                for old_pat in [old_ver, f"{r.declared_major}.x", f"^{r.declared_major}"]:
                    for new_pat in [new_ver, f"{latest_major}.x"]:
                        if old_pat in patched and old_pat != new_pat:
                            patched = patched.replace(old_pat, new_pat)
                            changes.append(
                                f"Version: `{r.ref.name}` {old_pat} → {new_pat} "
                                f"(v{r.declared_major} does not exist; latest major is v{latest_major})"
                            )
                            break

    return patched, changes


def _replace_package_name_in_spec(content: str, old: str, new: str) -> tuple[str, int]:
    """
    Replace package name in spec, only in likely package contexts:
    backtick spans, table cells, quoted strings.
    Returns (new_content, count_of_replacements).
    """
    count   = 0
    result  = content

    patterns = [
        (f"`{old}`",  f"`{new}`"),
        (f'"{old}"',  f'"{new}"'),
        (f"'{old}'",  f"'{new}'"),
        (f"| {old} ", f"| {new} "),
        (f"| {old}|", f"| {new}|"),
    ]
    for old_pat, new_pat in patterns:
        if old_pat in result:
            result = result.replace(old_pat, new_pat)
            count  += result.count(new_pat)  # approximate

    return result, count


# ─── Phase 4: Report ─────────────────────────────────────────────────────────

def generate_report(
    results:  list[CheckResult],
    changes:  list[str],
    spec_path: Path,
) -> str:
    """Generate human-readable markdown validation report."""
    not_found   = [r for r in results if r.exists is False]
    errored     = [r for r in results if r.exists is None]
    ver_bad     = [r for r in results if r.exists is True and r.version_ok is False]
    deprecated_ = [r for r in results if r.exists is True and r.deprecated]
    ok          = [r for r in results if r.exists is True and not r.has_issue]

    lines: list[str] = []
    lines += [
        "# Spec Validation Report",
        f"_Spec: `{spec_path.name}` — {len(results)} package(s) checked_",
        "",
    ]

    # Summary line
    issues = len(not_found) + len(errored) + len(ver_bad) + len(deprecated_)
    if issues == 0:
        lines += ["✓ **All packages validated successfully.**", ""]
    else:
        lines += [
            f"⚠ **{issues} issue(s) found** — "
            f"{len(not_found)} not found, "
            f"{len(ver_bad)} version mismatch, "
            f"{len(deprecated_)} deprecated, "
            f"{len(errored)} check failed.",
            "",
        ]

    # Not found
    if not_found:
        lines += ["## ✗ Not Found on npm", ""]
        for r in not_found:
            lines.append(f"- **`{r.ref.name}`**")
            lines.append(f"  - Context: {r.ref.context}")
            lines.append(f"  - Declared version: {r.ref.declared_version or '(none)'}")
            lines.append(f"  - Kind: {r.ref.kind}")
            lines.append(f"  - Action: verify package name — may be hallucinated")
            lines.append("")

    # Version mismatches
    if ver_bad:
        lines += ["## ⚠ Version Mismatch", ""]
        for r in ver_bad:
            lines.append(f"- **`{r.ref.name}`**")
            lines.append(f"  - Declared: `{r.ref.declared_version}`")
            lines.append(f"  - Available majors: {r.available_majors}")
            lines.append(f"  - Latest: `{r.latest_version}`")
            lines.append(f"  - Note: {r.version_note}")
            lines.append(f"  - Context: {r.ref.context}")
            lines.append("")

    # Deprecated
    if deprecated_:
        lines += ["## ⚠ Deprecated", ""]
        for r in deprecated_:
            lines.append(f"- **`{r.ref.name}`** (latest: `{r.latest_version}`)")
            lines.append(f"  - {r.deprecation_msg[:200]}")
            lines.append(f"  - Context: {r.ref.context}")
            lines.append("")

    # Check errors
    if errored:
        lines += ["## ? Check Failed (network/timeout)", ""]
        for r in errored:
            lines.append(f"- `{r.ref.name}`: {r.error}")
        lines.append("")

    # Auto-corrections
    if changes:
        lines += ["## Auto-corrections Applied to Spec", ""]
        for c in changes:
            lines.append(f"- {c}")
        lines.append("")

    # Valid
    if ok:
        lines += [f"## ✓ Valid ({len(ok)} packages confirmed)", ""]
        cols: list[str] = []
        for r in ok:
            ver = f" `{r.latest_version}`" if r.latest_version else ""
            cols.append(f"`{r.ref.name}`{ver}")
        lines.append(", ".join(cols))
        lines.append("")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="04b_spec_validator — fact-check package names and versions in spec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", metavar="NAME",
                        default=os.environ.get("PIPELINE_PROJECT"),
                        help="Project workspace name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate but do not patch spec or write report file")
    parser.add_argument("--no-patch", action="store_true",
                        help="Write report but do not auto-patch spec")
    args = parser.parse_args()

    project_name = args.project
    if not project_name:
        print("[spec_validator][error] No --project specified and PIPELINE_PROJECT not set.")
        sys.exit(1)

    os.environ["PIPELINE_PROJECT"] = project_name
    ensure_dirs()

    spec_path = get_spec_path()
    if not spec_path.exists():
        print(f"[spec_validator][error] Spec not found: {spec_path}")
        print("  Run 04_specwright.py first.")
        sys.exit(1)

    spec_content = spec_path.read_text(encoding="utf-8")
    print(f"[spec_validator] Spec: {spec_path} ({len(spec_content):,} chars)")
    print(f"[spec_validator] dry_run={args.dry_run}  no_patch={args.no_patch}")
    print()

    try:
        # Phase 1: Extract
        packages = extract_packages_from_spec(spec_content)
        if not packages:
            print("[spec_validator] No packages extracted — nothing to validate.")
            sys.exit(0)
        print(f"[spec_validator] Extracted {len(packages)} package reference(s):")
        for p in packages:
            ver = f" ({p.declared_version})" if p.declared_version else ""
            print(f"  [{p.kind:13s}] {p.name}{ver}  ← {p.context}")
        print()

        # Phase 2: Validate
        results = validate_packages(packages)
        print()

        # Summary
        issues = sum(1 for r in results if r.has_issue)
        print(f"[spec_validator] Validation complete: "
              f"{len(results) - issues} OK, {issues} issue(s)")

        # Phase 3: Patch spec
        changes: list[str] = []
        if issues > 0 and not args.no_patch and not args.dry_run:
            print("[spec_validator] Phase 3 — auto-patching spec …")
            patched_content, changes = patch_spec(spec_content, results)
            if changes:
                spec_path.write_text(patched_content, encoding="utf-8")
                print(f"[spec_validator] Applied {len(changes)} auto-correction(s):")
                for c in changes:
                    print(f"  • {c}")
            else:
                print("[spec_validator] No auto-corrections applicable — manual review needed.")
            print()

        # Phase 4: Report
        report = generate_report(results, changes, spec_path)
        report_path = spec_path.parent / "spec_validation_report.md"

        if args.dry_run:
            print("[spec_validator] Dry run — report:")
            print(report)
        else:
            report_path.write_text(report, encoding="utf-8")
            print(f"[spec_validator] Report written → {report_path}")

        # Exit code: 1 if issues found (lets CI fail or pipeline pause)
        if issues > 0:
            print(f"\n[spec_validator] ✗ {issues} issue(s) require attention.")
            sys.exit(1)
        print("\n[spec_validator] ✓ All packages validated.")
        sys.exit(0)

    finally:
        print()
        print_summary("[04b]")


if __name__ == "__main__":
    main()