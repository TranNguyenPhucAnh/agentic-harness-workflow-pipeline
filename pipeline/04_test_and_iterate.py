"""
pipeline/04_test_and_iterate.py
Step 4+5 — Clustered repair loop: tester_agent (surface) + logic_agent (logic).

Agent roles (configured in pipeline/pipeline_config.json):
    tester_agent  — Layer 1 (surface): selector, import path, style class,
                    float precision, text contract. ONE targeted pass per cluster.
                    If it detects a logic bug it signals LOGIC_BUG and steps aside.
    logic_agent   — Layer 2 (logic): state machines, data generation, range
                    constraints, acceptance criteria. Scope-locked to
                    the paths listed under "logic_scope_patterns" in config.

Cluster ownership:
    - Logic-scope files → skip tester_agent, go straight to logic_agent L2.
    - Other files → tester_agent L1 first; if LOGIC_BUG and in scope → logic_agent L2;
      if out of scope and unfixable → ESCALATED→human.
    - Once a cluster is owned by logic_agent, tester_agent will NOT touch it again.
    - logic_agent patches outside its scope are rejected silently.

Phase flow:
    Phase B : run test runner → parse output → list[FailureCluster]
    Phase C : per-cluster dispatch:
        P0 — Consistency: cross-check test vs code vs spec (no file writes)
             Verdicts: CODE_BUG | TEST_FRAGILE | SPEC_AMBIG | THRESHOLD_OK
             TEST_FRAGILE / THRESHOLD_OK → allowed to patch test file (query/threshold only)
             SPEC_AMBIG                  → escalate to human immediately, skip repair
             CODE_BUG                   → normal repair flow below
        L0 — Static     : esbuild/float fixes (no LLM)
        L1 — tester_agent : surface bugs; skipped for logic-scope files
        L2 — logic_agent  : logic bugs; activated on stale/scope/LOGIC_BUG
        L3 — Give-up    : ESCALATED after max_cluster_attempts
    Phase D : rerun tests → repeat

Writes:
    reports/test_iterations.json
    reports/escalated_clusters.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import httpx
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

ROOT        = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PLAN_PATH     = ROOT / "scaffold" / "plan.json"
FINDINGS_PATH = ROOT / "scaffold" / "judge_findings.md"

TAG = "[04]"


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    cfg_path = ROOT / "pipeline" / "pipeline_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def get_agent_cfg(cfg: dict, role: str) -> dict:
    return cfg.get("agents", {}).get(role, {})


def get_techstack(cfg: dict) -> str:
    return cfg.get("techstack", "unknown")


# ════════════════════════════════════════════════════════════════════════════
# Techstack helpers — test runner, language label, timeline keywords, fence
# ════════════════════════════════════════════════════════════════════════════

_TECHSTACK_TEST_CMD: dict[str, list[str]] = {
    "react-vite-ts":   ["npx", "vitest", "run", "--reporter=verbose"],
    "nextjs-ts":       ["npx", "vitest", "run", "--reporter=verbose"],
    "vue-vite-ts":     ["npx", "vitest", "run", "--reporter=verbose"],
    "svelte-ts":       ["npx", "vitest", "run", "--reporter=verbose"],
    "python-fastapi":  ["python", "-m", "pytest", "-v"],
    "python-flask":    ["python", "-m", "pytest", "-v"],
    "node-express-ts": ["npx", "jest", "--verbose"],
    "django":          ["python", "-m", "pytest", "-v"],
}

# Language label used in prompts ("TypeScript/React" etc.)
_TECHSTACK_LANG: dict[str, str] = {
    "react-vite-ts":   "TypeScript/React",
    "nextjs-ts":       "TypeScript/Next.js",
    "vue-vite-ts":     "TypeScript/Vue 3",
    "svelte-ts":       "TypeScript/SvelteKit",
    "python-fastapi":  "Python/FastAPI",
    "python-flask":    "Python/Flask",
    "node-express-ts": "TypeScript/Node.js",
    "django":          "Python/Django",
}

# Code fence language tag for prompts
_TECHSTACK_FENCE: dict[str, str] = {
    "react-vite-ts":   "typescript",
    "nextjs-ts":       "typescript",
    "vue-vite-ts":     "typescript",
    "svelte-ts":       "typescript",
    "python-fastapi":  "python",
    "python-flask":    "python",
    "node-express-ts": "typescript",
    "django":          "python",
}

# Timeline extraction — action keywords specific to the test framework
_TECHSTACK_TIMELINE_KEYWORDS: dict[str, list[str]] = {
    "react-vite-ts":   ["render(", "fireEvent", "userEvent", "act(", "screen."],
    "nextjs-ts":       ["render(", "fireEvent", "userEvent", "act(", "screen."],
    "vue-vite-ts":     ["mount(", "wrapper.", "trigger(", "nextTick"],
    "svelte-ts":       ["render(", "fireEvent", "userEvent", "act("],
    "python-fastapi":  ["client.", "response =", "assert response", ".post(", ".get("],
    "python-flask":    ["client.", "response =", "assert response", ".post(", ".get("],
    "node-express-ts": ["request(", "supertest", ".post(", ".get(", "expect("],
    "django":          ["client.", "response =", "self.client", ".post(", ".get("],
}

# Default logic-scope patterns (overridable via pipeline_config.json)
_DEFAULT_LOGIC_SCOPE_PATTERNS: list[str] = [
    r"^src/hooks/",
    r"^src/data/",
    r"^src/types/",
    r"^src/utils/",
]

# Python equivalent scopes
_PYTHON_LOGIC_SCOPE_PATTERNS: list[str] = [
    r"^src/services/",
    r"^src/models/",
    r"^src/schemas/",
    r"^src/repositories/",
    r"^app/services/",
    r"^app/models/",
]

_TECHSTACK_DEFAULT_SCOPE: dict[str, list[str]] = {
    "react-vite-ts":   _DEFAULT_LOGIC_SCOPE_PATTERNS,
    "nextjs-ts":       _DEFAULT_LOGIC_SCOPE_PATTERNS,
    "vue-vite-ts":     [r"^src/composables/", r"^src/stores/", r"^src/types/", r"^src/utils/"],
    "svelte-ts":       [r"^src/stores/", r"^src/types/", r"^src/utils/", r"^src/lib/"],
    "python-fastapi":  _PYTHON_LOGIC_SCOPE_PATTERNS,
    "python-flask":    _PYTHON_LOGIC_SCOPE_PATTERNS,
    "node-express-ts": [r"^src/services/", r"^src/models/", r"^src/types/", r"^src/utils/"],
    "django":          [r"^app/models/", r"^app/services/", r"^app/serializers/"],
}


def get_logic_scope_patterns(cfg: dict) -> list[re.Pattern]:
    """Return compiled scope patterns from config, falling back to techstack defaults."""
    techstack = get_techstack(cfg)
    raw = (
        cfg.get("logic_scope_patterns")
        or _TECHSTACK_DEFAULT_SCOPE.get(techstack, _DEFAULT_LOGIC_SCOPE_PATTERNS)
    )
    return [re.compile(p) for p in raw]


def get_test_cmd(cfg: dict) -> list[str]:
    techstack = get_techstack(cfg)
    return cfg.get("test_cmd") or _TECHSTACK_TEST_CMD.get(
        techstack, ["npx", "vitest", "run", "--reporter=verbose"]
    )


def get_lang_label(cfg: dict) -> str:
    return _TECHSTACK_LANG.get(get_techstack(cfg), "the project's language")


def get_fence(cfg: dict) -> str:
    return _TECHSTACK_FENCE.get(get_techstack(cfg), "")


def get_timeline_keywords(cfg: dict) -> list[str]:
    return _TECHSTACK_TIMELINE_KEYWORDS.get(
        get_techstack(cfg), ["render(", "fireEvent", "act(", "expect("]
    )


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TestFailure:
    test_file:     str
    test_name:     str
    error_snippet: str


@dataclass
class FailureCluster:
    """One cluster = one test file + its corresponding src file."""
    test_file:  str
    src_file:   str
    failures:   list[TestFailure] = field(default_factory=list)

    attempt_count:      int  = field(default=0)
    last_fingerprint:   str  = field(default="")
    escalated:          bool = field(default=False)
    is_transform_error: bool = field(default=False)
    # "tester" until logic_agent takes over; then "logic" permanently
    owner:              str  = field(default="tester")

    @property
    def key(self) -> str:
        return self.test_file

    def error_block(self) -> str:
        return "\n\n".join(
            f"  x {f.test_name}\n{f.error_snippet}" for f in self.failures
        )

    def fingerprint(self) -> str:
        return re.sub(r"\s+", " ", self.error_block()).strip()[:400]

    def is_logic_scope(self, scope_patterns: list[re.Pattern]) -> bool:
        return any(p.match(self.src_file) for p in scope_patterns)


@dataclass
class ClusterRepairRecord:
    cluster:             str
    src_file:            str
    failures:            int
    repaired:            bool
    layer_used:          str   # "static"|"tester_surface"|"logic_deep"|"test_rewrite"|"skipped"
    escalated:           bool
    escalated_to:        str   # ""|"logic"|"human"
    owner:               str   # "tester"|"logic"
    note:                str = ""
    consistency_verdict: str = ""   # P0 verdict


@dataclass
class IterationRecord:
    iteration:         int
    passed:            bool
    summary:           str
    clusters_found:    int
    clusters_repaired: int
    cluster_details:   list[dict]
    log_snippet:       str


# ════════════════════════════════════════════════════════════════════════════
# Spec + knowledge loaders
# ════════════════════════════════════════════════════════════════════════════

def _load_spec() -> str:
    compressed = ROOT / "scaffold" / "spec_compressed.md"
    return compressed.read_text() if compressed.exists() else (ROOT / "spec.md").read_text()


def _load_plan_global_notes() -> str:
    """Load global_notes from scaffold/plan.json (produced by planner_agent)."""
    if not PLAN_PATH.exists():
        return ""
    try:
        return json.loads(PLAN_PATH.read_text()).get("global_notes", "")
    except Exception:
        return ""


def _load_judge_findings() -> str:
    """Load scaffold/judge_findings.md — prevents re-introducing judged issues."""
    if not FINDINGS_PATH.exists():
        return ""
    try:
        return FINDINGS_PATH.read_text().strip()
    except Exception:
        return ""


def _load_knowledge_base() -> str:
    """Load scaffold/knowledge_base.md — accumulated human fix patterns."""
    kb = ROOT / "scaffold" / "knowledge_base.md"
    if not kb.exists():
        return ""
    content = kb.read_text().strip()
    lines = content.splitlines()
    body_lines = [l for l in lines if not l.startswith("# ") and not l.startswith("_")]
    return "\n".join(body_lines).strip()


# ════════════════════════════════════════════════════════════════════════════
# Provider adapters — same pattern as other pipeline steps
# ════════════════════════════════════════════════════════════════════════════

_PROVIDER_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "openai":     "https://api.openai.com/v1/chat/completions",
    "mistral":    "https://api.mistral.ai/v1/chat/completions",
    "together":   "https://api.together.xyz/v1/chat/completions",
}

_PROVIDER_ENV: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "together":   "TOGETHER_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "gemini":     "GEMINI_API_KEY",
}


def _call_openai_compat(provider: str, model: str, messages: list,
                         max_tokens: int = 32768, max_retries: int = 2) -> str:
    env_key = _PROVIDER_ENV.get(provider, f"{provider.upper()}_API_KEY")
    api_key = os.environ.get(env_key, "")
    if not api_key:
        raise RuntimeError(f"{env_key} not set")

    url = _PROVIDER_URLS.get(provider) or os.environ.get(f"{provider.upper()}_BASE_URL")
    if not url:
        raise RuntimeError(f"Unknown provider '{provider}'.")

    for attempt in range(max_retries):
        r = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages,
                  "temperature": 0.1, "max_tokens": max_tokens},
            timeout=300,
        )
        r.raise_for_status()
        data         = r.json()
        usage        = data.get("usage", {})
        print(f"    [tokens] {model}: "
              f"prompt={usage.get('prompt_tokens','?')}, "
              f"completion={usage.get('completion_tokens','?')}")
        content = data["choices"][0]["message"].get("content", "")
        if content and content.strip():
            return content
        if attempt < max_retries - 1:
            print(f"    [warn] {model} returned empty response, retrying in 3s …",
                  file=sys.stderr)
            time.sleep(3)
    return ""


def _call_anthropic(model: str, messages: list,
                    max_tokens: int = 32768, max_retries: int = 2) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    # Convert from OpenAI message format to Anthropic format
    system_content = next(
        (m["content"] for m in messages if m["role"] == "system"), ""
    )
    user_messages = [m for m in messages if m["role"] != "system"]

    for attempt in range(max_retries):
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"},
            json={"model": model, "max_tokens": max_tokens,
                  "system": system_content, "messages": user_messages},
            timeout=300,
        )
        r.raise_for_status()
        data    = r.json()
        content = "".join(
            b.get("text", "") for b in data.get("content", [])
            if b.get("type") == "text"
        ).strip()
        if content:
            return content
        if attempt < max_retries - 1:
            time.sleep(3)
    return ""


def _call_gemini(model: str, messages: list,
                 max_tokens: int = 32768, max_retries: int = 2) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    system_content = next(
        (m["content"] for m in messages if m["role"] == "system"), ""
    )
    user_content = next(
        (m["content"] for m in messages if m["role"] == "user"), ""
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_content}]},
        "contents":           [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig":   {"temperature": 0.1, "maxOutputTokens": max_tokens},
    }
    for attempt in range(max_retries):
        r = httpx.post(url, json=payload, timeout=300)
        r.raise_for_status()
        raw   = r.json()
        parts = raw["candidates"][0]["content"]["parts"]
        text  = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        if text:
            return text
        if attempt < max_retries - 1:
            time.sleep(3)
    return ""


def make_caller(agent_cfg: dict) -> Callable[[list], str]:
    """Return a bound call function for the given agent config."""
    provider = agent_cfg.get("provider", "openrouter")
    model    = agent_cfg.get("model", "")

    def _call(messages: list) -> str:
        if provider == "anthropic":
            return _call_anthropic(model, messages)
        elif provider == "gemini":
            return _call_gemini(model, messages)
        else:
            return _call_openai_compat(provider, model, messages)

    return _call


# ════════════════════════════════════════════════════════════════════════════
# Phase B — run test suite + parse failures
# ════════════════════════════════════════════════════════════════════════════

def run_tests(cfg: dict) -> tuple[bool, str]:
    cmd = get_test_cmd(cfg)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + "\n" + result.stderr


_RE_TEST_FILE   = re.compile(r"^\s*(FAIL|PASS)\s+(tests/\S+\.test\.[a-z]+)", re.MULTILINE)
_RE_FAIL_TEST   = re.compile(r"^\s+[x\u00d7\u2717\u274c]\s+(.+)$", re.MULTILINE)
_RE_ERROR_BLOCK = re.compile(
    r"(AssertionError|Error|TypeError|ReferenceError|assert)[^\n]*\n(?:[ \t]+[^\n]*\n)*",
    re.MULTILINE,
)
_RE_TRANSFORM_ERR = re.compile(
    r"(Transform failed|ERROR: Expected|SyntaxError.*esbuild|error TS\d+|ImportError|ModuleNotFoundError)",
    re.IGNORECASE,
)

# pytest output parser
_RE_PY_FAIL = re.compile(r"^FAILED\s+(.+?)\s*-", re.MULTILINE)
_RE_PY_ERROR = re.compile(r"(AssertionError|Error|assert)[^\n]*\n(?:[ \t]+[^\n]*\n)*", re.MULTILINE)


def _infer_src_file(test_file: str, cfg: dict) -> str:
    """Map a test file path to its corresponding src file."""
    techstack = get_techstack(cfg)
    # Python: tests/test_foo.py → src/foo.py
    if "python" in techstack or "django" in techstack:
        rel = re.sub(r"^tests/test_", "", test_file)
        rel = re.sub(r"^tests/", "", rel)
        return f"src/{rel}"
    # JS/TS: tests/components/Foo.test.tsx → src/components/Foo.tsx
    rel = test_file.replace("tests/", "", 1)
    rel = re.sub(r"\.test\.(tsx?)$", r".\1", rel)
    rel = re.sub(r"\.test\.(ts)$",   r".\1", rel)
    rel = re.sub(r"\.test\.(jsx?)$", r".\1", rel)
    return f"src/{rel}"


def parse_failures(output: str, cfg: dict) -> list[FailureCluster]:
    techstack = get_techstack(cfg)
    clusters: dict[str, FailureCluster] = {}

    if "python" in techstack or "django" in techstack:
        # pytest output format
        for m in _RE_PY_FAIL.finditer(output):
            test_id  = m.group(1).strip()
            # test_id format: tests/test_foo.py::TestClass::test_method
            parts    = test_id.split("::")
            test_file = parts[0]
            test_name = "::".join(parts[1:]) if len(parts) > 1 else test_id
            src_file  = _infer_src_file(test_file, cfg)
            cluster   = clusters.setdefault(
                test_file, FailureCluster(test_file=test_file, src_file=src_file)
            )
            errors = _RE_PY_ERROR.findall(output)
            snippet = errors[len(cluster.failures)] if len(errors) > len(cluster.failures) else output[:500]
            cluster.failures.append(TestFailure(
                test_file=test_file, test_name=test_name,
                error_snippet=snippet.strip()[:600],
            ))
    else:
        # vitest / jest output format
        file_matches = list(_RE_TEST_FILE.finditer(output))
        sections: list[tuple[str, str, str]] = []
        for i, m in enumerate(file_matches):
            start = m.end()
            end   = file_matches[i + 1].start() if i + 1 < len(file_matches) else len(output)
            sections.append((m.group(1), m.group(2), output[start:end]))

        for status, test_file, section in sections:
            if status != "FAIL":
                continue
            src_file = _infer_src_file(test_file, cfg)
            cluster  = clusters.setdefault(
                test_file, FailureCluster(test_file=test_file, src_file=src_file)
            )
            if _RE_TRANSFORM_ERR.search(section):
                cluster.is_transform_error = True
            fail_names = _RE_FAIL_TEST.findall(section)
            errors     = _RE_ERROR_BLOCK.findall(section)
            for j, name in enumerate(fail_names):
                snippet = errors[j] if j < len(errors) else section[:500]
                cluster.failures.append(TestFailure(
                    test_file=test_file, test_name=name.strip(),
                    error_snippet=snippet.strip()[:600],
                ))
            if not cluster.failures:
                cluster.failures.append(TestFailure(
                    test_file=test_file, test_name="(parse fallback)",
                    error_snippet=section[:800].strip(),
                ))

    return list(clusters.values())


def merge_cluster_state(
    new_clusters: list[FailureCluster],
    prev_state:   dict[str, FailureCluster],
) -> list[FailureCluster]:
    for c in new_clusters:
        if c.key in prev_state:
            prev = prev_state[c.key]
            c.attempt_count    = prev.attempt_count
            c.last_fingerprint = prev.last_fingerprint
            c.escalated        = prev.escalated
            c.owner            = prev.owner
    return new_clusters


# ════════════════════════════════════════════════════════════════════════════
# Layer 0 — Static pre-pass (no LLM)
# ════════════════════════════════════════════════════════════════════════════

_RE_JSX_GENERIC    = re.compile(
    r"(<\w[\w.]*)<(\w[\w,\s]*)>(\s*(?:events|data|items|props|value)\s*=)"
)
_RE_TEMPLATE_WIDTH = re.compile(r"(`\$\{)([^}]*\*\s*100)(\}%`)")
_RE_FLOAT_WIDTH    = re.compile(r"(width:\s*)(\d+\.\d+)(%)")


def _static_fix_transform(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "file not found"
    orig    = path.read_text()
    patched = _RE_JSX_GENERIC.sub(r"\1\3", orig)
    if patched != orig:
        path.write_text(patched)
        return True, "removed JSX generic type param causing esbuild parse error"
    return False, "no static transform pattern matched"


def _static_fix_src(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "file not found"
    orig    = path.read_text()
    patched = _RE_TEMPLATE_WIDTH.sub(r"`${Math.round(\2)}\3", orig)
    patched = _RE_FLOAT_WIDTH.sub(
        lambda m: f"{m.group(1)}{round(float(m.group(2)))}{m.group(3)}", patched,
    )
    if patched != orig:
        path.write_text(patched)
        return True, "rounded floating-point percentage widths"
    return False, "no static src pattern matched"


def layer0_static_prepass(cluster: FailureCluster, verbose: bool) -> tuple[bool, str]:
    if verbose:
        print(f"    [L0] Static pre-pass for {cluster.test_file} …")
    if cluster.is_transform_error:
        fixed, desc = _static_fix_transform(ROOT / cluster.test_file)
        if fixed:
            print(f"    [L0] ✓ {desc}")
            return True, desc
    fixed, desc = _static_fix_src(ROOT / cluster.src_file)
    if fixed:
        print(f"    [L0] ✓ {desc}")
        return True, desc
    if verbose:
        print("    [L0] No static pattern matched.")
    return False, "no static fix applicable"


# ════════════════════════════════════════════════════════════════════════════
# Prompts (techstack-aware, no hardcoded language)
# ════════════════════════════════════════════════════════════════════════════

def build_tester_system(cfg: dict, judge_findings: str = "") -> str:
    lang   = get_lang_label(cfg)
    fence  = get_fence(cfg)
    is_ts  = fence == "typescript"

    style_rules = (
        "TypeScript strict — no `any`. Tailwind only." if is_ts
        else "Follow project style conventions."
    )

    base = f"""\
You are a senior {lang} developer doing a SURFACE-LEVEL fix for ONE failing cluster.

Your scope is LIMITED to surface bugs:
  - Wrong DOM/query selector or missing test-id attribute
  - Incorrect import path or missing export
  - Style class name typo or wrong colour/state class
  - Floating-point precision in style values
  - Text content or label mismatch
  - Missing or wrong aria/accessibility attribute

DO NOT:
  - Rewrite hook/service logic, state machines, or data generation
  - Change public interfaces (types, props, function signatures)
  - Touch any file other than the one src file listed

If the bug is a LOGIC or ALGORITHM issue (state transition, math, data shape),
return the source file UNCHANGED and set:
  "explanation": "LOGIC_BUG — needs logic_agent debugger"

Return JSON (no other keys):
{{
  "file_path": "src/components/Example.{fence.replace('typescript','tsx').replace('python','py')}",
  "code": "<full file content — UNCHANGED if logic bug>",
  "explanation": "what was fixed, or LOGIC_BUG"
}}
{style_rules} Raw JSON. No markdown fences.
"""
    if not judge_findings:
        return base

    # Inject findings as a negative primer before the task instructions
    relevant_lines: list[str] = []
    capture = False
    for line in judge_findings.splitlines():
        if "## Non-blocking" in line or "## Patterns to avoid" in line:
            capture = True
        elif line.startswith("## ") and capture:
            if "Blocking" in line:
                capture = False
        if capture:
            relevant_lines.append(line)

    findings_block = "\n".join(relevant_lines).strip()
    if not findings_block:
        return base

    return (
        f"## Previous run — do NOT repeat these mistakes\n"
        f"{findings_block}\n\n---\n\n"
        + base
    )


def build_logic_system(cfg: dict, global_notes: str = "",
                        judge_findings: str = "") -> str:
    lang  = get_lang_label(cfg)
    fence = get_fence(cfg)
    scope_patterns = cfg.get("logic_scope_patterns") or _TECHSTACK_DEFAULT_SCOPE.get(
        get_techstack(cfg), _DEFAULT_LOGIC_SCOPE_PATTERNS
    )
    scope_display = "\n  ".join(f"- {p}" for p in scope_patterns)
    is_ts = fence == "typescript"
    style_rules = "TypeScript strict — no `any`." if is_ts else "Follow project style conventions."

    notes_block = (
        f"\n## Architect's Global Notes (MUST follow)\n{global_notes}\n"
        if global_notes else ""
    )
    findings_block = (
        f"\n## Judge findings from previous run — do NOT reintroduce\n{judge_findings}\n"
        if judge_findings else ""
    )
    kb_content = _load_knowledge_base()
    kb_block = (
        f"\n## Accumulated knowledge from human fixes — study these patterns\n"
        f"These are bugs the AI repair loop could NOT fix — a human intervened.\n"
        f"If the current cluster resembles any pattern here, apply the same fix strategy.\n\n"
        f"{kb_content}\n"
        if kb_content else ""
    )

    return f"""\
You are a senior {lang} logic debugger specialising in business logic and data processing.
{notes_block}{findings_block}{kb_block}
You receive a failing cluster for a logic or data file.
Fix the LOGIC — not the UI, not the styling.

SCOPE (STRICTLY ENFORCED):
  You may ONLY write to:
  {scope_display}
  Never touch presentation/component files — those are owned by tester_agent.
  Never change public interfaces unless they directly contradict spec + test.

WHAT TO FIX:
  - State machine transitions (play/pause/reset/scrub semantics)
  - Data generation (values, rates, cluster timing, constraints)
  - Range/constraint violations per acceptance criteria
  - NaN / undefined from uninitialised values or wrong array access
  - Off-by-one in window/index calculations

HOW:
  1. Read each test assertion as a hard requirement.
  2. Trace the state step by step.
  3. If this cluster resembles a pattern in the knowledge base above, apply that fix.
  4. Identify the root cause — one specific line or function.
  5. Rewrite ONLY the broken function(s). Leave everything else intact.

Return JSON:
{{
  "file_path": "src/example/file.py",
  "code": "<full corrected file>",
  "root_cause": "one sentence: what was logically wrong",
  "explanation": "what you changed and why"
}}
{style_rules} Raw JSON. No markdown fences.
"""


def build_consistency_system() -> str:
    return """\
You are a test-vs-code consistency auditor. You do NOT fix code or tests.
Your only job: read the spec, the failing test, the source file, and the assertion,
then classify who is wrong.

Return raw JSON only — no markdown fences, no preamble:
{
  "verdict": "CODE_BUG" | "TEST_FRAGILE" | "SPEC_AMBIG" | "THRESHOLD_OK",
  "confidence": "high" | "medium" | "low",
  "reasoning": "one paragraph — cite the specific spec section, test line, and code line",
  "test_patch_allowed": true | false,
  "test_patch_rationale": "only if test_patch_allowed=true — exactly what to change and why"
}

Verdict definitions:
  CODE_BUG      — code does not implement the spec correctly; test expectation is valid.
                  test_patch_allowed MUST be false.
  TEST_FRAGILE  — test query or assertion is brittle (wrong selector, DOM structure mismatch,
                  text split across nodes, timing issue) but the INTENT is correct.
                  The fix is to rewrite the query/assertion, NOT what is being tested.
                  test_patch_allowed = true.
  SPEC_AMBIG    — spec is genuinely ambiguous; test and code both have valid interpretations.
                  Cannot be resolved automatically. test_patch_allowed MUST be false.
  THRESHOLD_OK  — code behaviour is correct per spec, but test threshold is too strict.
                  test_patch_allowed = true ONLY if relaxing does NOT mask a real bug.

CRITICAL: never set test_patch_allowed=true for CODE_BUG.
"""


def build_test_repair_system() -> str:
    return """\
You are fixing a FRAGILE TEST or THRESHOLD. The test intent is correct — only the
implementation of the assertion is brittle or too strict.

Allowed changes ONLY:
  TEST_FRAGILE  — fix DOM query selectors, async patterns, or text-content matchers.
  THRESHOLD_OK  — relax a numerical threshold WITH explicit spec citation.

NOT allowed:
  - Changing what behaviour is being tested
  - Removing assertions or reducing assertion count
  - Changing expected values to match wrong code behaviour
  - Adding trivial pass-through assertions
  - Touching src/ files

Return raw JSON only:
{
  "file_path": "tests/example.test.ts",
  "code": "<full corrected test file>",
  "changes_made": ["one item per change, quoted verbatim before → after"],
  "explanation": "one sentence"
}
"""


# ════════════════════════════════════════════════════════════════════════════
# State timeline extractor (techstack-aware)
# ════════════════════════════════════════════════════════════════════════════

def _build_state_timeline(test_code: str, cfg: dict, max_entries: int = 12) -> str:
    keywords = get_timeline_keywords(cfg)
    lines    = test_code.splitlines()
    timeline: list[str] = []

    for line in lines:
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("#") or s.startswith("import"):
            continue
        if s.startswith("describe(") or s.startswith("it(") or s.startswith("test(") \
                or s.startswith("class Test") or s.startswith("def test_"):
            timeline.append(f"TEST: {s[:100]}")
        elif any(kw in s for kw in keywords):
            timeline.append(f"  ACTION: {s[:100]}")
        elif s.startswith("expect(") or s.startswith("assert ") or s.startswith("self.assert"):
            timeline.append(f"  ASSERT: {s[:100]}")
        else:
            continue
        if len(timeline) >= max_entries * 3:
            timeline.append("  … (truncated)")
            break

    return "\n".join(timeline) if timeline else "(could not extract timeline)"


def _read_file_safe(path: Path) -> str:
    return path.read_text() if path.exists() else f"# FILE NOT FOUND: {path}\n"


# ════════════════════════════════════════════════════════════════════════════
# Phase 0 — Consistency checker
# ════════════════════════════════════════════════════════════════════════════

def check_consistency(
    cluster:     FailureCluster,
    spec:        str,
    cfg:         dict,
    call_tester: Callable[[list], str],
    verbose:     bool = False,
) -> dict:
    fence     = get_fence(cfg)
    src_code  = _read_file_safe(ROOT / cluster.src_file)
    test_code = _read_file_safe(ROOT / cluster.test_file)
    error_log = cluster.error_block()

    user_content = (
        f"### spec.md\n\n{spec}\n\n"
        f"### Test file: {cluster.test_file}\n"
        f"```{fence}\n{test_code}\n```\n\n"
        f"### Source file: {cluster.src_file}\n"
        f"```{fence}\n{src_code}\n```\n\n"
        f"### Failing assertions\n```\n{error_log}\n```"
    )
    messages = [
        {"role": "system", "content": build_consistency_system()},
        {"role": "user",   "content": user_content},
    ]
    if verbose:
        print(f"    [P0] Consistency check → tester_agent ({cluster.test_file}) …")

    try:
        raw = call_tester(messages).strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",        "", raw)
        result  = json.loads(raw)
        verdict = result.get("verdict", "CODE_BUG")
        conf    = result.get("confidence", "low")
        print(f"    [P0] verdict={verdict} confidence={conf}")
        if verbose:
            print(f"    [P0] reasoning: {result.get('reasoning','')[:200]}")
        return result
    except Exception as e:
        print(f"    [P0] Check failed ({e}), defaulting to CODE_BUG.", file=sys.stderr)
        return {
            "verdict": "CODE_BUG", "confidence": "low",
            "test_patch_allowed": False,
            "reasoning": f"consistency check error: {e}",
            "test_patch_rationale": "",
        }


def repair_test_file(
    cluster:     FailureCluster,
    verdict:     dict,
    cfg:         dict,
    call_tester: Callable[[list], str],
    verbose:     bool = False,
) -> bool:
    fence     = get_fence(cfg)
    spec      = _load_spec()
    test_code = _read_file_safe(ROOT / cluster.test_file)
    src_code  = _read_file_safe(ROOT / cluster.src_file)
    error_log = cluster.error_block()

    user_content = (
        f"### spec.md\n\n{spec}\n\n"
        f"### Source file (DO NOT MODIFY): {cluster.src_file}\n"
        f"```{fence}\n{src_code}\n```\n\n"
        f"### Test file to fix: {cluster.test_file}\n"
        f"```{fence}\n{test_code}\n```\n\n"
        f"### Failing assertions\n```\n{error_log}\n```\n\n"
        f"### Auditor rationale\n{verdict.get('test_patch_rationale', '')}\n\n"
        f"Verdict: {verdict.get('verdict')} — fix ONLY what the rationale describes."
    )
    messages = [
        {"role": "system", "content": build_test_repair_system()},
        {"role": "user",   "content": user_content},
    ]
    if verbose:
        print(f"    [P0-fix] Rewriting test: {cluster.test_file} …")

    try:
        raw = call_tester(messages).strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",        "", raw)
        patch = json.loads(raw)
    except Exception as e:
        print(f"    [P0-fix] Parse error: {e}", file=sys.stderr)
        return False

    out_path = ROOT / patch.get("file_path", cluster.test_file)
    if not str(out_path.resolve()).startswith(str((ROOT / "tests").resolve())):
        print(f"    [P0-fix] ⚠ Scope violation: tried to write {out_path}. Rejected.",
              file=sys.stderr)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patch["code"])
    changes = patch.get("changes_made", [])
    print(f"    [P0-fix] ✓ Test updated — {patch.get('explanation','(no explanation)')}")
    for ch in changes:
        print(f"      • {ch}")
    return True


# ════════════════════════════════════════════════════════════════════════════
# Shared repair executor
# ════════════════════════════════════════════════════════════════════════════

def _call_repair(
    cluster:        FailureCluster,
    call_api:       Callable[[list], str],
    system:         str,
    cfg:            dict,
    extra_ctx:      str  = "",
    verbose:        bool = False,
    layer_name:     str  = "L1",
    scope_patterns: list[re.Pattern] | None = None,
) -> tuple[bool, str]:
    fence     = get_fence(cfg)
    spec      = _load_spec()
    src_code  = _read_file_safe(ROOT / cluster.src_file)
    test_code = _read_file_safe(ROOT / cluster.test_file)
    error_log = cluster.error_block()

    user_content = (
        f"### spec.md\n\n{spec}\n\n"
        f"### Test file (read-only): {cluster.test_file}\n"
        f"```{fence}\n{test_code}\n```\n\n"
        f"### Source file to fix: {cluster.src_file}\n"
        f"```{fence}\n{src_code}\n```\n\n"
        f"### Failing tests\n```\n{error_log}\n```"
        + (f"\n\n### Expected state timeline\n```\n{extra_ctx}\n```" if extra_ctx else "")
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]

    agent_label = "tester_agent" if layer_name == "L1" else "logic_agent"
    if verbose:
        print(f"    [{layer_name}] → {agent_label} "
              f"(attempt #{cluster.attempt_count + 1}, "
              f"{len(cluster.failures)} failure(s)) …")

    try:
        raw = call_api(messages).strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",        "", raw)
        patch = json.loads(raw)
    except Exception as e:
        print(f"    [{layer_name}] Parse error: {e}", file=sys.stderr)
        return False, f"parse error: {e}"

    explanation = patch.get("explanation", "")

    # tester_agent LOGIC_BUG signal — defer to logic_agent
    if layer_name == "L1" and "LOGIC_BUG" in explanation.upper():
        print(f"    [L1] tester_agent signalled LOGIC_BUG — deferring to logic_agent.")
        return False, "LOGIC_BUG"

    out_rel = patch.get("file_path", cluster.src_file)

    # Scope guard for logic_agent
    if scope_patterns and not any(p.match(out_rel) for p in scope_patterns):
        print(f"    [{layer_name}] ⚠ Scope violation: tried to write {out_rel}. "
              f"Patch rejected.")
        return False, f"scope violation: {out_rel}"

    out_path = ROOT / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patch["code"])

    root_cause = patch.get("root_cause", "")
    summary    = f"{root_cause} — {explanation}" if root_cause else explanation
    print(f"    [{layer_name}] ✓ Patched {out_rel} — {summary or '(no explanation)'}")
    return True, explanation


# ════════════════════════════════════════════════════════════════════════════
# Phase C — cluster dispatch
# ════════════════════════════════════════════════════════════════════════════

def repair_cluster(
    cluster:              FailureCluster,
    global_notes:         str,
    max_cluster_attempts: int,
    cfg:                  dict,
    call_tester:          Callable[[list], str],
    call_logic:           Callable[[list], str],
    scope_patterns:       list[re.Pattern],
    judge_findings:       str = "",
    verbose:              bool = False,
) -> ClusterRepairRecord:
    """
    Dispatch cluster through P0 → L0 → L1/L2 → L3.

    Full decision tree:
      P0  Consistency check (first attempt only):
            SPEC_AMBIG          → ESCALATED→human immediately
            TEST_FRAGILE / THRESHOLD_OK + test_patch_allowed → repair test → done
            CODE_BUG            → fall through to L0
      L3  Give-up guard (checked before every LLM call):
            cluster.escalated   → SKIP
            attempts ≥ max      → ESCALATED→human
      L0  Static pre-pass (no LLM)
      L1  tester_agent surface fix — non-logic files only:
            cluster.owner == "logic"  → skip (locked out)
            is_logic_scope            → skip (go straight to L2)
            is_stale                  → skip (→ L2)
            LOGIC_BUG signal + scope  → transfer owner→"logic", → L2
            LOGIC_BUG + out of scope  → ESCALATED→human
      L2  logic_agent deep fix — scope-locked
    """
    # ── L3 guard ─────────────────────────────────────────────────────────────
    if cluster.escalated:
        print(f"    [SKIP] {cluster.test_file} — ESCALATED, skipping.")
        return ClusterRepairRecord(
            cluster=cluster.key, src_file=cluster.src_file,
            failures=len(cluster.failures), repaired=False,
            layer_used="skipped", escalated=True,
            escalated_to=cluster.owner, owner=cluster.owner,
            note="previously escalated",
        )

    # ── Phase 0: consistency check (first attempt only) ───────────────────────
    spec = _load_spec()
    consistency_verdict_label = ""
    if cluster.attempt_count == 0:
        cv      = check_consistency(cluster, spec, cfg, call_tester, verbose=verbose)
        verdict = cv.get("verdict", "CODE_BUG")

        if verdict == "SPEC_AMBIG":
            cluster.escalated = True
            return ClusterRepairRecord(
                cluster=cluster.key, src_file=cluster.src_file,
                failures=len(cluster.failures), repaired=False,
                layer_used="skipped", escalated=True,
                escalated_to="human", owner=cluster.owner,
                note=f"spec ambiguous: {cv.get('reasoning','')[:150]}",
                consistency_verdict=verdict,
            )

        if verdict in ("TEST_FRAGILE", "THRESHOLD_OK") and cv.get("test_patch_allowed"):
            ok = repair_test_file(cluster, cv, cfg, call_tester, verbose=verbose)
            cluster.attempt_count   += 1
            cluster.last_fingerprint = cluster.fingerprint()
            return ClusterRepairRecord(
                cluster=cluster.key, src_file=cluster.src_file,
                failures=len(cluster.failures), repaired=ok,
                layer_used="test_rewrite", escalated=not ok,
                escalated_to="human" if not ok else "",
                owner=cluster.owner,
                note=(cv.get("test_patch_rationale", "")[:150] if ok else "test rewrite failed"),
                consistency_verdict=verdict,
            )

        consistency_verdict_label = verdict

    # ── L0: static pre-pass ───────────────────────────────────────────────────
    l0_fixed, l0_desc = layer0_static_prepass(cluster, verbose)
    if l0_fixed:
        cluster.attempt_count   += 1
        cluster.last_fingerprint = cluster.fingerprint()
        return ClusterRepairRecord(
            cluster=cluster.key, src_file=cluster.src_file,
            failures=len(cluster.failures), repaired=True,
            layer_used="static", escalated=False,
            escalated_to="", owner=cluster.owner, note=l0_desc,
            consistency_verdict=consistency_verdict_label,
        )

    # ── Give-up check ─────────────────────────────────────────────────────────
    if cluster.attempt_count >= max_cluster_attempts:
        cluster.escalated = True
        print(f"    [L3] ⚠ Gave up on {cluster.test_file} after "
              f"{cluster.attempt_count} attempt(s). ESCALATED→human.")
        return ClusterRepairRecord(
            cluster=cluster.key, src_file=cluster.src_file,
            failures=len(cluster.failures), repaired=False,
            layer_used="skipped", escalated=True, escalated_to="human",
            owner=cluster.owner,
            note=f"gave up after {cluster.attempt_count} LLM attempt(s)",
            consistency_verdict=consistency_verdict_label,
        )

    current_fp = cluster.fingerprint()
    is_stale   = cluster.attempt_count > 0 and cluster.last_fingerprint == current_fp

    skip_tester = (
        cluster.owner == "logic"
        or cluster.is_logic_scope(scope_patterns)
        or is_stale
    )

    if not skip_tester:
        # ── L1: tester_agent surface fix ─────────────────────────────────────
        tester_system = build_tester_system(cfg, judge_findings)
        ok, note = _call_repair(
            cluster, call_tester,
            system=tester_system, cfg=cfg,
            verbose=verbose, layer_name="L1",
        )
        cluster.attempt_count   += 1
        cluster.last_fingerprint = current_fp

        if ok:
            return ClusterRepairRecord(
                cluster=cluster.key, src_file=cluster.src_file,
                failures=len(cluster.failures), repaired=True,
                layer_used="tester_surface", escalated=False,
                escalated_to="", owner="tester",
                consistency_verdict=consistency_verdict_label,
            )

        # tester_agent failed: escalate to logic_agent if in scope
        if cluster.is_logic_scope(scope_patterns):
            print(f"    [L1→L2] Transferring {cluster.test_file} to logic_agent.")
            cluster.owner = "logic"
        else:
            cluster.escalated = True
            return ClusterRepairRecord(
                cluster=cluster.key, src_file=cluster.src_file,
                failures=len(cluster.failures), repaired=False,
                layer_used="tester_surface", escalated=True,
                escalated_to="human", owner="tester",
                note=f"tester_agent failed on file outside logic scope: {note}",
                consistency_verdict=consistency_verdict_label,
            )

    # ── L2: logic_agent deep fix ─────────────────────────────────────────────
    cluster.owner = "logic"
    test_code     = _read_file_safe(ROOT / cluster.test_file)
    timeline      = _build_state_timeline(test_code, cfg)
    logic_system  = build_logic_system(cfg, global_notes, judge_findings)

    ok, note = _call_repair(
        cluster, call_logic,
        system=logic_system, cfg=cfg,
        extra_ctx=timeline,
        verbose=verbose, layer_name="L2",
        scope_patterns=scope_patterns,
    )
    cluster.attempt_count   += 1
    cluster.last_fingerprint = current_fp

    return ClusterRepairRecord(
        cluster=cluster.key, src_file=cluster.src_file,
        failures=len(cluster.failures), repaired=ok,
        layer_used="logic_deep", escalated=False,
        escalated_to="", owner="logic", note=note,
        consistency_verdict=consistency_verdict_label,
    )


# ════════════════════════════════════════════════════════════════════════════
# Main — B / C / D loop
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test loop: tester_agent (surface) + logic_agent (logic)"
    )
    parser.add_argument("--max-iter",             type=int, default=3)
    parser.add_argument("--max-cluster-attempts", type=int, default=2)
    parser.add_argument("--verbose",              action="store_true")
    args = parser.parse_args()

    cfg            = load_config()
    tester_cfg     = get_agent_cfg(cfg, "tester")
    logic_cfg      = get_agent_cfg(cfg, "logic")
    scope_patterns = get_logic_scope_patterns(cfg)
    techstack      = get_techstack(cfg)

    print(f"{TAG} Techstack    : {techstack}")
    print(f"{TAG} tester_agent : {tester_cfg.get('model','?')} ({tester_cfg.get('provider','?')})")
    print(f"{TAG} logic_agent  : {logic_cfg.get('model','?')} ({logic_cfg.get('provider','?')})")

    call_tester = make_caller(tester_cfg)
    call_logic  = make_caller(logic_cfg)

    max_iter             = args.max_iter
    max_cluster_attempts = args.max_cluster_attempts
    verbose              = args.verbose

    global_notes = _load_plan_global_notes()
    if global_notes:
        print(f"{TAG} Plan global_notes loaded ({len(global_notes)} chars) "
              f"— injected into logic_agent prompts")

    judge_findings = _load_judge_findings()
    if judge_findings:
        print(f"{TAG} Judge findings loaded ({len(judge_findings)} chars) "
              f"— injected into tester + logic prompts (regression prevention)")

    iteration_records: list[IterationRecord]     = []
    cluster_state:     dict[str, FailureCluster] = {}
    escalated_log:     list[dict]                = []

    for iteration in range(1, max_iter + 1):
        itag = f"{TAG}[{iteration}/{max_iter}]"

        print(f"\n{itag} Phase B — running tests …")
        passed, output = run_tests(cfg)

        summary_line = next(
            (l.strip() for l in output.splitlines()
             if ("passed" in l or "failed" in l) and ("test" in l.lower() or "error" in l.lower())),
            output.strip().splitlines()[-1] if output.strip() else "no output",
        )
        print(f"{itag} {summary_line}")

        if passed:
            print(f"{itag} ✓ All tests passed.")
            iteration_records.append(IterationRecord(
                iteration=iteration, passed=True, summary=summary_line,
                clusters_found=0, clusters_repaired=0, cluster_details=[],
                log_snippet="",
            ))
            break

        clusters = parse_failures(output, cfg)
        clusters = merge_cluster_state(clusters, cluster_state)

        print(f"{itag} {len(clusters)} failing cluster(s):")
        for c in clusters:
            markers = []
            if c.attempt_count > 0 and c.last_fingerprint == c.fingerprint():
                markers.append("STALE")
            if c.escalated:
                markers.append("ESCALATED")
            if c.owner == "logic":
                markers.append("LOGIC")
            scope_label = "[logic-scope]" if c.is_logic_scope(scope_patterns) else "[surface]"
            marker_str  = f"  [{', '.join(markers)}]" if markers else ""
            print(f"  * {scope_label} {c.test_file} ({len(c.failures)} failure(s)){marker_str}")

        if not clusters:
            print(f"{itag} Could not parse clusters. Stopping.", file=sys.stderr)
            iteration_records.append(IterationRecord(
                iteration=iteration, passed=False, summary=summary_line,
                clusters_found=0, clusters_repaired=0, cluster_details=[],
                log_snippet=output[-1200:],
            ))
            break

        if iteration == max_iter:
            print(f"{itag} Reached max_iter — {len(clusters)} cluster(s) remaining.")
            iteration_records.append(IterationRecord(
                iteration=iteration, passed=False, summary=summary_line,
                clusters_found=len(clusters), clusters_repaired=0,
                cluster_details=[{
                    "cluster":  c.key, "failures": len(c.failures),
                    "escalated": c.escalated, "owner": c.owner,
                    "attempts": c.attempt_count,
                } for c in clusters],
                log_snippet=output[-1200:],
            ))
            break

        print(f"{itag} Phase C — dispatching {len(clusters)} cluster(s) …")
        repaired        = 0
        cluster_details = []

        for cluster in clusters:
            print(f"  -> {cluster.test_file} "
                  f"(owner={cluster.owner}, attempt #{cluster.attempt_count + 1})")
            record = repair_cluster(
                cluster=cluster,
                global_notes=global_notes,
                max_cluster_attempts=max_cluster_attempts,
                cfg=cfg,
                call_tester=call_tester,
                call_logic=call_logic,
                scope_patterns=scope_patterns,
                judge_findings=judge_findings,
                verbose=verbose,
            )
            cluster_state[cluster.key] = cluster
            repaired += int(record.repaired)
            detail = {
                "cluster":             record.cluster,
                "src_file":            record.src_file,
                "failures":            record.failures,
                "repaired":            record.repaired,
                "layer_used":          record.layer_used,
                "escalated":           record.escalated,
                "escalated_to":        record.escalated_to,
                "owner":               record.owner,
                "note":                record.note,
                "consistency_verdict": record.consistency_verdict,
            }
            cluster_details.append(detail)
            if record.escalated:
                escalated_log.append({"iteration": iteration, **detail})

        print(f"{itag} Phase C done — {repaired}/{len(clusters)} patched.")
        iteration_records.append(IterationRecord(
            iteration=iteration, passed=False, summary=summary_line,
            clusters_found=len(clusters), clusters_repaired=repaired,
            cluster_details=cluster_details,
            log_snippet=output[-1200:],
        ))

    # ── Reports ───────────────────────────────────────────────────────────────
    final_passed = bool(iteration_records and iteration_records[-1].passed)
    report = {
        "tester_agent":         tester_cfg.get("model", "unknown"),
        "logic_agent":          logic_cfg.get("model", "unknown"),
        "techstack":            techstack,
        "max_iter":             max_iter,
        "max_cluster_attempts": max_cluster_attempts,
        "total_iterations":     len(iteration_records),
        "final_status":         "PASS" if final_passed else "FAIL",
        "iterations":           [asdict(r) for r in iteration_records],
    }
    (REPORTS_DIR / "test_iterations.json").write_text(json.dumps(report, indent=2))
    print(f"\n{TAG} Report → reports/test_iterations.json")

    if escalated_log:
        esc_path = REPORTS_DIR / "escalated_clusters.json"
        esc_path.write_text(json.dumps({
            "total_escalated": len(escalated_log),
            "clusters":        escalated_log,
        }, indent=2))
        print(f"{TAG} ⚠ Escalated → {esc_path} ({len(escalated_log)} cluster(s))")

    if not final_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
