"""
toolkits/devops_mlops/infra_absorber.py
=======================================
Step 0 — Absorb an infrastructure / MLOps codebase into the knowledge layer.

Extends the SWE absorber pattern for DevOps/MLOps repos where "codebase"
means: Terraform modules, Helm charts, Kubernetes manifests, ArgoCD
Applications, Airflow DAGs, Ansible playbooks, Dockerfiles, Jenkinsfiles.

All downstream devops_mlops toolkit modules consume the artifacts produced
here — do not skip this step.

────────────────────────────────────────────────────────────────
Key differences from SWE absorber (01_absorber.py)
────────────────────────────────────────────────────────────────

  SWE absorber         Infra absorber
  ─────────────────    ────────────────────────────────────────
  Python/TS AST sigs   HCL resource/module/output extraction
  key-only for YAML    fingerprint-aware YAML (Helm vs K8s vs ArgoCD)
  env var detection    IRSA annotation + SA name + image ref extraction
  service detection    infra topology: namespaces, DAG graph, IAM roles
  _MAX_PER_FILE=2000   _MAX_PER_FILE_INFRA=4000 (HCL files are bigger)
  skip .terraform/     .terraform/ already in _BUILTIN_SKIP_DIRS ✓
  code call flows      ## Infrastructure Topology section (new)
  tech debt section    ## Drift Risk section (new)

────────────────────────────────────────────────────────────────
Infra-specific extraction modes (by file fingerprint)
────────────────────────────────────────────────────────────────

  .tf / .hcl     → HCL extraction: resource blocks, module calls,
                   variable names, output names, provider configs.
                   Uses python-hcl2 if available, regex fallback.

  .yaml / .yml   → fingerprint-aware:
    Helm values  → extract image.repository/tag, resources.limits/requests,
                   serviceAccount.annotations, ingress, replicaCount
    K8s manifest → extract kind, name, namespace, serviceAccountName, image
    ArgoCD App   → extract repoURL, targetRevision, helm.valueFiles, syncPolicy
    DAG file     → (Python) extract dag_id, task_ids, operators, connections
    Ansible      → extract hosts, roles, task names
    plain YAML   → key-only (redact values, keep keys)

  Dockerfile     → extract FROM, RUN apt/pip, EXPOSE, ENTRYPOINT/CMD
  Jenkinsfile    → extract stages, agent, environment blocks
  .py (DAG)      → extract dag_id, task_ids, operators, XCom keys, connections
  .py (non-DAG)  → signature-only (same as SWE absorber)

────────────────────────────────────────────────────────────────
Infra topology extraction (new vs SWE)
────────────────────────────────────────────────────────────────

After Phase 2 extraction, infra_absorber builds a structured
infra_topology dict injected into the LLM prompt as a dedicated
section. Contains:

  terraform_resources   list of {type, name, module} from .tf files
  k8s_namespaces        unique namespaces across manifests
  helm_releases         {chart, version, namespace} from Chart.yaml files
  argocd_apps           {name, repoURL, targetRevision} from ArgoCD apps
  dag_inventory         {dag_id, tasks, schedule} from Airflow DAG files
  iam_roles             IAM role names from Terraform aws_iam_role resources
  service_accounts      {name, namespace, irsa_annotation} from K8s SAs
  image_refs            unique container image refs across all manifests

  These are also written into infra_map.json["infra_topology"] for
  consumption by config_consistency_checker.py.

────────────────────────────────────────────────────────────────
Outputs written
────────────────────────────────────────────────────────────────

  infra_map.md          (short-term, OVERWRITE)
    LLM narrative. Sections: Project Overview, Infrastructure Topology,
    Module Inventory, Data Flow (pipeline DAG graph), Config & Secrets,
    Git/Blame, Drift Risk, Absorber Notes.

  infra_map.json        (short-term, OVERWRITE)
    Structured: meta, infra_topology, config, git, staleness.
    Primary input for config_consistency_checker.py.

  infra_absorber_log.json  (long-term, APPEND)

  cache/infra_snapshot.json  (internal SHA256 cache)

────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────

  python infra_absorber.py --project iot-mlops
  python infra_absorber.py --project iot-mlops --target /path/to/repo
  python infra_absorber.py --project iot-mlops --git-scope 6m
  python infra_absorber.py --project iot-mlops --force
  python infra_absorber.py --project iot-mlops --dry-run
  python infra_absorber.py --project iot-mlops --mode patch
  python infra_absorber.py --project iot-mlops --changed-since HEAD~1 --mode patch
  python infra_absorber.py --project iot-mlops --install-hook
  python infra_absorber.py --project iot-mlops --check-stale

────────────────────────────────────────────────────────────────
Artifact impact by command
────────────────────────────────────────────────────────────────

  Command                  infra_map.md   infra_map.json  log.json  cache
  ─────────────────────── ────────────── ──────────────  ────────  ─────
  (full run)               OVERWRITE      OVERWRITE       APPEND    OVERWRITE
  --force                  OVERWRITE      OVERWRITE       APPEND    OVERWRITE
  --dry-run                –              –               –         OVERWRITE
  --mode patch             REGEX REPLACE  IN-PLACE UPDATE APPEND    –
  --changed-since <ref>    OVERWRITE      OVERWRITE       APPEND    OVERWRITE (partial)
  --check-stale            –              staleness UPSERT –        –
  --install-hook           –              –               –         –

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Repo root resolution ──────────────────────────────────────────────────────
_TOOLKIT_DIR = Path(__file__).parent
_REPO_ROOT   = _TOOLKIT_DIR.parent.parent   # toolkits/devops_mlops/../../
sys.path.insert(0, str(_REPO_ROOT))

from artifacts.paths import (  # noqa: E402
    ensure_dirs,
    get_project_slug,
)
from artifacts.models import call_model, get_model, get_provider  # noqa: E402
from modules.artifact_tracking import (  # noqa: E402
    track_read, track_write,
    print_summary as print_artifact_summary,
)
from modules.cost import print_call, print_summary as print_cost_summary, record_usage  # noqa: E402
from modules.md_header import apply_header as apply_md_header  # noqa: E402

# ── Optional hcl2 parser ─────────────────────────────────────────────────────
try:
    import hcl2  # type: ignore
    _HCL2_AVAILABLE = True
except ImportError:
    _HCL2_AVAILABLE = False

ROLE            = "absorber"          # reuse absorber model config
_MAX_TOKENS_MAP = 16384
_MAX_FILE_BYTES = 512 * 1024          # 512 KB — HCL files can be larger
_IGNORED_FILE   = "absorber.ignored"  # same convention as SWE absorber

# Per-file content cap — larger than SWE (2000) because HCL resources are verbose
_MAX_PER_FILE_CODE  = 2_000
_MAX_PER_FILE_INFRA = 4_000           # .tf, Helm values, K8s manifests
_MAX_CONTEXT_CHARS  = 800_000


# ─────────────────────────────────────────────────────────────────────────────
# Skip lists — superset of SWE absorber
# ─────────────────────────────────────────────────────────────────────────────

_BUILTIN_SKIP_DIRS: frozenset[str] = frozenset({
    # from SWE absorber
    "node_modules", "vendor", ".git", "testdata", "dist", "build",
    ".next", "__pycache__", ".venv", "venv", ".tox", "target",
    "coverage", ".nyc_output", "storybook-static", ".parcel-cache",
    ".turbo", ".cache", "tmp", "temp",
    # infra-specific
    ".terraform",          # Terraform provider cache
    ".terraform.lock.hcl", # not a dir but guard anyway
    "charts",              # Helm dependency cache (if vendored)
    ".kube",               # kubeconfig
    "migrations",          # DB migration files (not infra config)
})

_ARTIFACT_CONTROL_DIRS: frozenset[str] = frozenset({
    # SWE toolkit artifacts
    "absorber", "clarificator", "enricher", "spectracker", "scaffolder",
    "planner", "executor", "debugger", "reporter", "judge", "patcher",
    "archivist", "spec", "output",
    # devops_mlops toolkit artifacts
    "infra_absorber", "consistency", "incident", "postmortem",
    "metrics", "infra_judge", "terraform_patch",
})

_BUILTIN_SKIP_PATTERNS: tuple[str, ...] = (
    "*.tfstate",            # Terraform state — binary JSON, can be MB
    "*.tfstate.backup",
    "*.tfplan",             # Binary plan file
    "*.lock.hcl",           # Dependency lock
    "*_test.go", "*.test.ts", "*.test.tsx", "*.test.js",
    "*.spec.ts", "*.spec.tsx", "*.spec.js",
    "test_*.py", "*_test.py",
)

# Extensions treated as infra config (larger per-file cap)
_INFRA_EXTENSIONS: frozenset[str] = frozenset({
    ".tf", ".hcl", ".yaml", ".yml", ".toml", ".ini",
    ".env", ".properties", ".cfg", ".conf",
})

# Extensions that are pure key-only (redact values, keep structure)
_KEY_ONLY_EXTENSIONS: frozenset[str] = frozenset({
    ".json", ".env", ".properties", ".cfg", ".conf", ".ini",
})


# ─────────────────────────────────────────────────────────────────────────────
# Ignore rules (same interface as SWE absorber)
# ─────────────────────────────────────────────────────────────────────────────

class InfraIgnoreRules:
    def __init__(self, rules_path: Path) -> None:
        self._forced_full:  list[str] = []
        self._forced_skip:  list[str] = []
        self._forced_key:   list[str] = []

        if not rules_path.exists():
            return
        track_read(rules_path)
        for raw in rules_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("full:"):
                self._forced_full.append(line[5:].strip())
            elif line.startswith("skip:"):
                self._forced_skip.append(line[5:].strip())
            elif line.startswith("key:"):
                self._forced_key.append(line[4:].strip())

    def mode_for(self, rel_path: str) -> str:
        import fnmatch
        for pat in self._forced_skip:
            if fnmatch.fnmatch(rel_path, pat):
                return "skip"
        for pat in self._forced_full:
            if fnmatch.fnmatch(rel_path, pat):
                return "full"
        for pat in self._forced_key:
            if fnmatch.fnmatch(rel_path, pat):
                return "key-only"
        return "auto"   # resolved per-file by fingerprint detector


# ─────────────────────────────────────────────────────────────────────────────
# Artifact root for devops_mlops toolkit
# ─────────────────────────────────────────────────────────────────────────────

def _devops_artifact_root() -> Path:
    """
    Artifact root lives OUTSIDE the repo to avoid .gitignore issues.
    Convention: ../outputs/devops_mlops/artifacts_<slug>/ relative to repo root.
    Override with DEVOPS_ARTIFACT_ROOT env var.
    """
    override = os.environ.get("DEVOPS_ARTIFACT_ROOT")
    if override:
        base = Path(override)
    else:
        base = _REPO_ROOT.parent / "outputs" / "devops_mlops"

    slug = os.environ.get("PIPELINE_PROJECT", "default")
    return base / f"artifacts_{slug}"


def _infra_absorber_dir() -> Path:
    return _devops_artifact_root() / "infra_absorber"


def _map_md_path() -> Path:
    return _infra_absorber_dir() / "infra_map.md"


def _map_json_path() -> Path:
    return _infra_absorber_dir() / "infra_map.json"


def _log_path() -> Path:
    return _infra_absorber_dir() / "infra_absorber_log.json"


def _cache_path() -> Path:
    return _infra_absorber_dir() / "cache" / "infra_snapshot.json"


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache() -> dict[str, Any]:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        track_read(p)
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    track_write(p)


def _file_hash(path: Path) -> str:
    import hashlib
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — File tree scan
# ─────────────────────────────────────────────────────────────────────────────

def _should_skip_dir(name: str) -> bool:
    return name in _BUILTIN_SKIP_DIRS or name.startswith(".")


def _should_skip_file(name: str) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(name, pat) for pat in _BUILTIN_SKIP_PATTERNS)


def _detect_infra_type(path: Path, raw_preview: str = "") -> str | None:
    """
    Returns infra file type string or None if not recognisable.
    Used to assign the right extraction mode.
    """
    name = path.name.lower()
    ext  = path.suffix.lower()

    # Dockerfile
    if name == "dockerfile" or name.endswith(".dockerfile"):
        return "dockerfile"
    # Jenkinsfile
    if name == "jenkinsfile" or name == "jenkinsfile.groovy":
        return "jenkinsfile"
    # HCL / Terraform
    if ext in (".tf", ".hcl"):
        return "terraform"

    if ext in (".yaml", ".yml"):
        # Fingerprint by content preview (first 800 chars)
        p = raw_preview[:800] if raw_preview else ""
        if not p and path.exists():
            try:
                p = path.read_text(errors="replace")[:800]
            except Exception:
                pass
        if "kind: Application" in p and ("repoURL" in p or "targetRevision" in p):
            return "argocd"
        if "apiVersion:" in p and "kind:" in p:
            return "k8s_manifest"
        if any(k in p for k in ("replicaCount:", "image:\n", "service:\n", "ingress:\n")):
            return "helm_values"
        if "Chart.yaml" in str(path) or name == "chart.yaml":
            return "helm_chart_yaml"
        if "- hosts:" in p or "- name:" in p and "tasks:" in p:
            return "ansible"
        return "yaml_generic"

    if ext == ".py":
        p = raw_preview[:600] if raw_preview else ""
        if not p and path.exists():
            try:
                p = path.read_text(errors="replace")[:600]
            except Exception:
                pass
        if "from airflow" in p or "DAG(" in p or "dag_id" in p:
            return "airflow_dag"
        return "python"

    if ext in _KEY_ONLY_EXTENSIONS:
        return "key_only"

    return None


def scan_infra_files(
    target: Path,
    rules:  InfraIgnoreRules,
) -> list[dict[str, Any]]:
    """Walk target and build file inventory with infra_type annotation."""
    inventory: list[dict[str, Any]] = []

    for root_dir, dirs, files in os.walk(target):
        root_path = Path(root_dir)
        dirs[:] = [d for d in dirs if not _should_skip_dir(d)]

        for fname in files:
            if _should_skip_file(fname):
                continue

            abs_path = root_path / fname
            try:
                rel_path = str(abs_path.relative_to(target))
            except ValueError:
                continue

            rule_mode = rules.mode_for(rel_path)
            if rule_mode == "skip":
                continue

            ext = abs_path.suffix.lower()
            try:
                size = abs_path.stat().st_size
            except OSError:
                continue

            if size <= 0 or size > _MAX_FILE_BYTES:
                continue

            infra_type = _detect_infra_type(abs_path)
            if infra_type is None:
                continue   # unrecognised file type — skip

            # Resolve mode
            if rule_mode == "full":
                mode = "full"
            elif rule_mode == "key-only":
                mode = "key-only"
            else:
                # auto: infra files get full extraction (smart extractors handle them)
                mode = "key-only" if infra_type == "key_only" else "full"

            inventory.append({
                "rel_path":   rel_path,
                "abs_path":   str(abs_path),
                "ext":        ext,
                "size":       size,
                "mode":       mode,
                "infra_type": infra_type,
            })

    inventory.sort(key=lambda x: x["rel_path"])
    return inventory


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Content extraction (infra-aware)
# ─────────────────────────────────────────────────────────────────────────────

def extract_infra_content(
    entry: dict[str, Any],
    cache: dict[str, Any],
    force: bool,
) -> tuple[str, bool]:
    abs_path   = Path(entry["abs_path"])
    rel_path   = entry["rel_path"]
    infra_type = entry["infra_type"]

    current_hash = _file_hash(abs_path)
    cached       = cache.get(rel_path, {})

    if (
        not force
        and cached.get("hash") == current_hash
        and cached.get("infra_type") == infra_type
        and "content" in cached
    ):
        return cached["content"], True

    content = _dispatch_extractor(abs_path, infra_type)

    cache[rel_path] = {
        "hash":       current_hash,
        "infra_type": infra_type,
        "content":    content,
        "size":       entry["size"],
    }
    return content, False


def _dispatch_extractor(path: Path, infra_type: str) -> str:
    extractors = {
        "terraform":     _extract_terraform,
        "helm_values":   _extract_helm_values,
        "helm_chart_yaml": _extract_helm_chart_yaml,
        "k8s_manifest":  _extract_k8s_manifest,
        "argocd":        _extract_argocd,
        "ansible":       _extract_ansible,
        "airflow_dag":   _extract_airflow_dag,
        "dockerfile":    _extract_dockerfile,
        "jenkinsfile":   _extract_jenkinsfile,
        "python":        _extract_python_signatures,
        "key_only":      _extract_key_only,
        "yaml_generic":  _extract_key_only,
    }
    fn = extractors.get(infra_type, _extract_key_only)
    try:
        return fn(path)
    except Exception as e:
        return f"[extraction error: {e}]"


def _read_raw(path: Path) -> str:
    try:
        track_read(path)
        return path.read_text(errors="replace")
    except Exception:
        return ""


# ── Terraform extractor ───────────────────────────────────────────────────────

def _extract_terraform(path: Path) -> str:
    raw = _read_raw(path)
    if not raw:
        return ""

    if _HCL2_AVAILABLE:
        return _extract_terraform_hcl2(raw, path)
    return _extract_terraform_regex(raw)


def _extract_terraform_hcl2(raw: str, path: Path) -> str:
    import io
    try:
        parsed = hcl2.load(io.StringIO(raw))
    except Exception:
        return _extract_terraform_regex(raw)

    lines: list[str] = [f"# {path.name}"]

    for block_type, blocks in parsed.items():
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            for resource_type, instances in block.items():
                if not isinstance(instances, dict):
                    continue
                for resource_name, body in instances.items():
                    lines.append(f"{block_type} \"{resource_type}\" \"{resource_name}\" {{")
                    if isinstance(body, dict):
                        for k, v in body.items():
                            if k in ("tags", "lifecycle", "timeouts"):
                                continue  # skip verbose blocks
                            v_str = json.dumps(v) if not isinstance(v, str) else v
                            lines.append(f"  {k} = {v_str[:120]}")
                    lines.append("}")
    return "\n".join(lines)


def _extract_terraform_regex(raw: str) -> str:
    """Regex fallback when python-hcl2 is not installed."""
    lines: list[str] = []
    # resource "type" "name"
    for m in re.finditer(
        r'^(resource|module|data|variable|output|locals|terraform|provider)\s+"?([^"\s{]+)"?\s*"?([^"\s{]*)"?\s*\{',
        raw, re.MULTILINE,
    ):
        block, type_, name = m.group(1), m.group(2), m.group(3)
        lines.append(f'{block} "{type_}" "{name}" {{...}}' if name else f'{block} "{type_}" {{...}}')

    # Pull specific high-value keys from body
    for key in ("source", "role_arn", "service_account_name", "namespace",
                 "cluster_name", "bucket", "iam_role_arn", "repository_url",
                 "image_uri", "handler", "runtime", "memory_size", "timeout"):
        for m in re.finditer(rf'^\s*{key}\s*=\s*"?([^"\n{{}}]+)"?', raw, re.MULTILINE):
            lines.append(f"  {key} = {m.group(1).strip()[:120]}")

    return "\n".join(lines) or raw[:_MAX_PER_FILE_INFRA]


# ── Helm values extractor ─────────────────────────────────────────────────────

_HELM_KEYS_OF_INTEREST = {
    "image", "replicaCount", "resources", "serviceAccount",
    "ingress", "service", "env", "extraEnv", "config",
    "persistence", "podAnnotations", "nodeSelector",
    "tolerations", "affinity", "autoscaling",
}

def _extract_helm_values(path: Path) -> str:
    raw = _read_raw(path)
    if not raw:
        return ""
    try:
        import yaml  # pyyaml
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            return raw[:_MAX_PER_FILE_INFRA]
        filtered = {k: v for k, v in data.items() if k in _HELM_KEYS_OF_INTEREST}
        return f"# helm_values: {path.name}\n" + yaml.dump(
            filtered, default_flow_style=False, allow_unicode=True,
        )[:_MAX_PER_FILE_INFRA]
    except Exception:
        # Regex extraction of key sections
        lines = [f"# helm_values: {path.name}"]
        for key in _HELM_KEYS_OF_INTEREST:
            for m in re.finditer(rf"^{key}:\s*\n((?:  .*\n)*)", raw, re.MULTILINE):
                lines.append(f"{key}:\n{m.group(1)[:400]}")
        return "\n".join(lines) or raw[:_MAX_PER_FILE_INFRA]


def _extract_helm_chart_yaml(path: Path) -> str:
    raw = _read_raw(path)
    try:
        import yaml
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            keep = {k: data[k] for k in ("name", "version", "appVersion", "description", "dependencies")
                    if k in data}
            return f"# Chart.yaml: {path.parent.name}\n" + yaml.dump(keep, allow_unicode=True)
    except Exception:
        pass
    return raw[:800]


# ── K8s manifest extractor ────────────────────────────────────────────────────

_K8S_KEYS = {"kind", "metadata", "spec"}
_METADATA_KEYS = {"name", "namespace", "labels", "annotations"}
_SPEC_KEYS = {"containers", "serviceAccountName", "nodeSelector",
              "tolerations", "replicas", "selector", "ports", "rules"}

def _extract_k8s_manifest(path: Path) -> str:
    raw = _read_raw(path)
    if not raw:
        return ""
    try:
        import yaml
        docs = list(yaml.safe_load_all(raw))
    except Exception:
        return raw[:_MAX_PER_FILE_INFRA]

    out_parts: list[str] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        kind = doc.get("kind", "?")
        meta = doc.get("metadata", {}) or {}
        name = meta.get("name", "?")
        ns   = meta.get("namespace", "")
        anns = meta.get("annotations", {}) or {}
        spec = doc.get("spec", {}) or {}

        header = f"kind: {kind}  name: {name}" + (f"  ns: {ns}" if ns else "")
        lines  = [header]

        # IRSA annotation
        irsa_key = "eks.amazonaws.com/role-arn"
        if irsa_key in anns:
            lines.append(f"  irsa: {anns[irsa_key]}")

        # SA name
        if "serviceAccountName" in spec:
            lines.append(f"  serviceAccountName: {spec['serviceAccountName']}")

        # Containers
        containers = spec.get("containers", []) or []
        for c in containers[:5]:
            if isinstance(c, dict):
                img = c.get("image", "")
                cname = c.get("name", "")
                if img:
                    lines.append(f"  container: {cname}  image: {img}")
                res = c.get("resources", {}) or {}
                if res:
                    lim = res.get("limits", {})
                    req = res.get("requests", {})
                    lines.append(f"    limits: {lim}  requests: {req}")

        out_parts.append("\n".join(lines))

    return "\n\n".join(out_parts)[:_MAX_PER_FILE_INFRA]


# ── ArgoCD Application extractor ──────────────────────────────────────────────

def _extract_argocd(path: Path) -> str:
    raw = _read_raw(path)
    try:
        import yaml
        doc = yaml.safe_load(raw)
        if not isinstance(doc, dict):
            return raw[:_MAX_PER_FILE_INFRA]
        meta = doc.get("metadata", {}) or {}
        spec = doc.get("spec", {}) or {}
        src  = spec.get("source", {}) or {}
        dest = spec.get("destination", {}) or {}
        helm = src.get("helm", {}) or {}
        sync = spec.get("syncPolicy", {}) or {}

        lines = [
            f"# ArgoCD Application: {meta.get('name', '?')}",
            f"  repoURL:         {src.get('repoURL', '?')}",
            f"  targetRevision:  {src.get('targetRevision', '?')}",
            f"  chart/path:      {src.get('chart') or src.get('path', '?')}",
            f"  destination:     {dest.get('namespace', '?')} / {dest.get('server', '?')}",
        ]
        if helm.get("valueFiles"):
            lines.append(f"  valueFiles:      {helm['valueFiles']}")
        if sync:
            lines.append(f"  syncPolicy:      automated={bool(sync.get('automated'))}")
        return "\n".join(lines)
    except Exception:
        return raw[:_MAX_PER_FILE_INFRA]


# ── Ansible extractor ─────────────────────────────────────────────────────────

def _extract_ansible(path: Path) -> str:
    raw = _read_raw(path)
    try:
        import yaml
        data = yaml.safe_load(raw)
        if not isinstance(data, list):
            return raw[:_MAX_PER_FILE_INFRA]
        lines = [f"# Ansible: {path.name}"]
        for play in data:
            if not isinstance(play, dict):
                continue
            hosts = play.get("hosts", "?")
            roles = play.get("roles", [])
            tasks = play.get("tasks", [])
            lines.append(f"play: hosts={hosts}  roles={roles}")
            for t in tasks[:10]:
                if isinstance(t, dict):
                    tname = t.get("name", "?")
                    lines.append(f"  task: {tname}")
        return "\n".join(lines)
    except Exception:
        return raw[:_MAX_PER_FILE_INFRA]


# ── Airflow DAG extractor ─────────────────────────────────────────────────────

def _extract_airflow_dag(path: Path) -> str:
    raw = _read_raw(path)
    lines = [f"# DAG: {path.name}"]

    # dag_id
    for m in re.finditer(r'dag_id\s*=\s*["\']([^"\']+)["\']', raw):
        lines.append(f"  dag_id: {m.group(1)}")

    # schedule
    for m in re.finditer(r'schedule(?:_interval)?\s*=\s*["\']?([^"\',()\n]+)["\']?', raw):
        val = m.group(1).strip()
        if val:
            lines.append(f"  schedule: {val}")
            break

    # operators used
    operators = set(re.findall(
        r"(PythonOperator|BashOperator|S3KeySensor|SqsSensor|"
        r"EmrServerlessStartJobOperator|AthenaOperator|"
        r"PostgresOperator|DummyOperator|EmptyOperator|"
        r"BranchPythonOperator|ShortCircuitOperator|[A-Z][A-Za-z]+Operator|"
        r"[A-Z][A-Za-z]+Sensor)",
        raw,
    ))
    if operators:
        lines.append(f"  operators: {', '.join(sorted(operators))}")

    # task IDs (task_id= strings)
    task_ids = re.findall(r'task_id\s*=\s*["\']([^"\']+)["\']', raw)
    if task_ids:
        lines.append(f"  tasks ({len(task_ids)}): {', '.join(task_ids[:15])}")

    # XCom keys
    xcom_keys = set(re.findall(r'key\s*=\s*["\']([^"\']+)["\']', raw))
    if xcom_keys:
        lines.append(f"  xcom_keys: {', '.join(sorted(xcom_keys)[:10])}")

    # Connections
    conns = set(re.findall(r'conn_id\s*=\s*["\']([^"\']+)["\']', raw))
    if conns:
        lines.append(f"  connections: {', '.join(sorted(conns))}")

    # S3 paths
    s3_paths = set(re.findall(r's3://([^"\')\s]+)', raw))
    if s3_paths:
        lines.append(f"  s3_paths: {', '.join(sorted(s3_paths)[:8])}")

    return "\n".join(lines)


# ── Dockerfile extractor ──────────────────────────────────────────────────────

def _extract_dockerfile(path: Path) -> str:
    raw = _read_raw(path)
    lines = [f"# Dockerfile: {path.parent.name}/{path.name}"]
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(("FROM ", "RUN ", "EXPOSE ", "ENTRYPOINT", "CMD",
                                 "COPY ", "ADD ", "ENV ", "ARG ", "WORKDIR ")):
            lines.append(f"  {stripped[:160]}")
    return "\n".join(lines)


# ── Jenkinsfile extractor ─────────────────────────────────────────────────────

def _extract_jenkinsfile(path: Path) -> str:
    raw = _read_raw(path)
    lines = [f"# Jenkinsfile: {path.parent.name}"]
    # Stages
    for m in re.finditer(r"stage\s*\(\s*['\"]([^'\"]+)['\"]", raw):
        lines.append(f"  stage: {m.group(1)}")
    # Agent
    for m in re.finditer(r"agent\s*\{([^}]+)\}", raw, re.DOTALL):
        lines.append(f"  agent: {m.group(1).strip()[:80]}")
    # Environment vars (keys only)
    for m in re.finditer(r"(\w+)\s*=\s*credentials\(['\"]([^'\"]+)['\"]\)", raw):
        lines.append(f"  env: {m.group(1)} = credentials({m.group(2)})")
    return "\n".join(lines)


# ── Python signature extractor (non-DAG) ─────────────────────────────────────

def _extract_python_signatures(path: Path) -> str:
    raw = _read_raw(path)
    lines = [f"# {path.name}"]
    import ast
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return raw[:_MAX_PER_FILE_CODE]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            lines.append(f"  def {node.name}({', '.join(args)})")
        elif isinstance(node, ast.ClassDef):
            lines.append(f"  class {node.name}")
    return "\n".join(lines)


# ── Key-only extractor (redact values, keep keys) ─────────────────────────────

def _extract_key_only(path: Path) -> str:
    raw = _read_raw(path)
    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        return _redact_yaml(raw)
    if ext == ".json":
        return _redact_json(raw)
    if ext == ".env":
        return _redact_env(raw)
    return _redact_generic(raw)


def _redact_yaml(raw: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines()[:200]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(line)
            continue
        if ":" in stripped and not stripped.startswith("-"):
            key_part = line.split(":", 1)[0]
            indent   = len(line) - len(line.lstrip())
            lines.append(" " * indent + key_part.strip() + ": <redacted>")
        else:
            lines.append(line)
    return "\n".join(lines)


def _redact_json(raw: str) -> str:
    try:
        data = json.loads(raw)
    except Exception:
        return raw[:800]

    def redact(obj: Any, depth: int = 0) -> Any:
        if depth > 4:
            return "..."
        if isinstance(obj, dict):
            return {k: redact(v, depth + 1) for k, v in obj.items()}
        if isinstance(obj, list):
            return [redact(i, depth + 1) for i in obj[:5]]
        return "<redacted>"

    return json.dumps(redact(data), indent=2)[:_MAX_PER_FILE_INFRA]


def _redact_env(raw: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(line)
        elif "=" in stripped:
            key = stripped.split("=", 1)[0]
            lines.append(f"{key}=<redacted>")
    return "\n".join(lines)


def _redact_generic(raw: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines()[:150]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(line)
        elif "=" in stripped or ":" in stripped:
            sep = "=" if "=" in stripped else ":"
            key = stripped.split(sep, 1)[0].strip()
            lines.append(f"  {key}: <redacted>")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Infra topology extractor (new, no SWE analog)
# ─────────────────────────────────────────────────────────────────────────────

def build_infra_topology(
    inventory: list[dict[str, Any]],
    cache:     dict[str, Any],
) -> dict[str, Any]:
    """
    Cross-file structural analysis. Extracts entities across all files
    and aggregates into a topology dict for LLM prompt + JSON artifact.

    Consumed by: config_consistency_checker.py (cross-file mismatch detection).
    """
    topo: dict[str, Any] = {
        "terraform_resources": [],
        "k8s_namespaces":      set(),
        "helm_releases":       [],
        "argocd_apps":         [],
        "dag_inventory":       [],
        "iam_roles":           [],
        "service_accounts":    [],
        "image_refs":          set(),
        "s3_buckets":          set(),
        "connections_used":    set(),
    }

    for entry in inventory:
        content    = cache.get(entry["rel_path"], {}).get("content", "")
        infra_type = entry["infra_type"]
        rel        = entry["rel_path"]

        if infra_type == "terraform":
            # Extract resource types and names
            for m in re.finditer(
                r'resource\s+"(aws_[^"]+)"\s+"([^"]+)"',
                content,
            ):
                topo["terraform_resources"].append({
                    "type": m.group(1), "name": m.group(2), "file": rel,
                })
            # IAM roles specifically
            for m in re.finditer(
                r'resource\s+"aws_iam_role"\s+"([^"]+)"',
                content,
            ):
                topo["iam_roles"].append({"name": m.group(1), "file": rel})
            # S3 buckets
            for m in re.finditer(r'bucket\s*=\s*"([^"]+)"', content):
                topo["s3_buckets"].add(m.group(1))

        elif infra_type == "k8s_manifest":
            # Namespaces
            for m in re.finditer(r"ns:\s*(\S+)", content):
                topo["k8s_namespaces"].add(m.group(1))
            # Service accounts with IRSA
            for m in re.finditer(
                r"kind: ServiceAccount.*?name:\s*(\S+).*?irsa:\s*(\S+)",
                content, re.DOTALL,
            ):
                topo["service_accounts"].append({
                    "name": m.group(1), "irsa": m.group(2), "file": rel,
                })
            # Image refs
            for m in re.finditer(r"image:\s*(\S+)", content):
                topo["image_refs"].add(m.group(1))

        elif infra_type == "helm_values":
            # SA annotations (IRSA)
            for m in re.finditer(
                r"eks\.amazonaws\.com/role-arn:\s*(\S+)",
                content,
            ):
                topo["service_accounts"].append({
                    "name": entry["rel_path"],
                    "irsa": m.group(1),
                    "file": rel,
                    "source": "helm_values",
                })
            # Image refs
            for m in re.finditer(r"repository:\s*(\S+)", content):
                topo["image_refs"].add(m.group(1))

        elif infra_type == "argocd":
            for m in re.finditer(
                r"ArgoCD Application:\s*(\S+).*?repoURL:\s*(\S+).*?targetRevision:\s*(\S+)",
                content, re.DOTALL,
            ):
                topo["argocd_apps"].append({
                    "name":            m.group(1),
                    "repoURL":         m.group(2),
                    "targetRevision":  m.group(3),
                    "file":            rel,
                })

        elif infra_type == "helm_chart_yaml":
            for m in re.finditer(r"name:\s*(\S+)", content):
                topo["helm_releases"].append({"chart": m.group(1), "file": rel})
                break

        elif infra_type == "airflow_dag":
            dag_id = ""
            for m in re.finditer(r"dag_id:\s*(\S+)", content):
                dag_id = m.group(1)
                break
            tasks = re.findall(r"tasks \(\d+\):\s*(.+)", content)
            conns = re.findall(r"connections:\s*(.+)", content)
            s3s   = re.findall(r"s3_paths:\s*(.+)", content)
            if dag_id:
                topo["dag_inventory"].append({
                    "dag_id": dag_id,
                    "tasks":  tasks[0].split(", ") if tasks else [],
                    "file":   rel,
                })
            if conns:
                for c in conns[0].split(", "):
                    topo["connections_used"].add(c.strip())

    # Convert sets to sorted lists for JSON serialisability
    topo["k8s_namespaces"]   = sorted(topo["k8s_namespaces"])
    topo["image_refs"]        = sorted(topo["image_refs"])
    topo["s3_buckets"]        = sorted(topo["s3_buckets"])
    topo["connections_used"]  = sorted(topo["connections_used"])

    return topo


def _topology_to_prompt(topo: dict[str, Any]) -> str:
    """Compact topology summary for LLM prompt injection."""
    lines: list[str] = ["--- INFRA TOPOLOGY ---"]

    resources = topo.get("terraform_resources", [])
    if resources:
        type_counts: dict[str, int] = {}
        for r in resources:
            type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
        lines.append(f"Terraform resources ({len(resources)}): "
                     + ", ".join(f"{t}×{n}" for t, n in sorted(type_counts.items())))

    iam = topo.get("iam_roles", [])
    if iam:
        lines.append(f"IAM roles: {', '.join(r['name'] for r in iam)}")

    sas = topo.get("service_accounts", [])
    if sas:
        for sa in sas[:10]:
            lines.append(f"ServiceAccount: {sa.get('name')}  IRSA: {sa.get('irsa', 'none')}")

    ns = topo.get("k8s_namespaces", [])
    if ns:
        lines.append(f"K8s namespaces: {', '.join(ns)}")

    apps = topo.get("argocd_apps", [])
    if apps:
        for a in apps:
            lines.append(f"ArgoCD: {a['name']}  rev={a.get('targetRevision')}")

    dags = topo.get("dag_inventory", [])
    if dags:
        lines.append(f"Airflow DAGs ({len(dags)}): {', '.join(d['dag_id'] for d in dags)}")

    imgs = topo.get("image_refs", [])
    if imgs:
        lines.append(f"Image refs: {', '.join(imgs[:12])}")

    s3 = topo.get("s3_buckets", [])
    if s3:
        lines.append(f"S3 buckets: {', '.join(s3[:10])}")

    conns = topo.get("connections_used", [])
    if conns:
        lines.append(f"Airflow connections: {', '.join(conns)}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Context builder (Phase 2 output → LLM input)
# ─────────────────────────────────────────────────────────────────────────────

def build_context(
    inventory: list[dict[str, Any]],
    cache:     dict[str, Any],
    force:     bool,
) -> tuple[str, int, int]:
    """Returns (context_str, cached_count, extracted_count)."""
    parts:          list[str] = []
    cached_count   = 0
    extracted_count = 0
    total_chars    = 0

    for entry in inventory:
        content, from_cache = extract_infra_content(entry, cache, force)
        if from_cache:
            cached_count += 1
        else:
            extracted_count += 1

        per_file_cap = (
            _MAX_PER_FILE_INFRA
            if entry["infra_type"] in (
                "terraform", "helm_values", "k8s_manifest",
                "argocd", "ansible", "airflow_dag",
            )
            else _MAX_PER_FILE_CODE
        )

        if len(content) > per_file_cap:
            content = content[:per_file_cap] + f"\n... ({len(content) - per_file_cap} chars truncated)"

        header = (
            f"### {entry['rel_path']}"
            f"  [{entry['infra_type']}  {entry['size']} bytes]\n"
        )
        block = header + content + "\n"

        if total_chars + len(block) > _MAX_CONTEXT_CHARS:
            parts.append(
                f"### ... {len(inventory) - len(parts)} more files truncated "
                f"(context limit {_MAX_CONTEXT_CHARS:,} chars)\n"
            )
            break

        parts.append(block)
        total_chars += len(block)

    return "\n".join(parts), cached_count, extracted_count


# ─────────────────────────────────────────────────────────────────────────────
# Git crawl (same logic as SWE absorber, reused directly)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_git_scope(scope: str) -> list[str]:
    if scope == "all":
        return []
    m = re.match(r"^(\d+)([mM]?)$", scope)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        if unit == "m":
            return [f"--since={n} months ago"]
        return [f"-n", str(n)]
    return []


def _git_log_stats(target: Path, scope: str) -> dict[str, Any] | None:
    git_args = _resolve_git_scope(scope)
    try:
        result = subprocess.run(
            ["git", "log", "--format=%H|%ae|%s", "--name-only"] + git_args,
            capture_output=True, text=True, cwd=target, timeout=60,
        )
        if result.returncode != 0:
            return None
    except Exception:
        return None

    commits: list[dict[str, Any]] = []
    file_counts: dict[str, int]   = {}
    file_authors: dict[str, set]  = {}
    authors: set[str]             = set()
    current: dict[str, Any] | None = None

    for line in result.stdout.splitlines():
        if "|" in line and len(line.split("|")) == 3:
            current = {"hash": line.split("|")[0], "author": line.split("|")[1]}
            authors.add(line.split("|")[1])
            commits.append(current)
        elif line.strip() and current:
            f = line.strip()
            file_counts[f]  = file_counts.get(f, 0) + 1
            file_authors.setdefault(f, set()).add(current["author"])

    hotspot_threshold_high   = 10
    hotspot_threshold_medium = 3
    high   = [f for f, c in file_counts.items() if c >= hotspot_threshold_high]
    medium = [
        f for f, c in file_counts.items()
        if hotspot_threshold_medium <= c < hotspot_threshold_high
    ]

    return {
        "total_commits":  len(commits),
        "unique_authors": len(authors),
        "authors":        sorted(authors),
        "hotspots": {
            "high":   sorted(high,   key=lambda f: -file_counts[f])[:20],
            "medium": sorted(medium, key=lambda f: -file_counts[f])[:20],
        },
        "file_change_counts": dict(sorted(
            file_counts.items(), key=lambda x: -x[1]
        )[:50]),
        "multi_author_files": [
            f for f, a in file_authors.items() if len(a) > 1
        ][:20],
    }


def _git_to_prompt(git: dict[str, Any]) -> str:
    if not git:
        return ""
    lines = [
        f"Total commits: {git['total_commits']}",
        f"Authors: {', '.join(git['authors'][:10])}",
    ]
    high = git.get("hotspots", {}).get("high", [])
    if high:
        lines.append(f"High-churn files (≥10 changes): {', '.join(high[:10])}")
    medium = git.get("hotspots", {}).get("medium", [])
    if medium:
        lines.append(f"Medium-churn files: {', '.join(medium[:10])}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Semantic compression → infra_map.md
# ─────────────────────────────────────────────────────────────────────────────

_MAP_SYSTEM = textwrap.dedent("""
    You are a senior DevOps / MLOps architect performing an infrastructure
    codebase intake. You will receive extracted content from Terraform, Helm
    charts, Kubernetes manifests, ArgoCD Applications, Airflow DAGs, Ansible
    playbooks, and other infra files.

    Your job is to produce a structured infra_map.md document.

    Output a SINGLE markdown document with these sections, no extra commentary:

    # Infra Map
    _Generated: {date} | infra_absorber v1_

    ## Project Overview
    [3-4 paragraph summary: what the system does, primary infra stack (EKS/AWS/etc.),
     data flow from edge to serving, and key patterns observed.]

    ## Infrastructure Topology
    [Structured inventory:
     - Terraform: which resource types, how many, key modules
     - Kubernetes: namespaces, workloads, service accounts, IRSA bindings
     - Helm releases: which charts, which versions
     - ArgoCD: which Applications, sync targets
     - Networking: ingress, Cloudflare tunnels, VPC details if present]

    ## Data Pipeline (DAG Graph)
    [For each Airflow DAG: name, schedule, task chain, data sources/sinks.
     Show dependency chain: edge device → SQS → S3 bronze → silver → gold → serving.]

    ## Config & Secrets
    [Services detected, env vars patterns, IRSA role bindings per workload,
     secret management approach detected.]

    ## Git/Blame
    [High-churn files, contributors, drift-prone areas.]

    ## Drift Risk
    [Files or patterns likely to cause config drift or synchronization failures.
     Specifically: IAM role names across Terraform vs Helm annotations,
     image tag references, SA names across manifests and values.yaml,
     Helm value keys that must match ArgoCD or K8s manifest fields.]

    ## Absorber Notes
    [Files that could not be parsed, ambiguities, recommended follow-ups.]
""").strip()


def call_llm_for_map(
    context:  str,
    target_name: str,
    topo:     dict[str, Any] | None = None,
    git:      dict[str, Any] | None = None,
) -> tuple[str, float]:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system   = _MAP_SYSTEM.replace("{date}", date_str)

    user_parts = [f"Infrastructure repo: {target_name}\n\nExtracted content:\n\n{context}"]

    if topo:
        user_parts.append(f"\n\n{_topology_to_prompt(topo)}")

    if git:
        user_parts.append(f"\n\n--- GIT/BLAME ---\n{_git_to_prompt(git)}")

    user       = "".join(user_parts)
    tokens_est = len(user) // 4
    print(f"[infra_absorber] LLM call: ~{tokens_est:,} input tokens")

    try:
        resp = call_model(
            ROLE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=_MAX_TOKENS_MAP,
            temperature=0.2,
        )
        choice    = resp.choices[0]
        usage     = getattr(resp, "usage", None)
        call_cost = 0.0
        if usage:
            pt        = getattr(usage, "prompt_tokens", 0) or 0
            ct        = getattr(usage, "completion_tokens", 0) or 0
            call_cost = record_usage(usage, model=get_model(ROLE), provider=get_provider(ROLE))
            print_call(__file__, pt, ct, call_cost)

        content       = choice.message.content
        finish_reason = getattr(choice, "finish_reason", "unknown")
        if finish_reason == "length":
            print("[infra_absorber][warn] LLM output truncated — consider --git-scope 3m")

        return content, call_cost

    except Exception as e:
        print(f"[infra_absorber][error] LLM call failed: {e}", file=sys.stderr)
        return _fallback_map(context, target_name, topo, git), 0.0


def _fallback_map(
    context:  str,
    target_name: str,
    topo:     dict[str, Any] | None = None,
    git:      dict[str, Any] | None = None,
) -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = [
        "# Infra Map",
        f"_Generated: {date_str} | infra_absorber v1 (fallback — LLM unavailable)_\n",
        f"## Project Overview\nTarget: {target_name}\n",
        "LLM call failed. Raw extraction below.\n",
        "## Raw Extraction\n",
        context[:8000],
    ]
    if topo:
        parts.append("\n## Infrastructure Topology\n")
        parts.append(_topology_to_prompt(topo)[:4000])
    if git:
        parts.append("\n## Git/Blame\n")
        parts.append(_git_to_prompt(git)[:2000])
    parts.append("\n## Absorber Notes\n- Fallback mode (no LLM). Re-run when model is available.")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Patch mode helpers (same pattern as SWE absorber)
# ─────────────────────────────────────────────────────────────────────────────

_GIT_SECTION_RE = re.compile(
    r"^## Git/Blame\s*\n.*?(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)


def run_patch_mode(target: Path, args: argparse.Namespace) -> None:
    """Re-run git crawl only. 0 LLM calls. Updates Git/Blame section in-place."""
    print("[infra_absorber] Mode: patch — updating git data (no LLM)")

    map_json = _map_json_path()
    if not map_json.exists():
        print("[infra_absorber][error] infra_map.json not found — run full absorb first.")
        sys.exit(2)

    track_read(map_json)
    existing = json.loads(map_json.read_text())

    new_git = _git_log_stats(target, args.git_scope)
    if new_git:
        print(f"  Commits: {new_git['total_commits']}  "
              f"high-churn: {len(new_git['hotspots']['high'])}")

    existing["git"] = new_git
    existing.setdefault("meta", {})["patched_at"] = datetime.now(timezone.utc).isoformat()
    existing["meta"]["stale_since"] = None

    map_json.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    track_write(map_json)

    md_path = _map_md_path()
    if md_path.exists():
        track_read(md_path)
        md_text = md_path.read_text(encoding="utf-8")
        new_git_section = "## Git/Blame\n" + _git_to_prompt(new_git or {}) + "\n\n"
        if _GIT_SECTION_RE.search(md_text):
            md_text = _GIT_SECTION_RE.sub(new_git_section, md_text)
        else:
            md_text = md_text.rstrip("\n") + "\n\n" + new_git_section
        md_path.write_text(md_text, encoding="utf-8")
        track_write(md_path)

    _append_log({"mode": "patch", "target": str(target), "git_scope": args.git_scope})
    print(f"  Patched: {map_json}")
    print(f"  Patched: {md_path}")
    print()
    print_artifact_summary()


# ─────────────────────────────────────────────────────────────────────────────
# Staleness check
# ─────────────────────────────────────────────────────────────────────────────

def check_stale(target: Path, threshold_days: int = 7) -> None:
    map_json = _map_json_path()
    print("[infra_absorber] Staleness check")

    if not map_json.exists():
        print("[infra_absorber][stale] No infra_map.json — never absorbed.")
        sys.exit(2)

    track_read(map_json)
    meta = json.loads(map_json.read_text()).get("meta", {})
    ts_str    = meta.get("patched_at") or meta.get("generated_at", "")
    generated = datetime.fromisoformat(ts_str) if ts_str else None

    if not generated:
        print("[infra_absorber][stale] Cannot parse timestamp.")
        sys.exit(2)

    now      = datetime.now(timezone.utc)
    days_old = (now - generated.replace(tzinfo=timezone.utc)).days
    stale    = days_old > threshold_days

    changed_count = 0
    try:
        since_iso = generated.strftime("%Y-%m-%dT%H:%M:%S")
        r = subprocess.run(
            ["git", "log", f"--since={since_iso}", "--name-only", "--format="],
            capture_output=True, text=True, cwd=target, timeout=15,
        )
        changed_count = len([l for l in r.stdout.splitlines() if l.strip()])
    except Exception:
        pass

    status = "STALE" if stale else "FRESH"
    print(f"  Status:        {status}")
    print(f"  Age:           {days_old}d  (threshold: {threshold_days}d)")
    print(f"  Files drifted: {changed_count}")

    # Write staleness signal into JSON
    try:
        data = json.loads(map_json.read_text())
        data["staleness"] = {
            "checked_at":    now.isoformat(),
            "stale":         stale,
            "days_old":      days_old,
            "changed_files": changed_count,
            "threshold_days": threshold_days,
        }
        map_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[infra_absorber][warn] Could not write staleness: {e}")

    sys.exit(2 if stale else 0)


# ─────────────────────────────────────────────────────────────────────────────
# Log writer
# ─────────────────────────────────────────────────────────────────────────────

def _append_log(entry: dict[str, Any]) -> None:
    log = _log_path()
    existing: list[dict[str, Any]] = []
    if log.exists():
        try:
            track_read(log)
            data     = json.loads(log.read_text())
            existing = data if isinstance(data, list) else data.get("entries", [])
        except Exception:
            pass

    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    existing.append(entry)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"entries": existing}, indent=2, ensure_ascii=False))
    track_write(log)


# ─────────────────────────────────────────────────────────────────────────────
# Git-hook installer
# ─────────────────────────────────────────────────────────────────────────────

_HOOK_SCRIPT = """\
#!/bin/sh
# infra_absorber post-commit hook
SCRIPT="$(git rev-parse --show-toplevel)/toolkits/devops_mlops/infra_absorber.py"
[ -f "$SCRIPT" ] && python "$SCRIPT" --changed-since HEAD~1 --mode patch
"""


def install_git_hook(target: Path) -> None:
    git_dir = target / ".git"
    if not git_dir.is_dir():
        print("[infra_absorber][error] Not a git repo.")
        sys.exit(1)
    hook = git_dir / "hooks" / "post-commit"
    marker = "# infra_absorber post-commit hook"
    if hook.exists() and marker in hook.read_text():
        print(f"[infra_absorber] Hook already installed: {hook}")
        return
    existing = hook.read_text() if hook.exists() else "#!/bin/sh\n"
    hook.write_text(existing.rstrip("\n") + "\n\n" + _HOOK_SCRIPT)
    hook.chmod(0o755)
    print(f"[infra_absorber] Hook installed: {hook}")


# ─────────────────────────────────────────────────────────────────────────────
# Changed-since cache invalidation
# ─────────────────────────────────────────────────────────────────────────────

def invalidate_changed_files(
    target:    Path,
    since_ref: str,
    cache:     dict[str, Any],
) -> list[str]:
    try:
        r = subprocess.run(
            ["git", "diff", "--name-only", since_ref, "HEAD"],
            capture_output=True, text=True, cwd=target, timeout=15,
        )
        changed = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    except Exception:
        return []

    invalidated = [f for f in changed if f in cache]
    for f in invalidated:
        del cache[f]

    if invalidated:
        print(f"[infra_absorber] Invalidated {len(invalidated)} cache entries "
              f"(changed since '{since_ref}')")
    return invalidated


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="infra_absorber.py",
        description="Absorb an infrastructure / MLOps codebase into the knowledge layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project",        default=None)
    p.add_argument("--target",         default=None)
    p.add_argument("--git-scope",      default="all")
    p.add_argument("--force",          action="store_true")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--mode",           default="full", choices=["full", "patch"])
    p.add_argument("--changed-since",  default=None, metavar="GIT_REF")
    p.add_argument("--install-hook",   action="store_true")
    p.add_argument("--check-stale",    action="store_true")
    p.add_argument("--stale-threshold",type=int, default=7, metavar="DAYS")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_infra_absorber(args: argparse.Namespace) -> None:
    if args.target:
        target = Path(args.target).expanduser().resolve()
    else:
        target = _REPO_ROOT

    print(f"  Target:    {target}")
    print(f"  Git scope: {args.git_scope}")
    print()

    # Phase 1 — scan
    rules_path = target / _IGNORED_FILE
    rules      = InfraIgnoreRules(rules_path)
    inventory  = scan_infra_files(target, rules)

    type_counts: dict[str, int] = {}
    for e in inventory:
        type_counts[e["infra_type"]] = type_counts.get(e["infra_type"], 0) + 1
    print(f"[infra_absorber] Phase 1 — {len(inventory)} files:")
    for itype, cnt in sorted(type_counts.items()):
        print(f"  {itype:<20} {cnt}")
    print()

    # Phase 2 — extract
    cache = _load_cache()
    if args.changed_since:
        invalidate_changed_files(target, args.changed_since, cache)

    context, cached_n, extracted_n = build_context(inventory, cache, args.force)
    print(f"[infra_absorber] Phase 2 — {extracted_n} extracted, {cached_n} from cache")
    print(f"  Context size: {len(context):,} chars")
    print()

    _save_cache(cache)

    # Infra topology
    topo = build_infra_topology(inventory, cache)
    print(f"[infra_absorber] Topology — "
          f"{len(topo['terraform_resources'])} tf resources, "
          f"{len(topo['dag_inventory'])} DAGs, "
          f"{len(topo['service_accounts'])} SAs")
    print()

    # Phase 4 — git crawl
    print(f"[infra_absorber] Phase 4 — git crawl (scope: {args.git_scope})")
    git_data = _git_log_stats(target, args.git_scope)
    if git_data:
        print(f"  {git_data['total_commits']} commits  "
              f"{git_data['unique_authors']} authors  "
              f"{len(git_data['hotspots']['high'])} high-churn")
    print()

    if args.dry_run:
        print("[infra_absorber] DRY RUN — skipping LLM and artifact writes")
        print(f"  Context: {len(context):,} chars  "
              f"~{len(context)//4:,} tokens")
        return

    # Phase 3 — LLM
    print("[infra_absorber] Phase 3 — semantic compression (LLM)…")
    t0 = time.time()
    map_text, call_cost = call_llm_for_map(
        context     = context,
        target_name = target.name,
        topo        = topo,
        git         = git_data,
    )
    print(f"  Elapsed: {time.time()-t0:.1f}s  cost: ${call_cost:.4f}")
    print()

    # Phase 5 — write artifacts
    _infra_absorber_dir().mkdir(parents=True, exist_ok=True)

    # infra_map.md
    md_path = _map_md_path()
    md_with_header = apply_md_header(
        content = map_text,
        path    = md_path,
        owner   = "infra_absorber.py",
    )
    md_path.write_text(md_with_header, encoding="utf-8")
    track_write(md_path)

    # infra_map.json
    map_json_data: dict[str, Any] = {
        "meta": {
            "generated_at":     datetime.now(timezone.utc).isoformat(),
            "target":           str(target),
            "git_scope":        args.git_scope,
            "absorber_version": 1,
            "run_mode":         "full",
            "changed_since":    args.changed_since,
            "total_files":      len(inventory),
            "cached_files":     cached_n,
            "extracted_files":  extracted_n,
            "map_md":           str(md_path),
            "map_size_bytes":   len(map_text.encode()),
            "cost":             round(call_cost, 6),
            "stale_since":      None,
            "file_types":       type_counts,
        },
        "infra_topology": topo,
        "git":            git_data,
    }
    json_path = _map_json_path()
    json_path.write_text(json.dumps(map_json_data, indent=2, ensure_ascii=False))
    track_write(json_path)

    # log
    _append_log({
        "mode":           "full",
        "target":         str(target),
        "git_scope":      args.git_scope,
        "total_files":    len(inventory),
        "cached_files":   cached_n,
        "extracted_files": extracted_n,
        "cost":           round(call_cost, 6),
        "file_types":     type_counts,
    })

    print(f"  Written: {md_path}")
    print(f"  Written: {json_path}")
    print(f"  Appended: {_log_path()}")
    print()
    print_artifact_summary()
    print()
    print_cost_summary()


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.project:
        os.environ["PIPELINE_PROJECT"] = args.project
    if not os.environ.get("PIPELINE_PROJECT"):
        parser.error("Use --project <name> or export PIPELINE_PROJECT=<name>.")

    target = Path(args.target).expanduser().resolve() if args.target else _REPO_ROOT

    print("=" * 60)
    print("  INFRA ABSORBER")
    print("=" * 60)
    print()

    try:
        if args.install_hook:
            install_git_hook(target)
            return
        if args.check_stale:
            check_stale(target, threshold_days=args.stale_threshold)
            return
        if args.mode == "patch":
            run_patch_mode(target, args)
            return
        run_infra_absorber(args)
    except KeyboardInterrupt:
        print("\n[infra_absorber] Interrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[infra_absorber][fatal] {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
