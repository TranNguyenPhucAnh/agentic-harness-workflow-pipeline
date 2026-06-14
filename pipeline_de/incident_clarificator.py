"""
toolkits/devops_mlops/incident_clarificator.py
===============================================
Interactive incident diagnosis assistant for DevOps/MLOps stack.

Flow:
  Phase 0 — Intake:    user paste error log / drag-drop file
  Phase 1 — Analyze:   LLM classify taxonomy + T1/T2/T3 questions
  Phase 2 — Q&A loop:  ask T1 (blocking) → T2 (narrowing) → T3 (context)
  Phase 3 — Diagnose:  LLM synthesize into structured diagnosis
  Phase 4 — Save:      write incident_session.json (overwrite) + incident_log.md (append)
             Ask:      save to postmortem KB? [y/n]

Each tier drives different questions per taxonomy:
  T1 — "Need immediately to diagnose" — full log, kubectl describe, IAM policy
  T2 — "Need to narrow down"          — Helm values, Terraform resource block, env vars
  T3 — "Context, timeline, changes"   — recent terraform apply, image tag change, etc.

Postmortem injection:
  At session start, loads relevant entries from postmortem_archivist KB
  and injects into system prompt as historical context. This gives the LLM
  memory of past incidents with the same taxonomy and failed attempts.

Artifacts:
  incident_clarificator/incident_session.json   (short-term overwrite)
  incident_clarificator/incident_log.md         (long-term append)

Usage:
  python incident_clarificator.py --project iot-mlops
  python incident_clarificator.py --project iot-mlops --text "pod OOMKilled exit 137"
  python incident_clarificator.py --project iot-mlops --auto   # skip Q&A, direct diagnosis
  PIPELINE_PROJECT=iot-mlops python incident_clarificator.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLKIT_DIR = Path(__file__).parent
_REPO_ROOT   = _TOOLKIT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from modules.artifact_tracking import (                                   # noqa: E402
    track_read, track_write,
    print_summary as print_artifact_summary,
)
from modules.cost import print_summary as print_cost_summary              # noqa: E402
from modules.call_llm import call_llm                                     # noqa: E402
from modules.drag_and_drop import gather_text_file_bundle                 # noqa: E402
from modules.post_interactive import prompt_next_step                     # noqa: E402
from artifacts.models import get_model                                    # noqa: E402

# Postmortem KB integration
try:
    from toolkits.devops_mlops.postmortem_archivist import (
        get_relevant_context, TAXONOMIES,
    )
    _POSTMORTEM_AVAILABLE = True
except ImportError:
    try:
        _HERE = Path(__file__).parent
        sys.path.insert(0, str(_HERE))
        from postmortem_archivist import get_relevant_context, TAXONOMIES  # type: ignore
        _POSTMORTEM_AVAILABLE = True
    except ImportError:
        _POSTMORTEM_AVAILABLE = False
        TAXONOMIES = {
            "auth_iam":           "IAM, IRSA, OIDC, ServiceAccount, AssumeRole, permissions",
            "resource_constraint":"OOMKilled, CPU throttle, memory limit, node capacity",
            "config_drift":       "values.yaml mismatch, annotation wrong, indent error, key path wrong",
            "image_registry":     "ECR auth, image pull, Image Updater, writeBackMethod, tag override",
            "data_pipeline":      "SQS, S3, Airflow DAG, XCom, SqsSensor, medallion architecture",
            "infra_connectivity": "connection refused, DNS, port, network, VPC, subnet, security group",
            "k8s_orchestration":  "ArgoCD sync, Helm chart, PVC, StorageClass, EBS CSI, finalizers",
            "ci_cd":              "Jenkins, JNLP, DinD, pipeline, gitSync, image build",
            "terraform":          "provider, module, state, destroy, dependency cycle, pagination",
            "observability":      "Grafana, Prometheus, Loki, Cloudflare Tunnel, node_exporter, Athena",
            "ml_model":           "training, inference, f1, precision, threshold, MLflow, synthetic anomaly",
            "other":              "does not fit any specific category",
        }
        def get_relevant_context(*args, **kwargs) -> str:  # type: ignore
            return ""

ROLE = "incident_clarificator"


# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths
# ─────────────────────────────────────────────────────────────────────────────

def _devops_artifact_root() -> Path:
    override = os.environ.get("DEVOPS_ARTIFACT_ROOT")
    if override:
        base = Path(override)
    else:
        base = _REPO_ROOT.parent / "outputs" / "devops_mlops"
    slug = os.environ.get("PIPELINE_PROJECT", "default")
    return base / f"artifacts_{slug}"


def _clarificator_dir() -> Path:
    return _devops_artifact_root() / "incident_clarificator"


def _session_path() -> Path:
    return _clarificator_dir() / "incident_session.json"


def _log_path() -> Path:
    return _clarificator_dir() / "incident_log.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_display() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ─────────────────────────────────────────────────────────────────────────────
# Tier question templates per taxonomy
# ─────────────────────────────────────────────────────────────────────────────
# Each taxonomy has a list of questions per tier.
# T1 = must have to diagnose; T2 = narrows down; T3 = context/timeline.
# The LLM will SELECT which questions are relevant from these lists
# based on the specific error provided.

_TIER_QUESTIONS: dict[str, dict[str, list[str]]] = {
    "auth_iam": {
        "T1": [
            "Paste the full error message including the AWS request ID if present.",
            "Run: `kubectl describe pod <pod-name> -n <namespace>` and paste the Events section.",
            "Run: `kubectl get serviceaccount <sa-name> -n <namespace> -o yaml` and paste the annotations block.",
            "Run: `aws iam get-role --role-name <role-name>` and paste the AssumeRolePolicyDocument.",
        ],
        "T2": [
            "Paste the IRSA module block from your Terraform (the `module \"*_irsa\"` block).",
            "Paste the serviceAccount section from your Helm values.yaml for the affected component.",
            "What is the OIDC issuer URL of your EKS cluster? (`aws eks describe-cluster --name <name> --query cluster.identity.oidc.issuer`)",
            "Does the IAM role trust policy `StringEquals` condition match the exact namespace:serviceaccount of the pod?",
        ],
        "T3": [
            "Did you recently apply Terraform changes to IAM or EKS modules?",
            "Was this working before? If so, what changed recently?",
            "Is this a new EKS cluster or an existing one?",
        ],
    },
    "resource_constraint": {
        "T1": [
            "Paste the full pod description: `kubectl describe pod <pod-name> -n <namespace>`",
            "What is the exit code? (OOMKilled = 137, CPU throttle has no exit code)",
            "Run: `kubectl top node` and `kubectl top pod -n <namespace>` and paste output.",
            "What are the current resource requests/limits in your values.yaml for this pod?",
        ],
        "T2": [
            "How many other pods/daemonsets are running on the same node? (`kubectl get pods -A -o wide | grep <node-name>`)",
            "Is this a t3.medium or similar small instance? What is the allocatable memory? (`kubectl describe node <node-name> | grep -A5 Allocatable`)",
            "Is the OOM happening during startup, steady-state, or during a specific task (e.g., model training)?",
        ],
        "T3": [
            "Did you recently add new pods or daemonsets to the cluster?",
            "Did you change the instance type or node group size recently?",
        ],
    },
    "config_drift": {
        "T1": [
            "Paste the exact error message from the pod logs or kubectl events.",
            "Paste the relevant section of your values.yaml (the block containing the misconfigured key).",
            "What is the key path you're trying to set? (e.g., `airflow.airflow.image.repository`)",
        ],
        "T2": [
            "Paste your Chart.yaml dependencies block.",
            "Is this a subchart/dependency (nested) or a top-level chart value?",
            "Run: `helm template <release> <chart> -f values.yaml | grep <key>` — does the rendered output match what you expect?",
            "For ArgoCD: paste your Application manifest's `helm.valueFiles` and `helm.parameters` sections.",
        ],
        "T3": [
            "Did you recently upgrade the Helm chart version?",
            "Did the chart maintainer change key paths between versions? (Check the chart CHANGELOG)",
            "Is ArgoCD detecting the change as OutOfSync? What does the diff show?",
        ],
    },
    "image_registry": {
        "T1": [
            "Paste the full error from `kubectl describe pod <pod-name>` — Events section.",
            "What is the exact image reference being pulled? (`kubectl get pod <name> -o jsonpath='{.spec.containers[*].image}'`)",
            "Can the node authenticate to ECR? (`aws ecr get-login-password | docker login ...` from the node)",
        ],
        "T2": [
            "Paste the Image Updater annotation block from your ArgoCD Application manifest.",
            "What is the `image-list` annotation value? Does it match the actual container name?",
            "What is the `writeBackMethod`? (git or argocd)",
            "Run: `kubectl get app <app> -n argocd -o jsonpath='{.status.summary.images}'` — does your custom image appear?",
            "What is the `helm.values` key path used in the image-list annotation? (community chart vs official chart key differ)",
        ],
        "T3": [
            "What version of ArgoCD Image Updater are you running?",
            "Did you recently change `writeBackMethod`?",
            "Is there a `.argocd-source-<app>.yaml` file being committed to your repo?",
        ],
    },
    "data_pipeline": {
        "T1": [
            "Paste the full Airflow task log for the failing task.",
            "What is the DAG ID and task ID?",
            "What operator is being used? (SqsSensor, S3KeySensor, PythonOperator, etc.)",
        ],
        "T2": [
            "Paste the Airflow Connection config for the AWS connection (from UI or `airflow connections get aws_default`).",
            "If SqsSensor: what are the `message_filtering` and `message_filtering_match_values` settings?",
            "If XCom: what type is being pushed and pulled? Paste the push/pull code.",
            "What version of `apache-airflow-providers-amazon` is installed?",
            "If S3: what is the exact S3 path pattern, including partition format?",
        ],
        "T3": [
            "Is the DAG reading from Bronze, Silver, or Gold layer?",
            "Did you recently change S3 path structure or partition format?",
            "Is MLflow accessible from the Airflow worker pod? (`kubectl exec -it -n airflow <scheduler-pod> -- curl http://mlflow.mlflow.svc.cluster.local:80`)",
        ],
    },
    "infra_connectivity": {
        "T1": [
            "Paste the full connection error including IP/hostname and port.",
            "What is trying to connect to what? (source pod/service → destination)",
            "Run: `kubectl exec -it <source-pod> -n <ns> -- curl -v <destination>:<port>` and paste output.",
        ],
        "T2": [
            "Paste the Security Group rules for both source and destination.",
            "Is the destination in a private subnet? Does the source have access to that subnet?",
            "For Cloudflare Tunnel: paste the `network_mode` setting of the cloudflared container.",
            "For K8s Service: what are `port` and `targetPort`? (`kubectl get svc <name> -n <ns> -o yaml`)",
        ],
        "T3": [
            "Did you recently change VPC/subnet/security group settings via Terraform?",
            "Is this cross-network (e.g., Pi 3 → EKS)? What tunnel/proxy is in between?",
        ],
    },
    "k8s_orchestration": {
        "T1": [
            "Paste: `kubectl describe <resource-type> <name> -n <namespace>`",
            "What is the exact ArgoCD sync status? (Synced/OutOfSync/Degraded/Unknown)",
            "Paste the ArgoCD Application Events section.",
        ],
        "T2": [
            "Is there a PVC involved? `kubectl get pvc -n <namespace>` — what is the STATUS?",
            "Are there Finalizers blocking deletion? `kubectl get <resource> -o jsonpath='{.metadata.finalizers}'`",
            "Paste your StorageClass manifest if PVC is stuck.",
            "Is EBS CSI driver installed? `kubectl get pods -n kube-system | grep ebs-csi`",
        ],
        "T3": [
            "Did you run `helm uninstall` or `kubectl delete` recently?",
            "Was the cluster destroyed and recreated? (orphan EBS volumes likely)",
            "Did you upgrade ArgoCD or Kubernetes version recently?",
        ],
    },
    "ci_cd": {
        "T1": [
            "Paste the full Jenkins build log for the failing stage.",
            "What stage is failing? (CI test, Docker build, Docker push)",
            "Paste the pod template YAML from your Jenkinsfile (the `yaml:` block).",
        ],
        "T2": [
            "Paste the `jenkinsUrl` and `jenkinsTunnel` values from your JCasC config.",
            "Is the Jenkins agent pod starting? `kubectl get pods -n jenkins`",
            "Does the Jenkins service account have ECR push permissions? `aws iam get-role --role-name <jenkins-irsa-role>`",
            "Is the SA name in Jenkinsfile pod template matching the SA created by Helm?",
        ],
        "T3": [
            "Did you recently change the Jenkinsfile or JCasC config?",
            "Did you recently upgrade the Jenkins Helm chart version?",
            "Is this a DNS resolution failure for the controller URL?",
        ],
    },
    "terraform": {
        "T1": [
            "Paste the full `terraform plan` or `terraform apply` error.",
            "What resource type is failing? (module, provider, resource)",
            "Run: `terraform state list | grep <resource-name>` — is the resource in state?",
        ],
        "T2": [
            "Is this a provider dependency cycle? Paste your `provider.tf` block.",
            "Is this a community provider (e.g., `gavinbunney/kubectl`)? Is `source` declared in both root `versions.tf` and child module `versions.tf`?",
            "Is the EKS endpoint resolved before the Helm/kubectl provider tries to use it?",
            "Run: `terraform state show <resource>` — what does the current state show?",
        ],
        "T3": [
            "Did you run `terraform init -upgrade` recently?",
            "Are you destroying? Did you remove ArgoCD apps before running destroy?",
            "Did you recently add a new module or provider?",
        ],
    },
    "observability": {
        "T1": [
            "Paste the full error from the relevant pod logs (Grafana/Prometheus/Loki/cloudflared).",
            "What component is failing? (Prometheus scrape, Loki push, Grafana datasource query)",
        ],
        "T2": [
            "For Cloudflare Tunnel: what is the `network_mode` of the cloudflared container? Is it `host`?",
            "For Grafana Athena: paste the Athena datasource config. What is the workgroup and output bucket?",
            "For Prometheus scrape target: paste the `additionalScrapeConfigs` section from KPS values.",
            "For Loki: is Promtail or Fluent Bit configured with the correct Loki push URL?",
        ],
        "T3": [
            "Did you recently change Cloudflare Tunnel config or token?",
            "For Athena: did you run `MSCK REPAIR TABLE` or update Partition Projection after adding new S3 paths?",
            "What protocol is cloudflared using? (QUIC may fail on some networks — try `--protocol http2`)",
        ],
    },
    "ml_model": {
        "T1": [
            "What metric values are you seeing? (f1, precision, recall, loss)",
            "Paste the MLflow run URL or experiment ID.",
            "What does `mlflow.list_registered_models()` return?",
        ],
        "T2": [
            "Is data too flat? Check the variance of your target variable in the training set.",
            "What threshold is used for model registration? What is the actual metric value?",
            "Is the model reading from Silver layer or directly from Bronze?",
            "Is the Airflow connection to MLflow correct? (`kubectl exec` into scheduler and test: `import mlflow; mlflow.set_tracking_uri(...)`)",
        ],
        "T3": [
            "Was the sensor moved recently? Is it in a location with more environmental variation?",
            "Have you considered synthetic anomaly injection (Data Augmentation)?",
            "Is the inference DAG using a registered model or a local artifact path?",
        ],
    },
    "other": {
        "T1": [
            "Paste the full error message and stack trace.",
            "What component produced this error? (pod name, service, script)",
            "What were you trying to do when this happened?",
        ],
        "T2": [
            "Paste relevant config files (values.yaml, .tf, Dockerfile, DAG file).",
            "What does the pod/container log show? (`kubectl logs <pod> -n <namespace>`)",
        ],
        "T3": [
            "What changed recently before this started happening?",
            "Has this worked before? When did it last work?",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

def _build_analyze_system(postmortem_ctx: str) -> str:
    taxonomy_list = "\n".join(f"  {k}: {v}" for k, v in TAXONOMIES.items())
    pm_section = ""
    if postmortem_ctx:
        pm_section = f"""
## Historical postmortems from this project
{postmortem_ctx}

Use these to inform your analysis — especially the "Did NOT work" items.
"""
    return f"""\
You are a senior DevOps/MLOps engineer diagnosing infrastructure and pipeline incidents.

Stack: EKS, Terraform, ArgoCD, Helm, Airflow, MLflow, Jenkins, Grafana, AWS (ECR, SQS, S3, IAM).
{pm_section}
## Your task
Given an error description or log snippet, classify the incident and identify
what information is needed to diagnose it.

## Taxonomy
{taxonomy_list}

## Tier definitions
T1 — Blocking: must have this information before any diagnosis is possible.
T2 — Narrowing: helps distinguish between 2-3 root cause hypotheses.
T3 — Context: timeline, recent changes, environment details.

Return ONLY valid JSON (no markdown fences, no prose):
{{
  "taxonomy":          "<one taxonomy key>",
  "initial_hypothesis": "<1-2 sentences: most likely root cause based on the error alone>",
  "confidence":        "low" | "medium" | "high",
  "t1_questions":      ["<question>", ...],   // 1-3 questions, from the taxonomy question bank
  "t2_questions":      ["<question>", ...],   // 1-3 questions
  "t3_questions":      ["<question>", ...],   // 1-2 questions
  "skip_to_diagnosis": false                  // true only if error + context is already conclusive
}}

RULES:
- t1_questions must be filled — never empty.
- Select questions from the taxonomy question bank that are most relevant to this specific error.
- Adapt the question text to reference the actual component names if visible in the error.
- If skip_to_diagnosis is true, still populate all question fields (for the record).
"""


_SYSTEM_DIAGNOSE = """\
You are a senior DevOps/MLOps engineer delivering a final incident diagnosis.

Given:
- The original error/symptom
- Answers collected through T1/T2/T3 Q&A rounds
- Historical postmortems if available

Return ONLY valid JSON (no markdown fences):
{
  "taxonomy":       "<taxonomy key>",
  "symptom":        "<what the engineer observed>",
  "causal_chain":   ["<step 1>", "<step 2>", "..."],
  "root_cause":     "<single sentence>",
  "resolution":     "<specific fix with exact commands or config changes>",
  "failed_attempts": [],
  "files_affected": ["<terraform/modules/iam/main.tf>", "..."],
  "tags":           ["<eks>", "<irsa>", "..."],
  "preventable_by": "<config_consistency_checker | resource_monitor | incident_clarificator | pre_deploy_validation | manual_review | documentation | null>",
  "confidence":     "low" | "medium" | "high",
  "follow_up":      "<optional: what to watch for after applying the fix>"
}

RULES:
- resolution must include specific commands or config snippets, not vague advice.
- causal_chain must be ordered: what caused what.
- If you cannot determine root cause with high confidence, set confidence to low/medium
  and explain in follow_up what additional info would help.
- Output ONLY the JSON object.
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(system: str, user: str, max_tokens: int = 4096) -> str:
    raw, _ = call_llm(
        ROLE, system, user,
        max_tokens=max_tokens,
        caller_file=__file__,
        label=f"[clarificator] {get_model(ROLE)}",
    )
    return raw


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", text)
    text = re.sub(r"\n?\s*```\s*$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Brace-counting fallback
    depth, start = 0, None
    candidates: list[str] = []
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if esc:
            esc = False; continue
        if ch == "\\" and in_str:
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None

    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(f"No valid JSON in response ({len(raw)} chars)", raw, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Flush stdin helper
# ─────────────────────────────────────────────────────────────────────────────

def _flush_stdin() -> None:
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def _read_input_file(path: Path) -> str:
    track_read(path)
    return path.read_text(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — Intake
# ─────────────────────────────────────────────────────────────────────────────

def phase0_intake(args: argparse.Namespace) -> str:
    """Gather initial error description from user."""
    print("[clarificator] Phase 0 — Intake")
    print()

    if args.text:
        print(f"  Input: {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
        return args.text

    if args.no_interactive or not sys.stdin.isatty():
        print("[clarificator] No input provided and non-interactive mode.")
        return ""

    try:
        bundle = gather_text_file_bundle(
            cli_text=None,
            cli_files=[],
            read_file_fn=_read_input_file,
            prompt_title="Error / incident description",
            prompt_body=(
                "Paste your error message, log snippet, or describe the incident.\n"
                "You can also drag-drop a log file, kubectl output, or config file.\n"
                "Press Enter twice to submit."
            ),
            attachment_prompt="Attach supporting file(s) if relevant",
            default_attachment_only_prompt="Analyze the attached file(s) for errors.",
            allow_interactive=True,
            ask_for_attachments_after_text=True,
        )
        return bundle.text.strip()
    except (RuntimeError, EOFError, KeyboardInterrupt):
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Analyze
# ─────────────────────────────────────────────────────────────────────────────

def phase1_analyze(
    error_text: str,
    postmortem_ctx: str,
) -> dict[str, Any]:
    """LLM call 1: classify taxonomy + generate tier questions."""
    print("[clarificator] Phase 1 — Analyzing error …")

    system = _build_analyze_system(postmortem_ctx)
    user   = f"## Error / Symptom\n\n{error_text}"

    raw    = _call_llm(system, user)
    result = _parse_json(raw)

    taxonomy   = result.get("taxonomy", "other")
    hypothesis = result.get("initial_hypothesis", "")
    confidence = result.get("confidence", "low")

    print(f"  Taxonomy:    {taxonomy}")
    print(f"  Hypothesis:  {hypothesis}")
    print(f"  Confidence:  {confidence}")
    print()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Q&A loop
# ─────────────────────────────────────────────────────────────────────────────

def _ask_tier(
    tier_label:  str,
    questions:   list[str],
    answers_acc: list[dict[str, str]],
    auto:        bool,
) -> bool:
    """
    Present tier questions to user, collect answers.
    Returns False if user wants to skip remaining questions (types 'skip' or 'done').
    """
    if not questions:
        return True

    print(f"  ── {tier_label} ──────────────────────────────────────")
    print()

    for i, q in enumerate(questions, 1):
        print(f"  Q{i}: {q}")
        print()

        if auto:
            print("  [auto mode — skipping answer]")
            answers_acc.append({"tier": tier_label, "question": q, "answer": "[auto-skipped]"})
            continue

        if not sys.stdin.isatty():
            answers_acc.append({"tier": tier_label, "question": q, "answer": "[non-interactive]"})
            continue

        _flush_stdin()
        print("  Paste answer (Enter twice to submit, 'skip' to skip this question,")
        print("  'done' to stop Q&A and proceed to diagnosis):")
        print()

        lines: list[str] = []
        blank_count = 0
        try:
            while blank_count < 2:
                line = input("  > ")
                stripped = line.strip().lower()
                if stripped == "done":
                    answers_acc.append({"tier": tier_label, "question": q, "answer": "\n".join(lines)})
                    return False
                if stripped == "skip":
                    answers_acc.append({"tier": tier_label, "question": q, "answer": "[skipped]"})
                    break
                if line.strip() == "":
                    blank_count += 1
                else:
                    blank_count = 0
                    lines.append(line)
            else:
                answers_acc.append({"tier": tier_label, "question": q, "answer": "\n".join(lines)})
        except (EOFError, KeyboardInterrupt):
            print()
            return False

        print()

    return True


def phase2_qa(
    analysis:   dict[str, Any],
    auto:       bool,
    max_rounds: int,
) -> list[dict[str, str]]:
    """
    Run T1 → T2 → T3 Q&A loop.

    max_rounds controls how many tiers to ask (1=T1 only, 2=T1+T2, 3=all).
    Default is 3 (all tiers). Use --max-rounds 1 for quick diagnosis when
    you only want to supply the blocking information.

    Returns accumulated answers list.
    """
    print("[clarificator] Phase 2 — Q&A")
    print()

    if analysis.get("skip_to_diagnosis"):
        print("  Sufficient context already — skipping Q&A.")
        return []

    taxonomy   = analysis.get("taxonomy", "other")
    t1_qs      = analysis.get("t1_questions", [])
    t2_qs      = analysis.get("t2_questions", [])
    t3_qs      = analysis.get("t3_questions", [])

    # Fallback to question bank if LLM returned empty
    bank = _TIER_QUESTIONS.get(taxonomy, _TIER_QUESTIONS["other"])
    if not t1_qs:
        t1_qs = bank["T1"][:2]
    if not t2_qs:
        t2_qs = bank["T2"][:2]
    if not t3_qs:
        t3_qs = bank["T3"][:2]

    answers: list[dict[str, str]] = []
    rounds = 0

    for tier_label, questions in [("T1 — Need immediately", t1_qs),
                                   ("T2 — Narrowing down", t2_qs),
                                   ("T3 — Context", t3_qs)]:
        if rounds >= max_rounds:
            break
        cont = _ask_tier(tier_label, questions, answers, auto)
        rounds += 1
        if not cont:
            print("  Stopping Q&A — proceeding to diagnosis.")
            break

    print(f"  Q&A complete — {len(answers)} answer(s) collected.")
    return answers


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Diagnose
# ─────────────────────────────────────────────────────────────────────────────

def phase3_diagnose(
    error_text:     str,
    analysis:       dict[str, Any],
    answers:        list[dict[str, str]],
    postmortem_ctx: str,
) -> dict[str, Any]:
    """LLM call 2: synthesize all context into structured diagnosis."""
    print("[clarificator] Phase 3 — Synthesizing diagnosis …")

    # Build Q&A summary
    qa_lines = []
    for a in answers:
        if a["answer"] not in ("[skipped]", "[auto-skipped]", "[non-interactive]"):
            qa_lines.append(f"**{a['tier']} — {a['question']}**\n{a['answer']}")
    qa_summary = "\n\n".join(qa_lines) if qa_lines else "(no additional context provided)"

    user = "\n\n---\n\n".join([
        f"## Original Error\n\n{error_text}",
        f"## Initial Analysis\n\nTaxonomy: {analysis.get('taxonomy', '?')}\n"
        f"Hypothesis: {analysis.get('initial_hypothesis', '?')}",
        f"## Q&A Answers\n\n{qa_summary}",
        *(
            [f"## Historical Context\n\n{postmortem_ctx}"]
            if postmortem_ctx else []
        ),
    ])

    raw    = _call_llm(_SYSTEM_DIAGNOSE, user, max_tokens=4096)
    result = _parse_json(raw)

    print()
    print(f"  Root cause:  {result.get('root_cause', '?')}")
    res = result.get('resolution', '?')
    print(f"  Resolution:  {res[:100]}{'…' if len(res) > 100 else ''}")
    print(f"  Confidence:  {result.get('confidence', '?')}")
    if result.get("follow_up"):
        print(f"  Follow-up:   {result['follow_up'][:80]}")
    print()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Save
# ─────────────────────────────────────────────────────────────────────────────

def _write_session(
    error_text: str,
    analysis:   dict[str, Any],
    answers:    list[dict[str, str]],
    diagnosis:  dict[str, Any],
) -> None:
    """Overwrite incident_session.json — current session state."""
    session = {
        "session_at":    _now_iso(),
        "error_text":    error_text[:3000],
        "taxonomy":      analysis.get("taxonomy", "other"),
        "hypothesis":    analysis.get("initial_hypothesis", ""),
        "qa_answers":    answers,
        "diagnosis":     diagnosis,
    }
    p = _session_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    track_write(p)


def _append_log(
    error_text: str,
    diagnosis:  dict[str, Any],
    saved_to_kb: bool,
) -> None:
    """Append session summary to incident_log.md."""
    log = _log_path()
    log.parent.mkdir(parents=True, exist_ok=True)

    rc  = diagnosis.get("root_cause", "?")
    res = diagnosis.get("resolution", "?")
    tax = diagnosis.get("taxonomy", "?")
    conf = diagnosis.get("confidence", "?")

    block = (
        f"## Session — {_now_display()}\n\n"
        f"- **Symptom**: {error_text[:120]}\n"
        f"- **Taxonomy**: {tax}\n"
        f"- **Root cause**: {rc}\n"
        f"- **Resolution**: {res[:200]}\n"
        f"- **Confidence**: {conf}\n"
        f"- **Saved to postmortem KB**: {'yes' if saved_to_kb else 'no'}\n\n"
        f"---\n\n"
    )
    with log.open("a", encoding="utf-8") as f:
        f.write(block)
    track_write(log)


def _maybe_save_to_kb(
    error_text: str,
    analysis:   dict[str, Any],
    answers:    list[dict[str, str]],
    diagnosis:  dict[str, Any],
    auto:       bool,
    save_kb:    bool = False,
) -> bool:
    """
    Ask user if they want to save this diagnosis to the postmortem KB.

    auto=True  (--auto / --no-interactive): skip prompt, do NOT save unless
               save_kb=True is also set.
    save_kb=True (--save-kb): save without prompting, even in auto mode.
    """
    if not _POSTMORTEM_AVAILABLE:
        return False

    # --save-kb: explicit opt-in, no prompt needed
    if save_kb:
        print("  [--save-kb] Saving to postmortem KB.")
        do_save = True

    elif not sys.stdin.isatty() and not auto:
        # Non-interactive, non-auto: skip silently
        print("  [non-interactive] Skipping KB save prompt.")
        return False

    elif auto:
        # --auto means skip Q&A, not skip human decisions about KB
        # Default to not saving in auto mode — use --save-kb to opt in
        print("  [auto] Skipping KB save (use --save-kb to save in auto mode).")
        return False

    else:
        print("=" * 60)
        print("  Save to postmortem knowledge base?")
        print("  This will add a structured entry to postmortem_kb.json")
        print("  so future sessions can learn from this incident.")
        print()
        _flush_stdin()
        try:
            ans = input("  Save? [Y/n]: ").strip().lower()
            do_save = ans in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            do_save = False

    if not do_save:
        print("  Skipped — not saved to postmortem KB.")
        return False

    try:
        from postmortem_archivist import capture_to_kb  # type: ignore
        iid = capture_to_kb(
            taxonomy        = diagnosis.get("taxonomy", "other"),
            symptom         = error_text[:300],
            causal_chain    = diagnosis.get("causal_chain", []),
            root_cause      = diagnosis.get("root_cause", ""),
            resolution      = diagnosis.get("resolution", ""),
            failed_attempts = diagnosis.get("failed_attempts", []),
            preventable_by  = diagnosis.get("preventable_by"),
            files_affected  = diagnosis.get("files_affected", []),
            tags            = diagnosis.get("tags", []),
        )
        print(f"  ✓ Saved as {iid} in postmortem KB.")
        return True
    except Exception as exc:
        print(f"  [warn] Could not save to KB: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Keyword extraction (no LLM — used for fast postmortem context pre-fetch)
# ─────────────────────────────────────────────────────────────────────────────

_TECH_KEYWORDS = {
    "irsa", "iam", "oidc", "serviceaccount", "assumerole",
    "oomkilled", "oom", "memory", "cpu", "resource",
    "values.yaml", "helm", "chart", "annotation",
    "ecr", "image", "docker", "updater", "writeback",
    "sqs", "s3", "airflow", "dag", "xcom", "sensor",
    "connection refused", "dns", "port", "vpc", "subnet",
    "argocd", "pvc", "storageclass", "ebs", "finalizer",
    "jenkins", "jnlp", "pipeline", "build",
    "terraform", "provider", "state", "destroy",
    "grafana", "prometheus", "loki", "cloudflare",
    "mlflow", "training", "inference", "f1", "precision",
    "eks", "kubernetes", "kubectl", "pod", "node",
}


def _extract_keywords(text: str) -> list[str]:
    text_lower = text.lower()
    found = [kw for kw in _TECH_KEYWORDS if kw in text_lower]
    return found[:10]  # cap to avoid over-fetching postmortems


def _load_consistency_context() -> str:
    """
    Pull recent HIGH findings from config_consistency_checker for LLM context.
    Closes the feedback loop: if checker already detected an IRSA mismatch,
    clarificator knows about it before diagnosing AccessDenied errors.
    """
    log_path = _clarificator_dir().parent / "consistency" / "consistency_log.json"
    if not log_path.exists():
        return ""
    try:
        data    = json.loads(log_path.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else data.get("entries", [])
        high    = [e for e in entries[-10:] if e.get("risk_level") == "HIGH"]
        if not high:
            return ""
        lines = [
            "## Recent HIGH consistency findings (config_consistency_checker)",
            "_These findings were detected before this incident — may be related._",
            "",
        ]
        for e in high[-3:]:
            run_at  = e.get("run_at", "?")
            summary = e.get("summary", "")[:80]
            n_high  = e.get("high_findings", 0)
            focus   = e.get("focus", "")
            lines.append(f"- [{run_at}] focus={focus} high={n_high}: {summary}")
        return "\n".join(lines)
    except Exception:
        return ""



# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="incident_clarificator.py",
        description="DevOps/MLOps incident diagnosis assistant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project",        default=os.environ.get("PIPELINE_PROJECT"))
    p.add_argument("--text",           default=None, help="Inline error description.")
    p.add_argument("--auto",           action="store_true",
                   help="Skip Q&A prompts, run direct diagnosis from initial input.")
    p.add_argument("--save-kb",        action="store_true",
                   help="Auto-save diagnosis to postmortem KB without prompting. "
                        "Use with --auto for fully non-interactive runs.")
    p.add_argument("--no-interactive", action="store_true",
                   help="Disable all TTY prompts.")
    p.add_argument("--max-rounds",     type=int, default=3,
                   help="Max Q&A tier rounds (default: 3 = T1+T2+T3).")
    p.add_argument("--no-postmortem",  action="store_true",
                   help="Skip postmortem KB context injection.")
    p.add_argument("--verbose",        action="store_true")
    return p


def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return
    if not os.environ.get("PIPELINE_PROJECT"):
        parser.error("Use --project <name> or export PIPELINE_PROJECT=<name>.")


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    _configure_project(args.project, parser)
    _clarificator_dir().mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  INCIDENT CLARIFICATOR")
    print("=" * 68)
    print()

    exit_code = 0
    error_text = ""
    analysis:  dict[str, Any] = {}
    answers:   list[dict]     = []
    diagnosis: dict[str, Any] = {}

    try:
        # Phase 0 — Intake
        error_text = phase0_intake(args)
        if not error_text:
            print("[clarificator] No input. Exiting.")
            sys.exit(0)

        print(f"[clarificator] Input received ({len(error_text)} chars)")
        print()

        # Load postmortem context
        postmortem_ctx = ""
        if _POSTMORTEM_AVAILABLE and not args.no_postmortem:
            # Pre-analyze to get taxonomy for targeted context pull
            # Use keyword extraction first (fast, no LLM)
            keywords = _extract_keywords(error_text)
            postmortem_ctx = get_relevant_context(keywords=keywords, max_items=4)
            if postmortem_ctx and args.verbose:
                print(f"[clarificator] Postmortem context injected ({len(postmortem_ctx)} chars)")

        # Load consistency checker context (recent HIGH findings)
        consistency_ctx = _load_consistency_context()
        if consistency_ctx:
            postmortem_ctx = (postmortem_ctx + "\n\n" + consistency_ctx).strip()
            if args.verbose:
                print(f"[clarificator] Consistency context injected ({len(consistency_ctx)} chars)")

        # Phase 1 — Analyze
        analysis = phase1_analyze(error_text, postmortem_ctx)

        # Re-fetch postmortem context with taxonomy if we didn't have it yet
        if _POSTMORTEM_AVAILABLE and not args.no_postmortem:
            taxonomy = analysis.get("taxonomy", "other")
            postmortem_ctx = get_relevant_context(
                taxonomy=taxonomy,
                keywords=_extract_keywords(error_text),
                max_items=4,
            )

        # Phase 2 — Q&A
        answers = phase2_qa(analysis, auto=args.auto, max_rounds=args.max_rounds)

        # Phase 3 — Diagnose
        diagnosis = phase3_diagnose(error_text, analysis, answers, postmortem_ctx)

        # Phase 4 — Save
        _write_session(error_text, analysis, answers, diagnosis)
        print(f"[clarificator] Session → {_session_path()}")

        saved_to_kb = _maybe_save_to_kb(
            error_text, analysis, answers, diagnosis,
            auto     = args.auto or args.no_interactive,
            save_kb  = getattr(args, "save_kb", False),
        )

        _append_log(error_text, diagnosis, saved_to_kb)
        print(f"[clarificator] Log     → {_log_path()}")
        print()
        print("=" * 68)
        print(f"  DIAGNOSIS COMPLETE")
        print("=" * 68)
        print(f"  Root cause:   {diagnosis.get('root_cause', '?')}")
        print(f"  Confidence:   {diagnosis.get('confidence', '?')}")
        print(f"  Resolution:   {diagnosis.get('resolution', '?')[:120]}")
        if diagnosis.get("follow_up"):
            print(f"  Follow-up:    {diagnosis['follow_up'][:100]}")
        print()

    except KeyboardInterrupt:
        print("\n[clarificator] Interrupted.")
        exit_code = 130
    except Exception as exc:
        print(f"[clarificator][error] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        print()
        print_artifact_summary("[clarificator]")
        print()
        print_cost_summary("[clarificator]")
        prompt_next_step(ROLE, prefix="[clarificator]")

    sys.exit(exit_code)


# ─────────────────────────────────────────────────────────────────────────────
# Keyword extraction (no LLM — used for fast postmortem context pre-fetch)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()