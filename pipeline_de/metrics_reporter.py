"""
toolkits/devops_mlops/metrics_reporter.py
==========================================
Multi-source metrics collector for the IoT MLOps pipeline.

Collects from 3 sources (4th deferred):
  1. CloudWatchCollector   — pod CPU/memory/restarts, node utilization, OOMKilled events
  2. MLflowCollector       — experiment runs, best metrics, model registry status
  3. CostCollector         — AWS Cost Explorer by tag, breakdown by service
  4. AthenaCollector       — deferred (add after infra stable)

Output:
  metrics_reporter/metrics_report.json   (short-term overwrite — current period snapshot)
  metrics_reporter/metrics_log.json      (long-term append — history across runs, user can discard)
  metrics_reporter/metrics_history.json  (long-term append — permanent trend store for infra_judge.py)

Usage:
  python metrics_reporter.py --project iot-mlops
  python metrics_reporter.py --project iot-mlops --period 7d
  python metrics_reporter.py --project iot-mlops --collectors cloudwatch mlflow
  python metrics_reporter.py --project iot-mlops --dry-run
  python metrics_reporter.py --project iot-mlops --show-last
  PIPELINE_PROJECT=iot-mlops python metrics_reporter.py

AWS credentials:
  Standard boto3 credential chain (env vars, ~/.aws/credentials, EC2 instance profile).
  Required permissions: cloudwatch:GetMetricStatistics, cloudwatch:ListMetrics,
  ce:GetCostAndUsage, ce:GetDimensionValues.

MLflow:
  Requires MLFLOW_TRACKING_URI env var or mlflow running locally.
  If unreachable, MLflow collector is skipped with a warning (non-fatal).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
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
from modules.post_interactive import prompt_next_step                     # noqa: E402

ROLE = "metrics_reporter"

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_PERIOD_DAYS  = 7
_DEFAULT_AWS_REGION   = "ap-southeast-1"
_DEFAULT_COST_TAG_KEY = "Project"
_CE_LAG_HOURS         = 24   # Cost Explorer data lags ~24h

# EKS namespaces to monitor (extend as needed)
_DEFAULT_NAMESPACES = [
    "airflow", "mlflow", "jenkins", "monitoring",
    "argocd", "kube-system",
]

# Services to break down in cost report
_COST_SERVICES = [
    "Amazon Elastic Kubernetes Service",
    "Amazon Elastic Compute Cloud - Compute",
    "Amazon Simple Storage Service",
    "Amazon Relational Database Service",
    "Amazon Simple Queue Service",
    "AWS Lambda",
    "Amazon API Gateway",
    "AWS Key Management Service",
]


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


def _reporter_dir() -> Path:
    return _devops_artifact_root() / "metrics_reporter"


def _report_path() -> Path:
    return _reporter_dir() / "metrics_report.json"


def _log_path() -> Path:
    return _reporter_dir() / "metrics_log.json"


def _history_path() -> Path:
    return _reporter_dir() / "metrics_history.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_display() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _period_window(days: int) -> tuple[datetime, datetime]:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return start, end


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _week_label(dt: datetime) -> str:
    return dt.strftime("%Y-W%V")


# ─────────────────────────────────────────────────────────────────────────────
# Safe boto3 import helper
# ─────────────────────────────────────────────────────────────────────────────

def _boto3_client(service: str, region: str):
    try:
        import boto3
        return boto3.client(service, region_name=region)
    except ImportError:
        raise RuntimeError("boto3 not installed. Run: pip install boto3")


# ─────────────────────────────────────────────────────────────────────────────
# CloudWatch Collector
# ─────────────────────────────────────────────────────────────────────────────

class CloudWatchCollector:
    """
    Collects EKS pod + node metrics from CloudWatch Container Insights.

    Requires Container Insights enabled on the EKS cluster:
      aws eks update-addon --cluster-name <name> --addon-name amazon-cloudwatch-observability

    Metrics collected:
      - pod_cpu_utilization        (% per pod per namespace)
      - pod_memory_utilization     (% per pod per namespace)
      - pod_restarts               (count per pod, last N days)
      - node_cpu_utilization       (% per node)
      - node_memory_utilization    (% per node)
      - oom_events                 (pod_status == OOMKilled, last N days)
    """

    def __init__(self, region: str, cluster_name: str, period_days: int, namespaces: list[str]):
        self.region       = region
        self.cluster_name = cluster_name
        self.period_days  = period_days
        self.namespaces   = namespaces
        self._client      = None

    def _cw(self):
        if self._client is None:
            self._client = _boto3_client("cloudwatch", self.region)
        return self._client

    def _cw_logs(self):
        try:
            import boto3
            return boto3.client("logs", region_name=self.region)
        except ImportError:
            raise RuntimeError("boto3 not installed")

    def _get_metric_stats(
        self,
        namespace:   str,
        metric_name: str,
        dimensions:  list[dict],
        stat:        str,
        period_secs: int = 3600,
    ) -> list[dict]:
        start, end = _period_window(self.period_days)
        try:
            resp = self._cw().get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dimensions,
                StartTime=start,
                EndTime=end,
                Period=period_secs,
                Statistics=[stat],
            )
            return resp.get("Datapoints", [])
        except Exception as exc:
            if "AccessDenied" in str(exc):
                raise
            return []

    def _avg(self, datapoints: list[dict]) -> float | None:
        vals = [d.get("Average", d.get("Sum", 0)) for d in datapoints]
        return round(sum(vals) / len(vals), 2) if vals else None

    def _max(self, datapoints: list[dict]) -> float | None:
        vals = [d.get("Maximum", d.get("Average", 0)) for d in datapoints]
        return round(max(vals), 2) if vals else None

    def _sum(self, datapoints: list[dict]) -> float:
        return sum(d.get("Sum", d.get("Average", 0)) for d in datapoints)

    def collect_node_metrics(self) -> dict[str, Any]:
        """Collect cluster-level node CPU/memory averages."""
        result: dict[str, Any] = {
            "node_cpu_avg_pct":    None,
            "node_memory_avg_pct": None,
            "node_count":          None,
        }

        try:
            cw = self._cw()
            # List node metrics to count nodes
            resp = cw.list_metrics(
                Namespace="ContainerInsights",
                MetricName="node_cpu_utilization",
                Dimensions=[{"Name": "ClusterName", "Value": self.cluster_name}],
            )
            node_names = list({
                d["Value"]
                for m in resp.get("Metrics", [])
                for d in m.get("Dimensions", [])
                if d["Name"] == "NodeName"
            })
            result["node_count"] = len(node_names)

            cpu_avgs:    list[float] = []
            memory_avgs: list[float] = []

            for node_name in node_names[:20]:   # cap at 20 nodes
                dims = [
                    {"Name": "ClusterName", "Value": self.cluster_name},
                    {"Name": "NodeName",    "Value": node_name},
                ]
                cpu_pts = self._get_metric_stats(
                    "ContainerInsights", "node_cpu_utilization", dims, "Average"
                )
                mem_pts = self._get_metric_stats(
                    "ContainerInsights", "node_memory_utilization", dims, "Average"
                )
                if cpu_pts:
                    cpu_avgs.append(self._avg(cpu_pts) or 0)
                if mem_pts:
                    memory_avgs.append(self._avg(mem_pts) or 0)

            if cpu_avgs:
                result["node_cpu_avg_pct"]    = round(sum(cpu_avgs) / len(cpu_avgs), 2)
            if memory_avgs:
                result["node_memory_avg_pct"] = round(sum(memory_avgs) / len(memory_avgs), 2)

        except Exception as exc:
            result["error"] = str(exc)

        return result

    def collect_pod_metrics(self) -> dict[str, Any]:
        """Collect per-namespace pod CPU/memory + restart counts."""
        result: dict[str, Any] = {
            "by_namespace":    {},
            "total_restarts":  0,
            "oom_events":      0,
        }

        try:
            cw = self._cw()
            start, end = _period_window(self.period_days)

            for ns in self.namespaces:
                ns_data: dict[str, Any] = {
                    "pod_count":        0,
                    "cpu_avg_pct":      None,
                    "memory_avg_pct":   None,
                    "pod_restarts":     {},
                }

                # List pods in this namespace
                resp = cw.list_metrics(
                    Namespace="ContainerInsights",
                    MetricName="pod_cpu_utilization",
                    Dimensions=[
                        {"Name": "ClusterName", "Value": self.cluster_name},
                        {"Name": "Namespace",   "Value": ns},
                    ],
                )
                pod_names = list({
                    d["Value"]
                    for m in resp.get("Metrics", [])
                    for d in m.get("Dimensions", [])
                    if d["Name"] == "PodName"
                })
                ns_data["pod_count"] = len(pod_names)

                cpu_avgs:    list[float] = []
                memory_avgs: list[float] = []

                for pod in pod_names[:10]:   # cap per namespace
                    dims = [
                        {"Name": "ClusterName", "Value": self.cluster_name},
                        {"Name": "Namespace",   "Value": ns},
                        {"Name": "PodName",     "Value": pod},
                    ]
                    # CPU
                    cpu_pts = self._get_metric_stats(
                        "ContainerInsights", "pod_cpu_utilization", dims, "Average"
                    )
                    if cpu_pts:
                        cpu_avgs.append(self._avg(cpu_pts) or 0)

                    # Memory
                    mem_pts = self._get_metric_stats(
                        "ContainerInsights", "pod_memory_utilization", dims, "Average"
                    )
                    if mem_pts:
                        memory_avgs.append(self._avg(mem_pts) or 0)

                    # Restarts
                    restart_pts = self._get_metric_stats(
                        "ContainerInsights", "pod_number_of_container_restarts",
                        dims, "Sum", period_secs=86400,
                    )
                    total_restarts = int(self._sum(restart_pts))
                    if total_restarts > 0:
                        ns_data["pod_restarts"][pod] = total_restarts
                        result["total_restarts"] += total_restarts

                if cpu_avgs:
                    ns_data["cpu_avg_pct"]    = round(sum(cpu_avgs) / len(cpu_avgs), 2)
                if memory_avgs:
                    ns_data["memory_avg_pct"] = round(sum(memory_avgs) / len(memory_avgs), 2)

                if ns_data["pod_count"] > 0:
                    result["by_namespace"][ns] = ns_data

            # OOMKilled events via CloudWatch Logs Insights query
            result["oom_events"] = self._count_oom_events()

        except Exception as exc:
            result["error"] = str(exc)

        return result

    def _count_oom_events(self) -> int:
        """Count OOMKilled events from Container Insights logs."""
        try:
            logs_client = self._cw_logs()
            start, end  = _period_window(self.period_days)

            log_group   = f"/aws/containerinsights/{self.cluster_name}/performance"
            query       = (
                "fields @timestamp, pod_status | "
                "filter pod_status = 'OOMKilled' | "
                "stats count() as oom_count"
            )
            start_resp  = logs_client.start_query(
                logGroupName = log_group,
                startTime    = int(start.timestamp()),
                endTime      = int(end.timestamp()),
                queryString  = query,
                limit        = 1,
            )
            query_id    = start_resp["queryId"]

            import time
            for _ in range(20):
                time.sleep(1)
                result = logs_client.get_query_results(queryId=query_id)
                if result["status"] == "Complete":
                    rows = result.get("results", [])
                    if rows:
                        for field in rows[0]:
                            if field.get("field") == "oom_count":
                                return int(field.get("value", 0))
                    return 0
            return 0
        except Exception:
            return 0

    def collect(self) -> dict[str, Any]:
        print("  [cloudwatch] Collecting node metrics …", end=" ", flush=True)
        nodes = self.collect_node_metrics()
        print("done")

        print("  [cloudwatch] Collecting pod metrics …", end=" ", flush=True)
        pods = self.collect_pod_metrics()
        print(f"done ({pods.get('total_restarts', 0)} restarts, {pods.get('oom_events', 0)} OOM)")

        return {
            "nodes": nodes,
            "pods":  pods,
            "collected_at": _now_iso(),
            "period_days":  self.period_days,
            "cluster":      self.cluster_name,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Collector
# ─────────────────────────────────────────────────────────────────────────────

class MLflowCollector:
    """
    Collects experiment and model registry data from MLflow.

    Requires:
      - MLFLOW_TRACKING_URI env var (e.g. http://localhost:5000 or internal cluster URL)
      - mlflow Python package: pip install mlflow

    If MLflow is running inside EKS and not accessible from the machine running
    this script, use kubectl port-forward first:
      kubectl port-forward svc/mlflow 5000:80 -n mlflow

    Non-fatal: if MLflow is unreachable, collector returns {"status": "unavailable"}.
    """

    def __init__(self, tracking_uri: str | None, period_days: int):
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
        self.period_days  = period_days

    def _get_client(self):
        try:
            import mlflow
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            # Probe connectivity
            client.search_experiments(max_results=1)
            return client
        except ImportError:
            raise RuntimeError("mlflow not installed. Run: pip install mlflow")
        except Exception as exc:
            raise RuntimeError(f"MLflow unreachable at {self.tracking_uri!r}: {exc}")

    def collect_experiments(self, client) -> list[dict[str, Any]]:
        """Collect recent runs from all experiments."""
        try:
            import mlflow
            experiments = client.search_experiments()
        except Exception as exc:
            return [{"error": str(exc)}]

        results = []
        start, _ = _period_window(self.period_days)
        cutoff_ms = int(start.timestamp() * 1000)

        for exp in experiments:
            if exp.name == "Default" and not exp.tags:
                continue   # skip empty default experiment

            try:
                runs = client.search_runs(
                    experiment_ids = [exp.experiment_id],
                    filter_string  = f"attributes.start_time >= {cutoff_ms}",
                    max_results    = 50,
                    order_by       = ["attributes.start_time DESC"],
                )
            except Exception:
                runs = []

            if not runs:
                continue

            # Best metrics across runs
            best_metrics: dict[str, float] = {}
            metric_keys: set[str] = set()
            for run in runs:
                metric_keys.update(run.data.metrics.keys())

            for key in metric_keys:
                vals = [
                    run.data.metrics[key]
                    for run in runs
                    if key in run.data.metrics
                ]
                if vals:
                    # Higher is better for f1/precision/recall/auc
                    best_metrics[key] = max(vals)

            # Last run status
            last_run    = runs[0]
            last_status = last_run.info.status
            last_ts     = datetime.fromtimestamp(
                last_run.info.start_time / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")

            results.append({
                "experiment_id":   exp.experiment_id,
                "experiment_name": exp.name,
                "run_count":       len(runs),
                "last_run_at":     last_ts,
                "last_run_status": last_status,
                "best_metrics":    {k: round(v, 4) for k, v in best_metrics.items()},
            })

        return results

    def collect_model_registry(self, client) -> list[dict[str, Any]]:
        """Collect registered models and their versions."""
        try:
            models = client.search_registered_models(max_results=50)
        except Exception as exc:
            return [{"error": str(exc)}]

        results = []
        for model in models:
            versions = client.get_latest_versions(model.name)
            version_summary = []
            for v in versions:
                version_summary.append({
                    "version": v.version,
                    "stage":   v.current_stage,
                    "status":  v.status,
                    "run_id":  v.run_id,
                })
            results.append({
                "model_name":       model.name,
                "latest_versions":  version_summary,
                "total_versions":   len(versions),
                "has_production":   any(v.current_stage == "Production" for v in versions),
                "has_staging":      any(v.current_stage == "Staging"    for v in versions),
            })

        return results

    def collect(self) -> dict[str, Any]:
        print("  [mlflow] Connecting …", end=" ", flush=True)

        try:
            client = self._get_client()
            print("connected")
        except RuntimeError as exc:
            print(f"unavailable — {exc}")
            return {
                "status":      "unavailable",
                "reason":      str(exc),
                "hint":        (
                    "If MLflow runs in EKS, use: "
                    "kubectl port-forward svc/mlflow 5000:80 -n mlflow"
                ),
                "collected_at": _now_iso(),
            }

        print("  [mlflow] Collecting experiments …", end=" ", flush=True)
        experiments = self.collect_experiments(client)
        print(f"done ({len(experiments)} experiments)")

        print("  [mlflow] Collecting model registry …", end=" ", flush=True)
        registry = self.collect_model_registry(client)
        prod_count = sum(1 for m in registry if m.get("has_production"))
        print(f"done ({len(registry)} models, {prod_count} in Production)")

        # Data quality flags
        flags = []
        for exp in experiments:
            if "error" in exp:
                continue
            f1 = exp.get("best_metrics", {}).get("f1", None)
            precision = exp.get("best_metrics", {}).get("precision", None)
            if f1 is not None and f1 == 0.0:
                flags.append({
                    "experiment": exp["experiment_name"],
                    "flag":       "f1_zero",
                    "hint":       "f1=0 may indicate flat data or model not qualifying threshold",
                })
            if precision is not None and precision == 0.0:
                flags.append({
                    "experiment": exp["experiment_name"],
                    "flag":       "precision_zero",
                    "hint":       "precision=0 — check data variance and anomaly injection",
                })

        return {
            "status":         "ok",
            "experiments":    experiments,
            "model_registry": registry,
            "data_flags":     flags,
            "collected_at":   _now_iso(),
            "period_days":    self.period_days,
            "tracking_uri":   self.tracking_uri,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Cost Collector
# ─────────────────────────────────────────────────────────────────────────────

class CostCollector:
    """
    Collects AWS cost data from Cost Explorer.

    Requirements:
      - ce:GetCostAndUsage permission on the IAM role/user running this script
      - Cost allocation tags enabled: activate the tag in AWS Billing console first
        (Billing → Cost allocation tags → Activate)

    Rate limit: Cost Explorer allows ~1 request/second. This collector makes
    2-3 calls total so it is well within limits.

    Data lag: Cost Explorer data lags ~24h. The report period end is set to
    yesterday to avoid misleading partial-day totals.
    """

    def __init__(
        self,
        region:      str,
        period_days: int,
        tag_key:     str,
        tag_value:   str,
    ):
        self.region      = region
        self.period_days = period_days
        self.tag_key     = tag_key
        self.tag_value   = tag_value
        self._client     = None

    def _ce(self):
        if self._client is None:
            # Cost Explorer is global, always us-east-1
            self._client = _boto3_client("ce", "us-east-1")
        return self._client

    def _date_range(self) -> tuple[str, str]:
        """
        Cost Explorer uses YYYY-MM-DD dates.
        End = yesterday (data lag ~24h).
        Start = end - period_days.
        """
        end   = datetime.now(timezone.utc).date() - timedelta(days=1)
        start = end - timedelta(days=self.period_days)
        return str(start), str(end)

    def collect_total(self) -> dict[str, Any]:
        """Collect total cost filtered by tag."""
        start, end = self._date_range()
        try:
            resp = self._ce().get_cost_and_usage(
                TimePeriod = {"Start": start, "End": end},
                Granularity = "MONTHLY",
                Filter = {
                    "Tags": {
                        "Key":    self.tag_key,
                        "Values": [self.tag_value],
                    }
                },
                Metrics = ["UnblendedCost"],
            )
            total_usd = sum(
                float(
                    r["Total"]["UnblendedCost"]["Amount"]
                )
                for r in resp.get("ResultsByTime", [])
            )
            return {
                "total_usd": round(total_usd, 4),
                "period":    f"{start} to {end}",
                "tag":       f"{self.tag_key}={self.tag_value}",
            }
        except Exception as exc:
            return {"error": str(exc), "period": f"{start} to {end}"}

    def collect_by_service(self) -> dict[str, float]:
        """Collect cost breakdown by AWS service."""
        start, end = self._date_range()
        try:
            resp = self._ce().get_cost_and_usage(
                TimePeriod  = {"Start": start, "End": end},
                Granularity = "MONTHLY",
                Filter      = {
                    "Tags": {
                        "Key":    self.tag_key,
                        "Values": [self.tag_value],
                    }
                },
                Metrics     = ["UnblendedCost"],
                GroupBy     = [{"Type": "DIMENSION", "Key": "SERVICE"}],
            )
            breakdown: dict[str, float] = {}
            for period_result in resp.get("ResultsByTime", []):
                for group in period_result.get("Groups", []):
                    svc_name = group["Keys"][0]
                    amount   = float(group["Total"]["UnblendedCost"]["Amount"])
                    if amount > 0:
                        # Map to short service name
                        short = _shorten_service_name(svc_name)
                        breakdown[short] = round(
                            breakdown.get(short, 0) + amount, 4
                        )
            return breakdown
        except Exception as exc:
            return {"error": str(exc)}

    def collect_week_over_week(self) -> dict[str, Any]:
        """Compare this week vs last week cost (anomaly detection)."""
        try:
            end   = datetime.now(timezone.utc).date() - timedelta(days=1)
            w1_end   = end
            w1_start = end - timedelta(days=7)
            w2_end   = w1_start
            w2_start = w2_end - timedelta(days=7)

            def _week_cost(s, e) -> float:
                r = self._ce().get_cost_and_usage(
                    TimePeriod  = {"Start": str(s), "End": str(e)},
                    Granularity = "MONTHLY",
                    Filter      = {
                        "Tags": {
                            "Key":    self.tag_key,
                            "Values": [self.tag_value],
                        }
                    },
                    Metrics = ["UnblendedCost"],
                )
                return sum(
                    float(row["Total"]["UnblendedCost"]["Amount"])
                    for row in r.get("ResultsByTime", [])
                )

            this_week = _week_cost(w1_start, w1_end)
            last_week = _week_cost(w2_start, w2_end)
            delta_pct = (
                round((this_week - last_week) / last_week * 100, 1)
                if last_week > 0 else None
            )
            return {
                "this_week_usd": round(this_week, 4),
                "last_week_usd": round(last_week, 4),
                "delta_pct":     delta_pct,
                "anomaly":       delta_pct is not None and abs(delta_pct) > 20,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def collect(self) -> dict[str, Any]:
        print("  [cost] Collecting total cost …", end=" ", flush=True)
        total = self.collect_total()
        print(f"done (${total.get('total_usd', '?')})")

        print("  [cost] Collecting cost by service …", end=" ", flush=True)
        by_service = self.collect_by_service()
        print(f"done ({len(by_service)} services)")

        print("  [cost] Collecting week-over-week delta …", end=" ", flush=True)
        wow = self.collect_week_over_week()
        delta_str = f"{wow.get('delta_pct', '?')}%" if "error" not in wow else "error"
        print(f"done (WoW: {delta_str})")

        result = {
            "total":           total,
            "by_service":      by_service,
            "week_over_week":  wow,
            "collected_at":    _now_iso(),
            "period_days":     self.period_days,
            "data_lag_note":   f"Cost Explorer lags ~{_CE_LAG_HOURS}h — today's costs not included.",
        }

        # Flag cost anomaly
        if wow.get("anomaly"):
            result["cost_anomaly_flag"] = {
                "flag":    "COST_SPIKE",
                "delta_pct": wow["delta_pct"],
                "hint":    f"Cost increased {wow['delta_pct']}% week-over-week — investigate scaling or new resources.",
            }

        return result


def _shorten_service_name(name: str) -> str:
    """Map verbose AWS service names to short keys."""
    mapping = {
        "Amazon Elastic Kubernetes Service":               "eks",
        "Amazon Elastic Compute Cloud - Compute":          "ec2",
        "Amazon Simple Storage Service":                   "s3",
        "Amazon Relational Database Service":              "rds",
        "Amazon Simple Queue Service":                     "sqs",
        "AWS Lambda":                                      "lambda",
        "Amazon API Gateway":                              "api_gateway",
        "AWS Key Management Service":                      "kms",
        "Amazon Elastic Container Registry Public":        "ecr_public",
        "Amazon EC2 Container Registry (ECR)":             "ecr",
        "Amazon Virtual Private Cloud":                    "vpc",
        "AWS Data Transfer":                               "data_transfer",
        "Amazon CloudWatch":                               "cloudwatch",
        "AWS Secrets Manager":                             "secrets_manager",
        "Amazon Route 53":                                 "route53",
        "AWS Certificate Manager":                         "acm",
        "Amazon Athena":                                   "athena",
        "AWS Glue":                                        "glue",
    }
    for long_name, short in mapping.items():
        if long_name.lower() in name.lower():
            return short
    # Fallback: lowercase + underscores
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


import re  # used by _shorten_service_name — ensure available


# ─────────────────────────────────────────────────────────────────────────────
# Report assembly
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_report(
    period_days:  int,
    cloudwatch:   dict[str, Any] | None,
    mlflow:       dict[str, Any] | None,
    cost:         dict[str, Any] | None,
    run_at:       str,
) -> dict[str, Any]:
    """Assemble all collector outputs into a single structured report."""

    # ── Infra health summary ──────────────────────────────────────────────────
    infra_health: dict[str, Any] = {"status": "no_data"}
    if cloudwatch:
        nodes = cloudwatch.get("nodes", {})
        pods  = cloudwatch.get("pods", {})
        infra_health = {
            "status":              "ok" if "error" not in nodes else "error",
            "node_count":          nodes.get("node_count"),
            "node_cpu_avg_pct":    nodes.get("node_cpu_avg_pct"),
            "node_memory_avg_pct": nodes.get("node_memory_avg_pct"),
            "total_pod_restarts":  pods.get("total_restarts", 0),
            "oom_events":          pods.get("oom_events", 0),
            "pod_restarts_by_ns":  {
                ns: sum(data.get("pod_restarts", {}).values())
                for ns, data in pods.get("by_namespace", {}).items()
                if data.get("pod_restarts")
            },
            "namespaces_monitored": list(pods.get("by_namespace", {}).keys()),
        }

    # ── ML health summary ─────────────────────────────────────────────────────
    ml_health: dict[str, Any] = {"status": "no_data"}
    if mlflow:
        ml_status = mlflow.get("status", "unavailable")
        if ml_status == "ok":
            exps      = mlflow.get("experiments", [])
            registry  = mlflow.get("model_registry", [])
            data_flags = mlflow.get("data_flags", [])

            last_run_at = None
            for exp in exps:
                if "last_run_at" in exp:
                    if last_run_at is None or exp["last_run_at"] > last_run_at:
                        last_run_at = exp["last_run_at"]

            prod_models = sum(1 for m in registry if m.get("has_production"))

            ml_health = {
                "status":            "ok",
                "experiment_count":  len(exps),
                "registered_models": len(registry),
                "production_models": prod_models,
                "last_training_at":  last_run_at,
                "data_quality_flags": data_flags,
                "experiments":       exps,
            }
        else:
            ml_health = {
                "status": ml_status,
                "reason": mlflow.get("reason", ""),
                "hint":   mlflow.get("hint", ""),
            }

    # ── Cost summary ─────────────────────────────────────────────────────────
    cost_summary: dict[str, Any] = {"status": "no_data"}
    if cost:
        total   = cost.get("total", {})
        by_svc  = cost.get("by_service", {})
        wow     = cost.get("week_over_week", {})

        cost_summary = {
            "status":           "ok" if "error" not in total else "error",
            "total_usd":        total.get("total_usd"),
            "period":           total.get("period"),
            "by_service":       {k: v for k, v in by_svc.items() if isinstance(v, float)},
            "week_over_week":   wow,
            "cost_anomaly":     cost.get("cost_anomaly_flag"),
            "data_lag_note":    cost.get("data_lag_note"),
        }

    # ── Flags and alerts ──────────────────────────────────────────────────────
    flags: list[dict] = []

    if cloudwatch:
        pods = cloudwatch.get("pods", {})
        if pods.get("oom_events", 0) > 0:
            flags.append({
                "level":   "WARNING",
                "source":  "cloudwatch",
                "code":    "OOM_EVENTS",
                "detail":  f"{pods['oom_events']} OOMKilled event(s) in the last {period_days} days.",
                "hint":    "Check pod memory limits. Common culprit: Airflow workers during training.",
            })
        if pods.get("total_restarts", 0) > 5:
            flags.append({
                "level":  "WARNING",
                "source": "cloudwatch",
                "code":   "HIGH_RESTART_COUNT",
                "detail": f"{pods['total_restarts']} total pod restarts.",
                "hint":   "Check CrashLoopBackOff pods: kubectl get pods -A | grep -v Running",
            })
        nodes = cloudwatch.get("nodes", {})
        if nodes.get("node_memory_avg_pct") and nodes["node_memory_avg_pct"] > 85:
            flags.append({
                "level":  "WARNING",
                "source": "cloudwatch",
                "code":   "HIGH_NODE_MEMORY",
                "detail": f"Average node memory: {nodes['node_memory_avg_pct']}%",
                "hint":   "Consider adding nodes or upgrading instance type.",
            })

    if mlflow and mlflow.get("status") == "ok":
        for flag in mlflow.get("data_flags", []):
            flags.append({
                "level":  "INFO",
                "source": "mlflow",
                "code":   flag.get("flag", "").upper(),
                "detail": f"Experiment '{flag.get('experiment')}': {flag.get('flag')}",
                "hint":   flag.get("hint", ""),
            })

    if cost and cost.get("cost_anomaly_flag"):
        anomaly = cost["cost_anomaly_flag"]
        flags.append({
            "level":  "WARNING",
            "source": "cost",
            "code":   "COST_SPIKE",
            "detail": anomaly.get("hint", ""),
            "hint":   "Run: aws ce get-cost-and-usage with GROUP BY SERVICE to find the spike source.",
        })

    return {
        "report_at":    run_at,
        "period_days":  period_days,
        "infra_health": infra_health,
        "ml_health":    ml_health,
        "cost":         cost_summary,
        "flags":        flags,
        "flag_count":   len(flags),
        "_raw": {
            "cloudwatch": cloudwatch,
            "mlflow":     mlflow,
            "cost":       cost,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(report: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    p = _report_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    track_write(p)


def _append_log(report: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    log = _log_path()
    log.parent.mkdir(parents=True, exist_ok=True)

    try:
        track_read(log)
        data    = json.loads(log.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else data.get("entries", [])
    except Exception:
        entries = []

    # Summary entry (compact — not full raw data)
    infra  = report.get("infra_health", {})
    ml     = report.get("ml_health", {})
    cost   = report.get("cost", {})
    flags  = report.get("flags", [])

    entries.append({
        "report_at":           report["report_at"],
        "period_days":         report["period_days"],
        "node_cpu_avg_pct":    infra.get("node_cpu_avg_pct"),
        "node_memory_avg_pct": infra.get("node_memory_avg_pct"),
        "oom_events":          infra.get("oom_events", 0),
        "pod_restarts":        infra.get("total_pod_restarts", 0),
        "ml_status":           ml.get("status"),
        "production_models":   ml.get("production_models"),
        "total_cost_usd":      cost.get("total_usd"),
        "cost_anomaly":        cost.get("cost_anomaly") is not None,
        "flag_count":          len(flags),
        "flag_codes":          [f.get("code") for f in flags],
    })

    log.write_text(
        json.dumps({"entries": entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(log)


def _maybe_commit_log() -> None:
    log = _log_path()
    if not log.exists():
        return
    try:
        data    = json.loads(log.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else data.get("entries", [])
    except Exception:
        return
    if not entries:
        return
    try:
        ans = input(f"  Keep this entry in {log.name}? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("  [reporter] Entry kept (non-interactive).")
        return
    if ans in ("n", "no"):
        entries.pop()
        try:
            log.write_text(
                json.dumps({"entries": entries}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("  [reporter] Entry discarded.")
        except Exception as exc:
            print(f"  [reporter][warn] Could not revert log: {exc}")
    else:
        print(f"  [reporter] Entry kept (total: {len(entries)}).")


def _append_history(report: dict[str, Any]) -> None:
    """
    Long-term trend store — append-only, never overwritten.

    Differs from metrics_log.json:
      metrics_log.json     — compact summary, user can discard via _maybe_commit_log()
      metrics_history.json — permanent record, append-only, consumed by infra_judge.py
                             for trend detection across multiple periods (OOM increasing
                             over 4 weeks, cost above budget, model never reaching
                             Production stage, etc.).
    """
    hist = _history_path()
    hist.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []
    if hist.exists():
        try:
            track_read(hist)
            data     = json.loads(hist.read_text(encoding="utf-8"))
            existing = data if isinstance(data, list) else data.get("entries", [])
        except Exception:
            pass

    infra = report.get("infra_health", {})
    ml    = report.get("ml_health",   {})
    cost  = report.get("cost",        {})
    flags = report.get("flags",       [])

    entry: dict[str, Any] = {
        "report_at":           report.get("report_at"),
        "period_days":         report.get("period_days"),
        # infra snapshot
        "node_count":          infra.get("node_count"),
        "node_cpu_avg_pct":    infra.get("node_cpu_avg_pct"),
        "node_memory_avg_pct": infra.get("node_memory_avg_pct"),
        "oom_events":          infra.get("oom_events", 0),
        "total_pod_restarts":  infra.get("total_pod_restarts", 0),
        # ml snapshot
        "ml_status":           ml.get("status"),
        "experiment_count":    ml.get("experiment_count"),
        "registered_models":   ml.get("registered_models"),
        "production_models":   ml.get("production_models"),
        "last_training_at":    ml.get("last_training_at"),
        # cost snapshot
        "total_cost_usd":      cost.get("total_usd"),
        "cost_period":         cost.get("period"),
        "wow_delta_pct":       cost.get("week_over_week", {}).get("delta_pct"),
        "cost_anomaly":        cost.get("cost_anomaly") is not None,
        # flags
        "flag_count":          len(flags),
        "flag_codes":          [f.get("code") for f in flags],
        "warning_count":       sum(1 for f in flags if f.get("level") == "WARNING"),
    }

    existing.append(entry)
    hist.write_text(
        json.dumps({"entries": existing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(hist)


def _print_summary(report: dict[str, Any]) -> None:
    """Print human-readable summary to terminal."""
    infra = report.get("infra_health", {})
    ml    = report.get("ml_health", {})
    cost  = report.get("cost", {})
    flags = report.get("flags", [])

    print()
    print("=" * 68)
    print("  METRICS SUMMARY")
    print("=" * 68)

    # Infra
    print()
    print("  Infrastructure:")
    if infra.get("status") == "no_data":
        print("    (no cloudwatch data)")
    elif infra.get("status") == "error":
        print(f"    ⚠  Error: {infra.get('error', '?')}")
    else:
        print(f"    Nodes:    {infra.get('node_count', '?')} nodes  "
              f"CPU avg {infra.get('node_cpu_avg_pct', '?')}%  "
              f"Mem avg {infra.get('node_memory_avg_pct', '?')}%")
        print(f"    Pods:     {infra.get('total_pod_restarts', 0)} restarts  "
              f"{infra.get('oom_events', 0)} OOM events")

    # ML
    print()
    print("  ML Pipeline:")
    if ml.get("status") == "no_data":
        print("    (no mlflow data)")
    elif ml.get("status") == "unavailable":
        print(f"    ⚠  MLflow unavailable: {ml.get('reason', '')}")
    elif ml.get("status") == "ok":
        print(f"    Experiments:     {ml.get('experiment_count', 0)}")
        print(f"    Registered:      {ml.get('registered_models', 0)} models "
              f"({ml.get('production_models', 0)} in Production)")
        print(f"    Last training:   {ml.get('last_training_at', 'never')}")
        if ml.get("data_quality_flags"):
            for flag in ml["data_quality_flags"]:
                print(f"    ⚠  {flag.get('flag')}: {flag.get('hint', '')}")

    # Cost
    print()
    print("  Cost:")
    if cost.get("status") == "no_data":
        print("    (no cost data)")
    elif cost.get("status") == "error":
        print(f"    ⚠  Error: {cost.get('error', '?')}")
    else:
        print(f"    Total:   ${cost.get('total_usd', '?')}  ({cost.get('period', '')})")
        by_svc = cost.get("by_service", {})
        if by_svc:
            top = sorted(by_svc.items(), key=lambda x: x[1] if isinstance(x[1], float) else 0, reverse=True)
            for svc, amt in top[:5]:
                if isinstance(amt, float):
                    print(f"    {svc:<20} ${amt:.4f}")
        wow = cost.get("week_over_week", {})
        if wow.get("delta_pct") is not None:
            delta = wow["delta_pct"]
            icon  = "⬆" if delta > 0 else "⬇"
            print(f"    WoW:  {icon} {delta:+.1f}%  "
                  f"(this week: ${wow.get('this_week_usd', '?')}, "
                  f"last week: ${wow.get('last_week_usd', '?')})")

    # Flags
    if flags:
        print()
        print(f"  ⚠  {len(flags)} flag(s):")
        for flag in flags:
            lvl = flag.get("level", "")
            code = flag.get("code", "")
            detail = flag.get("detail", "")[:60]
            print(f"    [{lvl}] {code}: {detail}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_ALL_COLLECTORS = ["cloudwatch", "mlflow", "cost"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="metrics_reporter.py",
        description="Multi-source metrics collector for IoT MLOps pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project",    default=os.environ.get("PIPELINE_PROJECT"))
    p.add_argument(
        "--period",
        default="7d",
        help="Collection period: e.g. 7d, 14d, 30d (default: 7d).",
    )
    p.add_argument(
        "--collectors",
        nargs="+",
        choices=_ALL_COLLECTORS,
        default=_ALL_COLLECTORS,
        help="Which collectors to run (default: all).",
    )
    p.add_argument(
        "--cluster",
        default=os.environ.get("EKS_CLUSTER_NAME", ""),
        help="EKS cluster name (or set EKS_CLUSTER_NAME env var).",
    )
    p.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", _DEFAULT_AWS_REGION),
        help=f"AWS region (default: {_DEFAULT_AWS_REGION}).",
    )
    p.add_argument(
        "--cost-tag-key",
        default=os.environ.get("COST_TAG_KEY", _DEFAULT_COST_TAG_KEY),
        help=f"AWS cost allocation tag key (default: {_DEFAULT_COST_TAG_KEY}).",
    )
    p.add_argument(
        "--cost-tag-value",
        default=os.environ.get("COST_TAG_VALUE", os.environ.get("PIPELINE_PROJECT", "")),
        help="AWS cost allocation tag value (default: PIPELINE_PROJECT).",
    )
    p.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", ""),
        help="MLflow tracking URI (or set MLFLOW_TRACKING_URI env var).",
    )
    p.add_argument(
        "--namespaces",
        nargs="+",
        default=_DEFAULT_NAMESPACES,
        help="K8s namespaces to monitor.",
    )
    p.add_argument("--dry-run",   action="store_true",
                   help="Collect and print but do not write artifacts.")
    p.add_argument("--show-last", action="store_true",
                   help="Print last report and exit.")
    p.add_argument("--bootstrap", action="store_true",
                   help="Collect only — no verdict/flags (use for first run with no baseline).")
    p.add_argument("--verbose",   action="store_true")
    return p


def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return
    if not os.environ.get("PIPELINE_PROJECT"):
        parser.error("Use --project <name> or export PIPELINE_PROJECT=<name>.")


def _parse_period(period_str: str) -> int:
    m = re.match(r"^(\d+)d$", period_str.strip())
    if not m:
        raise ValueError(f"Invalid period format: {period_str!r}. Use e.g. '7d', '14d'.")
    return int(m.group(1))


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    _configure_project(args.project, parser)
    _reporter_dir().mkdir(parents=True, exist_ok=True)

    # --show-last short circuit
    if args.show_last:
        p = _report_path()
        if not p.exists():
            print("[reporter] No report found.")
            sys.exit(1)
        track_read(p)
        print(p.read_text(encoding="utf-8"))
        sys.exit(0)

    try:
        period_days = _parse_period(args.period)
    except ValueError as exc:
        print(f"[reporter] {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 68)
    print("  METRICS REPORTER")
    print("=" * 68)
    print(f"  Period:      {period_days} days")
    print(f"  Collectors:  {', '.join(args.collectors)}")
    print(f"  Region:      {args.region}")
    print(f"  Cluster:     {args.cluster or '(not set)'}")
    print(f"  Dry-run:     {args.dry_run}")
    print(f"  Bootstrap:   {args.bootstrap}")
    print()

    exit_code = 0
    run_at    = _now_display()

    cw_data:    dict[str, Any] | None = None
    mlflow_data: dict[str, Any] | None = None
    cost_data:  dict[str, Any] | None = None

    try:
        # ── CloudWatch ────────────────────────────────────────────────────────
        if "cloudwatch" in args.collectors:
            if not args.cluster:
                print("[reporter][warn] --cluster not set — skipping CloudWatch collection.")
                print("  Set EKS_CLUSTER_NAME or use --cluster <name>.")
            else:
                collector = CloudWatchCollector(
                    region       = args.region,
                    cluster_name = args.cluster,
                    period_days  = period_days,
                    namespaces   = args.namespaces,
                )
                cw_data = collector.collect()

        # ── MLflow ────────────────────────────────────────────────────────────
        if "mlflow" in args.collectors:
            collector = MLflowCollector(
                tracking_uri = args.mlflow_uri,
                period_days  = period_days,
            )
            mlflow_data = collector.collect()

        # ── Cost ──────────────────────────────────────────────────────────────
        if "cost" in args.collectors:
            tag_value = args.cost_tag_value or os.environ.get("PIPELINE_PROJECT", "")
            if not tag_value:
                print("[reporter][warn] --cost-tag-value not set — skipping Cost collection.")
                print("  Set COST_TAG_VALUE or use --cost-tag-value <value>.")
            else:
                collector = CostCollector(
                    region      = args.region,
                    period_days = period_days,
                    tag_key     = args.cost_tag_key,
                    tag_value   = tag_value,
                )
                cost_data = collector.collect()

        # ── Assemble ──────────────────────────────────────────────────────────
        report = _assemble_report(
            period_days = period_days,
            cloudwatch  = cw_data,
            mlflow      = mlflow_data,
            cost        = cost_data,
            run_at      = run_at,
        )

        # In bootstrap mode, clear flags (no baseline to compare against)
        if args.bootstrap:
            report["flags"]      = []
            report["flag_count"] = 0
            report["_bootstrap"] = True

        _print_summary(report)

        # ── Write artifacts ───────────────────────────────────────────────────
        _write_report(report, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"  Report: {_report_path()}")

        _append_log(report, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"  Log:     {_log_path()}")

        _append_history(report)
        if not args.dry_run:
            print(f"  History: {_history_path()}")

        # Exit code: 1 if any WARNING flags
        warning_flags = [f for f in report.get("flags", []) if f.get("level") == "WARNING"]
        exit_code = 1 if warning_flags else 0

    except KeyboardInterrupt:
        print("\n[reporter] Interrupted.")
        exit_code = 130
    except Exception as exc:
        print(f"[reporter][error] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        print()
        print_artifact_summary("[reporter]")
        print()
        print_cost_summary("[reporter]")
        prompt_next_step(ROLE, prefix="[reporter]")

    # Long-term artifact commit
    if not args.dry_run and exit_code in (0, 1):
        _maybe_commit_log()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
