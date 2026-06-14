"""
toolkits/devops_mlops/live_discovery.py
========================================
Crawl a live AWS account and extract deep metadata for every reachable
service — without downloading real data.

Complements infra_absorber.py:
  infra_absorber   reads IaC source (.tf, values.yaml, DAGs)   → DESIRED state
  live_discovery   calls AWS APIs directly                      → ACTUAL state

Running both and feeding both outputs to config_consistency_checker.py
enables true drift detection: desired vs actual.

────────────────────────────────────────────────────────────────
When to use each
────────────────────────────────────────────────────────────────

  No IaC codebase            → live_discovery only
  IaC only, no AWS access    → infra_absorber only
  Both available (ideal)     → live_discovery first, then infra_absorber,
                               then config_consistency_checker --mode drift

────────────────────────────────────────────────────────────────
Services crawled (default: all, skip gracefully if no permission)
────────────────────────────────────────────────────────────────

  s3          Buckets, ARNs, regions, policies, ACLs, versioning,
              encryption, tags, object listing (keys + sizes, no content)

  iam         Users, roles, groups, policies (inline + attached),
              trust relationships, permission boundaries, access keys
              (existence only, not values)

  eks         Clusters, versions, endpoint access, node groups,
              Fargate profiles, add-ons, OIDC config

  redshift    Clusters, databases, schemas, tables, column counts,
              external schemas (Spectrum), Glue catalog tables,
              row counts via SVV_TABLE_INFO (requires cluster access)

  rds         DB instances and clusters, engine versions, parameter
              groups, subnet groups, security groups, tags

  lambda      Functions, runtimes, handlers, memory/timeout,
              environment variable keys (values redacted), layers,
              triggers (event source mappings)

  sqs         Queues, ARNs, attributes (visibility timeout, retention,
              max message size, DLQ config), tags

  sns         Topics, ARNs, subscriptions, access policies, tags

  ec2         VPCs, subnets, security groups (rules), route tables,
              internet gateways, instances (state, type, tags),
              EBS volumes (no snapshots), AMIs (owned only)

  ecr         Repositories, image count, tags, scan config,
              lifecycle policies, repository policies

  glue        Databases, tables, columns, partition keys,
              crawlers, jobs, connections

  athena      Workgroups, named queries, data catalogs

  secretsmanager
              Secret names, ARNs, rotation config (values never extracted)

  ssm         Parameter names, types, tiers (values never extracted)

  cloudwatch  Alarms (name, state, metric, threshold),
              log groups (name, retention, size)

  ────────────────────────────────────────────────────────────────
  Use --services to crawl a subset:
    --services s3 iam eks redshift

────────────────────────────────────────────────────────────────
Safety guarantees
────────────────────────────────────────────────────────────────

  Never reads object content, secret values, SSM parameter values,
  or environment variable values. Only metadata, ARNs, names,
  counts, policies, and structural information.

  Graceful permission degradation: AccessDenied on any call → skip
  that resource/service with a warning, continue crawling the rest.

────────────────────────────────────────────────────────────────
Outputs written
────────────────────────────────────────────────────────────────

  live_discovery/discovery_map.json  (short-term, OVERWRITE)
    Deep structured metadata. Top-level keys: meta, services.
    Each service key contains its full extracted data.
    Consumed by: config_consistency_checker.py, infra_judge.py

  live_discovery/discovery_map.md    (short-term, OVERWRITE)
    Human-readable summary per service. No LLM — pure formatting.

  live_discovery/discovery_log.json  (long-term, APPEND)
    One entry per run: timestamp, services crawled, resource counts,
    skipped services (permission denied).

────────────────────────────────────────────────────────────────
Artifact impact by command
────────────────────────────────────────────────────────────────

  Command                      discovery_map.json  .md   log.json
  ─────────────────────────── ──────────────────  ───── ─────────
  (full run)                   OVERWRITE           OVERWRITE  APPEND
  --services s3 iam            OVERWRITE           OVERWRITE  APPEND
  --dry-run                    –                   –          –
  --show-last                  –                   –          –

────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────

  python live_discovery.py --project my-co
    Crawl all services. Uses default AWS credentials.

  python live_discovery.py --project my-co --services s3 iam eks
    Crawl only specified services.

  python live_discovery.py --project my-co --region ap-southeast-1
    Crawl specific region (default: from AWS config / ap-southeast-1).

  python live_discovery.py --project my-co --all-regions
    Crawl all enabled regions. Slower but complete.

  python live_discovery.py --project my-co --profile prod-readonly
    Use a specific AWS CLI profile.

  python live_discovery.py --project my-co --dry-run
    Discover what would be crawled without making API calls.

  python live_discovery.py --project my-co --show-last
    Print most recent discovery_map.md without re-crawling.

────────────────────────────────────────────────────────────────
Environment variables
────────────────────────────────────────────────────────────────

  PIPELINE_PROJECT       Project slug (required if --project not set)
  AWS_REGION             Default region (fallback: ap-southeast-1)
  AWS_PROFILE            AWS CLI profile name
  DEVOPS_ARTIFACT_ROOT   Override artifact output root

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLKIT_DIR = Path(__file__).parent
_REPO_ROOT   = _TOOLKIT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from modules.artifact_tracking import (  # noqa: E402
    track_read, track_write,
    print_summary as print_artifact_summary,
)
from modules.md_header import apply_header as apply_md_header  # noqa: E402

_DEFAULT_REGION  = os.environ.get("AWS_REGION", "ap-southeast-1")
_ALL_SERVICES    = (
    "s3", "iam", "eks", "redshift", "rds", "lambda",
    "sqs", "sns", "ec2", "ecr", "glue", "athena",
    "secretsmanager", "ssm", "cloudwatch",
)

# Max items per paginated list to avoid runaway API calls
_MAX_OBJECTS_PER_BUCKET = 5_000
_MAX_LOG_GROUPS         = 500
_MAX_ITEMS              = 1_000


# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths
# ─────────────────────────────────────────────────────────────────────────────

def _devops_artifact_root() -> Path:
    override = os.environ.get("DEVOPS_ARTIFACT_ROOT")
    base     = Path(override) if override else _REPO_ROOT.parent / "outputs" / "devops_mlops"
    slug     = os.environ.get("PIPELINE_PROJECT", "default")
    return base / f"artifacts_{slug}"

def _discovery_dir()  -> Path: return _devops_artifact_root() / "live_discovery"
def _json_path()      -> Path: return _discovery_dir() / "discovery_map.json"
def _md_path()        -> Path: return _discovery_dir() / "discovery_map.md"
def _log_path()       -> Path: return _discovery_dir() / "discovery_log.json"

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# boto3 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _boto3_client(service: str, region: str, profile: str | None = None) -> Any:
    import boto3  # type: ignore
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client(service, region_name=region)


def _paginate(client: Any, method: str, result_key: str, **kwargs: Any) -> list[Any]:
    """Generic paginator wrapper. Returns flat list of items."""
    items: list[Any] = []
    try:
        paginator = client.get_paginator(method)
        for page in paginator.paginate(**kwargs):
            items.extend(page.get(result_key, []))
            if len(items) >= _MAX_ITEMS:
                break
    except Exception:
        # Paginator may not exist for some methods — return what we have
        pass
    return items


def _safe(fn: Any, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """Call fn, return default on any exception (AccessDenied, NoSuchBucket, etc.)"""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _redact(d: dict[str, Any] | None, keys_to_keep: list[str]) -> dict[str, Any]:
    """Keep only specified keys from a dict, redact the rest."""
    if not d:
        return {}
    return {k: v for k, v in d.items() if k in keys_to_keep}


# ─────────────────────────────────────────────────────────────────────────────
# CrawlerStatus — track per-service outcome
# ─────────────────────────────────────────────────────────────────────────────

class CrawlerStatus:
    def __init__(self) -> None:
        self._s: dict[str, dict[str, Any]] = {}

    def ok(self, svc: str, counts: dict[str, int], note: str = "") -> None:
        self._s[svc] = {"status": "ok", "counts": counts, "note": note}
        total = sum(counts.values())
        print(f"  [{svc}] ✓  {total} resources  {counts}")

    def skip(self, svc: str, reason: str) -> None:
        self._s[svc] = {"status": "skipped", "reason": reason}
        print(f"  [{svc}] ⏭  Skipped: {reason}")

    def fail(self, svc: str, error: str) -> None:
        self._s[svc] = {"status": "failed", "error": error}
        print(f"  [{svc}] ✗  Failed: {error}")

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return dict(self._s)


# ─────────────────────────────────────────────────────────────────────────────
# Crawlers — one function per service
# ─────────────────────────────────────────────────────────────────────────────

def _crawl_s3(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        import boto3  # type: ignore
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        s3 = session.client("s3", region_name=region)

        raw_buckets = _safe(lambda: s3.list_buckets().get("Buckets", []), default=[])
        buckets: list[dict[str, Any]] = []

        for b in raw_buckets:
            name = b["Name"]
            created = b.get("CreationDate", "").isoformat() if hasattr(b.get("CreationDate", ""), "isoformat") else str(b.get("CreationDate", ""))

            # Location
            location = _safe(
                lambda n=name: s3.get_bucket_location(Bucket=n)
                          .get("LocationConstraint") or "us-east-1",
                default="unknown",
            )

            # Policy
            policy_raw = _safe(
                lambda n=name: s3.get_bucket_policy(Bucket=n).get("Policy"),
                default=None,
            )
            policy = json.loads(policy_raw) if policy_raw else None

            # ACL — single API call, capture name with default arg
            acl = _safe(
                lambda n=name: (
                    lambda r: {
                        "owner":  r.get("Owner", {}),
                        "grants": [
                            {
                                "grantee": g.get("Grantee", {}),
                                "permission": g.get("Permission"),
                            }
                            for g in r.get("Grants", [])
                        ],
                    }
                )(s3.get_bucket_acl(Bucket=n)),
                default={},
            )

            # Versioning
            versioning = _safe(
                lambda n=name: s3.get_bucket_versioning(Bucket=n).get("Status", "Disabled"),
                default="unknown",
            )

            # Encryption
            enc = _safe(
                lambda n=name: s3.get_bucket_encryption(Bucket=n)
                          .get("ServerSideEncryptionConfiguration", {})
                          .get("Rules", [{}])[0]
                          .get("ApplyServerSideEncryptionByDefault", {}),
                default={},
            )

            # Logging
            logging_cfg = _safe(
                lambda n=name: s3.get_bucket_logging(Bucket=n).get("LoggingEnabled"),
                default=None,
            )

            # Public access block
            public_block = _safe(
                lambda n=name: s3.get_public_access_block(Bucket=n)
                          .get("PublicAccessBlockConfiguration", {}),
                default={},
            )

            # Tags
            tags = _safe(
                lambda n=name: {
                    t["Key"]: t["Value"]
                    for t in s3.get_bucket_tagging(Bucket=n).get("TagSet", [])
                },
                default={},
            )

            # Lifecycle rules (names only)
            lifecycle = _safe(
                lambda n=name: [
                    r.get("ID", f"rule-{i}")
                    for i, r in enumerate(
                        s3.get_bucket_lifecycle_configuration(Bucket=n).get("Rules", [])
                    )
                ],
                default=[],
            )

            # Object listing — keys + sizes only, no content
            objects: list[dict[str, Any]] = []
            total_size_bytes = 0
            object_count = 0
            try:
                pager = s3.get_paginator("list_objects_v2")
                for page in pager.paginate(Bucket=name):
                    for obj in page.get("Contents", []):
                        total_size_bytes += obj.get("Size", 0)
                        object_count += 1
                        if object_count <= _MAX_OBJECTS_PER_BUCKET:
                            objects.append({
                                "key":           obj["Key"],
                                "size_bytes":    obj.get("Size", 0),
                                "last_modified": obj.get("LastModified", "").isoformat()
                                    if hasattr(obj.get("LastModified", ""), "isoformat")
                                    else str(obj.get("LastModified", "")),
                                "storage_class": obj.get("StorageClass", "STANDARD"),
                            })
            except Exception:
                pass

            buckets.append({
                "name":              name,
                "arn":               f"arn:aws:s3:::{name}",
                "region":            location,
                "created_at":        created,
                "versioning":        versioning,
                "encryption":        enc,
                "public_access_block": public_block,
                "logging":           logging_cfg,
                "policy":            policy,
                "acl":               acl,
                "tags":              tags,
                "lifecycle_rules":   lifecycle,
                "object_count":      object_count,
                "total_size_bytes":  total_size_bytes,
                "objects":           objects,
                "objects_truncated": object_count > _MAX_OBJECTS_PER_BUCKET,
            })

        status.ok("s3", {"buckets": len(buckets),
                          "objects": sum(b["object_count"] for b in buckets)})
        return {"buckets": buckets}

    except ImportError:
        status.skip("s3", "boto3 not installed")
        return {}
    except Exception as e:
        status.fail("s3", str(e)[:120])
        return {}


def _crawl_iam(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        iam = _boto3_client("iam", region, profile)

        # ── Roles ────────────────────────────────────────────────────────────
        raw_roles = _safe(lambda: _paginate(iam, "list_roles", "Roles"), default=[])
        roles: list[dict[str, Any]] = []
        for r in raw_roles:
            rname = r["RoleName"]
            # Inline policies
            inline_names = _safe(
                lambda rn=rname: _paginate(iam, "list_role_policies", "PolicyNames", RoleName=rn),
                default=[],
            )
            inline_docs: dict[str, Any] = {}
            for pname in (inline_names or []):
                doc = _safe(
                    lambda rn=rname, pn=pname: iam.get_role_policy(RoleName=rn, PolicyName=pn)
                               .get("PolicyDocument"),
                    default=None,
                )
                if doc:
                    inline_docs[pname] = doc

            # Attached managed policies
            attached = _safe(
                lambda rn=rname: _paginate(
                    iam, "list_attached_role_policies", "AttachedPolicies", RoleName=rn
                ),
                default=[],
            )

            # Permission boundary
            boundary_arn = r.get("PermissionsBoundary", {}).get("PermissionsBoundaryArn")

            roles.append({
                "name":                rname,
                "arn":                 r["Arn"],
                "path":                r.get("Path", "/"),
                "created_at":          r.get("CreateDate", "").isoformat()
                    if hasattr(r.get("CreateDate", ""), "isoformat")
                    else str(r.get("CreateDate", "")),
                "trust_policy":        r.get("AssumeRolePolicyDocument", {}),
                "description":         r.get("Description", ""),
                "max_session_duration": r.get("MaxSessionDuration", 3600),
                "permission_boundary": boundary_arn,
                "inline_policies":     inline_docs,
                "attached_policies":   [
                    {"name": p["PolicyName"], "arn": p["PolicyArn"]}
                    for p in (attached or [])
                ],
                "tags": {
                    t["Key"]: t["Value"]
                    for t in r.get("Tags", [])
                },
            })

        # ── Users ─────────────────────────────────────────────────────────────
        raw_users = _safe(lambda: _paginate(iam, "list_users", "Users"), default=[])
        users: list[dict[str, Any]] = []
        for u in (raw_users or []):
            uname = u["UserName"]
            access_keys = _safe(
                lambda un=uname: [
                    {
                        "key_id":     k["AccessKeyId"],
                        "status":     k["Status"],
                        "created_at": k.get("CreateDate", "").isoformat()
                            if hasattr(k.get("CreateDate", ""), "isoformat")
                            else str(k.get("CreateDate", "")),
                    }
                    for k in iam.list_access_keys(UserName=un)
                               .get("AccessKeyMetadata", [])
                ],
                default=[],
            )
            groups = _safe(
                lambda un=uname: [
                    g["GroupName"]
                    for g in _paginate(
                        iam, "list_groups_for_user", "Groups", UserName=un
                    )
                ],
                default=[],
            )
            users.append({
                "username":          uname,
                "arn":               u["Arn"],
                "path":              u.get("Path", "/"),
                "created_at":        u.get("CreateDate", "").isoformat()
                    if hasattr(u.get("CreateDate", ""), "isoformat")
                    else str(u.get("CreateDate", "")),
                "password_last_used": str(u.get("PasswordLastUsed", "")),
                "access_keys":       access_keys,  # IDs and status only, never secret
                "groups":            groups,
            })

        # ── Groups ────────────────────────────────────────────────────────────
        raw_groups = _safe(lambda: _paginate(iam, "list_groups", "Groups"), default=[])
        groups_out: list[dict[str, Any]] = []
        for g in (raw_groups or []):
            gname = g["GroupName"]
            attached = _safe(
                lambda gn=gname: _paginate(
                    iam, "list_attached_group_policies", "AttachedPolicies", GroupName=gn
                ),
                default=[],
            )
            groups_out.append({
                "name":             gname,
                "arn":              g["Arn"],
                "attached_policies": [
                    {"name": p["PolicyName"], "arn": p["PolicyArn"]}
                    for p in (attached or [])
                ],
            })

        status.ok("iam", {
            "roles": len(roles),
            "users": len(users),
            "groups": len(groups_out),
        })
        return {"roles": roles, "users": users, "groups": groups_out}

    except ImportError:
        status.skip("iam", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("iam", "AccessDenied — need iam:List* permissions")
        else:
            status.fail("iam", str(e)[:120])
        return {}


def _crawl_eks(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        eks = _boto3_client("eks", region, profile)

        cluster_names = _safe(
            lambda: eks.list_clusters().get("clusters", []),
            default=[],
        )
        clusters: list[dict[str, Any]] = []

        for cname in (cluster_names or []):
            c = _safe(
                lambda: eks.describe_cluster(name=cname).get("cluster", {}),
                default={},
            )

            # Node groups
            ng_names = _safe(
                lambda: _paginate(
                    eks, "list_nodegroups", "nodegroups", clusterName=cname
                ),
                default=[],
            )
            node_groups: list[dict[str, Any]] = []
            for ngname in (ng_names or []):
                ng = _safe(
                    lambda: eks.describe_nodegroup(
                        clusterName=cname, nodegroupName=ngname
                    ).get("nodegroup", {}),
                    default={},
                )
                node_groups.append({
                    "name":           ng.get("nodegroupName"),
                    "status":         ng.get("status"),
                    "instance_types": ng.get("instanceTypes", []),
                    "ami_type":       ng.get("amiType"),
                    "capacity_type":  ng.get("capacityType"),
                    "scaling": {
                        "min":     ng.get("scalingConfig", {}).get("minSize"),
                        "max":     ng.get("scalingConfig", {}).get("maxSize"),
                        "desired": ng.get("scalingConfig", {}).get("desiredSize"),
                    },
                    "disk_size_gb":   ng.get("diskSize"),
                    "node_role_arn":  ng.get("nodeRole"),
                    "labels":         ng.get("labels", {}),
                    "taints":         ng.get("taints", []),
                    "tags":           ng.get("tags", {}),
                })

            # Add-ons
            addon_names = _safe(
                lambda: _paginate(
                    eks, "list_addons", "addons", clusterName=cname
                ),
                default=[],
            )
            addons: list[dict[str, Any]] = []
            for aname in (addon_names or []):
                ad = _safe(
                    lambda: eks.describe_addon(
                        clusterName=cname, addonName=aname
                    ).get("addon", {}),
                    default={},
                )
                addons.append({
                    "name":            ad.get("addonName"),
                    "version":         ad.get("addonVersion"),
                    "status":          ad.get("status"),
                    "service_account": ad.get("serviceAccountRoleArn"),
                })

            # OIDC
            oidc_url = c.get("identity", {}).get("oidc", {}).get("issuer", "")

            clusters.append({
                "name":               cname,
                "arn":                c.get("arn"),
                "status":             c.get("status"),
                "kubernetes_version": c.get("version"),
                "endpoint":           c.get("endpoint"),
                "role_arn":           c.get("roleArn"),
                "vpc_config":         _redact(
                    c.get("resourcesVpcConfig", {}),
                    ["vpcId", "subnetIds", "securityGroupIds",
                     "endpointPublicAccess", "endpointPrivateAccess",
                     "publicAccessCidrs"],
                ),
                "logging":            c.get("logging", {}),
                "oidc_issuer":        oidc_url,
                "platform_version":   c.get("platformVersion"),
                "tags":               c.get("tags", {}),
                "node_groups":        node_groups,
                "addons":             addons,
            })

        status.ok("eks", {"clusters": len(clusters),
                           "node_groups": sum(len(c["node_groups"]) for c in clusters)})
        return {"clusters": clusters}

    except ImportError:
        status.skip("eks", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("eks", "AccessDenied — need eks:List*/Describe* permissions")
        else:
            status.fail("eks", str(e)[:120])
        return {}


def _crawl_redshift(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        rs  = _boto3_client("redshift",      region, profile)
        rsd = _boto3_client("redshift-data", region, profile)

        clusters_raw = _safe(
            lambda: _paginate(rs, "describe_clusters", "Clusters"),
            default=[],
        )
        clusters: list[dict[str, Any]] = []

        for c in (clusters_raw or []):
            cid  = c["ClusterIdentifier"]
            db   = c.get("DBName", "dev")

            # ── Schema + table inventory via redshift-data ────────────────────
            databases:  list[dict[str, Any]] = []
            db_names = _safe(
                lambda: [
                    d["databaseName"]
                    for d in _paginate(
                        rsd, "list_databases", "Databases",
                        ClusterIdentifier=cid, Database=db,
                    )
                ],
                default=[db],
            )

            for dbname in (db_names or []):
                schemas_raw = _safe(
                    lambda: _paginate(
                        rsd, "list_schemas", "Schemas",
                        ClusterIdentifier=cid, Database=dbname,
                    ),
                    default=[],
                )
                schemas: list[dict[str, Any]] = []
                for schema in (schemas_raw or []):
                    sname = schema.get("schemaName", schema) if isinstance(schema, dict) else schema

                    tables_raw = _safe(
                        lambda: _paginate(
                            rsd, "list_tables", "Tables",
                            ClusterIdentifier=cid, Database=dbname,
                            SchemaPattern=sname,
                        ),
                        default=[],
                    )
                    tables: list[dict[str, Any]] = []
                    for t in (tables_raw or []):
                        tname = t.get("name") or t.get("tableName", "")
                        ttype = t.get("type", "TABLE")

                        # Column metadata
                        cols = _safe(
                            lambda: _paginate(
                                rsd, "list_table_metadata", "TableList",
                                ClusterIdentifier=cid, Database=dbname,
                                SchemaPattern=sname, TablePattern=tname,
                            ),
                            default=[],
                        )
                        col_count = len(cols) if cols else None

                        # Row count via SVV_TABLE_INFO (best-effort)
                        row_count = None
                        stmt = (
                            f"SELECT tbl_rows FROM svv_table_info "
                            f"WHERE schema='{sname}' AND \"table\"='{tname}'"
                        )
                        exec_resp = _safe(
                            lambda: rsd.execute_statement(
                                ClusterIdentifier=cid, Database=dbname, Sql=stmt,
                            ),
                            default=None,
                        )
                        if exec_resp:
                            import time
                            for _ in range(8):
                                time.sleep(1)
                                desc = _safe(
                                    lambda: rsd.describe_statement(Id=exec_resp["Id"]),
                                    default={},
                                )
                                if desc.get("Status") in ("FINISHED", "FAILED", "ABORTED"):
                                    break
                            if desc.get("Status") == "FINISHED":
                                rows_resp = _safe(
                                    lambda: rsd.get_statement_result(Id=exec_resp["Id"]),
                                    default={},
                                )
                                records = rows_resp.get("Records", [])
                                if records:
                                    row_count = int(records[0][0].get("longValue", 0))

                        tables.append({
                            "name":       tname,
                            "type":       ttype,
                            "column_count": col_count,
                            "row_count":  row_count,
                        })

                    schemas.append({
                        "name":         sname,
                        "is_external":  sname.lower() not in ("public", "information_schema",
                                                               "pg_catalog", "pg_toast"),
                        "table_count":  len(tables),
                        "tables":       tables,
                    })

                databases.append({"name": dbname, "schemas": schemas})

            clusters.append({
                "identifier":             cid,
                "arn":                    f"arn:aws:redshift:{region}:{c.get('MasterUsername', '')}:cluster:{cid}",
                "status":                 c.get("ClusterStatus"),
                "node_type":              c.get("NodeType"),
                "number_of_nodes":        c.get("NumberOfNodes"),
                "db_name":                db,
                "endpoint":               c.get("Endpoint", {}).get("Address"),
                "port":                   c.get("Endpoint", {}).get("Port"),
                "vpc_id":                 c.get("VpcId"),
                "encrypted":              c.get("Encrypted"),
                "publicly_accessible":    c.get("PubliclyAccessible"),
                "parameter_groups":       [
                    pg["ParameterGroupName"]
                    for pg in c.get("ClusterParameterGroups", [])
                ],
                "tags":                   {t["Key"]: t["Value"] for t in c.get("Tags", [])},
                "databases":              databases,
            })

        status.ok("redshift", {
            "clusters":  len(clusters),
            "databases": sum(len(c["databases"]) for c in clusters),
            "tables":    sum(
                sum(len(s["tables"]) for s in db["schemas"])
                for c in clusters for db in c["databases"]
            ),
        })
        return {"clusters": clusters}

    except ImportError:
        status.skip("redshift", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("redshift", "AccessDenied — need redshift:Describe*, redshift-data:* permissions")
        else:
            status.fail("redshift", str(e)[:120])
        return {}


def _crawl_rds(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        rds = _boto3_client("rds", region, profile)

        instances_raw = _safe(
            lambda: _paginate(rds, "describe_db_instances", "DBInstances"),
            default=[],
        )
        instances = [
            {
                "identifier":        i["DBInstanceIdentifier"],
                "arn":               i["DBInstanceArn"],
                "engine":            i["Engine"],
                "engine_version":    i["EngineVersion"],
                "instance_class":    i["DBInstanceClass"],
                "status":            i["DBInstanceStatus"],
                "multi_az":          i.get("MultiAZ"),
                "publicly_accessible": i.get("PubliclyAccessible"),
                "storage_type":      i.get("StorageType"),
                "allocated_storage": i.get("AllocatedStorage"),
                "db_name":           i.get("DBName"),
                "endpoint":          i.get("Endpoint", {}).get("Address"),
                "port":              i.get("Endpoint", {}).get("Port"),
                "parameter_groups":  [pg["DBParameterGroupName"] for pg in i.get("DBParameterGroups", [])],
                "security_groups":   [sg["VpcSecurityGroupId"] for sg in i.get("VpcSecurityGroups", [])],
                "tags":              {t["Key"]: t["Value"] for t in i.get("TagList", [])},
            }
            for i in (instances_raw or [])
        ]

        clusters_raw = _safe(
            lambda: _paginate(rds, "describe_db_clusters", "DBClusters"),
            default=[],
        )
        clusters = [
            {
                "identifier":     c["DBClusterIdentifier"],
                "arn":            c["DBClusterArn"],
                "engine":         c["Engine"],
                "engine_version": c.get("EngineVersion"),
                "status":         c["Status"],
                "multi_az":       c.get("MultiAZ"),
                "db_name":        c.get("DatabaseName"),
                "endpoint":       c.get("Endpoint"),
                "port":           c.get("Port"),
                "members":        [m["DBInstanceIdentifier"] for m in c.get("DBClusterMembers", [])],
                "tags":           {t["Key"]: t["Value"] for t in c.get("TagList", [])},
            }
            for c in (clusters_raw or [])
        ]

        status.ok("rds", {"instances": len(instances), "clusters": len(clusters)})
        return {"instances": instances, "clusters": clusters}

    except ImportError:
        status.skip("rds", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("rds", "AccessDenied — need rds:Describe* permissions")
        else:
            status.fail("rds", str(e)[:120])
        return {}


def _crawl_lambda(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        lm = _boto3_client("lambda", region, profile)

        fns_raw = _safe(
            lambda: _paginate(lm, "list_functions", "Functions"),
            default=[],
        )
        functions: list[dict[str, Any]] = []

        for f in (fns_raw or []):
            fname = f["FunctionName"]

            # Event source mappings (triggers)
            esm = _safe(
                lambda fn=fname: [
                    {
                        "event_source_arn": m.get("EventSourceArn"),
                        "state":            m.get("State"),
                        "batch_size":       m.get("BatchSize"),
                    }
                    for m in _paginate(
                        lm, "list_event_source_mappings", "EventSourceMappings",
                        FunctionName=fn,
                    )
                ],
                default=[],
            )

            # Env var keys only (never values)
            env_keys = list((f.get("Environment", {}) or {}).get("Variables", {}).keys())

            # Layers
            layers = [
                {"name": l.get("Arn", "").split(":")[-2], "arn": l.get("Arn")}
                for l in f.get("Layers", [])
            ]

            functions.append({
                "name":            fname,
                "arn":             f["FunctionArn"],
                "runtime":         f.get("Runtime"),
                "handler":         f.get("Handler"),
                "role_arn":        f.get("Role"),
                "memory_mb":       f.get("MemorySize"),
                "timeout_seconds": f.get("Timeout"),
                "code_size_bytes": f.get("CodeSize"),
                "description":     f.get("Description", ""),
                "env_var_keys":    env_keys,
                "layers":          layers,
                "triggers":        esm,
                "tags":            f.get("Tags", {}),
                "last_modified":   f.get("LastModified", ""),
            })

        status.ok("lambda", {"functions": len(functions)})
        return {"functions": functions}

    except ImportError:
        status.skip("lambda", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("lambda", "AccessDenied — need lambda:List* permissions")
        else:
            status.fail("lambda", str(e)[:120])
        return {}


def _crawl_sqs(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        sqs = _boto3_client("sqs", region, profile)

        queue_urls = _safe(
            lambda: sqs.list_queues().get("QueueUrls", []),
            default=[],
        )
        queues: list[dict[str, Any]] = []

        attrs_to_get = [
            "QueueArn", "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
            "VisibilityTimeout", "MaximumMessageSize",
            "MessageRetentionPeriod", "Policy",
            "RedrivePolicy", "FifoQueue",
            "ContentBasedDeduplication",
        ]

        for url in (queue_urls or []):
            attrs = _safe(
                lambda u=url: sqs.get_queue_attributes(
                    QueueUrl=u, AttributeNames=attrs_to_get
                ).get("Attributes", {}),
                default={},
            )
            tags = _safe(
                lambda u=url: sqs.list_queue_tags(QueueUrl=u).get("Tags", {}),
                default={},
            )
            dlq = None
            if attrs.get("RedrivePolicy"):
                try:
                    dlq = json.loads(attrs["RedrivePolicy"])
                except Exception:
                    pass

            queues.append({
                "name":             url.split("/")[-1],
                "url":              url,
                "arn":              attrs.get("QueueArn"),
                "messages_available": int(attrs.get("ApproximateNumberOfMessages", 0)),
                "messages_in_flight": int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0)),
                "visibility_timeout": int(attrs.get("VisibilityTimeout", 30)),
                "retention_seconds":  int(attrs.get("MessageRetentionPeriod", 345600)),
                "max_message_bytes":  int(attrs.get("MaximumMessageSize", 262144)),
                "policy":             json.loads(attrs["Policy"]) if attrs.get("Policy") else None,
                "dlq":               dlq,
                "is_fifo":           attrs.get("FifoQueue") == "true",
                "tags":              tags,
            })

        status.ok("sqs", {"queues": len(queues)})
        return {"queues": queues}

    except ImportError:
        status.skip("sqs", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("sqs", "AccessDenied — need sqs:ListQueues, sqs:GetQueueAttributes")
        else:
            status.fail("sqs", str(e)[:120])
        return {}


def _crawl_sns(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        sns = _boto3_client("sns", region, profile)

        topics_raw = _safe(
            lambda: _paginate(sns, "list_topics", "Topics"),
            default=[],
        )
        topics: list[dict[str, Any]] = []

        for t in (topics_raw or []):
            arn  = t["TopicArn"]
            name = arn.split(":")[-1]

            attrs = _safe(
                lambda a=arn: sns.get_topic_attributes(TopicArn=a).get("Attributes", {}),
                default={},
            )
            subs = _safe(
                lambda a=arn: _paginate(
                    sns, "list_subscriptions_by_topic", "Subscriptions", TopicArn=a
                ),
                default=[],
            )
            tags = _safe(
                lambda a=arn: {
                    t2["Key"]: t2["Value"]
                    for t2 in sns.list_tags_for_resource(ResourceArn=a).get("Tags", [])
                },
                default={},
            )

            topics.append({
                "name":              name,
                "arn":               arn,
                "display_name":      attrs.get("DisplayName", ""),
                "subscription_count": int(attrs.get("SubscriptionsConfirmed", 0)),
                "subscriptions":     [
                    {
                        "protocol":  s.get("Protocol"),
                        "endpoint":  s.get("Endpoint"),
                        "arn":       s.get("SubscriptionArn"),
                    }
                    for s in (subs or [])
                ],
                "policy": json.loads(attrs["Policy"]) if attrs.get("Policy") else None,
                "tags":   tags,
            })

        status.ok("sns", {"topics": len(topics)})
        return {"topics": topics}

    except ImportError:
        status.skip("sns", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("sns", "AccessDenied — need sns:List*, sns:GetTopicAttributes")
        else:
            status.fail("sns", str(e)[:120])
        return {}


def _crawl_ec2(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        ec2 = _boto3_client("ec2", region, profile)

        # VPCs
        vpcs = _safe(
            lambda: [
                {
                    "id":           v["VpcId"],
                    "cidr":         v.get("CidrBlock"),
                    "is_default":   v.get("IsDefault"),
                    "state":        v.get("State"),
                    "tags":         {t["Key"]: t["Value"] for t in v.get("Tags", [])},
                }
                for v in ec2.describe_vpcs().get("Vpcs", [])
            ],
            default=[],
        )

        # Subnets
        subnets = _safe(
            lambda: [
                {
                    "id":         s["SubnetId"],
                    "vpc_id":     s["VpcId"],
                    "cidr":       s.get("CidrBlock"),
                    "az":         s.get("AvailabilityZone"),
                    "public":     s.get("MapPublicIpOnLaunch"),
                    "available_ips": s.get("AvailableIpAddressCount"),
                    "tags":       {t["Key"]: t["Value"] for t in s.get("Tags", [])},
                }
                for s in _paginate(ec2, "describe_subnets", "Subnets")
            ],
            default=[],
        )

        # Security groups
        sgs = _safe(
            lambda: [
                {
                    "id":          sg["GroupId"],
                    "name":        sg["GroupName"],
                    "vpc_id":      sg.get("VpcId"),
                    "description": sg.get("Description"),
                    "ingress":     [
                        {
                            "protocol": r.get("IpProtocol"),
                            "from_port": r.get("FromPort"),
                            "to_port":   r.get("ToPort"),
                            "cidrs":     [ip["CidrIp"] for ip in r.get("IpRanges", [])],
                            "sources":   [sg2["GroupId"] for sg2 in r.get("UserIdGroupPairs", [])],
                        }
                        for r in sg.get("IpPermissions", [])
                    ],
                    "egress": [
                        {
                            "protocol": r.get("IpProtocol"),
                            "from_port": r.get("FromPort"),
                            "to_port":   r.get("ToPort"),
                            "cidrs":    [ip["CidrIp"] for ip in r.get("IpRanges", [])],
                        }
                        for r in sg.get("IpPermissionsEgress", [])
                    ],
                    "tags": {t["Key"]: t["Value"] for t in sg.get("Tags", [])},
                }
                for sg in _paginate(ec2, "describe_security_groups", "SecurityGroups")
            ],
            default=[],
        )

        # Instances (running + stopped, skip terminated)
        instances = _safe(
            lambda: [
                {
                    "id":           i["InstanceId"],
                    "type":         i["InstanceType"],
                    "state":        i["State"]["Name"],
                    "az":           i.get("Placement", {}).get("AvailabilityZone"),
                    "ami_id":       i.get("ImageId"),
                    "subnet_id":    i.get("SubnetId"),
                    "vpc_id":       i.get("VpcId"),
                    "private_ip":   i.get("PrivateIpAddress"),
                    "public_ip":    i.get("PublicIpAddress"),
                    "iam_profile":  i.get("IamInstanceProfile", {}).get("Arn"),
                    "security_groups": [sg["GroupId"] for sg in i.get("SecurityGroups", [])],
                    "tags":         {t["Key"]: t["Value"] for t in i.get("Tags", [])},
                }
                for r in ec2.describe_instances(
                    Filters=[{"Name": "instance-state-name",
                               "Values": ["running", "stopped", "pending"]}]
                ).get("Reservations", [])
                for i in r.get("Instances", [])
            ],
            default=[],
        )

        status.ok("ec2", {
            "vpcs": len(vpcs),
            "subnets": len(subnets),
            "security_groups": len(sgs),
            "instances": len(instances),
        })
        return {
            "vpcs":            vpcs,
            "subnets":         subnets,
            "security_groups": sgs,
            "instances":       instances,
        }

    except ImportError:
        status.skip("ec2", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("ec2", "AccessDenied — need ec2:Describe* permissions")
        else:
            status.fail("ec2", str(e)[:120])
        return {}


def _crawl_ecr(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        ecr = _boto3_client("ecr", region, profile)

        repos_raw = _safe(
            lambda: _paginate(ecr, "describe_repositories", "repositories"),
            default=[],
        )
        repos: list[dict[str, Any]] = []

        for r in (repos_raw or []):
            rname = r["repositoryName"]

            images = _safe(
                lambda: _paginate(
                    ecr, "describe_images", "imageDetails",
                    repositoryName=rname,
                ),
                default=[],
            )
            policy = _safe(
                lambda: json.loads(
                    ecr.get_repository_policy(repositoryName=rname).get("policyText", "{}")
                ),
                default=None,
            )
            lifecycle = _safe(
                lambda: json.loads(
                    ecr.get_lifecycle_policy(repositoryName=rname).get("lifecyclePolicyText", "{}")
                ),
                default=None,
            )
            scan_cfg = r.get("imageScanningConfiguration", {})

            repos.append({
                "name":          rname,
                "arn":           r["repositoryArn"],
                "uri":           r["repositoryUri"],
                "image_count":   len(images),
                "image_tags":    sorted(
                    {
                        tag
                        for img in (images or [])
                        for tag in img.get("imageTags", [])
                    }
                )[:50],
                "scan_on_push":  scan_cfg.get("scanOnPush", False),
                "encryption":    r.get("encryptionConfiguration", {}),
                "policy":        policy,
                "lifecycle":     lifecycle,
                "tags":          {t["Key"]: t["Value"] for t in r.get("tags", [])},
            })

        status.ok("ecr", {"repositories": len(repos)})
        return {"repositories": repos}

    except ImportError:
        status.skip("ecr", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("ecr", "AccessDenied — need ecr:Describe*, ecr:List* permissions")
        else:
            status.fail("ecr", str(e)[:120])
        return {}


def _crawl_glue(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        glue = _boto3_client("glue", region, profile)

        # Databases
        dbs_raw = _safe(
            lambda: _paginate(glue, "get_databases", "DatabaseList"),
            default=[],
        )
        databases: list[dict[str, Any]] = []

        for db in (dbs_raw or []):
            dbname = db["Name"]
            tables_raw = _safe(
                lambda: _paginate(glue, "get_tables", "TableList", DatabaseName=dbname),
                default=[],
            )
            tables: list[dict[str, Any]] = []
            for t in (tables_raw or []):
                cols = t.get("StorageDescriptor", {}).get("Columns", [])
                part_keys = t.get("PartitionKeys", [])
                tables.append({
                    "name":           t["Name"],
                    "type":           t.get("TableType", ""),
                    "location":       t.get("StorageDescriptor", {}).get("Location"),
                    "input_format":   t.get("StorageDescriptor", {}).get("InputFormat", "").split(".")[-1],
                    "column_count":   len(cols),
                    "columns":        [{"name": c["Name"], "type": c["Type"]} for c in cols],
                    "partition_keys": [{"name": k["Name"], "type": k["Type"]} for k in part_keys],
                    "parameters":     t.get("Parameters", {}),
                })
            databases.append({
                "name":        dbname,
                "location":    db.get("LocationUri", ""),
                "description": db.get("Description", ""),
                "table_count": len(tables),
                "tables":      tables,
            })

        # Crawlers
        crawlers_raw = _safe(
            lambda: _paginate(glue, "list_crawlers", "CrawlerNames"),
            default=[],
        )
        crawlers: list[dict[str, Any]] = []
        for cname in (crawlers_raw or []):
            c = _safe(
                lambda: glue.get_crawler(Name=cname).get("Crawler", {}),
                default={},
            )
            crawlers.append({
                "name":      cname,
                "state":     c.get("State"),
                "targets":   c.get("Targets", {}),
                "schedule":  c.get("Schedule", {}).get("ScheduleExpression"),
                "database":  c.get("DatabaseName"),
                "last_run":  str(c.get("LastCrawl", {}).get("StartTime", "")),
                "last_status": c.get("LastCrawl", {}).get("Status"),
            })

        # Jobs
        jobs_raw = _safe(
            lambda: _paginate(glue, "list_jobs", "JobNames"),
            default=[],
        )
        jobs: list[dict[str, Any]] = []
        for jname in (jobs_raw or []):
            j = _safe(
                lambda: glue.get_job(JobName=jname).get("Job", {}),
                default={},
            )
            jobs.append({
                "name":         jname,
                "role":         j.get("Role"),
                "command":      j.get("Command", {}).get("Name"),
                "glue_version": j.get("GlueVersion"),
                "worker_type":  j.get("WorkerType"),
                "num_workers":  j.get("NumberOfWorkers"),
                "timeout":      j.get("Timeout"),
            })

        status.ok("glue", {
            "databases": len(databases),
            "tables":    sum(db["table_count"] for db in databases),
            "crawlers":  len(crawlers),
            "jobs":      len(jobs),
        })
        return {"databases": databases, "crawlers": crawlers, "jobs": jobs}

    except ImportError:
        status.skip("glue", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("glue", "AccessDenied — need glue:GetDatabases, GetTables permissions")
        else:
            status.fail("glue", str(e)[:120])
        return {}


def _crawl_athena(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        ath = _boto3_client("athena", region, profile)

        # Workgroups
        wgs_raw = _safe(
            lambda: _paginate(ath, "list_work_groups", "WorkGroups"),
            default=[],
        )
        workgroups: list[dict[str, Any]] = []
        for wg in (wgs_raw or []):
            wgname = wg.get("Name", "")
            detail = _safe(
                lambda: ath.get_work_group(WorkGroup=wgname)
                           .get("WorkGroup", {})
                           .get("Configuration", {}),
                default={},
            )
            workgroups.append({
                "name":                 wgname,
                "state":                wg.get("State"),
                "output_location":      detail.get("ResultConfiguration", {}).get("OutputLocation"),
                "enforce_config":       detail.get("EnforceWorkGroupConfiguration"),
                "bytes_scanned_cutoff": detail.get("BytesScannedCutoffPerQuery"),
            })

        # Data catalogs
        catalogs = _safe(
            lambda: [
                {"name": c["CatalogName"], "type": c["Type"]}
                for c in _paginate(ath, "list_data_catalogs", "DataCatalogsSummary")
            ],
            default=[],
        )

        # Named queries
        nq_ids = _safe(
            lambda: _paginate(ath, "list_named_queries", "NamedQueryIds"),
            default=[],
        )
        named_queries: list[dict[str, Any]] = []
        for qid in (nq_ids or [])[:50]:
            q = _safe(
                lambda: ath.get_named_query(NamedQueryId=qid).get("NamedQuery", {}),
                default={},
            )
            if q:
                named_queries.append({
                    "name":        q.get("Name"),
                    "database":    q.get("Database"),
                    "description": q.get("Description", ""),
                })

        status.ok("athena", {
            "workgroups": len(workgroups),
            "named_queries": len(named_queries),
        })
        return {
            "workgroups":   workgroups,
            "data_catalogs": catalogs,
            "named_queries": named_queries,
        }

    except ImportError:
        status.skip("athena", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("athena", "AccessDenied — need athena:List* permissions")
        else:
            status.fail("athena", str(e)[:120])
        return {}


def _crawl_secretsmanager(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    """Extract secret names, ARNs, rotation config. Never extract values."""
    try:
        sm = _boto3_client("secretsmanager", region, profile)

        secrets_raw = _safe(
            lambda: _paginate(sm, "list_secrets", "SecretList"),
            default=[],
        )
        secrets = [
            {
                "name":               s["Name"],
                "arn":                s["ARN"],
                "description":        s.get("Description", ""),
                "rotation_enabled":   s.get("RotationEnabled", False),
                "rotation_lambda_arn": s.get("RotationLambdaARN"),
                "last_rotated":       str(s.get("LastRotatedDate", "")),
                "last_accessed":      str(s.get("LastAccessedDate", "")),
                "tags":               {t["Key"]: t["Value"] for t in s.get("Tags", [])},
                # value: NEVER extracted
            }
            for s in (secrets_raw or [])
        ]

        status.ok("secretsmanager", {"secrets": len(secrets)})
        return {"secrets": secrets}

    except ImportError:
        status.skip("secretsmanager", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("secretsmanager", "AccessDenied — need secretsmanager:ListSecrets")
        else:
            status.fail("secretsmanager", str(e)[:120])
        return {}


def _crawl_ssm(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    """Extract parameter names, types, tiers. Never extract values."""
    try:
        ssm = _boto3_client("ssm", region, profile)

        params_raw = _safe(
            lambda: _paginate(ssm, "describe_parameters", "Parameters"),
            default=[],
        )
        params = [
            {
                "name":        p["Name"],
                "type":        p["Type"],   # String / StringList / SecureString
                "tier":        p.get("Tier", "Standard"),
                "data_type":   p.get("DataType", "text"),
                "description": p.get("Description", ""),
                "last_modified": str(p.get("LastModifiedDate", "")),
                # value: NEVER extracted
            }
            for p in (params_raw or [])
        ]

        status.ok("ssm", {"parameters": len(params)})
        return {"parameters": params}

    except ImportError:
        status.skip("ssm", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("ssm", "AccessDenied — need ssm:DescribeParameters")
        else:
            status.fail("ssm", str(e)[:120])
        return {}


def _crawl_cloudwatch(region: str, profile: str | None, status: CrawlerStatus) -> dict[str, Any]:
    try:
        cw  = _boto3_client("cloudwatch", region, profile)
        cwl = _boto3_client("logs",       region, profile)

        # Alarms
        alarms_raw = _safe(
            lambda: _paginate(cw, "describe_alarms", "MetricAlarms"),
            default=[],
        )
        alarms = [
            {
                "name":              a["AlarmName"],
                "arn":               a.get("AlarmArn"),
                "state":             a.get("StateValue"),
                "metric":            a.get("MetricName"),
                "namespace":         a.get("Namespace"),
                "threshold":         a.get("Threshold"),
                "comparison":        a.get("ComparisonOperator"),
                "evaluation_periods": a.get("EvaluationPeriods"),
                "treat_missing":     a.get("TreatMissingData"),
                "actions_ok":        a.get("OKActions", []),
                "actions_alarm":     a.get("AlarmActions", []),
                "description":       a.get("AlarmDescription", ""),
            }
            for a in (alarms_raw or [])
        ]

        # Log groups
        lgs_raw = _safe(
            lambda: _paginate(cwl, "describe_log_groups", "logGroups"),
            default=[],
        )
        log_groups = [
            {
                "name":              lg["logGroupName"],
                "arn":               lg.get("arn"),
                "retention_days":    lg.get("retentionInDays"),
                "stored_bytes":      lg.get("storedBytes", 0),
                "created_at":        str(lg.get("creationTime", "")),
                "kms_key_id":        lg.get("kmsKeyId"),
            }
            for lg in (lgs_raw or [])[:_MAX_LOG_GROUPS]
        ]

        status.ok("cloudwatch", {
            "alarms":      len(alarms),
            "log_groups":  len(log_groups),
        })
        return {"alarms": alarms, "log_groups": log_groups}

    except ImportError:
        status.skip("cloudwatch", "boto3 not installed")
        return {}
    except Exception as e:
        if "AccessDenied" in str(e):
            status.skip("cloudwatch", "AccessDenied — need cloudwatch:Describe*, logs:Describe*")
        else:
            status.fail("cloudwatch", str(e)[:120])
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_CRAWLERS = {
    "s3":             _crawl_s3,
    "iam":            _crawl_iam,
    "eks":            _crawl_eks,
    "redshift":       _crawl_redshift,
    "rds":            _crawl_rds,
    "lambda":         _crawl_lambda,
    "sqs":            _crawl_sqs,
    "sns":            _crawl_sns,
    "ec2":            _crawl_ec2,
    "ecr":            _crawl_ecr,
    "glue":           _crawl_glue,
    "athena":         _crawl_athena,
    "secretsmanager": _crawl_secretsmanager,
    "ssm":            _crawl_ssm,
    "cloudwatch":     _crawl_cloudwatch,
}


# ─────────────────────────────────────────────────────────────────────────────
# Markdown generator (no LLM — pure formatting)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_md(report: dict[str, Any]) -> str:
    meta   = report.get("meta", {})
    svcs   = report.get("services", {})
    cstat  = report.get("crawler_status", {})

    L: list[str] = [
        "# Live Discovery Map",
        "",
        f"**Generated:** {meta.get('run_at', '')}  ",
        f"**Region:** {meta.get('region', '')}  ",
        f"**Account:** {meta.get('account_id', 'unknown')}  ",
        f"**Services crawled:** {', '.join(meta.get('services_requested', []))}  ",
        "",
        "---",
        "",
        "## Crawler Status", "",
        "| Service | Status | Resources |",
        "|---------|--------|-----------|",
    ]
    for svc in _ALL_SERVICES:
        s     = cstat.get(svc, {})
        st    = s.get("status", "not_run")
        icon  = {"ok": "✅", "skipped": "⏭️", "failed": "❌", "not_run": "—"}.get(st, "—")
        counts = s.get("counts", {})
        counts_str = "  ".join(f"{k}={v}" for k, v in counts.items()) if counts else (
            s.get("reason") or s.get("error") or "—"
        )
        L.append(f"| {svc} | {icon} {st} | {counts_str} |")
    L += [""]

    # ── Per-service summary sections ─────────────────────────────────────────
    def _section(title: str, lines: list[str]) -> None:
        L.extend([f"## {title}", ""] + lines + [""])

    if "s3" in svcs and svcs["s3"].get("buckets"):
        rows = []
        for b in svcs["s3"]["buckets"]:
            rows.append(
                f"| `{b['name']}` | {b['region']} | {b['object_count']} | "
                f"{b['total_size_bytes']:,} | {b['versioning']} | "
                f"{'✅' if b.get('encryption') else '❌'} |"
            )
        _section("S3 Buckets", [
            "| Name | Region | Objects | Size (bytes) | Versioning | Encrypted |",
            "|------|--------|---------|--------------|------------|-----------|",
        ] + rows)

    if "iam" in svcs:
        iam_d = svcs["iam"]
        lines = [
            f"**Roles ({len(iam_d.get('roles', []))}):**",
        ]
        for r in iam_d.get("roles", []):
            principals = []
            for stmt in r.get("trust_policy", {}).get("Statement", []):
                p = stmt.get("Principal", {})
                if isinstance(p, dict):
                    principals += p.get("Service", []) + p.get("AWS", [])
                elif isinstance(p, str):
                    principals.append(p)
            trust_str = ", ".join(principals[:3])
            lines.append(f"- `{r['name']}` — trusts: {trust_str or '—'}")
        lines += [
            "",
            f"**Users ({len(iam_d.get('users', []))}):**",
        ]
        for u in iam_d.get("users", []):
            key_count = len(u.get("access_keys", []))
            lines.append(f"- `{u['username']}` — {key_count} access key(s)")
        _section("IAM", lines)

    if "eks" in svcs and svcs["eks"].get("clusters"):
        lines = []
        for c in svcs["eks"]["clusters"]:
            lines.append(f"### Cluster: `{c['name']}`")
            lines.append(f"- Version: {c['kubernetes_version']}  Status: {c['status']}")
            lines.append(f"- OIDC issuer: {c.get('oidc_issuer') or '—'}")
            for ng in c.get("node_groups", []):
                lines.append(
                    f"- Node group `{ng['name']}`: {ng['instance_types']}  "
                    f"desired={ng['scaling'].get('desired')}  "
                    f"max={ng['scaling'].get('max')}"
                )
        _section("EKS", lines)

    if "redshift" in svcs and svcs["redshift"].get("clusters"):
        lines = []
        for c in svcs["redshift"]["clusters"]:
            lines.append(f"### Cluster: `{c['identifier']}`")
            lines.append(f"- Node type: {c['node_type']}  Nodes: {c['number_of_nodes']}")
            for db in c.get("databases", []):
                schema_summary = ", ".join(
                    f"{s['name']}({s['table_count']} tables)"
                    for s in db.get("schemas", [])
                )
                lines.append(f"- DB `{db['name']}`: {schema_summary}")
        _section("Redshift", lines)

    if "glue" in svcs and svcs["glue"].get("databases"):
        lines = []
        for db in svcs["glue"]["databases"]:
            lines.append(f"- `{db['name']}`: {db['table_count']} table(s)  {db.get('location','')}")
        _section("Glue Catalog", lines)

    if "lambda" in svcs and svcs["lambda"].get("functions"):
        rows = []
        for f in svcs["lambda"]["functions"]:
            rows.append(
                f"| `{f['name']}` | {f.get('runtime','—')} | "
                f"{f.get('memory_mb','—')}MB | {f.get('timeout_seconds','—')}s |"
            )
        _section("Lambda", [
            "| Function | Runtime | Memory | Timeout |",
            "|----------|---------|--------|---------|",
        ] + rows)

    if "sqs" in svcs and svcs["sqs"].get("queues"):
        rows = []
        for q in svcs["sqs"]["queues"]:
            rows.append(
                f"| `{q['name']}` | {q['messages_available']} | "
                f"{q['messages_in_flight']} | {'Yes' if q.get('dlq') else 'No'} |"
            )
        _section("SQS Queues", [
            "| Queue | Available | In-flight | DLQ |",
            "|-------|-----------|-----------|-----|",
        ] + rows)

    if "ec2" in svcs:
        ec2_d = svcs["ec2"]
        lines = [
            f"**VPCs:** {len(ec2_d.get('vpcs', []))}",
            f"**Subnets:** {len(ec2_d.get('subnets', []))}",
            f"**Security groups:** {len(ec2_d.get('security_groups', []))}",
            f"**Instances:** {len(ec2_d.get('instances', []))}",
        ]
        for i in ec2_d.get("instances", []):
            name = i.get("tags", {}).get("Name", i["id"])
            lines.append(f"- `{name}` ({i['id']}) — {i['type']}  {i['state']}")
        _section("EC2", lines)

    if "ecr" in svcs and svcs["ecr"].get("repositories"):
        rows = []
        for r in svcs["ecr"]["repositories"]:
            rows.append(f"| `{r['name']}` | {r['image_count']} | {r['uri']} |")
        _section("ECR Repositories", [
            "| Repository | Images | URI |",
            "|------------|--------|-----|",
        ] + rows)

    if "cloudwatch" in svcs:
        cw_d = svcs["cloudwatch"]
        alarms = cw_d.get("alarms", [])
        alarm_states: dict[str, int] = {}
        for a in alarms:
            s2 = a.get("state", "UNKNOWN")
            alarm_states[s2] = alarm_states.get(s2, 0) + 1
        lines = [
            f"**Alarms ({len(alarms)}):** "
            + "  ".join(f"{s2}={c}" for s2, c in sorted(alarm_states.items())),
            f"**Log groups:** {len(cw_d.get('log_groups', []))}",
        ]
        for a in alarms:
            if a.get("state") == "ALARM":
                lines.append(f"- 🔴 `{a['name']}` — {a.get('metric')} {a.get('comparison')} {a.get('threshold')}")
        _section("CloudWatch", lines)

    # ── Next step ─────────────────────────────────────────────────────────────
    slug = os.environ.get("PIPELINE_PROJECT", "<name>")
    L += [
        "---", "",
        "## Next steps", "",
        "Cross-check with IaC (if available):",
        "```",
        f"python toolkits/devops_mlops/infra_absorber.py --project {slug}",
        f"python toolkits/devops_mlops/config_consistency_checker.py --project {slug} --mode drift",
        "```", "",
    ]

    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Account ID helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_account_id(region: str, profile: str | None) -> str:
    try:
        sts = _boto3_client("sts", region, profile)
        return sts.get_caller_identity().get("Account", "unknown")
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Log writer
# ─────────────────────────────────────────────────────────────────────────────

def _append_log(
    services:   list[str],
    run_at:     str,
    region:     str,
    account_id: str,
    cstatus:    CrawlerStatus,
) -> None:
    log = _log_path()
    existing: list[dict[str, Any]] = []
    if log.exists():
        try:
            track_read(log)
            data     = json.loads(log.read_text(encoding="utf-8"))
            existing = data if isinstance(data, list) else data.get("entries", [])
        except Exception:
            pass

    entry: dict[str, Any] = {
        "run_at":     run_at,
        "region":     region,
        "account_id": account_id,
        "services":   services,
        "results":    {
            svc: d.get("status", "not_run")
            for svc, d in cstatus.to_dict().items()
        },
        "resource_counts": {
            svc: d.get("counts", {})
            for svc, d in cstatus.to_dict().items()
            if d.get("status") == "ok"
        },
    }
    existing.append(entry)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps({"entries": existing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(log)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live_discovery.py",
        description="Crawl live AWS account metadata — service by service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            Available services: {', '.join(_ALL_SERVICES)}

            Examples:
              python live_discovery.py --project my-co
              python live_discovery.py --project my-co --services s3 iam eks
              python live_discovery.py --project my-co --region us-east-1
              python live_discovery.py --project my-co --profile prod-readonly
              python live_discovery.py --project my-co --dry-run
              python live_discovery.py --project my-co --show-last
        """),
    )
    p.add_argument("--project",    default=os.environ.get("PIPELINE_PROJECT"),
                   help="Project slug. Sets PIPELINE_PROJECT.")
    p.add_argument("--services",   nargs="+", choices=list(_ALL_SERVICES),
                   default=None,
                   help=f"Services to crawl. Default: all ({len(_ALL_SERVICES)} services).")
    p.add_argument("--region",     default=_DEFAULT_REGION,
                   help=f"AWS region (default: {_DEFAULT_REGION}).")
    p.add_argument("--all-regions", action="store_true",
                   help="Crawl all enabled regions (slower).")
    p.add_argument("--profile",    default=os.environ.get("AWS_PROFILE"),
                   help="AWS CLI profile name.")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print what would be crawled without making API calls.")
    p.add_argument("--show-last",  action="store_true",
                   help="Print most recent discovery_map.md and exit.")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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

    # --show-last
    if args.show_last:
        p = _md_path()
        if not p.exists():
            print("[live_discovery] No previous report. Run without --show-last first.")
            sys.exit(1)
        track_read(p)
        print(p.read_text())
        sys.exit(0)

    if args.all_regions:
        print("[live_discovery][warn] --all-regions is not yet implemented — crawling single region only.")
        print(f"  Region: {args.region}")
        print()

    services = args.services or list(_ALL_SERVICES)
    run_at   = _now_iso()

    print("=" * 60)
    print("  LIVE DISCOVERY")
    print("=" * 60)
    print(f"  Region:   {args.region}")
    print(f"  Profile:  {args.profile or '(default)'}")
    print(f"  Services: {', '.join(services)}")
    print()

    if args.dry_run:
        print("[live_discovery] DRY RUN — would crawl:")
        for svc in services:
            print(f"  {svc}")
        sys.exit(0)

    # ── Boto3 availability check ──────────────────────────────────────────────
    try:
        import boto3  # noqa: F401  type: ignore
    except ImportError:
        print("[live_discovery][error] boto3 not installed.")
        print("  pip install boto3")
        sys.exit(2)

    account_id = _get_account_id(args.region, args.profile)
    print(f"  Account:  {account_id}")
    print()

    # ── Crawl ─────────────────────────────────────────────────────────────────
    cstatus  = CrawlerStatus()
    svc_data: dict[str, Any] = {}
    exit_code = 0

    try:
        for svc in services:
            print(f"[live_discovery] {svc} …")
            crawler       = _CRAWLERS[svc]
            svc_data[svc] = crawler(args.region, args.profile, cstatus)

        print()

        # ── Assemble report ───────────────────────────────────────────────────
        report: dict[str, Any] = {
            "meta": {
                "run_at":             run_at,
                "region":             args.region,
                "account_id":         account_id,
                "services_requested": services,
                "discovery_version":  1,
            },
            "services":       svc_data,
            "crawler_status": cstatus.to_dict(),
        }
        report_md = _generate_md(report)

        # ── Write ─────────────────────────────────────────────────────────────
        _discovery_dir().mkdir(parents=True, exist_ok=True)

        json_path = _json_path()
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        track_write(json_path)

        md_out = apply_md_header(
            content = report_md,
            path    = _md_path(),
            owner   = "live_discovery.py",
        )
        _md_path().write_text(md_out, encoding="utf-8")
        track_write(_md_path())

        _append_log(services, run_at, args.region, account_id, cstatus)

        print(f"  Written:  {json_path}")
        print(f"  Written:  {_md_path()}")
        print(f"  Appended: {_log_path()}")

    except KeyboardInterrupt:
        print("\n[live_discovery] Interrupted.")
        exit_code = 130
    except Exception as exc:
        print(f"[live_discovery][error] {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        exit_code = 1
    finally:
        print()
        print_artifact_summary("[live_discovery]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
