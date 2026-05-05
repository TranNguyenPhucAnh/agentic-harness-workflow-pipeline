"""
test_absorber.py
================
Unit + integration tests for 01_absorber.py.

Test groups:
  A. AbsorberIgnoreRules  — parsing + mode_for()
  B. File scanning        — scan_files(), _should_skip_dir/file
  C. Content extraction   — key-only redaction, signature extraction, caching
  D. Config inventory     — build_config_map(), service/env detection
  E. Git crawl            — _parse_git_log(), _scope_to_git_args(), build_blame_map()
  F. Context assembly     — _build_extraction_context() size limits
  G. Integration          — full pipeline dry-run on a temp codebase fixture

Run:
    python test_absorber.py                # all tests
    python test_absorber.py TestA          # single group
    python test_absorber.py -v             # verbose

No API keys needed. No network calls. No LLM calls.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Load module under test ─────────────────────────────────────────────────────
_MOD_PATH = Path(__file__).parent / "01_absorber.py"

# Stub artifacts.paths before import
_mock_paths = MagicMock()
_mock_paths.ROOT         = Path("/tmp/absorber_test_root")
_mock_paths.CACHE_DIR    = Path("/tmp/absorber_test/cache")
_mock_paths.CURRENT_DIR  = Path("/tmp/absorber_test/current")
_mock_paths.HISTORY_DIR  = Path("/tmp/absorber_test/history")
_mock_paths.ensure_dirs  = MagicMock()

with patch.dict("sys.modules", {
    "artifacts":       MagicMock(),
    "artifacts.paths": _mock_paths,
    "httpx":           MagicMock(),
}):
    _spec = importlib.util.spec_from_file_location("absorber", str(_MOD_PATH))
    mod   = types.ModuleType("absorber")
    mod.__file__ = str(_MOD_PATH)
    sys.modules["absorber"] = mod
    _spec.loader.exec_module(mod)

# Convenience aliases
AbsorberIgnoreRules      = mod.AbsorberIgnoreRules
scan_files               = mod.scan_files
extract_content          = mod.extract_content
build_config_map         = mod.build_config_map
build_blame_map          = mod.build_blame_map
_parse_git_log           = mod._parse_git_log
_scope_to_git_args       = mod._scope_to_git_args
_should_skip_dir         = mod._should_skip_dir
_should_skip_file        = mod._should_skip_file
_redact_json             = mod._redact_json
_redact_yaml             = mod._redact_yaml
_redact_toml             = mod._redact_toml
_redact_env              = mod._redact_env
_extract_python_signatures = mod._extract_python_signatures
_extract_ts_signatures   = mod._extract_ts_signatures
_build_extraction_context = mod._build_extraction_context
_file_hash               = mod._file_hash
_load_cache              = mod._load_cache
_save_cache              = mod._save_cache
_resolve_model           = mod._resolve_model
_detect_language         = mod._detect_language
ABSORBER_CACHE           = Path("/tmp/absorber_test/cache/absorber_cache.json")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_tmp_codebase() -> Path:
    """
    Create a minimal but realistic temp codebase for integration tests.
    Structure:
      src/
        app.ts               — TypeScript source
        utils/helper.py      — Python source
      config/
        appsettings.json     — config with secrets (auto key-only)
        .env                 — env file
      migrations/
        001_init.sql         — should be signature-only if absorber.ignored says so
      node_modules/          — should be skipped (builtin)
        lodash/index.js
      dist/
        bundle.js            — should be skipped (builtin)
      absorber.ignored       — project rules
    """
    root = Path(tempfile.mkdtemp(prefix="absorber_test_"))

    # src/app.ts
    (root / "src").mkdir()
    (root / "src" / "app.ts").write_text(textwrap.dedent("""
        export interface User {
          id: string;
          email: string;
        }

        export class UserService {
          constructor(private db: Database) {}

          async findById(id: string): Promise<User | null> {
            return this.db.query('SELECT * FROM users WHERE id = $1', [id]);
          }

          async create(email: string): Promise<User> {
            return this.db.query('INSERT INTO users(email) VALUES ($1)', [email]);
          }
        }

        export function hashPassword(plain: string): string {
          return bcrypt.hash(plain, 10);
        }
    """))

    # src/utils/helper.py
    (root / "src" / "utils").mkdir()
    (root / "src" / "utils" / "helper.py").write_text(textwrap.dedent("""
        import hashlib
        from typing import Optional


        class CacheManager:
            \"\"\"Manages in-memory cache with TTL support.\"\"\"

            def __init__(self, ttl_seconds: int = 300) -> None:
                self._store: dict = {}
                self.ttl = ttl_seconds

            def get(self, key: str) -> Optional[str]:
                \"\"\"Retrieve a value by key.\"\"\"
                return self._store.get(key)

            def set(self, key: str, value: str) -> None:
                self._store[key] = value


        def compute_hash(data: str) -> str:
            return hashlib.sha256(data.encode()).hexdigest()
    """))

    # config/appsettings.json — auto-promoted to key-only (contains "settings")
    (root / "config").mkdir()
    (root / "config" / "appsettings.json").write_text(json.dumps({
        "database": {
            "host": "prod-postgres.company.internal",
            "port": 5432,
            "password": "super-secret-db-password",
            "name": "myapp_prod",
        },
        "auth": {
            "jwt_secret": "my-jwt-secret-key-never-share",
            "oauth_client_id": "client-12345",
        },
        "features": {
            "dark_mode": True,
            "beta_users": False,
        },
        "redis_url": "redis://default:password@prod-redis:6379",
    }, indent=2))

    # config/.env
    (root / "config" / ".env").write_text(textwrap.dedent("""
        DATABASE_URL=postgres://user:password@localhost:5432/mydb
        JWT_SECRET=supersecretkey
        AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
        AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
        PORT=3000
        DEBUG=true
        REDIS_URL=redis://default:secret@prod-redis:6379
    """))

    # migrations/ — signature-only via absorber.ignored
    (root / "migrations").mkdir()
    (root / "migrations" / "001_init.sql").write_text(textwrap.dedent("""
        CREATE TABLE users (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          email VARCHAR(255) UNIQUE NOT NULL,
          created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX idx_users_email ON users(email);
    """))

    # node_modules/ — should be auto-skipped
    (root / "node_modules" / "lodash").mkdir(parents=True)
    (root / "node_modules" / "lodash" / "index.js").write_text("// lodash")

    # dist/ — should be auto-skipped
    (root / "dist").mkdir()
    (root / "dist" / "bundle.js").write_text("// minified bundle")

    # absorber.ignored
    (root / "absorber.ignored").write_text(textwrap.dedent("""
        # Standard skips
        *.lock
        *.log

        # Key-only — extract structure only
        [key-only]
        config/**

        # Signature-only — entry points and exports only
        [signature-only]
        migrations/**
    """))

    return root


# ─────────────────────────────────────────────────────────────────────────────
# A. AbsorberIgnoreRules
# ─────────────────────────────────────────────────────────────────────────────

class TestAAbsorberIgnoreRules(unittest.TestCase):
    """Tests for absorber.ignored parsing and mode_for()."""

    def _make_rules(self, content: str) -> AbsorberIgnoreRules:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ignored", delete=False
        ) as f:
            f.write(content)
            f.flush()   # ensure bytes reach disk before _parse() reads the file
            return AbsorberIgnoreRules(Path(f.name))

    def test_empty_file_gives_full_mode(self):
        rules = self._make_rules("")
        self.assertEqual(rules.mode_for("src/app.ts"), "full")

    def test_skip_pattern_exact(self):
        # *.lock matches files that literally end in ".lock" (yarn.lock, Gemfile.lock)
        # package-lock.json ends in ".json" — to skip it, use "package-lock.json" exactly
        rules = self._make_rules("*.lock\npackage-lock.json\ndist/\n")
        self.assertEqual(rules.mode_for("package-lock.json"), "skip")
        self.assertEqual(rules.mode_for("yarn.lock"), "skip")

    def test_skip_pattern_glob(self):
        rules = self._make_rules("vendor/**\n")
        self.assertEqual(rules.mode_for("vendor/lib/foo.go"), "skip")

    def test_key_only_section(self):
        rules = self._make_rules(
            "[key-only]\n**/appsettings*.json\n**/.env*\n"
        )
        self.assertEqual(rules.mode_for("config/appsettings.json"), "key-only")
        self.assertEqual(rules.mode_for("config/appsettings.production.json"), "key-only")
        self.assertEqual(rules.mode_for("config/.env.local"), "key-only")
        self.assertEqual(rules.mode_for("src/app.ts"), "full")

    def test_signature_only_section(self):
        rules = self._make_rules(
            "[signature-only]\nmigrations/**\nsrc/generated/**\n"
        )
        self.assertEqual(rules.mode_for("migrations/001_init.sql"), "signature-only")
        self.assertEqual(rules.mode_for("src/generated/types.ts"), "signature-only")
        self.assertEqual(rules.mode_for("src/app.ts"), "full")

    def test_signature_only_takes_precedence_over_key_only(self):
        rules = self._make_rules(
            "[key-only]\nmigrations/**\n[signature-only]\nmigrations/**\n"
        )
        # signature-only checked first in mode_for()
        self.assertEqual(rules.mode_for("migrations/001.sql"), "signature-only")

    def test_comments_and_blank_lines_ignored(self):
        rules = self._make_rules(
            "# This is a comment\n\n*.lock\n\n# Another comment\n"
        )
        self.assertEqual(len(rules.skip_patterns), 1)
        self.assertEqual(rules.skip_patterns[0], "*.lock")

    def test_missing_file_gives_full_mode(self):
        rules = AbsorberIgnoreRules(Path("/nonexistent/absorber.ignored"))
        self.assertEqual(rules.mode_for("anything/file.py"), "full")

    def test_mixed_sections(self):
        rules = self._make_rules(textwrap.dedent("""
            *.log
            *.lock

            [key-only]
            config/**
            secrets/**

            [signature-only]
            src/generated/**
            dist/**
        """))
        self.assertEqual(rules.mode_for("app.log"), "skip")
        self.assertEqual(rules.mode_for("config/db.yaml"), "key-only")
        self.assertEqual(rules.mode_for("secrets/vault.json"), "key-only")
        self.assertEqual(rules.mode_for("src/generated/client.ts"), "signature-only")
        self.assertEqual(rules.mode_for("src/app.ts"), "full")


# ─────────────────────────────────────────────────────────────────────────────
# B. File scanning
# ─────────────────────────────────────────────────────────────────────────────

class TestBFileScanning(unittest.TestCase):
    """Tests for scan_files() and builtin skip rules."""

    @classmethod
    def setUpClass(cls):
        cls.root = make_tmp_codebase()
        cls.rules = AbsorberIgnoreRules(cls.root / "absorber.ignored")

    def test_builtin_dirs_are_skipped(self):
        inventory = scan_files(self.root, self.rules)
        rel_paths = [e["rel_path"] for e in inventory]
        # node_modules/ and dist/ should not appear
        self.assertFalse(
            any("node_modules" in p for p in rel_paths),
            "node_modules should be skipped"
        )
        self.assertFalse(
            any(p.startswith("dist/") for p in rel_paths),
            "dist/ should be skipped"
        )

    def test_source_files_are_included(self):
        inventory = scan_files(self.root, self.rules)
        rel_paths = [e["rel_path"] for e in inventory]
        self.assertIn("src/app.ts", rel_paths)
        self.assertIn("src/utils/helper.py", rel_paths)

    def test_config_files_get_key_only_mode(self):
        inventory = scan_files(self.root, self.rules)
        by_path = {e["rel_path"]: e for e in inventory}
        self.assertEqual(by_path["config/appsettings.json"]["mode"], "key-only")
        self.assertEqual(by_path["config/.env"]["mode"], "key-only")

    def test_migrations_get_signature_only_mode(self):
        inventory = scan_files(self.root, self.rules)
        by_path = {e["rel_path"]: e for e in inventory}
        self.assertEqual(
            by_path["migrations/001_init.sql"]["mode"], "signature-only"
        )

    def test_language_detection(self):
        inventory = scan_files(self.root, self.rules)
        by_path = {e["rel_path"]: e for e in inventory}
        self.assertEqual(by_path["src/app.ts"]["lang"], "TypeScript")
        self.assertEqual(by_path["src/utils/helper.py"]["lang"], "Python")
        self.assertEqual(by_path["config/appsettings.json"]["lang"], "JSON")

    def test_absorber_ignored_file_itself_not_in_inventory(self):
        """absorber.ignored has no source extension — should not appear."""
        inventory = scan_files(self.root, self.rules)
        rel_paths = [e["rel_path"] for e in inventory]
        self.assertNotIn("absorber.ignored", rel_paths)

    def test_should_skip_dir_builtin(self):
        for d in ("node_modules", "vendor", ".git", "dist", "__pycache__"):
            self.assertTrue(_should_skip_dir(d), f"{d} should be skipped")

    def test_should_skip_dir_hidden(self):
        self.assertTrue(_should_skip_dir(".hidden"))
        self.assertTrue(_should_skip_dir(".vscode"))

    def test_should_not_skip_src(self):
        self.assertFalse(_should_skip_dir("src"))
        self.assertFalse(_should_skip_dir("config"))
        self.assertFalse(_should_skip_dir("migrations"))

    def test_should_skip_test_files(self):
        for f in ("app_test.go", "foo.test.ts", "bar.spec.tsx",
                  "test_utils.py", "UserTest.java"):
            self.assertTrue(_should_skip_file(f), f"{f} should be skipped")

    def test_should_not_skip_source_files(self):
        for f in ("app.ts", "helper.py", "main.go", "Service.java"):
            self.assertFalse(_should_skip_file(f), f"{f} should not be skipped")


# ─────────────────────────────────────────────────────────────────────────────
# C. Content extraction & redaction
# ─────────────────────────────────────────────────────────────────────────────

class TestCContentExtraction(unittest.TestCase):
    """Tests for redaction functions and signature extraction."""

    # ── JSON redaction ────────────────────────────────────────────────────────

    def test_redact_json_strings_replaced(self):
        raw = json.dumps({"host": "prod.server.com", "password": "secret123"})
        result = _redact_json(raw)
        self.assertNotIn("prod.server.com", result)
        self.assertNotIn("secret123", result)
        self.assertIn('"host"', result)
        self.assertIn('"password"', result)
        self.assertIn("<redacted>", result)

    def test_redact_json_keeps_numeric_primitives(self):
        """Port numbers, counts, timeouts are not secrets — keep them."""
        raw = json.dumps({"port": 5432, "timeout": 30, "enabled": True})
        result = _redact_json(raw)
        self.assertIn("5432", result)
        self.assertIn("30", result)
        self.assertIn("True", result)

    def test_redact_json_nested(self):
        raw = json.dumps({
            "database": {
                "host": "db.internal",
                "password": "supersecret",
                "port": 5432,
            }
        })
        result = _redact_json(raw)
        self.assertNotIn("db.internal", result)
        self.assertNotIn("supersecret", result)
        self.assertIn("5432", result)
        self.assertIn('"database"', result)

    def test_redact_json_array_summarized(self):
        raw = json.dumps({"allowed_origins": ["https://app.com", "https://api.com"]})
        result = _redact_json(raw)
        self.assertIn("2 item(s)", result)
        self.assertNotIn("https://app.com", result)

    def test_redact_json_empty_object(self):
        result = _redact_json("{}")
        self.assertEqual(result.strip(), "{}")

    def test_redact_json_invalid_falls_back_to_generic(self):
        """Malformed JSON should not crash — fallback to generic redaction."""
        result = _redact_json("this is not json at all")
        self.assertIsInstance(result, str)

    # ── YAML redaction ────────────────────────────────────────────────────────

    def test_redact_yaml_values_replaced(self):
        yaml = textwrap.dedent("""
            database:
              host: prod.server.com
              password: supersecret
              port: 5432
        """)
        result = _redact_yaml(yaml)
        self.assertNotIn("prod.server.com", result)
        self.assertNotIn("supersecret", result)
        self.assertIn("host:", result)
        self.assertIn("password:", result)

    def test_redact_yaml_preserves_comments(self):
        yaml = "# This is a comment\nkey: value\n"
        result = _redact_yaml(yaml)
        self.assertIn("# This is a comment", result)

    # ── TOML redaction ────────────────────────────────────────────────────────

    def test_redact_toml_values_replaced(self):
        toml = textwrap.dedent("""
            [database]
            host = "prod.server.com"
            password = "secret"

            [server]
            port = 8080
        """)
        result = _redact_toml(toml)
        self.assertNotIn("prod.server.com", result)
        self.assertNotIn("secret", result)
        self.assertIn("[database]", result)
        self.assertIn("[server]", result)

    # ── ENV redaction ─────────────────────────────────────────────────────────

    def test_redact_env_values_replaced(self):
        env = textwrap.dedent("""
            DATABASE_URL=postgres://user:password@localhost:5432/mydb
            JWT_SECRET=supersecretkey
            PORT=3000
            DEBUG=true
        """)
        result = _redact_env(env)
        self.assertNotIn("postgres://", result)
        self.assertNotIn("supersecretkey", result)
        self.assertIn("DATABASE_URL=<redacted>", result)
        self.assertIn("JWT_SECRET=<redacted>", result)
        self.assertIn("PORT=<redacted>", result)

    def test_redact_env_comments_preserved(self):
        env = "# DB settings\nDATABASE_URL=secret\n"
        result = _redact_env(env)
        self.assertIn("# DB settings", result)

    # ── Python signature extraction ───────────────────────────────────────────

    def test_python_signatures_extracts_functions(self):
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(textwrap.dedent("""
                def process_order(order_id: str, user: User) -> Order:
                    \"\"\"Process and validate an order.\"\"\"
                    pass

                async def fetch_data(url: str, timeout: int = 30) -> dict:
                    pass

                def _private_helper():
                    pass
            """))
            path = Path(f.name)

        result = _extract_python_signatures(path)
        self.assertIn("def process_order", result)
        self.assertIn("async def fetch_data", result)
        self.assertIn("def _private_helper", result)
        # Docstring should be included (truncated)
        self.assertIn("Process and validate an order", result)

    def test_python_signatures_extracts_classes(self):
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(textwrap.dedent("""
                class OrderService(BaseService):
                    \"\"\"Handles order business logic.\"\"\"

                    def create(self, data: dict) -> Order:
                        pass
            """))
            path = Path(f.name)

        result = _extract_python_signatures(path)
        self.assertIn("class OrderService", result)
        self.assertIn("BaseService", result)

    def test_python_signatures_no_implementation_details(self):
        """Signature extraction must not include function bodies."""
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(textwrap.dedent("""
                SECRET_KEY = "this-should-not-appear"

                def get_secret():
                    return "also-should-not-appear"
            """))
            path = Path(f.name)

        result = _extract_python_signatures(path)
        # The return value in body should not appear in signatures
        self.assertNotIn("also-should-not-appear", result)

    # ── TypeScript signature extraction ───────────────────────────────────────

    def test_ts_signatures_exports(self):
        with tempfile.NamedTemporaryFile(
            suffix=".ts", mode="w", delete=False
        ) as f:
            f.write(textwrap.dedent("""
                export interface UserRepository {
                  findById(id: string): Promise<User>;
                }

                export class UserService {
                  constructor(private repo: UserRepository) {}
                }

                export function hashPassword(plain: string): string {
                  return bcrypt.hash(plain);
                }

                const internalHelper = () => 'internal';
            """))
            path = Path(f.name)

        result = _extract_ts_signatures(path)
        self.assertIn("export interface UserRepository", result)
        self.assertIn("export class UserService", result)
        self.assertIn("export function hashPassword", result)

    # ── Change detection cache ────────────────────────────────────────────────

    def test_cache_hit_returns_cached_content(self):
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write("def hello(): pass\n")
            path = Path(f.name)

        h = _file_hash(path)
        cache = {
            "test/file.py": {
                "hash": h, "mode": "full",
                "content": "cached_content",
                "lang": "Python", "size": 20,
            }
        }
        entry = {
            "rel_path": "test/file.py",
            "abs_path": str(path),
            "ext": ".py", "size": 20,
            "mode": "full", "lang": "Python",
        }
        content, from_cache = extract_content(entry, cache, force=False)
        self.assertTrue(from_cache)
        self.assertEqual(content, "cached_content")

    def test_cache_miss_on_changed_file(self):
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write("def hello(): pass\n")
            path = Path(f.name)

        cache = {
            "test/file.py": {
                "hash": "stale_hash_00000000",  # wrong hash
                "mode": "full",
                "content": "stale_content",
                "lang": "Python", "size": 20,
            }
        }
        entry = {
            "rel_path": "test/file.py",
            "abs_path": str(path),
            "ext": ".py", "size": 20,
            "mode": "full", "lang": "Python",
        }
        content, from_cache = extract_content(entry, cache, force=False)
        self.assertFalse(from_cache)
        self.assertNotEqual(content, "stale_content")

    def test_force_flag_bypasses_cache(self):
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write("def hello(): pass\n")
            path = Path(f.name)

        h = _file_hash(path)
        cache = {
            "test/file.py": {
                "hash": h,  # hash matches, but force=True
                "mode": "full",
                "content": "old_content",
                "lang": "Python", "size": 20,
            }
        }
        entry = {
            "rel_path": "test/file.py",
            "abs_path": str(path),
            "ext": ".py", "size": 20,
            "mode": "full", "lang": "Python",
        }
        content, from_cache = extract_content(entry, cache, force=True)
        self.assertFalse(from_cache)
        self.assertNotEqual(content, "old_content")


# ─────────────────────────────────────────────────────────────────────────────
# D. Config inventory
# ─────────────────────────────────────────────────────────────────────────────

class TestDConfigInventory(unittest.TestCase):
    """Tests for build_config_map() — service detection, env var extraction."""

    def _make_inventory_and_cache(
        self, files: dict[str, str]
    ) -> tuple[list[dict], dict]:
        inventory = []
        cache = {}
        for rel_path, content in files.items():
            inventory.append({
                "rel_path": rel_path,
                "ext": Path(rel_path).suffix,
                "mode": "key-only",
                "lang": "JSON",
                "size": len(content),
            })
            cache[rel_path] = {"content": content, "mode": "key-only"}
        return inventory, cache

    def test_detects_database_service(self):
        inv, cache = self._make_inventory_and_cache({
            "config/db.json": json.dumps({
                "postgres_host": "prod-db.internal",
                "database_name": "myapp",
            })
        })
        result = build_config_map(inv, cache)
        self.assertIn("database", result["services_detected"])

    def test_detects_multiple_services(self):
        inv, cache = self._make_inventory_and_cache({
            "config/app.json": json.dumps({
                "redis_url": "redis://...",
                "kafka_brokers": ["broker1:9092"],
                "aws_region": "us-east-1",
                "s3_bucket": "my-uploads",      # triggers 'storage' pattern
                "auth_provider": "keycloak",
                "smtp_host": "mail.server.com",
            })
        })
        result = build_config_map(inv, cache)
        detected = result["services_detected"]
        self.assertIn("messaging", detected)
        self.assertIn("storage", detected)   # s3_bucket → storage
        self.assertIn("cloud", detected)     # aws_region → cloud
        self.assertIn("auth", detected)
        self.assertIn("email", detected)

    def test_extracts_env_var_references(self):
        content = textwrap.dedent("""
            DATABASE_URL=${DATABASE_URL}
            API_KEY=${API_KEY}
            port: ${PORT}
        """)
        inv, cache = self._make_inventory_and_cache({"config/app.yaml": content})
        result = build_config_map(inv, cache)
        self.assertIn("DATABASE_URL", result["env_vars_detected"])
        self.assertIn("API_KEY", result["env_vars_detected"])

    def test_skips_non_key_only_entries(self):
        """build_config_map should only process key-only entries."""
        inventory = [{
            "rel_path": "src/app.ts",
            "ext": ".ts",
            "mode": "full",          # not key-only
            "lang": "TypeScript",
            "size": 500,
        }]
        cache = {"src/app.ts": {"content": "export const DB_HOST = 'prod'", "mode": "full"}}
        result = build_config_map(inventory, cache)
        self.assertEqual(result["total_configs"], 0)

    def test_empty_inventory_returns_valid_structure(self):
        result = build_config_map([], {})
        self.assertEqual(result["total_configs"], 0)
        self.assertEqual(result["services_detected"], [])
        self.assertEqual(result["env_vars_detected"], [])
        self.assertIn("generated", result)
        self.assertIn("files", result)


# ─────────────────────────────────────────────────────────────────────────────
# E. Git crawl
# ─────────────────────────────────────────────────────────────────────────────

class TestEGitCrawl(unittest.TestCase):
    """Tests for git log parsing, scope translation, and blame map generation."""

    _SAMPLE_GIT_LOG = textwrap.dedent("""\
        a1b2c3d4|||2026-04-28T10:00:00+07:00|||dev@company.com|||fix: SLA timer not resetting
        12\t3\tsrc/sla/timer.ts
        5\t2\ttests/sla.test.ts

        e5f6a7b8|||2026-04-27T15:30:00+07:00|||pm@company.com|||feat: add customer filter
        25\t0\tsrc/filters/customer.ts
        10\t0\tsrc/filters/index.ts
        8\t4\tsrc/app.ts

        c9d0e1f2|||2026-04-26T09:00:00+07:00|||dev@company.com|||refactor: extract utils
        15\t20\tsrc/utils/helper.ts
        0\t5\tsrc/old-helper.ts
    """)

    def test_parse_git_log_commit_count(self):
        commits = _parse_git_log(self._SAMPLE_GIT_LOG)
        self.assertEqual(len(commits), 3)

    def test_parse_git_log_fields(self):
        commits = _parse_git_log(self._SAMPLE_GIT_LOG)
        first = commits[0]
        self.assertEqual(first["hash"], "a1b2c3d")
        self.assertEqual(first["date"], "2026-04-28")
        self.assertEqual(first["author"], "dev@company.com")
        self.assertIn("SLA timer", first["message"])

    def test_parse_git_log_files_changed(self):
        commits = _parse_git_log(self._SAMPLE_GIT_LOG)
        first = commits[0]
        self.assertIn("src/sla/timer.ts", first["files_changed"])
        self.assertIn("tests/sla.test.ts", first["files_changed"])

    def test_parse_git_log_insertions_deletions(self):
        commits = _parse_git_log(self._SAMPLE_GIT_LOG)
        first = commits[0]
        # First commit touches 2 files: 12+5=17 insertions, 3+2=5 deletions
        self.assertEqual(first["insertions"], 17)
        self.assertEqual(first["deletions"], 5)

    def test_parse_git_log_empty_input(self):
        self.assertEqual(_parse_git_log(""), [])

    def test_parse_git_log_binary_files_handled(self):
        """numstat shows '-' for binary files — should not crash."""
        log = "abc123|||2026-04-28|||dev@co.com|||add images\n-\t-\tassets/logo.png\n"
        commits = _parse_git_log(log)
        self.assertEqual(len(commits), 1)
        self.assertIn("assets/logo.png", commits[0]["files_changed"])

    def test_scope_to_git_args_months(self):
        args = _scope_to_git_args("3m")
        self.assertTrue(any("3 months ago" in a for a in args))

    def test_scope_to_git_args_years(self):
        args = _scope_to_git_args("1y")
        self.assertTrue(any("1 years ago" in a for a in args))

    def test_scope_to_git_args_all(self):
        args = _scope_to_git_args("all")
        self.assertEqual(args, [])

    def test_scope_to_git_args_commit_count(self):
        args = _scope_to_git_args("500")
        self.assertIn("500", str(args))

    def test_scope_to_git_args_date(self):
        args = _scope_to_git_args("2024-01-01")
        self.assertTrue(any("2024-01-01" in a for a in args))

    def test_scope_to_git_args_unknown_defaults_to_6m(self):
        args = _scope_to_git_args("???")
        self.assertTrue(any("6 months ago" in a for a in args))

    def test_build_blame_map_sections(self):
        git_data = {
            "scope": "6m",
            "generated": "2026-04-30T00:00:00",
            "total_commits": 3,
            "authors": ["dev@co.com", "pm@co.com"],
            "hotspots": [
                {"file": "src/sla/timer.ts",    "change_count": 15, "authors": ["dev@co.com"]},
                {"file": "src/filters/customer.ts", "change_count": 7, "authors": ["pm@co.com"]},
                {"file": "src/app.ts",           "change_count": 3,  "authors": ["dev@co.com", "pm@co.com"]},
            ],
            "commits": [],
        }
        result = build_blame_map(git_data)
        self.assertIn("# Codebase Hotspot Map", result)
        self.assertIn("src/sla/timer.ts", result)    # high churn
        self.assertIn("src/filters/customer.ts", result)  # medium churn
        self.assertIn("dev@co.com", result)
        self.assertIn("6m", result)

    def test_build_blame_map_module_activity(self):
        git_data = {
            "scope": "all",
            "generated": "2026-04-30T00:00:00",
            "total_commits": 5,
            "authors": ["dev@co.com"],
            "hotspots": [
                {"file": "src/sla/timer.ts",    "change_count": 20, "authors": ["dev@co.com"]},
                {"file": "src/sla/calculator.ts","change_count": 10, "authors": ["dev@co.com"]},
                {"file": "api/routes.ts",        "change_count": 5,  "authors": ["dev@co.com"]},
            ],
            "commits": [],
        }
        result = build_blame_map(git_data)
        # Module activity section groups by top-level dir
        self.assertIn("sla", result)


# ─────────────────────────────────────────────────────────────────────────────
# F. Context assembly
# ─────────────────────────────────────────────────────────────────────────────

class TestFContextAssembly(unittest.TestCase):
    """Tests for _build_extraction_context() — grouping, truncation, size limits."""

    def _make_inventory(self, paths_contents: dict[str, str]) -> tuple[list, dict]:
        inventory = []
        cache = {}
        for rel_path, content in paths_contents.items():
            ext  = Path(rel_path).suffix
            lang = _detect_language(ext) or "Unknown"
            inventory.append({
                "rel_path": rel_path,
                "ext": ext,
                "mode": "full",
                "lang": lang,
                "size": len(content),
            })
            cache[rel_path] = {"content": content}
        return inventory, cache

    def test_groups_by_top_level_directory(self):
        inv, cache = self._make_inventory({
            "src/app.ts": "export class App {}",
            "src/utils.ts": "export function util() {}",
            "config/db.json": '{"host": "<redacted>"}',
        })
        result = _build_extraction_context(inv, cache)
        self.assertIn("## src/", result)
        self.assertIn("## config/", result)

    def test_root_files_grouped_under_root(self):
        inv, cache = self._make_inventory({
            "README.md": "# My Project",
            "package.json": '{"name": "my-app"}',
        })
        result = _build_extraction_context(inv, cache)
        self.assertIn("(root)", result)

    def test_per_file_truncation(self):
        """Files exceeding _MAX_PER_FILE chars should be truncated."""
        long_content = "x" * 5000  # well above 2000 char limit
        inv, cache = self._make_inventory({"src/big.ts": long_content})
        result = _build_extraction_context(inv, cache)
        self.assertIn("[truncated", result)
        self.assertNotIn("x" * 5000, result)

    def test_empty_content_files_skipped(self):
        inv, cache = self._make_inventory({
            "src/app.ts": "export class App {}",
            "src/empty.ts": "",  # empty — should be skipped
        })
        result = _build_extraction_context(inv, cache)
        self.assertNotIn("empty.ts", result)

    def test_language_and_mode_in_header(self):
        inv, cache = self._make_inventory({
            "src/app.ts": "export const x = 1;",
        })
        result = _build_extraction_context(inv, cache)
        self.assertIn("TypeScript", result)
        self.assertIn("full", result)


# ─────────────────────────────────────────────────────────────────────────────
# G. Integration — full pipeline on fixture codebase
# ─────────────────────────────────────────────────────────────────────────────

class TestGIntegration(unittest.TestCase):
    """
    End-to-end pipeline test on the fixture codebase.
    Skips LLM and git phases — tests everything else.
    """

    @classmethod
    def setUpClass(cls):
        cls.root = make_tmp_codebase()
        cls.rules = AbsorberIgnoreRules(cls.root / "absorber.ignored")
        cls.inventory = scan_files(cls.root, cls.rules)

        # Run extraction
        cls.cache: dict = {}
        for entry in cls.inventory:
            extract_content(entry, cls.cache, force=False)

        cls.config_map = build_config_map(cls.inventory, cls.cache)

    def test_inventory_excludes_node_modules_and_dist(self):
        rel_paths = [e["rel_path"] for e in self.inventory]
        self.assertFalse(any("node_modules" in p for p in rel_paths))
        self.assertFalse(any("dist/" in p for p in rel_paths))

    def test_config_files_are_key_only(self):
        by_path = {e["rel_path"]: e for e in self.inventory}
        self.assertEqual(by_path["config/appsettings.json"]["mode"], "key-only")
        self.assertEqual(by_path["config/.env"]["mode"], "key-only")

    def test_secrets_not_in_cache(self):
        """After key-only extraction, secrets must not appear in cache."""
        appsettings = self.cache.get("config/appsettings.json", {}).get("content", "")
        self.assertNotIn("super-secret-db-password", appsettings)
        self.assertNotIn("my-jwt-secret-key-never-share", appsettings)

        env_content = self.cache.get("config/.env", {}).get("content", "")
        self.assertNotIn("supersecretkey", env_content)
        self.assertNotIn("wJalrXUtnFEMI", env_content)  # AWS secret

    def test_numeric_values_preserved_in_config(self):
        """Port 5432 should survive key-only extraction."""
        appsettings = self.cache.get("config/appsettings.json", {}).get("content", "")
        self.assertIn("5432", appsettings)

    def test_python_signatures_extracted(self):
        py_content = self.cache.get("src/utils/helper.py", {}).get("content", "")
        self.assertIn("class CacheManager", py_content)
        self.assertIn("def get", py_content)
        self.assertIn("def set", py_content)

    def test_ts_signatures_extracted(self):
        ts_content = self.cache.get("src/app.ts", {}).get("content", "")
        self.assertIn("export interface User", ts_content)
        self.assertIn("export class UserService", ts_content)

    def test_config_map_detects_database_service(self):
        self.assertIn("database", self.config_map["services_detected"])

    def test_config_map_detects_auth_service(self):
        self.assertIn("auth", self.config_map["services_detected"])

    def test_config_map_detects_env_vars(self):
        env_vars = self.config_map["env_vars_detected"]
        # .env file contains these
        self.assertIn("DATABASE_URL", env_vars)
        self.assertIn("JWT_SECRET", env_vars)

    def test_context_assembly_contains_all_modules(self):
        context = _build_extraction_context(self.inventory, self.cache)
        self.assertIn("## src/", context)
        self.assertIn("## config/", context)
        self.assertIn("## migrations/", context)

    def test_context_assembly_no_secrets_leaked(self):
        """Final LLM context must not contain any secrets."""
        context = _build_extraction_context(self.inventory, self.cache)
        self.assertNotIn("super-secret-db-password", context)
        self.assertNotIn("supersecretkey", context)
        self.assertNotIn("wJalrXUtnFEMI", context)
        self.assertNotIn("my-jwt-secret-key-never-share", context)

    def test_cache_hit_on_second_pass(self):
        """Running extraction again should yield 100% cache hits."""
        cache2 = dict(self.cache)  # copy existing cache
        hits = 0
        for entry in self.inventory:
            _, from_cache = extract_content(entry, cache2, force=False)
            if from_cache:
                hits += 1
        self.assertEqual(hits, len(self.inventory), "All files should be cache hits on second pass")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("Test"):
        group = sys.argv[1]
        # Map short group name to full class name
        class_map = {
            "TestA": "TestAAbsorberIgnoreRules",
            "TestB": "TestBFileScanning",
            "TestC": "TestCContentExtraction",
            "TestD": "TestDConfigInventory",
            "TestE": "TestEGitCrawl",
            "TestF": "TestFContextAssembly",
            "TestG": "TestGIntegration",
        }
        cls_name = class_map.get(group, group)
        suite = unittest.TestLoader().loadTestsFromName(
            f"test_absorber.{cls_name}"
        )
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        unittest.main(verbosity=2)
