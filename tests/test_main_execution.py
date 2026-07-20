"""Tests for main_execution module: artifact paths and template lists."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import aetherdialect._main_execution
from aetherdialect._config import (
    ConfigError,
    DatabricksRuntimeConfig,
    EngineConfig,
    PostgresRuntimeConfig,
)
from aetherdialect._constants import ARTIFACT_DIRECTORY_SEGMENT
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import (
    QSimSummary,
    SeedWarmupSummary,
)
from aetherdialect._main_execution import (
    _activate_engine,
    _apply_runtime_environments,
    _select_engine_name,
    compute_connection_storage_slug,
    compute_engine_storage_dir,
    configure_runtime_from_environment,
    find_latest_seed_warmup_summary,
    get_seed_warmup_summary_from_dir,
    load_qsim_summaries,
    resolve_qsim_path,
)


def _snapshot_os_environ() -> dict[str, str]:
    """Return a string copy of ``os.environ`` for passing into configure helpers."""
    return {str(k): str(v) for k, v in os.environ.items()}


class TestComputeEngineStorageDir:
    """Tests for :func:`aetherdialect._main_execution.compute_engine_storage_dir`."""

    def test_custom_root_joins_aetherdialect_and_slug(self) -> None:
        root = tempfile.mkdtemp()
        merged = {"PGDATABASE": "mydb", "PGUSER": "u", "PGPASSWORD": "p"}
        _apply_runtime_environments(merged)
        path = compute_engine_storage_dir(root, "postgresql")
        slug = compute_connection_storage_slug("postgresql")
        assert path == os.path.join(os.path.abspath(root), ARTIFACT_DIRECTORY_SEGMENT, slug)

    def test_none_root_uses_platformdirs_parent(self) -> None:
        from platformdirs import user_data_dir

        merged = {"PGDATABASE": "mydb", "PGUSER": "u", "PGPASSWORD": "p"}
        _apply_runtime_environments(merged)
        path = compute_engine_storage_dir(None, "postgresql")
        parent = user_data_dir(appname="aetherdialect", appauthor=False)
        slug = compute_connection_storage_slug("postgresql")
        assert path == os.path.join(parent, ARTIFACT_DIRECTORY_SEGMENT, slug)


class TestLoadConfigFile:
    """Tests for :func:`aetherdialect._main_execution._load_config_file`."""

    def test_full_coverage_flatten(self, tmp_path) -> None:
        path = tmp_path / "full.toml"
        path.write_text(
            "\n".join(
                (
                    "[openai]",
                    'api_key = "oak"',
                    'base_url = "https://example-openai/v1"',
                    "",
                    "[azure_openai]",
                    'endpoint = "https://ex.azure.com"',
                    'api_key = "aak"',
                    'api_version = "2024-01-01"',
                    'base_url = "https://ex.azure.com/base"',
                    "",
                    "[azure_openai.deployments]",
                    'light = "al"',
                    'medium = "am"',
                    'heavy = "ah"',
                    "",
                    "[postgresql]",
                    'host = "h"',
                    "port = 5433",
                    'database = "d"',
                    'schema = "sch"',
                    'user = "u"',
                    'password = "pw"',
                    "",
                    "[databricks]",
                    'host = "dh"',
                    'http_path = "/sql"',
                    'access_token = "tok"',
                    'catalog = "cat"',
                    'schema = "ds"',
                    "",
                    "[engine]",
                    'selected = "postgresql"',
                    "",
                    "[llm]",
                    'provider = "openai"',
                    "",
                    "[execution]",
                    "max_query_cost_rows = 100",
                    "max_query_cost_bytes = 200",
                    "statement_timeout_ms = 300",
                    "llm_timeout_ms = 400",
                    "profile_timeout_ms = 500",
                    "explain_timeout_ms = 600",
                ),
            ),
            encoding="utf-8",
        )
        got, claimed = aetherdialect._main_execution._load_config_file(str(path))
        expected = {
            "OPENAI_API_KEY": "oak",
            "OPENAI_BASE_URL": "https://example-openai/v1",
            "AZURE_OPENAI_ENDPOINT": "https://ex.azure.com",
            "AZURE_OPENAI_API_KEY": "aak",
            "AZURE_OPENAI_API_VERSION": "2024-01-01",
            "AZURE_OPENAI_BASE_URL": "https://ex.azure.com/base",
            "AZURE_OPENAI_DEPLOYMENT_LIGHT": "al",
            "AZURE_OPENAI_DEPLOYMENT_MEDIUM": "am",
            "AZURE_OPENAI_DEPLOYMENT_HEAVY": "ah",
            "POSTGRES_HOST": "h",
            "POSTGRES_PORT": "5433",
            "POSTGRES_DB": "d",
            "POSTGRES_SCHEMA": "sch",
            "POSTGRES_USER": "u",
            "POSTGRES_PASSWORD": "pw",
            "DATABRICKS_HOST": "dh",
            "DATABRICKS_HTTP_PATH": "/sql",
            "DATABRICKS_ACCESS_TOKEN": "tok",
            "DATABRICKS_CATALOG": "cat",
            "DATABRICKS_SCHEMA": "ds",
            "AETHERDIALECT_ENGINE": "postgresql",
            "AETHERDIALECT_LLM_PROVIDER": "openai",
            "AETHERDIALECT_MAX_QUERY_COST_ROWS": "100",
            "AETHERDIALECT_MAX_QUERY_COST_BYTES": "200",
            "AETHERDIALECT_STATEMENT_TIMEOUT_MS": "300",
            "AETHERDIALECT_LLM_TIMEOUT_MS": "400",
            "AETHERDIALECT_PROFILE_TIMEOUT_MS": "500",
            "AETHERDIALECT_EXPLAIN_TIMEOUT_MS": "600",
        }
        assert got == expected
        assert frozenset(expected.keys()) <= claimed


def test_load_config_file_claims_empty_openai_api_key_without_flat_value(
    tmp_path,
) -> None:
    path = tmp_path / "empty_key.toml"
    path.write_text('[openai]\napi_key = ""\n', encoding="utf-8")
    flat, claimed = aetherdialect._main_execution._load_config_file(str(path))
    assert flat == {}
    assert "OPENAI_API_KEY" in claimed


def test_merge_configuration_environment_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "from_os")
    config_values = {"POSTGRES_HOST": "from_toml"}
    merged, _keys = aetherdialect._main_execution._merge_configuration_environment(config_values)
    assert merged["POSTGRES_HOST"] == "from_toml"


def test_merge_toml_diagnostic_when_toml_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "os_host")
    merged, keys = aetherdialect._main_execution._merge_configuration_environment({"POSTGRES_HOST": "toml_host"})
    assert merged["POSTGRES_HOST"] == "toml_host"
    assert "POSTGRES_HOST" in keys


def test_merge_configuration_environment_ssot_clears_claimed_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from_env")
    merged, _diag = aetherdialect._main_execution._merge_configuration_environment(
        {},
        toml_claimed_keys=frozenset({"OPENAI_API_KEY"}),
    )
    assert "OPENAI_API_KEY" not in merged


def test_merge_configuration_environment_ssot_toml_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "from_os")
    merged, keys = aetherdialect._main_execution._merge_configuration_environment(
        {"POSTGRES_HOST": "from_toml"},
        toml_claimed_keys=frozenset({"POSTGRES_HOST"}),
    )
    assert merged["POSTGRES_HOST"] == "from_toml"
    assert "POSTGRES_HOST" in keys


class TestPrepareSchemaContextForInit:
    """Tests for :func:`aetherdialect._main_execution._prepare_schema_co ntext_for_init`."""

    def test_reuses_cached_sql_when_missing_in_explicit_context(self, tmp_path) -> None:
        from aetherdialect._main_execution import (
            _prepare_schema_context_for_init,
            write_schema_context_cache,
        )

        sql = tmp_path / "ddl.sql"
        sql.write_text("SELECT 1;", encoding="utf-8")
        engine_dir = str(tmp_path / "engine")
        os.makedirs(engine_dir, exist_ok=True)
        write_schema_context_cache(engine_dir, EngineContext(sql_file=str(sql)))
        logs: list[str] = []
        out = _prepare_schema_context_for_init(EngineContext(), engine_dir, logs.append)
        assert out.sql_file is not None
        assert os.path.isfile(out.sql_file)


class TestResolveQsimPath:
    """Tests for resolve_qsim_path."""

    def test_from_int_version(self):
        """Integer version should produce correct filename."""
        path = resolve_qsim_path(3, "/artifacts")
        assert path.endswith("qsim_questions_v3.txt")
        assert path.startswith("/artifacts")

    def test_from_qsim_summary(self):
        """QSimSummary should use its version attribute."""
        summary = QSimSummary(version=7, num_intents=10, num_questions=50, seed=42)
        path = resolve_qsim_path(summary, "/out")
        assert path.endswith("qsim_questions_v7.txt")


class TestLoadQsimSummaries:
    """Tests for :func:`aetherdialect._main_execution.load_qsim_summaries`."""

    def test_missing_summary_returns_empty(self):
        """Missing summary file should return empty list."""
        with tempfile.TemporaryDirectory() as td:
            result = load_qsim_summaries(td)
            assert result == []

    def test_loads_existing_summaries(self):
        """Should load and parse existing summary entries."""
        with tempfile.TemporaryDirectory() as td:
            data = [
                {
                    "version": 1,
                    "num_intents": 5,
                    "num_questions": 20,
                    "seed": 1,
                },
                {
                    "version": 2,
                    "num_intents": 10,
                    "num_questions": 50,
                    "seed": 2,
                },
            ]
            with open(os.path.join(td, "qsim_summary.json"), "w") as f:
                json.dump(data, f)
            result = load_qsim_summaries(td)
            assert len(result) == 2
            assert isinstance(result[0], QSimSummary)
            assert result[0].version == 1


class TestFindLatestSeedWarmupSummary:
    """Tests for :func:`aetherdialect._main_execution.find_latest_seed_w armup_summary`."""

    def test_empty_dir_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            assert find_latest_seed_warmup_summary(td) is None

    def test_picks_highest_version(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            for ver, total in ((1, 3), (3, 5)):
                with open(os.path.join(td, f"seed_warmup_report_v{ver}.json"), "w") as f:
                    json.dump({"total": total, "success": total, "failed": 0}, f)
            got = find_latest_seed_warmup_summary(td)
            assert got is not None
            assert got.version == 3
            assert got.total == 5


class TestGetSeedWarmupSummaryFromDir:
    """Tests for get_seed_warmup_summary_from_dir."""

    def test_missing_report_raises(self):
        """Missing report file should raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                get_seed_warmup_summary_from_dir(td, 1)

    def test_loads_valid_report(self):
        """Valid report JSON should produce SeedWarmupSummary."""
        with tempfile.TemporaryDirectory() as td:
            report = {"total": 10, "success": 8, "failed": 2}
            with open(os.path.join(td, "seed_warmup_report_v1.json"), "w") as f:
                json.dump(report, f)
            result = get_seed_warmup_summary_from_dir(td, 1)
            assert isinstance(result, SeedWarmupSummary)
            assert result.total == 10
            assert result.success == 8
            assert result.success_rate == 0.8


def _import_module_side_effect_pg_only(name: str):
    if name == "databricks.sql":
        raise ImportError()
    if name == "psycopg2":
        raise ImportError()
    if name == "psycopg":
        return object()
    if name in ("duckdb", "openpyxl"):
        return object()
    if name.startswith("aetherdialect._dialect"):
        return object()
    return importlib.import_module(name)


def _import_module_side_effect_dbx_only(name: str):
    if name in ("psycopg2", "psycopg"):
        raise ImportError()
    if name == "databricks.sql":
        return object()
    if name in ("duckdb", "openpyxl"):
        return object()
    if name.startswith("aetherdialect._dialect"):
        return object()
    return importlib.import_module(name)


def _import_module_side_effect_both(name: str):
    if name in ("psycopg2", "psycopg", "databricks.sql"):
        return object()
    if name in ("duckdb", "openpyxl"):
        return object()
    if name.startswith("aetherdialect._dialect"):
        return object()
    return importlib.import_module(name)


class TestSelectEngineFromEnvironment:
    """Engine selection honours per-field env alias lists from ``config``."""

    def test_postgresql_from_libpq_without_pghost(self) -> None:
        env = {
            "PGDATABASE": "rental_shop",
            "PGUSER": "postgres",
            "PGPASSWORD": "secret",
        }
        with patch.object(importlib, "import_module", side_effect=_import_module_side_effect_pg_only):
            assert _select_engine_name(env) == "postgresql"

    def test_postgresql_from_postgres_docker_style_names(self) -> None:
        env = {
            "POSTGRES_DB": "rental_shop",
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": "secret",
        }
        with patch.object(importlib, "import_module", side_effect=_import_module_side_effect_pg_only):
            assert _select_engine_name(env) == "postgresql"

    def test_postgresql_with_pghost(self) -> None:
        env = {
            "PGHOST": "db.internal",
            "PGDATABASE": "d1",
            "PGUSER": "u1",
            "PGPASSWORD": "p1",
        }
        with patch.object(importlib, "import_module", side_effect=_import_module_side_effect_pg_only):
            assert _select_engine_name(env) == "postgresql"

    def test_databricks_from_datatabricks_env(self) -> None:
        env = {
            "DATABRICKS_HOST": "adb-1.azuredatabricks.net",
            "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/abc",
            "DATABRICKS_TOKEN": "dapi_test_token",
            "DATABRICKS_CATALOG": "hive_metastore",
            "DATABRICKS_SCHEMA": "default",
        }
        with patch.object(importlib, "import_module", side_effect=_import_module_side_effect_dbx_only):
            assert _select_engine_name(env) == "databricks"

    def test_databricks_alternate_host_and_token_names(self) -> None:
        env = {
            "DATABRICKS_SERVER_HOSTNAME": "adb-1.azuredatabricks.net",
            "DATABRICKS_SQL_HTTP_PATH": "/sql/1.0/warehouses/abc",
            "ACCESS_TOKEN": "dapi_test_token",
            "SPARK_DEFAULT_CATALOG": "hive_metastore",
            "DATABRICKS_DEFAULT_SCHEMA": "default",
        }
        with patch.object(importlib, "import_module", side_effect=_import_module_side_effect_dbx_only):
            assert _select_engine_name(env) == "databricks"


class TestSelectEngineExplicit:
    def test_explicit_databricks_when_both_envs_present(self) -> None:
        env = {
            "PGDATABASE": "rental_shop",
            "PGUSER": "postgres",
            "PGPASSWORD": "secret",
            "DATABRICKS_HOST": "adb-1.azuredatabricks.net",
            "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/abc",
            "DATABRICKS_TOKEN": "dapi_test_token",
            "DATABRICKS_CATALOG": "hive_metastore",
            "DATABRICKS_SCHEMA": "default",
            "AETHERDIALECT_ENGINE": "databricks",
        }
        with patch.object(importlib, "import_module", side_effect=_import_module_side_effect_both):
            assert _select_engine_name(env) == "databricks"

    def test_explicit_postgresql_when_both_envs_present(self) -> None:
        env = {
            "PGDATABASE": "rental_shop",
            "PGUSER": "postgres",
            "PGPASSWORD": "secret",
            "DATABRICKS_HOST": "adb-1.azuredatabricks.net",
            "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/abc",
            "DATABRICKS_TOKEN": "dapi_test_token",
            "DATABRICKS_CATALOG": "hive_metastore",
            "DATABRICKS_SCHEMA": "default",
            "AETHERDIALECT_ENGINE": "postgresql",
        }
        with patch.object(importlib, "import_module", side_effect=_import_module_side_effect_both):
            assert _select_engine_name(env) == "postgresql"

    def test_both_drivers_and_both_env_without_explicit_raises(self) -> None:
        env = {
            "PGDATABASE": "rental_shop",
            "PGUSER": "postgres",
            "PGPASSWORD": "secret",
            "DATABRICKS_HOST": "adb-1.azuredatabricks.net",
            "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/abc",
            "DATABRICKS_TOKEN": "dapi_test_token",
            "DATABRICKS_CATALOG": "hive_metastore",
            "DATABRICKS_SCHEMA": "default",
        }
        with patch.object(importlib, "import_module", side_effect=_import_module_side_effect_both):
            with pytest.raises(ConfigError, match="AETHERDIALECT_ENGINE"):
                _select_engine_name(env)

    def test_explicit_databricks_with_missing_env_raises(self) -> None:
        env = {
            "PGDATABASE": "rental_shop",
            "PGUSER": "postgres",
            "PGPASSWORD": "secret",
            "AETHERDIALECT_ENGINE": "databricks",
        }
        with patch.object(importlib, "import_module", side_effect=_import_module_side_effect_pg_only):
            with pytest.raises(ConfigError, match="Cannot select databricks engine"):
                _select_engine_name(env)

    def test_invalid_aether_engine_raises(self) -> None:
        with pytest.raises(ConfigError, match="Unsupported AETHERDIALECT_ENGINE"):
            _select_engine_name({"AETHERDIALECT_ENGINE": "not_a_registered_engine"})


class TestApplyDatabaseEnv:
    """``_apply_*_env`` maps process env into runtime ClassVars."""

    def test_apply_postgres_libpq(self) -> None:
        pg_fields = ("HOST", "PORT", "USER", "PASSWORD", "DATABASE", "SCHEMA")
        snap_pg = {k: getattr(PostgresRuntimeConfig, k) for k in pg_fields}
        eng_type, eng_rt = EngineConfig.TYPE, EngineConfig.RUNTIME
        try:
            _apply_runtime_environments(
                {
                    "POSTGRES_HOST": "10.0.0.1",
                    "POSTGRES_PORT": "5433",
                    "POSTGRES_USER": "app",
                    "POSTGRES_PASSWORD": "pw",
                    "POSTGRES_DB": "appdb",
                    "POSTGRES_SCHEMA": "sales",
                },
            )
            assert PostgresRuntimeConfig.HOST == "10.0.0.1"
            assert PostgresRuntimeConfig.PORT == 5433
            assert PostgresRuntimeConfig.USER == "app"
            assert PostgresRuntimeConfig.PASSWORD == "pw"
            assert PostgresRuntimeConfig.DATABASE == "appdb"
            assert PostgresRuntimeConfig.SCHEMA == "sales"
            assert EngineConfig.TYPE == eng_type
            assert EngineConfig.RUNTIME is eng_rt
        finally:
            for k, v in snap_pg.items():
                setattr(PostgresRuntimeConfig, k, v)
            EngineConfig.TYPE = eng_type
            EngineConfig.RUNTIME = eng_rt

    def test_apply_databricks_datatabricks_token(self) -> None:
        dbx_fields = (
            "SERVER_HOSTNAME",
            "HTTP_PATH",
            "ACCESS_TOKEN",
            "CATALOG",
            "SCHEMA",
        )
        snap_dbx = {k: getattr(DatabricksRuntimeConfig, k) for k in dbx_fields}
        eng_type, eng_rt = EngineConfig.TYPE, EngineConfig.RUNTIME
        try:
            _apply_runtime_environments(
                {
                    "DATABRICKS_SERVER_HOSTNAME": "adb-1.azuredatabricks.net",
                    "DATABRICKS_WAREHOUSE_HTTP_PATH": "/sql/1.0/warehouses/x",
                    "ACCESS_TOKEN": "dapi_xyz",
                    "SPARK_DEFAULT_CATALOG": "main",
                    "SPARK_DEFAULT_SCHEMA": "dbo",
                },
            )
            assert DatabricksRuntimeConfig.SERVER_HOSTNAME == "adb-1.azuredatabricks.net"
            assert DatabricksRuntimeConfig.HTTP_PATH == "/sql/1.0/warehouses/x"
            assert DatabricksRuntimeConfig.ACCESS_TOKEN == "dapi_xyz"
            assert DatabricksRuntimeConfig.CATALOG == "main"
            assert DatabricksRuntimeConfig.SCHEMA == "dbo"
            assert EngineConfig.TYPE == eng_type
            assert EngineConfig.RUNTIME is eng_rt
        finally:
            for k, v in snap_dbx.items():
                setattr(DatabricksRuntimeConfig, k, v)
            EngineConfig.TYPE = eng_type
            EngineConfig.RUNTIME = eng_rt

    def test_activate_engine_switches_engine_type_and_runtime(self) -> None:
        eng_type, eng_rt = EngineConfig.TYPE, EngineConfig.RUNTIME
        try:
            _activate_engine("databricks")
            assert EngineConfig.TYPE == "databricks"
            assert EngineConfig.RUNTIME is DatabricksRuntimeConfig
            _activate_engine("postgresql")
            assert EngineConfig.TYPE == "postgresql"
            assert EngineConfig.RUNTIME is PostgresRuntimeConfig
        finally:
            EngineConfig.TYPE = eng_type
            EngineConfig.RUNTIME = eng_rt

    def test_activate_engine_rejects_unknown_name(self) -> None:
        with pytest.raises(ConfigError, match="Unsupported engine activation"):
            _activate_engine("bogus_engine")


class TestDatabricksEngineSelectionPySpark:
    """Databricks is selectable with Unity Catalog scope plus PySpark when warehouse env is absent."""

    def test_selects_databricks_when_pyspark_reachable(self) -> None:
        env = {
            "DATABRICKS_CATALOG": "main",
            "SPARK_DEFAULT_SCHEMA": "default",
        }
        with patch.object(importlib, "import_module", side_effect=_import_module_side_effect_dbx_only):
            with patch.object(DatabricksRuntimeConfig, "pyspark_session_reachable", return_value=True):
                assert _select_engine_name(env) == "databricks"

    def test_raises_when_catalog_schema_only_and_no_pyspark(self) -> None:
        env = {
            "DATABRICKS_CATALOG": "main",
            "DATABRICKS_SCHEMA": "dbo",
        }
        with patch.object(importlib, "import_module", side_effect=_import_module_side_effect_dbx_only):
            with patch.object(DatabricksRuntimeConfig, "pyspark_session_reachable", return_value=False):
                with pytest.raises(ConfigError, match="Cannot select database engine"):
                    _select_engine_name(env)


class TestConfigureRuntimeFromEnvironment:
    """``configure_runtime_from_environment`` wires LLM ClassVars from env."""

    def test_openai_ignores_optional_model_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for k in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
        ):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PGDATABASE", "db")
        monkeypatch.setenv("PGUSER", "u")
        monkeypatch.setenv("PGPASSWORD", "p")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "should-be-ignored")
        eng_fields = (
            "TYPE",
            "RUNTIME",
            "LLM_PROVIDER",
            "API_TOKEN",
            "AZURE_API_TOKEN",
            "OPENAI_MODEL",
            "OPENAI_MODEL_INTENT",
            "OPENAI_MODEL_JOIN",
            "OPENAI_MODEL_SCHEMA",
            "OPENAI_BASE_URL",
        )
        snap_eng = {k: getattr(EngineConfig, k) for k in eng_fields}
        snap_pg = {
            k: getattr(PostgresRuntimeConfig, k) for k in ("HOST", "PORT", "USER", "PASSWORD", "DATABASE", "SCHEMA")
        }
        try:
            with patch.object(
                importlib,
                "import_module",
                side_effect=_import_module_side_effect_pg_only,
            ):
                configure_runtime_from_environment(EngineContext(), _snapshot_os_environ())
            assert EngineConfig.LLM_PROVIDER == "openai"
            assert EngineConfig.OPENAI_MODEL == "gpt-4o-mini"
            assert EngineConfig.OPENAI_MODEL_JOIN == "gpt-5.4-mini"
        finally:
            for k, v in snap_eng.items():
                setattr(EngineConfig, k, v)
            for k, v in snap_pg.items():
                setattr(PostgresRuntimeConfig, k, v)

    def test_azure_deployment_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("PGDATABASE", "db")
        monkeypatch.setenv("PGUSER", "u")
        monkeypatch.setenv("PGPASSWORD", "p")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "ak")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_LIGHT", "dep-a")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_MEDIUM", "dep-b")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_HEAVY", "dep-c")
        eng_fields = (
            "TYPE",
            "RUNTIME",
            "LLM_PROVIDER",
            "API_TOKEN",
            "AZURE_API_TOKEN",
            "OPENAI_MODEL",
            "OPENAI_MODEL_INTENT",
            "OPENAI_MODEL_JOIN",
            "OPENAI_MODEL_SCHEMA",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_BASE_URL",
        )
        snap_eng = {k: getattr(EngineConfig, k) for k in eng_fields}
        snap_pg = {
            k: getattr(PostgresRuntimeConfig, k) for k in ("HOST", "PORT", "USER", "PASSWORD", "DATABASE", "SCHEMA")
        }
        try:
            with patch.object(
                importlib,
                "import_module",
                side_effect=_import_module_side_effect_pg_only,
            ):
                configure_runtime_from_environment(EngineContext(), _snapshot_os_environ())
            from aetherdialect._llm_provider import _azure_deployment_for_model

            assert EngineConfig.LLM_PROVIDER == "azure"
            assert _azure_deployment_for_model("gpt-4o-mini") == "dep-a"
            assert _azure_deployment_for_model("gpt-4.1-mini") == "dep-b"
            assert _azure_deployment_for_model("gpt-5.4-mini") == "dep-c"
        finally:
            for k, v in snap_eng.items():
                setattr(EngineConfig, k, v)
            for k, v in snap_pg.items():
                setattr(PostgresRuntimeConfig, k, v)

    def test_both_llm_providers_configured_requires_explicit_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PGDATABASE", "db")
        monkeypatch.setenv("PGUSER", "u")
        monkeypatch.setenv("PGPASSWORD", "p")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "ak")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        eng_fields = (
            "TYPE",
            "RUNTIME",
            "LLM_PROVIDER",
            "API_TOKEN",
            "AZURE_API_TOKEN",
            "OPENAI_MODEL",
            "OPENAI_MODEL_INTENT",
            "OPENAI_MODEL_JOIN",
            "OPENAI_MODEL_SCHEMA",
            "OPENAI_BASE_URL",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_BASE_URL",
        )
        snap_eng = {k: getattr(EngineConfig, k) for k in eng_fields}
        snap_pg = {
            k: getattr(PostgresRuntimeConfig, k) for k in ("HOST", "PORT", "USER", "PASSWORD", "DATABASE", "SCHEMA")
        }
        from aetherdialect._config import ConfigError

        try:
            with patch.object(
                importlib,
                "import_module",
                side_effect=_import_module_side_effect_pg_only,
            ):
                with pytest.raises(ConfigError, match="AETHERDIALECT_LLM_PROVIDER"):
                    configure_runtime_from_environment(EngineContext(), _snapshot_os_environ())
        finally:
            for k, v in snap_eng.items():
                setattr(EngineConfig, k, v)
            for k, v in snap_pg.items():
                setattr(PostgresRuntimeConfig, k, v)

    def test_both_llm_providers_explicit_openai_configures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PGDATABASE", "db")
        monkeypatch.setenv("PGUSER", "u")
        monkeypatch.setenv("PGPASSWORD", "p")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "ak")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AETHERDIALECT_LLM_PROVIDER", "openai")
        eng_fields = (
            "TYPE",
            "RUNTIME",
            "LLM_PROVIDER",
            "API_TOKEN",
            "AZURE_API_TOKEN",
            "OPENAI_MODEL",
            "OPENAI_MODEL_INTENT",
            "OPENAI_MODEL_JOIN",
            "OPENAI_MODEL_SCHEMA",
            "OPENAI_BASE_URL",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_BASE_URL",
        )
        snap_eng = {k: getattr(EngineConfig, k) for k in eng_fields}
        snap_pg = {
            k: getattr(PostgresRuntimeConfig, k) for k in ("HOST", "PORT", "USER", "PASSWORD", "DATABASE", "SCHEMA")
        }
        try:
            with patch.object(
                importlib,
                "import_module",
                side_effect=_import_module_side_effect_pg_only,
            ):
                configure_runtime_from_environment(EngineContext(), _snapshot_os_environ())
            assert EngineConfig.LLM_PROVIDER == "openai"
        finally:
            for k, v in snap_eng.items():
                setattr(EngineConfig, k, v)
            for k, v in snap_pg.items():
                setattr(PostgresRuntimeConfig, k, v)


class TestSchemaContextCache:
    """Tests for write/load helpers on the persisted EngineContext."""

    def test_write_inlines_sql_and_notes(self, tmp_path) -> None:
        sql_path = tmp_path / "ddl.sql"
        notes_file = tmp_path / "notes.md"
        sql_path.write_text("CREATE TABLE t (id INT);\n", encoding="utf-8")
        notes_file.write_text("hello notes", encoding="utf-8")
        adir = str(tmp_path / "artifacts")
        os.makedirs(adir, exist_ok=True)
        ctx = EngineContext(
            allow_objects=frozenset({"public.t"}),
            deny_columns=frozenset({"t.secret"}),
            sql_file=str(sql_path),
            notes_file=str(notes_file),
        )
        out = aetherdialect._main_execution.write_schema_context_cache(adir, ctx)
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == aetherdialect._main_execution.SCHEMA_CONTEXT_CACHE_VERSION
        assert data["sql_text"] == "CREATE TABLE t (id INT);\n"
        assert data["notes_text"] == "hello notes"
        assert "public.t" in data["allow_objects"]
        assert "t.secret" in data["deny_columns"]

    def test_load_round_trip_materializes_files(self, tmp_path) -> None:
        sql_path = tmp_path / "ddl.sql"
        sql_path.write_text("SELECT 1;", encoding="utf-8")
        adir = str(tmp_path / "artifacts")
        os.makedirs(adir, exist_ok=True)
        ctx = EngineContext(
            allow_objects=frozenset({"public.t"}),
            sql_file=str(sql_path),
        )
        aetherdialect._main_execution.write_schema_context_cache(adir, ctx)
        loaded = aetherdialect._main_execution.load_schema_context_cache(adir)
        assert loaded is not None
        assert "public.t" in loaded.allow_objects
        assert loaded.sql_file is not None
        assert os.path.exists(loaded.sql_file)
        with open(loaded.sql_file, encoding="utf-8") as f:
            assert f.read() == "SELECT 1;"

    def test_load_missing_returns_none(self, tmp_path) -> None:
        loaded = aetherdialect._main_execution.load_schema_context_cache(str(tmp_path))
        assert loaded is None

    def test_load_bad_version_returns_none(self, tmp_path) -> None:
        adir = str(tmp_path)
        path = os.path.join(adir, aetherdialect._main_execution.SCHEMA_CONTEXT_CACHE_NAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 999}, f)
        assert aetherdialect._main_execution.load_schema_context_cache(adir) is None


class TestPipelineSessionSuspendToStep:
    """Tests for :meth:`PipelineSession._suspend_to_step` prompt and payload assembly."""

    def test_sql_feedback_suspend_populates_message_prompt(self) -> None:
        from aetherdialect._constants import PIPELINE_SUSPEND_ID_SQL, GenerationPath
        from aetherdialect._contracts_base import PipelineSuspended
        from aetherdialect._contracts_core import (
            InteractiveTailSnapshot,
            RuntimeIntent,
            SqlFeedbackSuspendContext,
            SqlGenerationOutcome,
        )
        from aetherdialect._main_execution import PipelineSession

        ri = RuntimeIntent(
            tables=[],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        tail = InteractiveTailSnapshot(
            q_norm="q",
            intent=ri,
            schema=None,
            store={},
            templates={},
            rejected={},
            schema_terms=set(),
            dialect=None,
            semantic_warnings=(),
            has_union_match=False,
            cols_changed=False,
            matched_template=None,
            union_select_cols=None,
            structural_match_templates=(),
            ikey="k",
            intent_sim=0.0,
        )
        gen_out = SqlGenerationOutcome(
            sql="SELECT 1",
            success=True,
            generation_path=GenerationPath.EXACT_QUESTION_REUSE,
            matched_template=None,
        )
        ctx = SqlFeedbackSuspendContext(
            tail=tail,
            execution_intent=ri,
            sql="SELECT 1",
            rows=(),
            conf=0.5,
            tmpl_sd=None,
            gen_out=gen_out,
            matched_rejected_template=None,
            force_feedback=True,
        )
        owner = MagicMock()
        owner._audit_emit = MagicMock()
        owner._schema_graph = MagicMock(effective_structural_hash="h")
        sess = PipelineSession(owner)
        ex = PipelineSuspended(PIPELINE_SUSPEND_ID_SQL, "ignored", ctx)
        step = PipelineSession._suspend_to_step(sess, ex)
        assert step.message == ""
        assert step.prompt == "Is this correct? (y/n): "
        assert step.sql == "SELECT 1"
        assert step.data is None

    def test_direct_reuse_suspend_populates_sql_data_message(self) -> None:
        from aetherdialect._constants import PIPELINE_SUSPEND_ID_DIRECT_REUSE, GenerationPath
        from aetherdialect._contracts_base import PipelineSuspended
        from aetherdialect._contracts_core import (
            DirectReuseSuspendContext,
            RuntimeIntent,
        )
        from aetherdialect._main_execution import PipelineSession

        ri = RuntimeIntent(
            tables=[],
            grain="scalar",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        ctx = DirectReuseSuspendContext(
            q_norm="how many",
            ref_tmpl=MagicMock(),
            dialect=None,
            store={},
            templates={},
            rejected={},
            schema=None,
            intent=ri,
            sql="SELECT 1",
            rows=((42,),),
            display_sql="SELECT 42 AS n",
            headers=("n",),
            is_exact=True,
            reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
            sd_reuse=None,
        )
        sess = PipelineSession(MagicMock())
        ex = PipelineSuspended(PIPELINE_SUSPEND_ID_DIRECT_REUSE, "ignored", ctx)
        step = PipelineSession._suspend_to_step(sess, ex)
        assert step.message == ""
        assert step.prompt == "Is this correct? (y/n): "
        assert step.sql == "SELECT 42 AS n"
        assert step.data is not None
        assert list(step.data.columns) == ["n"]
        assert step.data.iloc[0, 0] == 42

    def test_user_feedback_reject_suspend_reason_prompt(self) -> None:
        from aetherdialect._constants import PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT
        from aetherdialect._contracts_base import PipelineSuspended
        from aetherdialect._main_execution import SESSION_PROMPT_REASON, PipelineSession

        sess = PipelineSession(MagicMock())
        ex = PipelineSuspended(PIPELINE_SUSPEND_ID_USER_FEEDBACK_REJECT, "What was wrong?", None)
        step = PipelineSession._suspend_to_step(sess, ex)
        assert step.prompt == SESSION_PROMPT_REASON
        assert "What was wrong?" in (step.message or "")


def test_reader_mode_does_not_save_templates(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from aetherdialect._config import EngineConfig
    from aetherdialect._contracts_core import RuntimeIntent
    from aetherdialect._pipeline import _intent_decline_feedback_bucket
    from aetherdialect._templates import empty_template_store

    store_dir = tmp_path / "intent_templates"
    store_dir.mkdir()
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(store_dir))

    saved: list[int] = []
    monkeypatch.setattr("aetherdialect._pipeline.save_template_store", lambda *_a, **_k: saved.append(1))

    store = empty_template_store("hash_a")
    schema = MagicMock()
    schema.effective_structural_hash = "hash_a"
    intent = RuntimeIntent(
        tables=[],
        grain="scalar",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
    )
    choice_port = MagicMock()
    choice_port.has_pending_choice.return_value = True
    choice_port._consume_next_queued_choice = MagicMock(return_value="wrong table")
    with patch("aetherdialect._pipeline.print_info", lambda *a, **k: None):
        bucket = _intent_decline_feedback_bucket(
            intent,
            store,
            "how many widgets",
            schema,
            choice_port,
            None,
            "default",
            persist_template_learning=False,
        )
    assert saved == []
    assert bucket is not None
    wq = tmp_path / "write_queue.jsonl"
    assert wq.is_file()
    assert b"feedback_record" in wq.read_bytes()


def test_writer_drain_applies_reader_event(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from datetime import datetime, timezone

    from aetherdialect._config import EngineConfig
    from aetherdialect._constants import AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD
    from aetherdialect._contracts_base import WriteQueueEvent
    from aetherdialect._contracts_core import (
        FeedbackKind,
        QuestionFeedbackEntry,
        RejectionBucket,
    )
    from aetherdialect._core_utils import emit_write_queue_event
    from aetherdialect._main_execution import drain_write_queue
    from aetherdialect._templates import empty_template_store, store_to_templates

    store_dir = tmp_path / "intent_templates"
    store_dir.mkdir()
    monkeypatch.setattr(EngineConfig, "TEMPLATE_STORE_DIR", str(store_dir))

    store = empty_template_store("h1")
    templates = store_to_templates(store)
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.schema_graph_id = "h1"
    owner._schema_graph.effective_structural_hash = "h1"
    owner._store = store
    owner._templates = templates
    owner._rejected = {}
    owner._dialect = None
    owner._artifacts_dir = str(tmp_path)
    owner._audit_emit = MagicMock()

    ts = datetime.now(timezone.utc).isoformat()
    entry = QuestionFeedbackEntry(
        summary="s",
        buckets=(RejectionBucket.OTHER,),
        kind=FeedbackKind.INTENT_REJECTED,
        effective_structural_hash="h1",
        intent_structural_hash="ik",
        intent_payload="{}",
        created_at=ts,
        updated_at=ts,
    )
    ev = WriteQueueEvent(
        kind="feedback_record",
        schema_graph_id="h1",
        schema_hash="h1",
        produced_at=ts,
        payload=(("q_norm", "q1"), ("entry_json", json.dumps(entry.to_dict()))),
    )
    emit_write_queue_event(str(tmp_path), ev)

    saves: list[int] = []
    monkeypatch.setattr(
        "aetherdialect._main_execution.save_template_store",
        lambda s: saves.append(1),
    )
    n = drain_write_queue(owner, str(tmp_path))
    assert n == 1
    assert saves == [1]
    assert "q1" in store.question_feedback
    owner._audit_emit.assert_called_once()
    assert owner._audit_emit.call_args[0][0] == AUDIT_EVENT_WRITE_QUEUE_FEEDBACK_RECORD


def test_writer_lock_reentered_on_resume() -> None:
    from contextlib import nullcontext

    from aetherdialect._contracts_base import PipelineSuspended
    from aetherdialect._main_execution import PipelineSession

    lock = MagicMock()
    owner = MagicMock()
    owner._pipeline_writer_lock = lock
    owner._runtime_config = MagicMock()
    owner._runtime_config.llm_execution = None

    sess = PipelineSession(owner, mode="writer")
    sess._suspended = PipelineSuspended("st", "m", MagicMock())

    with patch(
        "aetherdialect._main_execution.llm_execution_scope",
        lambda *_a, **_k: nullcontext(),
    ):
        with patch("aetherdialect._main_execution.dispatch_pipeline_resume", lambda s, e: None):
            sess._resume_from_suspend()
    assert lock.__enter__.call_count >= 1
