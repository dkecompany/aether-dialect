"""Pytest fixtures for live pipeline tests against a real database. Bootstraps a ``AetherEngine`` instance using a temporary TOML file built from the live ``KEY=value`` env file, redirects artifact storage to a ``livetest_`` prefixed directory (separate from interactive artifacts), wipes only the template store at the start of every session so each run starts clean, and builds a ``LiveTestRunner``."""

from __future__ import annotations

import copy
import glob
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from aetherdialect import AetherEngine
from aetherdialect._config import (
    EngineConfig,
    PolicyConfig,
    PostgresRuntimeConfig,
    QSimConfig,
)
from aetherdialect._contracts_base import EngineContext, SensitivityClassification
from aetherdialect._core_utils import (
    StepResult,
    append_failure_trace,
    llm_usage_build_scope,
    llm_usage_question_scope,
    llm_usage_session_scope,
)
from aetherdialect._live_testing import LiveTestRunner
from aetherdialect._main_execution import load_schema_context_cache, write_schema_context_cache
from aetherdialect._schema_overrides import (
    apply_schema_overrides_to_graph,
    load_schema_overrides_file,
)
from aetherdialect._templates import (
    load_template_store,
    store_to_templates,
)

from ._invoice import clear_invoice_file, write_invoice_file

_RENTAL_SHOP_OVERRIDES = Path(__file__).resolve().parent.parent / "scripts" / "data" / "rental_shop_overrides.json"

_RESULTS_FILE = Path(__file__).parent / "results.txt"

_step_results: dict[str, StepResult] = {}
_NODEID_SCENARIO_IDS: dict[str, list[str]] = {}
_CURRENT_TEST_NODEID: str | None = None
_results_trace_pending_sep = False


def _clear_results_file() -> None:
    _RESULTS_FILE.write_text("", encoding="utf-8")


def _append_failure_trace(step: StepResult | list[StepResult] | object | None) -> None:
    append_failure_trace(step, _RESULTS_FILE)


def pytest_sessionstart(session: pytest.Session) -> None:
    global _results_trace_pending_sep
    _ = session
    _results_trace_pending_sep = False
    _clear_results_file()
    clear_invoice_file()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _ = session, exitstatus
    write_invoice_file()


def _is_databricks_live_nodeid(nodeid: str) -> bool:
    """Return True when the test lives under Databricks-only live modules."""
    path = nodeid.replace("\\", "/")
    return "test_databricks.py::" in path or "test_databricks_dialect.py::" in path


def _is_engine_specific_nodeid(nodeid: str) -> bool:
    """Return True when the test lives under a non-PostgreSQL engine- specific live module."""
    from ._engine_live import ENGINE_MODULE_FRAGMENTS

    path = nodeid.replace("\\", "/")
    return any(f"{fragment}" in path for fragment in ENGINE_MODULE_FRAGMENTS)


def pytest_collection_modifyitems(config: pytest.Config, items: list[Any]) -> None:
    """Order engine-specific live items last (live tests are never tagged ``fast``)."""
    _ = config
    engine_items = [it for it in items if _is_engine_specific_nodeid(it.nodeid)]
    rest = [it for it in items if it not in engine_items]
    items[:] = rest + engine_items


def pytest_runtest_setup(item: Any) -> None:
    _ = item


@pytest.fixture(autouse=True)
def _bind_live_step_nodeid(request: pytest.FixtureRequest) -> Any:
    """Bind the active pytest nodeid so captured StepResults map to ``results.txt`` for non-parametrised tests."""
    global _CURRENT_TEST_NODEID
    previous = _CURRENT_TEST_NODEID
    _CURRENT_TEST_NODEID = request.node.nodeid
    yield
    _CURRENT_TEST_NODEID = previous


def _relax_rental_shop_selectability(schema: Any, database_name: str) -> None:
    """Mark every non-selectable column as selectable and clear restricted/hidden sensitivity for rental_shop-shaped live databases."""
    if "rental_shop" not in (database_name or "").lower():
        return
    for table in schema.tables.values():
        for column in table.columns.values():
            if getattr(column, "sensitivity", None) not in (
                None,
                SensitivityClassification.NONE,
            ):
                column.sensitivity = SensitivityClassification.NONE
            column.distinct_count = max(column.distinct_count or 0, 2)
            column.null_ratio = 0.0
            column.mode_frequency_ratio = 0.0


def _parse_live_env_file(path: str) -> dict[str, str]:
    """Parse a UTF-8 ``KEY=value`` environment file into a flat mapping of configuration keys."""
    raw = Path(path).read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    out: dict[str, str] = {}
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        out[key] = value
    return out


def _toml_emit_section(full_name: str, table: Mapping[str, Any], lines: list[str]) -> None:
    scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
    nested = {k: v for k, v in table.items() if isinstance(v, dict)}
    if scalars:
        lines.append(f"[{full_name}]")
        for k, v in scalars.items():
            lines.append(f"{k} = {json.dumps(str(v))}")
    for child_name, child_table in nested.items():
        _toml_emit_section(f"{full_name}.{child_name}", child_table, lines)


def _flat_live_env_to_nested_document(flat: dict[str, str]) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    openai: dict[str, str] = {}
    if v := flat.get("OPENAI_API_KEY"):
        openai["api_key"] = v
    if v := flat.get("OPENAI_BASE_URL"):
        openai["base_url"] = v
    if openai:
        doc["openai"] = openai
    azure: dict[str, Any] = {}
    if v := flat.get("AZURE_OPENAI_ENDPOINT"):
        azure["endpoint"] = v
    if v := flat.get("AZURE_OPENAI_API_KEY"):
        azure["api_key"] = v
    if v := flat.get("AZURE_OPENAI_API_VERSION"):
        azure["api_version"] = v
    if v := flat.get("AZURE_OPENAI_BASE_URL"):
        azure["base_url"] = v
    deployments: dict[str, str] = {}
    if v := flat.get("AZURE_OPENAI_DEPLOYMENT_LIGHT"):
        deployments["light"] = v
    if v := flat.get("AZURE_OPENAI_DEPLOYMENT_HEAVY"):
        deployments["heavy"] = v
    if deployments:
        azure["deployments"] = deployments
    if azure:
        doc["azure_openai"] = azure

    def _first_nonempty(*keys: str) -> str:
        for k in keys:
            raw = flat.get(k)
            if raw is None:
                continue
            t = str(raw).strip()
            if t:
                return t
        return ""

    pg: dict[str, str] = {}
    if v := _first_nonempty("POSTGRES_HOST", "PGHOST", "PGHOSTADDR"):
        pg["host"] = v
    if v := _first_nonempty("POSTGRES_PORT", "PGPORT"):
        pg["port"] = v
    if v := _first_nonempty("POSTGRES_DB", "PGDATABASE"):
        pg["database"] = v
    if v := _first_nonempty("POSTGRES_SCHEMA", "PGSCHEMA"):
        pg["schema"] = v
    if v := _first_nonempty("POSTGRES_USER", "PGUSER"):
        pg["user"] = v
    if v := _first_nonempty("POSTGRES_PASSWORD", "PGPASSWORD"):
        pg["password"] = v
    if pg:
        doc["postgresql"] = pg
    dbx: dict[str, str] = {}
    if v := flat.get("DATABRICKS_HOST"):
        dbx["host"] = v
    if v := flat.get("DATABRICKS_HTTP_PATH"):
        dbx["http_path"] = v
    if v := _first_nonempty("DATABRICKS_ACCESS_TOKEN", "DATABRICKS_TOKEN"):
        dbx["access_token"] = v
    if v := flat.get("DATABRICKS_CATALOG"):
        dbx["catalog"] = v
    if v := flat.get("DATABRICKS_SCHEMA"):
        dbx["schema"] = v
    if dbx:
        doc["databricks"] = dbx
    mysql: dict[str, str] = {}
    if v := _first_nonempty("MYSQL_HOST", "MYSQL_SERVER", "MYSQL_HOSTNAME"):
        mysql["host"] = v
    if v := _first_nonempty("MYSQL_PORT"):
        mysql["port"] = v
    if v := _first_nonempty("MYSQL_USER"):
        mysql["user"] = v
    if v := _first_nonempty("MYSQL_PASSWORD"):
        mysql["password"] = v
    if v := _first_nonempty("MYSQL_DATABASE"):
        mysql["database"] = v
    if mysql:
        doc["mysql"] = mysql
    mariadb: dict[str, str] = {}
    if v := _first_nonempty("MARIADB_HOST", "MARIADB_SERVER"):
        mariadb["host"] = v
    if v := _first_nonempty("MARIADB_PORT"):
        mariadb["port"] = v
    if v := _first_nonempty("MARIADB_USER", "MARIADB_USERNAME"):
        mariadb["user"] = v
    if v := _first_nonempty("MARIADB_PASSWORD", "MARIADB_PWD"):
        mariadb["password"] = v
    if v := _first_nonempty("MARIADB_DATABASE", "MARIADB_DB"):
        mariadb["database"] = v
    if mariadb:
        doc["mariadb"] = mariadb
    duckdb_doc: dict[str, str] = {}
    if v := _first_nonempty(
        "DUCKDB_PATH",
        "DUCKDB_DATABASE",
        "DUCKDB_DATABASE_PATH",
        "DUCKDB_FILE",
        "DUCKDB_DB",
        "DUCKDB_DSN",
    ):
        duckdb_doc["path"] = v
    if v := _first_nonempty("DUCKDB_SCHEMA", "DUCKDB_DEFAULT_SCHEMA"):
        duckdb_doc["schema"] = v
    if duckdb_doc:
        doc["duckdb"] = duckdb_doc
    sqlite_doc: dict[str, str] = {}
    if v := _first_nonempty(
        "SQLITE_PATH",
        "SQLITE_DATABASE",
        "SQLITE_DATABASE_PATH",
        "SQLITE_FILE",
        "SQLITE_DB",
        "SQLITE_DSN",
        "SQLITE3_DATABASE",
    ):
        sqlite_doc["path"] = v
    if sqlite_doc:
        doc["sqlite"] = sqlite_doc
    sqlserver: dict[str, str] = {}
    if v := _first_nonempty("SQLSERVER_HOST", "MSSQL_HOST"):
        sqlserver["host"] = v
    if v := _first_nonempty("SQLSERVER_PORT", "MSSQL_PORT"):
        sqlserver["port"] = v
    if v := _first_nonempty("SQLSERVER_USER", "MSSQL_USER"):
        sqlserver["user"] = v
    if v := _first_nonempty("SQLSERVER_PASSWORD", "MSSQL_PASSWORD"):
        sqlserver["password"] = v
    if v := _first_nonempty("SQLSERVER_DATABASE", "MSSQL_DATABASE"):
        sqlserver["database"] = v
    if v := _first_nonempty("SQLSERVER_SCHEMA", "MSSQL_SCHEMA"):
        sqlserver["schema"] = v
    if v := _first_nonempty("SQLSERVER_DRIVER", "MSSQL_DRIVER"):
        sqlserver["driver"] = v
    if v := _first_nonempty("SQLSERVER_AUTH_MODE", "MSSQL_AUTH_MODE"):
        sqlserver["auth_mode"] = v
    if v := flat.get("SQLSERVER_TENANT_ID"):
        sqlserver["tenant_id"] = v
    if v := flat.get("SQLSERVER_CLIENT_ID"):
        sqlserver["client_id"] = v
    if v := flat.get("SQLSERVER_CLIENT_SECRET"):
        sqlserver["client_secret"] = v
    if sqlserver:
        doc["sqlserver"] = sqlserver
    snowflake: dict[str, str] = {}
    if v := flat.get("SNOWFLAKE_ACCOUNT"):
        snowflake["account"] = v
    if v := flat.get("SNOWFLAKE_USER"):
        snowflake["user"] = v
    if v := flat.get("SNOWFLAKE_PASSWORD"):
        snowflake["password"] = v
    if v := flat.get("SNOWFLAKE_DATABASE"):
        snowflake["database"] = v
    if v := flat.get("SNOWFLAKE_SCHEMA"):
        snowflake["schema"] = v
    if v := flat.get("SNOWFLAKE_WAREHOUSE"):
        snowflake["warehouse"] = v
    if v := flat.get("SNOWFLAKE_ROLE"):
        snowflake["role"] = v
    if v := flat.get("SNOWFLAKE_PRIVATE_KEY_PATH"):
        snowflake["private_key_path"] = v
    if v := flat.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"):
        snowflake["private_key_passphrase"] = v
    if v := flat.get("SNOWFLAKE_AUTHENTICATOR"):
        snowflake["authenticator"] = v
    if v := flat.get("SNOWFLAKE_OAUTH_TOKEN"):
        snowflake["oauth_token"] = v
    if snowflake:
        doc["snowflake"] = snowflake
    bigquery: dict[str, str] = {}
    if v := _first_nonempty("BIGQUERY_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"):
        bigquery["project"] = v
    if v := flat.get("BIGQUERY_DATASET"):
        bigquery["dataset"] = v
    if v := _first_nonempty("BIGQUERY_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"):
        bigquery["credentials_path"] = v
    if v := flat.get("BIGQUERY_LOCATION"):
        bigquery["location"] = v
    if bigquery:
        doc["bigquery"] = bigquery
    redshift: dict[str, str] = {}
    if v := _first_nonempty("REDSHIFT_HOST", "REDSHIFT_SERVER"):
        redshift["host"] = v
    if v := _first_nonempty("REDSHIFT_PORT", "REDSHIFT_TCP_PORT"):
        redshift["port"] = v
    if v := _first_nonempty("REDSHIFT_USER", "REDSHIFT_USERNAME"):
        redshift["user"] = v
    if v := _first_nonempty("REDSHIFT_PASSWORD", "REDSHIFT_PWD"):
        redshift["password"] = v
    if v := _first_nonempty("REDSHIFT_DATABASE", "REDSHIFT_DB"):
        redshift["database"] = v
    if v := _first_nonempty("REDSHIFT_SCHEMA"):
        redshift["schema"] = v
    if v := _first_nonempty("REDSHIFT_USE_IAM", "REDSHIFT_IAM"):
        redshift["use_iam"] = v
    if v := _first_nonempty("REDSHIFT_CLUSTER_IDENTIFIER", "REDSHIFT_CLUSTER_ID"):
        redshift["cluster_identifier"] = v
    if v := _first_nonempty("REDSHIFT_WORKGROUP", "REDSHIFT_SERVERLESS_WORKGROUP"):
        redshift["workgroup"] = v
    if v := _first_nonempty("REDSHIFT_REGION", "REDSHIFT_AWS_REGION"):
        redshift["region"] = v
    if redshift:
        doc["redshift"] = redshift
    engine: dict[str, str] = {}
    if v := flat.get("AETHERDIALECT_ENGINE"):
        engine["selected"] = v
    if engine:
        doc["engine"] = engine
    execution: dict[str, str] = {}
    if v := flat.get("AETHERDIALECT_MAX_QUERY_COST_ROWS"):
        execution["max_query_cost_rows"] = v
    if v := flat.get("AETHERDIALECT_MAX_QUERY_COST_BYTES"):
        execution["max_query_cost_bytes"] = v
    if v := flat.get("AETHERDIALECT_STATEMENT_TIMEOUT_MS"):
        execution["statement_timeout_ms"] = v
    if v := flat.get("AETHERDIALECT_LLM_TIMEOUT_MS"):
        execution["llm_timeout_ms"] = v
    if v := flat.get("AETHERDIALECT_PROFILE_TIMEOUT_MS"):
        execution["profile_timeout_ms"] = v
    if v := flat.get("AETHERDIALECT_EXPLAIN_TIMEOUT_MS"):
        execution["explain_timeout_ms"] = v
    llm_flat: dict[str, str] = {}
    if v := flat.get("AETHERDIALECT_LLM_PROVIDER"):
        llm_flat["provider"] = v
    if llm_flat:
        doc["llm"] = llm_flat
    return doc


def _nested_document_to_toml_str(doc: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for name in (
        "openai",
        "azure_openai",
        "postgresql",
        "databricks",
        "mysql",
        "mariadb",
        "duckdb",
        "sqlite",
        "sqlserver",
        "snowflake",
        "bigquery",
        "redshift",
        "engine",
        "llm",
        "execution",
    ):
        if name not in doc:
            continue
        _toml_emit_section(name, doc[name], lines)
    return "\n".join(lines) + ("\n" if lines else "")


def write_live_env_file_to_temp_config_toml(env_path: str, extra_flat: dict[str, str] | None = None) -> str:
    """Materialise a ``KEY=value`` live env file as a temporary TOML file understood by :func:`_load_config_file`. Callers must delete the returned path when finished."""
    flat = _parse_live_env_file(env_path)
    if extra_flat:
        flat = {**flat, **extra_flat}
    doc = _flat_live_env_to_nested_document(flat)
    fd, path = tempfile.mkstemp(prefix="live_aetherdialect_", suffix=".toml")
    os.close(fd)
    Path(path).write_text(_nested_document_to_toml_str(doc), encoding="utf-8")
    return path


def write_sandbox_recording_toml(env_path: str) -> str:
    """Materialise sandbox corpus recording config: LLM creds from *env_path*, in-memory DuckDB."""
    return write_live_env_file_to_temp_config_toml(
        env_path,
        {
            "AETHERDIALECT_ENGINE": "duckdb",
            "AETHERDIALECT_LLM_PROVIDER": "openai",
            "DUCKDB_PATH": ":memory:",
            "DUCKDB_SCHEMA": "main",
        },
    )


def _env_file() -> str:
    raw = os.environ.get("LIVE_ENV_FILE", "env.env")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / raw
    return str(p)


def _domain_notes_path() -> Path | None:
    """Schema-graph domain notes (``scripts/data/rental_shop_notes.txt`` by default)."""
    raw = os.environ.get(
        "LIVE_DOMAIN_NOTES_FILE",
        os.path.join("scripts", "data", "rental_shop_notes.txt"),
    )
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / raw
    return p if p.is_file() else None


def _pg_param(name: str, default: str) -> str:
    return os.environ.get(name, default)


_RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS = frozenset(
    {
        "actor",
        "address",
        "category",
        "city",
        "country",
        "customer",
        "film",
        "film_actor",
        "item",
        "item_category",
        "inventory",
        "language",
        "payment",
        "rental",
        "store",
    }
)

_CONSUMER_CREDENTIALS_SKIP_REASON = "Set PGUSER2 and PGPASSWORD2 in the live env file for permission live tests."
_POSTGRES_SKIP_REASON = "Set PGHOST, PGUSER, PGPASSWORD, and PGDATABASE in the live env file for PostgreSQL live tests."


def _consumer_credentials_from_env() -> tuple[str, str]:
    """Read restricted consumer Postgres credentials from the live env file or process environment."""
    flat = _parse_live_env_file(_env_file())
    user = (flat.get("PGUSER2") or os.environ.get("PGUSER2") or "").strip()
    password = (flat.get("PGPASSWORD2") or os.environ.get("PGPASSWORD2") or "").strip()
    return user, password


def _consumer_credentials_configured() -> bool:
    """Return True when both PGUSER2 and PGPASSWORD2 are available for consumer initialization."""
    user, password = _consumer_credentials_from_env()
    return bool(user and password)


def _postgres_credentials_configured() -> bool:
    """Return True when primary PostgreSQL credentials are present in the live env file."""
    flat = _parse_live_env_file(_env_file())
    for key in ("PGUSER", "PGPASSWORD", "PGHOST", "PGDATABASE"):
        if not (flat.get(key) or os.environ.get(key) or "").strip():
            return False
    return True


def _rbac_consumer_configured() -> bool:
    """Return True when both owner and consumer PostgreSQL credentials are configured."""
    return _postgres_credentials_configured() and _consumer_credentials_configured()


def _owner_engine_context(
    *, allow_objects: frozenset[str] | None = None, deny_columns: frozenset[str] | None = None
) -> EngineContext:
    """Build a master EngineContext for rental_shop owner live fixtures."""
    notes = _domain_notes_path()
    kwargs: dict[str, Any] = {
        "notes_file": str(notes) if notes else None,
        "sql_file": _pg_param("SQL_FILE", os.path.join("scripts", "data", "rental_shop.sql")),
    }
    if allow_objects is not None:
        kwargs["allow_objects"] = allow_objects
    if deny_columns is not None:
        kwargs["deny_columns"] = deny_columns
    return EngineContext(**kwargs)


def _consumer_engine_context() -> EngineContext:
    """Build the restricted EngineContext aligned with pguser2 database grants."""
    return _owner_engine_context(allow_objects=_RENTAL_SHOP_CONSUMER_ALLOW_OBJECTS)


def _copy_session_schema_cache(source_artifacts_dir: str, dest_artifacts_dir: str) -> None:
    """Copy a built schema graph cache into a fresh RBAC artifacts dir (skip full reflect)."""
    os.makedirs(dest_artifacts_dir, exist_ok=True)
    for name in ("schema_graph.json.gz", "artifact_manifest.json", "schema_context.json"):
        src = os.path.join(source_artifacts_dir, name)
        dst = os.path.join(dest_artifacts_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    src_dir = source_artifacts_dir
    for sidecar in glob.glob(os.path.join(src_dir, "schema_context*.json")):
        base = os.path.basename(sidecar)
        if base == "schema_context.json":
            continue
        dst = os.path.join(dest_artifacts_dir, base)
        if not os.path.isfile(dst):
            shutil.copy2(sidecar, dst)


def _build_rbac_owner_engine(
    *,
    engine_context: EngineContext | None = None,
    relax_sensitivity: bool = True,
    config_file: str | None = None,
    schema_cache_source: str | None = None,
) -> AetherEngine:
    """Construct an owner ``AetherEngine`` for RBAC live tests."""
    ctx = engine_context or _owner_engine_context()
    owns_config = config_file is None
    if config_file is None:
        config_file = write_live_env_file_to_temp_config_toml(
            _env_file(),
            {"AETHERDIALECT_ENGINE": "postgresql"},
        )
    seed_cache = schema_cache_source is not None
    try:
        artifacts_dir = tempfile.mkdtemp(prefix="live_rbac_owner_")
        if seed_cache:
            _copy_session_schema_cache(schema_cache_source, artifacts_dir)
        prev_regen_graph = PolicyConfig.REGENERATE_SCHEMA_GRAPH
        if seed_cache:
            PolicyConfig.REGENERATE_SCHEMA_GRAPH = False
        try:
            instance = AetherEngine(
                ctx,
                artifacts_dir=artifacts_dir,
                config_file=config_file,
                role="owner",
            )
        finally:
            PolicyConfig.REGENERATE_SCHEMA_GRAPH = prev_regen_graph
        _redirect_to_livetest_dir(instance)
        master_ctx = instance._runtime_config.engine_context
        if load_schema_context_cache(str(instance._artifacts_dir)) is None:
            write_schema_context_cache(str(instance._artifacts_dir), master_ctx)
        if relax_sensitivity:
            _relax_rental_shop_selectability(
                instance._schema_graph,
                _pg_param("PGDATABASE", "rental_shop"),
            )
        return instance
    finally:
        if owns_config:
            Path(config_file).unlink(missing_ok=True)


def _build_rbac_consumer_engine(
    owner: AetherEngine,
    *,
    engine_context: EngineContext | str | None = None,
    config_file: str | None = None,
) -> AetherEngine:
    """Construct a consumer ``AetherEngine`` sharing the owner artifact directory."""
    pguser2, pgpassword2 = _consumer_credentials_from_env()
    owns_config = config_file is None
    if config_file is None:
        config_file = write_live_env_file_to_temp_config_toml(
            _env_file(),
            {
                "AETHERDIALECT_ENGINE": "postgresql",
                "PGUSER": pguser2,
                "PGPASSWORD": pgpassword2,
            },
        )
    ctx = engine_context if engine_context is not None else _consumer_engine_context()
    try:
        instance = AetherEngine(
            ctx,
            artifacts_dir=str(owner._artifacts_dir),
            config_file=config_file,
            role="consumer",
        )
        _relax_rental_shop_selectability(
            instance._schema_graph,
            _pg_param("PGDATABASE", "rental_shop"),
        )
        return instance
    finally:
        if owns_config:
            Path(config_file).unlink(missing_ok=True)


def _redirect_to_livetest_dir(t2s: AetherEngine) -> str:
    """Swap artifact paths to the livetest directory and return its absolute path."""
    original = t2s._artifacts_dir
    parent = os.path.dirname(original)
    folder = os.path.basename(original)
    live_folder = folder.replace("artifacts_", "livetest_", 1)
    if live_folder == folder:
        live_folder = f"livetest_{folder}"
    live_dir = os.path.join(parent, live_folder)

    if os.path.isdir(original):
        if os.path.isdir(live_dir):
            shutil.rmtree(live_dir, ignore_errors=True)
        shutil.copytree(original, live_dir, dirs_exist_ok=True)
    else:
        os.makedirs(live_dir, exist_ok=True)

    schema_dst = os.path.join(live_dir, "schema_graph.json.gz")

    template_store_dir = os.path.join(live_dir, "intent_templates")
    if os.path.isdir(template_store_dir):
        shutil.rmtree(template_store_dir, ignore_errors=True)

    t2s._artifacts_dir = live_dir
    EngineConfig.SCHEMA_JSON_PATH = schema_dst
    EngineConfig.TEMPLATE_STORE_DIR = template_store_dir
    QSimConfig.SKELETONS_JSON_PATH = os.path.join(live_dir, "qsim_skeletons.json.gz")

    return live_dir


@pytest.fixture(scope="session", autouse=True)
def _llm_usage_invoice_session() -> Any:
    """Keep one session-scoped LLM usage accumulator for ``invoice.txt``."""
    with llm_usage_session_scope():
        yield


def _build_live_aether_engine(*, relax_sensitivity: bool) -> AetherEngine:
    """Construct a session ``AetherEngine`` instance for rental_shop live tests."""
    notes = _domain_notes_path()
    cfg_path = write_live_env_file_to_temp_config_toml(_env_file(), {"AETHERDIALECT_ENGINE": "postgresql"})
    try:
        with llm_usage_build_scope():
            instance = AetherEngine(
                EngineContext(
                    notes_file=str(notes) if notes else None,
                    sql_file=_pg_param("SQL_FILE", os.path.join("scripts", "data", "rental_shop.sql")),
                ),
                artifacts_dir=tempfile.mkdtemp(prefix="live_pg_artifacts_"),
                config_file=cfg_path,
            )

            _redirect_to_livetest_dir(instance)

            prev_regen_graph = PolicyConfig.REGENERATE_SCHEMA_GRAPH
            prev_regen_skeleton = PolicyConfig.REGENERATE_SKELETON_CACHE
            PolicyConfig.REGENERATE_SCHEMA_GRAPH = True
            PolicyConfig.REGENERATE_SKELETON_CACHE = True
            try:
                print("Live tests: building schema graph from PostgreSQL...", flush=True)
                schema_graph = instance._schema_graph
                table_count = len(schema_graph.tables)
                column_count = sum(len(table.columns) for table in schema_graph.tables.values())
                print(
                    f"Live tests: schema graph ready ({table_count} tables, {column_count} columns)",
                    flush=True,
                )
            finally:
                PolicyConfig.REGENERATE_SCHEMA_GRAPH = prev_regen_graph
                PolicyConfig.REGENERATE_SKELETON_CACHE = prev_regen_skeleton

            fresh_store = load_template_store(
                instance._schema_graph.effective_structural_hash,
                instance._schema_graph,
            )
            instance._store = fresh_store
            instance._templates = store_to_templates(fresh_store)
            instance._rejected = {}

            if relax_sensitivity:
                _relax_rental_shop_selectability(instance._schema_graph, _pg_param("PGDATABASE", "rental_shop"))
            elif _RENTAL_SHOP_OVERRIDES.is_file():
                apply_schema_overrides_to_graph(
                    instance._schema_graph,
                    load_schema_overrides_file(_RENTAL_SHOP_OVERRIDES),
                )

            return instance
    finally:
        Path(cfg_path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def t2s() -> AetherEngine:
    """Session-scoped ``AetherEngine`` instance with a clean livetest artifact dir."""
    return _build_live_aether_engine(relax_sensitivity=True)


def _derive_enforce_sensitivity_engine(t2s: AetherEngine) -> AetherEngine:
    """Derive a session engine with bundled sensitivity overrides from the primary ``t2s`` build."""
    enforced_graph = copy.deepcopy(t2s._schema_graph)
    if _RENTAL_SHOP_OVERRIDES.is_file():
        apply_schema_overrides_to_graph(
            enforced_graph,
            load_schema_overrides_file(_RENTAL_SHOP_OVERRIDES),
        )
    derived = object.__new__(AetherEngine)
    for slot in AetherEngine.__slots__:
        setattr(derived, slot, getattr(t2s, slot))
    derived._schema_graph = enforced_graph
    derived._store = load_template_store(enforced_graph.effective_structural_hash, enforced_graph)
    derived._templates = store_to_templates(derived._store)
    derived._rejected = {}
    return derived


@pytest.fixture(scope="session")
def t2s_enforce_sensitivity(t2s: AetherEngine) -> AetherEngine:
    """Session ``AetherEngine`` with bundled sensitivity overrides and no selectability relax."""
    return _derive_enforce_sensitivity_engine(t2s)


@pytest.fixture(autouse=True)
def _enforce_postgresql_dialect(request: pytest.FixtureRequest) -> None:
    """Restore PostgreSQL engine config and owner credentials before each non-engine-module test."""
    from ._engine_live import ENGINE_MODULE_FRAGMENTS

    if any(fragment in request.node.nodeid for fragment in ENGINE_MODULE_FRAGMENTS):
        return
    EngineConfig.TYPE = "postgresql"
    EngineConfig.RUNTIME = PostgresRuntimeConfig
    flat = _parse_live_env_file(_env_file())
    pg_env = {
        k: flat[k]
        for k in ("PGUSER", "PGPASSWORD", "PGHOST", "PGPORT", "PGDATABASE")
        if k in flat and str(flat[k]).strip()
    }
    if pg_env:
        PostgresRuntimeConfig.apply_environment(pg_env)


@pytest.fixture(scope="session")
def schema(t2s: AetherEngine) -> Any:
    """Profiled ``SchemaGraph`` from the session ``AetherEngine`` instance."""
    return t2s._schema_graph


@pytest.fixture(scope="session")
def store(t2s: AetherEngine) -> dict[str, Any]:
    """Template store dict from the session ``AetherEngine`` instance."""
    return t2s._store


@pytest.fixture(scope="session")
def templates(t2s: AetherEngine) -> dict:
    """Accepted templates dict from the session ``AetherEngine`` instance."""
    return t2s._templates


@pytest.fixture(scope="session")
def rejected(t2s: AetherEngine) -> dict:
    """Rejected templates dict from the session ``AetherEngine`` instance."""
    return t2s._rejected


@pytest.fixture(scope="session")
def schema_terms(t2s: AetherEngine) -> set[str]:
    """Schema term tokens from the session ``AetherEngine`` instance."""
    return t2s._schema_terms


@pytest.fixture(scope="session")
def runner(schema, store, templates, rejected, schema_terms, t2s) -> LiveTestRunner:
    """Session-scoped ``LiveTestRunner`` wired to the test database resources."""
    r = LiveTestRunner(
        schema=schema,
        store=store,
        templates=templates,
        rejected=rejected,
        schema_terms=schema_terms,
        csv_dir=t2s._artifacts_dir,
        dialect=t2s._dialect,
    )
    _instrument_runner(r)
    return r


@pytest.fixture(scope="session")
def schema_enforce_sensitivity(t2s_enforce_sensitivity: AetherEngine) -> Any:
    """Profiled schema graph with bundled sensitivity overrides applied."""
    return t2s_enforce_sensitivity._schema_graph


@pytest.fixture(scope="session")
def store_enforce_sensitivity(t2s_enforce_sensitivity: AetherEngine) -> dict[str, Any]:
    return t2s_enforce_sensitivity._store


@pytest.fixture(scope="session")
def templates_enforce_sensitivity(t2s_enforce_sensitivity: AetherEngine) -> dict:
    return t2s_enforce_sensitivity._templates


@pytest.fixture(scope="session")
def rejected_enforce_sensitivity(t2s_enforce_sensitivity: AetherEngine) -> dict:
    return t2s_enforce_sensitivity._rejected


@pytest.fixture(scope="session")
def schema_terms_enforce_sensitivity(t2s_enforce_sensitivity: AetherEngine) -> set[str]:
    return t2s_enforce_sensitivity._schema_terms


@pytest.fixture(scope="session")
def runner_enforce_sensitivity(
    schema_enforce_sensitivity,
    store_enforce_sensitivity,
    templates_enforce_sensitivity,
    rejected_enforce_sensitivity,
    schema_terms_enforce_sensitivity,
    t2s_enforce_sensitivity,
) -> LiveTestRunner:
    """Live runner that keeps rental_shop sensitivity tiers enforced."""
    r = LiveTestRunner(
        schema=schema_enforce_sensitivity,
        store=store_enforce_sensitivity,
        templates=templates_enforce_sensitivity,
        rejected=rejected_enforce_sensitivity,
        schema_terms=schema_terms_enforce_sensitivity,
        csv_dir=t2s_enforce_sensitivity._artifacts_dir,
        dialect=t2s_enforce_sensitivity._dialect,
    )
    _instrument_runner(r)
    return r


@pytest.fixture(scope="session")
def rbac_postgres_config_path() -> str:
    """Session-persistent TOML config for RBAC PostgreSQL live tests."""
    if not _postgres_credentials_configured():
        pytest.skip(_POSTGRES_SKIP_REASON)
    path = write_live_env_file_to_temp_config_toml(
        _env_file(),
        {"AETHERDIALECT_ENGINE": "postgresql"},
    )
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def rbac_consumer_config_path() -> str:
    """Session-persistent TOML config for pguser2 consumer live tests."""
    if not _consumer_credentials_configured():
        pytest.skip(_CONSUMER_CREDENTIALS_SKIP_REASON)
    pguser2, pgpassword2 = _consumer_credentials_from_env()
    path = write_live_env_file_to_temp_config_toml(
        _env_file(),
        {
            "AETHERDIALECT_ENGINE": "postgresql",
            "PGUSER": pguser2,
            "PGPASSWORD": pgpassword2,
        },
    )
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def t2s_rbac_owner(t2s: AetherEngine, rbac_postgres_config_path: str) -> AetherEngine:
    """Session-scoped owner ``AetherEngine`` for RBAC live tests (full master context)."""
    return _build_rbac_owner_engine(
        config_file=rbac_postgres_config_path,
        schema_cache_source=str(t2s._artifacts_dir),
    )


@pytest.fixture(scope="session")
def t2s_deny_columns_owner(
    t2s: AetherEngine,
    rbac_postgres_config_path: str,
) -> AetherEngine:
    """Session-scoped owner ``AetherEngine`` with ``deny_columns`` on hidden staff columns."""
    return _build_rbac_owner_engine(
        engine_context=_owner_engine_context(
            deny_columns=frozenset({"staff.ssn", "staff.password"}),
        ),
        relax_sensitivity=False,
        config_file=rbac_postgres_config_path,
        schema_cache_source=str(t2s._artifacts_dir),
    )


@pytest.fixture(scope="session")
def t2s_consumer_pguser2(
    t2s_rbac_owner: AetherEngine,
    rbac_consumer_config_path: str,
) -> AetherEngine:
    """Session-scoped consumer ``AetherEngine`` using PGUSER2/PGPASSWORD2 database grants."""
    return _build_rbac_consumer_engine(t2s_rbac_owner, config_file=rbac_consumer_config_path)


@pytest.fixture(scope="session")
def runner_consumer_pguser2(t2s_consumer_pguser2: AetherEngine) -> LiveTestRunner:
    """``LiveTestRunner`` wired to the restricted pguser2 consumer engine."""
    r = LiveTestRunner(
        schema=t2s_consumer_pguser2._schema_graph,
        store=t2s_consumer_pguser2._store,
        templates=t2s_consumer_pguser2._templates,
        rejected=t2s_consumer_pguser2._rejected,
        schema_terms=t2s_consumer_pguser2._schema_terms,
        csv_dir=t2s_consumer_pguser2._artifacts_dir,
        dialect=t2s_consumer_pguser2._dialect,
    )
    _instrument_runner(r)
    return r


def _capture_result(result: StepResult, scenario: Any) -> None:
    """Store a step result so it appears in results.txt diagnostics."""
    _step_results[scenario.id] = result
    nodeid = _CURRENT_TEST_NODEID
    if nodeid:
        bucket = _NODEID_SCENARIO_IDS.setdefault(nodeid, [])
        bucket.append(scenario.id)
    seq_id = getattr(scenario, "sequence_id", None)
    if seq_id:
        _step_results.setdefault(seq_id, [])
        bucket = _step_results[seq_id]
        if isinstance(bucket, list):
            bucket.append(result)


def _instrument_runner(target: LiveTestRunner) -> None:
    """Patch *run*, *run_deferred*, and *clone* to capture every step result."""
    bound_run = LiveTestRunner.run.__get__(target, LiveTestRunner)
    bound_deferred = LiveTestRunner.run_deferred.__get__(target, LiveTestRunner)
    bound_clone = LiveTestRunner.clone.__get__(target, LiveTestRunner)

    def _capturing_run(scenario: Any, retries: int = 0) -> StepResult:
        with llm_usage_question_scope():
            result = bound_run(scenario, retries=retries)
        _capture_result(result, scenario)
        return result

    def _capturing_run_deferred(scenario: Any, retries: int = 0) -> StepResult:
        with llm_usage_question_scope():
            result = bound_deferred(scenario, retries=retries)
        _capture_result(result, scenario)
        return result

    def _capturing_clone() -> LiveTestRunner:
        cloned = bound_clone()
        _instrument_runner(cloned)
        return cloned

    target.run = _capturing_run
    target.run_deferred = _capturing_run_deferred
    target.clone = _capturing_clone


_SCENARIO_ID_RE = re.compile(r"\[([A-Z0-9_-]+(?:-[A-Z0-9]+)*)\]$")


def _resolve_scenario_ids(nodeid: str) -> list[str]:
    """Return scenario ids captured for *nodeid*, else extract from parametrized ``[ID]`` suffix."""
    registered = _NODEID_SCENARIO_IDS.get(nodeid)
    if registered:
        return list(registered)
    match = _SCENARIO_ID_RE.search(nodeid)
    return [match.group(1)] if match else []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or report.outcome != "failed":
        return
    scenario_ids = _resolve_scenario_ids(report.nodeid)
    step: StepResult | list[StepResult] | None = None
    if scenario_ids:
        ordered_steps: list[StepResult] = []
        seen_ids: set[int] = set()
        for sid in scenario_ids:
            candidate = _step_results.get(sid)
            if candidate is None:
                continue
            cid = id(candidate)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            ordered_steps.append(candidate)
        if len(ordered_steps) == 1:
            step = ordered_steps[0]
        elif len(ordered_steps) > 1:
            step = ordered_steps
    _append_failure_trace(step)
