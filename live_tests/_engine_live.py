"""Shared fixture builders for per-engine live test modules. Covers connection, reflection, and engine-specific dialect scenarios for sqlite, duckdb, mysql, mariadb, sqlserver, redshift, databricks, snowflake, and bigquery. PostgreSQL full pipeline coverage runs via ``test_core_pipeline.py``. This module centralises ``AetherEngine`` bootstrap and ``LiveTestRunner`` wiring."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from aetherdialect import AetherEngine
from aetherdialect._config import EngineConfig, QSimConfig
from aetherdialect._contracts_base import EngineContext, SchemaInclude
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._core_utils import write_gzip_json_atomic
from aetherdialect._live_testing import LiveTestRunner
from aetherdialect._templates import load_template_store, store_to_templates

from ._rental_partition_metadata import apply_synthetic_rental_partition_metadata
from .conftest import (
    _domain_notes_path,
    _env_file,
    _instrument_runner,
    _relax_rental_shop_selectability,
    write_live_env_file_to_temp_config_toml,
)

ENGINE_MODULE_FRAGMENTS = (
    "test_databricks",
    "test_mysql",
    "test_sqlserver",
    "test_snowflake",
    "test_bigquery",
    "test_redshift",
    "test_mariadb",
    "test_duckdb",
    "test_sqlite",
    "test_mysql_dialect",
    "test_sqlserver_dialect",
    "test_snowflake_dialect",
    "test_bigquery_dialect",
    "test_redshift_dialect",
    "test_duckdb_dialect",
    "test_sqlite_dialect",
)


_SESSION_ENGINE_CACHE: dict[tuple[str, str], AetherEngine] = {}


_RENTAL_SHOP_VIEWS_SQL = Path(__file__).resolve().parent.parent / "scripts" / "data" / "rental_shop_views.sql"


_RENTAL_SHOP_VIEW_NAMES = ("active_customer_v", "store_revenue_v", "film_catalog_v")


def _apply_rental_shop_views(instance: AetherEngine) -> None:
    """Create rental_shop views on the live engine connection when the DDL file is present."""
    if not _RENTAL_SHOP_VIEWS_SQL.is_file():
        return
    dialect = instance._dialect
    for name in _RENTAL_SHOP_VIEW_NAMES:
        dialect.execute(f'DROP VIEW IF EXISTS "{name}"')
    for stmt in _RENTAL_SHOP_VIEWS_SQL.read_text(encoding="utf-8").split(";"):
        stripped = stmt.strip()
        if stripped:
            dialect.execute(stripped)


def _runtime_schema_name() -> str:
    """Return the active engine schema/database name for information_schema queries."""
    runtime = EngineConfig.RUNTIME
    schema = getattr(runtime, "SCHEMA", None)
    if schema:
        return str(schema)
    database = getattr(runtime, "DATABASE", None)
    if database:
        return str(database)
    return "main"


def _rental_shop_views_from_catalog(instance: AetherEngine) -> dict[str, TableMetadata]:
    """Load rental_shop view columns from the engine catalog."""
    dialect_name = str(getattr(instance._dialect, "name", "") or EngineConfig.TYPE or "").lower()
    if dialect_name == "sqlite":
        grouped: dict[str, dict[str, ColumnMetadata]] = {}
        for view_name in _RENTAL_SHOP_VIEW_NAMES:
            rows = instance._dialect.execute(f'PRAGMA table_info("{view_name}")')
            columns: dict[str, ColumnMetadata] = {}
            for row in rows or ():
                if len(row) < 2:
                    continue
                col_name = str(row[1])
                col_type = str(row[2]) if len(row) > 2 and row[2] is not None else "TEXT"
                columns[col_name] = ColumnMetadata(name=col_name, data_type=col_type)
            if columns:
                grouped[view_name] = columns
        return {
            tname: TableMetadata(
                name=tname,
                columns=columns,
                primary_key=[],
                foreign_keys=[],
                kind="view",
            )
            for tname, columns in grouped.items()
        }

    schema_name = "main" if dialect_name == "duckdb" else _runtime_schema_name()
    names_sql = ", ".join(f"'{name}'" for name in _RENTAL_SHOP_VIEW_NAMES)
    sql = (
        "SELECT table_name, column_name, data_type "
        "FROM information_schema.columns "
        f"WHERE table_schema = '{schema_name}' "
        f"AND table_name IN ({names_sql}) "
        "ORDER BY table_name, ordinal_position"
    )
    rows = instance._dialect.execute(sql)
    grouped: dict[str, dict[str, ColumnMetadata]] = {}
    for table_name, column_name, data_type in rows:
        tname = str(table_name)
        grouped.setdefault(tname, {})
        grouped[tname][str(column_name)] = ColumnMetadata(
            name=str(column_name),
            data_type=str(data_type),
        )
    return {
        tname: TableMetadata(
            name=tname,
            columns=columns,
            primary_key=[],
            foreign_keys=[],
            kind="view",
        )
        for tname, columns in grouped.items()
    }


def _merge_rental_shop_views_into_graph(instance: AetherEngine) -> None:
    """Ensure bundled rental_shop views exist in the database and schema graph."""
    _apply_rental_shop_views(instance)
    for name, table in _rental_shop_views_from_catalog(instance).items():
        instance._schema_graph.tables[name] = table


def reflect_rental_shop_schema_for_live_test(
    instance: AetherEngine,
    *,
    include: SchemaInclude,
) -> SchemaGraph:
    """Reflect rental_shop tables and/or views for local-engine live view tests."""
    view_tables = _rental_shop_views_from_catalog(instance)
    base = instance._schema_graph
    if include == "views":
        tables = dict(view_tables)
        if not tables:
            tables = {name: table for name, table in base.tables.items() if table.kind == "view"}
    elif include == "both":
        tables = {name: table for name, table in base.tables.items() if table.kind != "view"}
        tables.update(view_tables)
    else:
        tables = {name: table for name, table in base.tables.items() if table.kind != "view"}
    return replace(base, tables=tables, include=include)


def _redirect_to_livetest_dir(instance: AetherEngine) -> str:
    """Swap artifact paths from ``artifacts_...`` to ``livetest_...`` for the engine instance."""
    original = instance._artifacts_dir
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
    if not os.path.exists(schema_dst):
        schema_src_gz = os.path.join(original, "schema_graph.json.gz")
        schema_src_json = os.path.join(original, "schema_graph.json")
        if os.path.exists(schema_src_gz):
            shutil.copy2(schema_src_gz, schema_dst)
        elif os.path.exists(schema_src_json):
            with open(schema_src_json, encoding="utf-8") as sf:
                schema_payload = json.load(sf)
            write_gzip_json_atomic(schema_dst, schema_payload, sort_keys=True)

    template_store_dir = os.path.join(live_dir, "intent_templates")
    if os.path.isdir(template_store_dir):
        shutil.rmtree(template_store_dir, ignore_errors=True)

    instance._artifacts_dir = live_dir
    EngineConfig.SCHEMA_JSON_PATH = schema_dst
    EngineConfig.TEMPLATE_STORE_DIR = template_store_dir
    QSimConfig.SKELETONS_JSON_PATH = os.path.join(live_dir, "qsim_skeletons.json.gz")

    return live_dir


def build_engine_t2s(engine_name: str, schema: str) -> AetherEngine:
    """Build a ``AetherEngine`` instance configured for *engine_name* over the rental_shop schema."""
    cache_key = (engine_name, schema)
    cached = _SESSION_ENGINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    sql_file = os.environ.get("SQL_FILE", os.path.join("scripts", "data", "rental_shop.sql"))
    notes = _domain_notes_path()

    cfg_path = write_live_env_file_to_temp_config_toml(_env_file(), {"AETHERDIALECT_ENGINE": engine_name})
    try:
        instance = AetherEngine(
            EngineContext(
                notes_file=str(notes) if notes else None,
                sql_file=sql_file,
            ),
            artifacts_dir=tempfile.mkdtemp(prefix=f"live_{engine_name}_artifacts_"),
            config_file=cfg_path,
        )

        _redirect_to_livetest_dir(instance)

        fresh_store = load_template_store(instance._schema_graph.effective_structural_hash, instance._schema_graph)
        instance._store = fresh_store
        instance._templates = store_to_templates(fresh_store)
        instance._rejected = {}

        _relax_rental_shop_selectability(instance._schema_graph, schema)
        if engine_name in ("duckdb", "sqlite"):
            _merge_rental_shop_views_into_graph(instance)
            apply_synthetic_rental_partition_metadata(instance._schema_graph)

        _SESSION_ENGINE_CACHE[cache_key] = instance
        return instance
    finally:
        Path(cfg_path).unlink(missing_ok=True)


def build_runner(instance: AetherEngine) -> LiveTestRunner:
    """Wire a ``LiveTestRunner`` to an engine ``AetherEngine`` instance and instrument it."""
    r = LiveTestRunner(
        schema=instance._schema_graph,
        store=instance._store,
        templates=instance._templates,
        rejected=instance._rejected,
        schema_terms=instance._schema_terms,
        csv_dir=instance._artifacts_dir,
        dialect=instance._dialect,
    )
    _instrument_runner(r)
    return r


def engine_schema(name: str, default: str) -> str:
    """Resolve the target schema/dataset env var for an engine, falling back to *default*."""
    return os.environ.get(name, default)


def skip_unless_configured(engine_name: str) -> Any:
    """Return a pytest skip marker reason when *engine_name* is not configured in the env file."""
    flat_present = False
    try:
        cfg_path = write_live_env_file_to_temp_config_toml(_env_file(), {"AETHERDIALECT_ENGINE": engine_name})
        text = Path(cfg_path).read_text(encoding="utf-8")
        Path(cfg_path).unlink(missing_ok=True)
        flat_present = f"[{engine_name}]" in text
    except Exception:
        flat_present = False
    return None if flat_present else f"{engine_name} not configured in live env file"
