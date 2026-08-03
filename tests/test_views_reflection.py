"""Unit tests for views-only and mixed table/view schema reflection."""

from __future__ import annotations

import pytest

from aetherdialect._config import DuckDBRuntimeConfig, EngineConfig
from aetherdialect._contracts_base import ConfigError, EngineContext
from aetherdialect._dialect import get_dialect

_ORIG_ENGINE_TYPE = EngineConfig.TYPE
_ORIG_ENGINE_RUNTIME = EngineConfig.RUNTIME


@pytest.fixture(autouse=True)
def _reset_duckdb_runtime_config() -> None:
    orig_path = DuckDBRuntimeConfig.DATABASE_PATH
    orig_schema = DuckDBRuntimeConfig.SCHEMA
    orig_connection = DuckDBRuntimeConfig.NATIVE_CONNECTION
    EngineConfig.SCHEMA_JSON_PATH = ""
    EngineConfig.TYPE = "duckdb"
    EngineConfig.RUNTIME = DuckDBRuntimeConfig
    try:
        DuckDBRuntimeConfig.DATABASE_PATH = ":memory:"
        DuckDBRuntimeConfig.SCHEMA = "main"
        DuckDBRuntimeConfig.clear_attached_connection()
        yield
    finally:
        DuckDBRuntimeConfig.DATABASE_PATH = orig_path
        DuckDBRuntimeConfig.SCHEMA = orig_schema
        DuckDBRuntimeConfig.NATIVE_CONNECTION = orig_connection
        EngineConfig.TYPE = _ORIG_ENGINE_TYPE
        EngineConfig.RUNTIME = _ORIG_ENGINE_RUNTIME


def _seed_views_catalog(connection: object) -> None:
    connection.execute("CREATE TABLE payments (store_id INTEGER, amount DOUBLE)")
    connection.execute("INSERT INTO payments VALUES (1, 10.0), (2, 20.0)")
    connection.execute(
        "CREATE VIEW store_revenue_v AS SELECT store_id, SUM(amount) AS total_revenue FROM payments GROUP BY store_id"
    )


def test_views_only_reflection() -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect(":memory:")
    _seed_views_catalog(connection)
    dialect = get_dialect("duckdb", DuckDBRuntimeConfig, native_connection=connection)
    ctx = EngineContext(include="views")
    graph = dialect.reflect_schema_graph(include=ctx.include)
    assert set(graph.tables) == {"store_revenue_v"}
    assert all(tbl.kind == "view" for tbl in graph.tables.values())


def test_tables_only_reflection() -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect(":memory:")
    _seed_views_catalog(connection)
    dialect = get_dialect("duckdb", DuckDBRuntimeConfig, native_connection=connection)
    ctx = EngineContext(include="tables")
    graph = dialect.reflect_schema_graph(include=ctx.include)
    assert set(graph.tables) == {"payments"}
    assert all(tbl.kind == "table" for tbl in graph.tables.values())


def test_engine_context_rejects_include_both() -> None:
    with pytest.raises(ConfigError, match="include must be 'tables' or 'views'"):
        EngineContext(include="both")
