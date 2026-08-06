"""Unit tests for in-memory DuckDB shared-connection behavior."""

from __future__ import annotations

import pytest

from aetherdialect._config import DuckDBRuntimeConfig, EngineConfig
from aetherdialect._contracts_base import EngineIdentity
from aetherdialect._core_utils import pop_engine_identity, push_engine_identity
from aetherdialect._dialect import DialectRegistry
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect

_ORIG_ENGINE_TYPE = EngineConfig.TYPE
_ORIG_ENGINE_RUNTIME = EngineConfig.RUNTIME


def _seed_memory_table(connection: object) -> None:
    connection.execute("CREATE TABLE items (id INTEGER, name VARCHAR)")
    connection.execute("INSERT INTO items VALUES (1, 'alpha')")


@pytest.fixture(autouse=True)
def _reset_duckdb_runtime_config() -> None:
    orig_path = DuckDBRuntimeConfig.DATABASE_PATH
    orig_schema = DuckDBRuntimeConfig.SCHEMA
    orig_connection = DuckDBRuntimeConfig.NATIVE_CONNECTION
    EngineConfig.SCHEMA_JSON_PATH = ""
    EngineConfig.TYPE = "duckdb"
    EngineConfig.RUNTIME = DuckDBRuntimeConfig
    identity_token = push_engine_identity(EngineIdentity("duckdb", DuckDBRuntimeConfig))
    try:
        DuckDBRuntimeConfig.DATABASE_PATH = ":memory:"
        DuckDBRuntimeConfig.SCHEMA = "main"
        DuckDBRuntimeConfig.clear_attached_connection()
        yield
    finally:
        pop_engine_identity(identity_token)
        DuckDBRuntimeConfig.DATABASE_PATH = orig_path
        DuckDBRuntimeConfig.SCHEMA = orig_schema
        DuckDBRuntimeConfig.clear_attached_connection()
        if orig_connection is not None:
            DuckDBRuntimeConfig.attach_connection(orig_connection)
        EngineConfig.TYPE = _ORIG_ENGINE_TYPE
        EngineConfig.RUNTIME = _ORIG_ENGINE_RUNTIME


def test_in_memory_reflect_and_execute_share_database() -> None:
    duckdb = pytest.importorskip("duckdb")

    connection = duckdb.connect(":memory:")
    _seed_memory_table(connection)
    dialect = DialectRegistry.get("duckdb", DuckDBRuntimeConfig(), native_connection=connection)
    graph = dialect.reflect_schema_graph(include="tables")
    assert "items" in graph.tables
    rows = dialect.execute("SELECT id, name FROM items ORDER BY id")
    assert rows == [(1, "alpha")]


def test_in_memory_via_attach_connection() -> None:
    duckdb = pytest.importorskip("duckdb")

    connection = duckdb.connect(":memory:")
    _seed_memory_table(connection)
    DuckDBRuntimeConfig.attach_connection(connection)
    dialect = DialectRegistry.get("duckdb", DuckDBRuntimeConfig())
    graph = dialect.reflect_schema_graph(include="tables")
    assert "items" in graph.tables
    rows = dialect.execute("SELECT name FROM items")
    assert rows == [("alpha",)]


def test_in_memory_via_static_pool_execution_engine() -> None:
    duckdb = pytest.importorskip("duckdb")

    connection = duckdb.connect(":memory:")
    _seed_memory_table(connection)
    execution_engine = DuckDBDialect.create_duckdb_sqlalchemy_engine(connection)
    dialect = DialectRegistry.get("duckdb", DuckDBRuntimeConfig(), sqlalchemy_engine=execution_engine)
    rows = dialect.execute("SELECT id FROM items")
    assert rows == [(1,)]


def test_owned_connection_disposed() -> None:
    duckdb = pytest.importorskip("duckdb")

    dialect = DialectRegistry.get("duckdb", DuckDBRuntimeConfig())
    assert dialect._native_connection is not None
    assert dialect._owns_native_connection is True
    connection = dialect._native_connection
    dialect.dispose_native_connection()
    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_injected_connection_not_disposed() -> None:
    duckdb = pytest.importorskip("duckdb")

    connection = duckdb.connect(":memory:")
    _seed_memory_table(connection)
    dialect = DialectRegistry.get("duckdb", DuckDBRuntimeConfig(), native_connection=connection)
    assert dialect._owns_native_connection is False
    dialect.dispose_native_connection()
    rows = connection.execute("SELECT COUNT(*) FROM items").fetchall()
    assert rows == [(1,)]
