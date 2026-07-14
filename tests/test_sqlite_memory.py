"""Unit tests for in-memory SQLite shared-connection behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aetherdialect._config import EngineConfig, SQLiteRuntimeConfig
from aetherdialect._dialect import get_dialect
from aetherdialect._dialect_sqlglot_engines import create_sqlite_sqlalchemy_engine

_ORIG_ENGINE_TYPE = EngineConfig.TYPE
_ORIG_ENGINE_RUNTIME = EngineConfig.RUNTIME


def _seed_memory_table(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    connection.execute("INSERT INTO items VALUES (1, 'alpha')")
    connection.commit()


@pytest.fixture(autouse=True)
def _reset_sqlite_runtime_config() -> None:
    orig_path = SQLiteRuntimeConfig.DATABASE_PATH
    orig_schema = SQLiteRuntimeConfig.SCHEMA
    orig_connection = SQLiteRuntimeConfig.NATIVE_CONNECTION
    EngineConfig.SCHEMA_JSON_PATH = ""
    EngineConfig.TYPE = "sqlite"
    EngineConfig.RUNTIME = SQLiteRuntimeConfig
    try:
        SQLiteRuntimeConfig.DATABASE_PATH = ":memory:"
        SQLiteRuntimeConfig.SCHEMA = "main"
        SQLiteRuntimeConfig.clear_attached_connection()
        yield
    finally:
        SQLiteRuntimeConfig.DATABASE_PATH = orig_path
        SQLiteRuntimeConfig.SCHEMA = orig_schema
        SQLiteRuntimeConfig.NATIVE_CONNECTION = orig_connection
        EngineConfig.TYPE = _ORIG_ENGINE_TYPE
        EngineConfig.RUNTIME = _ORIG_ENGINE_RUNTIME


def test_in_memory_reflect_and_execute_share_database() -> None:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    _seed_memory_table(connection)
    dialect = get_dialect("sqlite", SQLiteRuntimeConfig, native_connection=connection)
    graph = dialect.reflect_schema_graph(include="tables")
    assert "items" in graph.tables
    rows = dialect.execute("SELECT id, name FROM items ORDER BY id")
    assert rows == [(1, "alpha")]


def test_in_memory_via_attach_connection() -> None:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    _seed_memory_table(connection)
    SQLiteRuntimeConfig.attach_connection(connection)
    dialect = get_dialect("sqlite", SQLiteRuntimeConfig)
    graph = dialect.reflect_schema_graph(include="tables")
    assert "items" in graph.tables
    rows = dialect.execute("SELECT name FROM items")
    assert rows == [("alpha",)]


def test_in_memory_via_static_pool_execution_engine() -> None:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    _seed_memory_table(connection)
    execution_engine = create_sqlite_sqlalchemy_engine(connection)
    dialect = get_dialect("sqlite", SQLiteRuntimeConfig, sqlalchemy_engine=execution_engine)
    rows = dialect.execute("SELECT id FROM items")
    assert rows == [(1,)]


def test_rental_shop_sqlite_translate_rental_date_timestamp() -> None:
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from aetherdialect._schema_graph import data_type_to_value_type
    from load_rental_shop_engines import iter_create_table_blocks, translate_create

    ddl = (Path(__file__).resolve().parents[1] / "scripts" / "data" / "rental_shop.sql").read_text(encoding="utf-8")
    rental_block = next(b for b in iter_create_table_blocks(ddl) if "CREATE TABLE rental" in b)
    create_sql = translate_create("sqlite", rental_block, schema="main")
    assert "rental_date TIMESTAMP" in create_sql
    assert data_type_to_value_type("TIMESTAMP") == "date"


def test_owned_connection_disposed() -> None:
    dialect = get_dialect("sqlite", SQLiteRuntimeConfig)
    assert dialect._native_connection is not None
    assert dialect._owns_native_connection is True
    connection = dialect._native_connection
    dialect.dispose_native_connection()
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_injected_connection_not_disposed() -> None:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    _seed_memory_table(connection)
    dialect = get_dialect("sqlite", SQLiteRuntimeConfig, native_connection=connection)
    assert dialect._owns_native_connection is False
    dialect.dispose_native_connection()
    rows = connection.execute("SELECT COUNT(*) FROM items").fetchall()
    assert rows == [(1,)]
