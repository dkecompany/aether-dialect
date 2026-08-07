"""Per-engine runtime configuration and embedded connection isolation."""

from __future__ import annotations

import pytest

from aetherdialect._config import DuckDBRuntimeConfig, PostgresRuntimeConfig
from aetherdialect._dialect import DialectRegistry


@pytest.mark.fast
def test_two_engines_keep_separate_credentials() -> None:
    left = PostgresRuntimeConfig()
    right = PostgresRuntimeConfig()
    left.apply_connection_credentials("secret-left")
    right.apply_connection_credentials("secret-right")
    assert left.PASSWORD == "secret-left"
    assert right.PASSWORD == "secret-right"


@pytest.mark.fast
def test_two_embedded_engines_keep_separate_connections() -> None:
    duckdb = pytest.importorskip("duckdb")

    conn_a = duckdb.connect(":memory:")
    conn_b = duckdb.connect(":memory:")
    conn_a.execute("CREATE TABLE items (id INTEGER)")
    conn_a.execute("INSERT INTO items VALUES (1)")
    conn_b.execute("CREATE TABLE items (id INTEGER)")
    conn_b.execute("INSERT INTO items VALUES (2)")

    dialect_a = DialectRegistry.get("duckdb", DuckDBRuntimeConfig(), native_connection=conn_a)
    dialect_b = DialectRegistry.get("duckdb", DuckDBRuntimeConfig(), native_connection=conn_b)

    assert dialect_a.execute("SELECT id FROM items") == [(1,)]
    assert dialect_b.execute("SELECT id FROM items") == [(2,)]
