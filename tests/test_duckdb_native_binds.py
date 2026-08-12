"""DuckDB native and federation coordinator bind conversion tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._dialect_sqlglot_engines import DuckDBNativeBackend
from aetherdialect._federation_execute import _execute_coordinator_sql_with_timeout


@pytest.mark.fast
def test_colon_params_converted_before_execute() -> None:
    connection = MagicMock()
    result = MagicMock()
    result.fetchmany.side_effect = [[(9,)], []]
    connection.execute.return_value = result
    backend = DuckDBNativeBackend(connection)
    params = {"p1": "horror"}
    assert backend.fetch_rows("SELECT 1 WHERE name = :p1", params) == [(9,)]
    connection.execute.assert_called_once_with("SELECT 1 WHERE name = ?", ["horror"])


@pytest.mark.fast
def test_coordinator_colon_params_converted_before_execute() -> None:
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE t AS SELECT 'horror' AS name")
    result = _execute_coordinator_sql_with_timeout(
        conn,
        "SELECT 1 AS n FROM t WHERE name = :p1",
        {"p1": "horror"},
        timeout_ms=None,
    )
    rows = result.fetchall()
    assert rows == [(1,)]
