"""Databricks connector and Snowflake Arrow backends must pass bind maps to execute."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect._dialect_sqlglot_engines import DatabricksConnectorBackend, SnowflakeArrowBackend


def _cursor_with_rows(rows: list[tuple[Any, ...]]) -> MagicMock:

    cursor = MagicMock()
    cursor.fetchmany.side_effect = [rows, []]
    return cursor


@pytest.mark.fast
def test_databricks_connector_receives_bound_values() -> None:
    conn = MagicMock()
    cursor = _cursor_with_rows([(42,)])
    conn.cursor.return_value = cursor
    backend = DatabricksConnectorBackend(conn)
    sql = "SELECT amount FROM orders WHERE category = :p1"
    params = {"p1": "electronics"}
    rows = backend.fetch_rows(sql, params)
    assert rows == [(42,)]
    assert cursor.execute.call_count == 1
    executed_sql, executed_params = cursor.execute.call_args.args
    assert ":p1" not in executed_sql
    assert executed_params == {"p1": "electronics"}


@pytest.mark.fast
def test_snowflake_arrow_receives_bound_values() -> None:
    conn = MagicMock()
    cursor = MagicMock(spec=["execute", "fetchmany", "close"])
    cursor.fetchmany.side_effect = [[(7,)], []]
    conn.cursor.return_value = cursor
    backend = SnowflakeArrowBackend(connection=conn)
    sql = "SELECT id FROM orders WHERE status = :p1"
    params = {"p1": "open"}
    rows = backend.fetch_rows(sql, params)
    assert rows == [(7,)]
    assert cursor.execute.call_count >= 1
    executed_sql, executed_params = cursor.execute.call_args.args
    assert ":p1" not in executed_sql
    assert executed_params == {"p1": "open"}
