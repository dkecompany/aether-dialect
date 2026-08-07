"""SQL Server execute paths must forward filter bind maps to the backend."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._dialect_sqlglot_engines import SQLServerDialect


@pytest.mark.fast
def test_filter_params_reach_backend() -> None:
    dialect = object.__new__(SQLServerDialect)
    mock_backend = MagicMock()
    mock_backend.fetch_rows.return_value = [(1,)]
    dialect._backend = mock_backend
    sql = "SELECT id FROM orders WHERE status = :p1"
    params = {"p1": "active"}
    rows = SQLServerDialect.execute(dialect, sql, params)
    assert rows == [(1,)]
    mock_backend.fetch_rows.assert_called_once()
    assert mock_backend.fetch_rows.call_args.args[0] == sql
    assert mock_backend.fetch_rows.call_args.args[1] == params
