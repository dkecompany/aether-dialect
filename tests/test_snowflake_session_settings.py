"""Snowflake session settings apply on the executing connection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._dialect_sqlglot_engines import SnowflakeArrowBackend


@pytest.mark.fast
def test_timeout_applied_on_executing_connection() -> None:
    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value = cursor
    cursor.fetchall.return_value = [(1,)]

    backend = SnowflakeArrowBackend(connection=connection)
    with patch(
        "aetherdialect._dialect_sqlglot_engines.cost_cap_active",
        return_value=True,
    ):
        backend.fetch_rows("SELECT 1", timeout_ms=5000)

    assert connection.cursor.call_count == 1
    calls = [str(call.args[0]) for call in cursor.execute.call_args_list]
    assert any("STATEMENT_TIMEOUT_IN_SECONDS" in sql for sql in calls)
    assert any(sql.strip().upper().startswith("SELECT") for sql in calls)
    assert calls.index(next(sql for sql in calls if "STATEMENT_TIMEOUT" in sql)) < calls.index(
        next(sql for sql in calls if sql.strip().upper().startswith("SELECT"))
    )
