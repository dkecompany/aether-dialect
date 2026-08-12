"""Federation coordinator egress preserves exact numeric Python types."""

from __future__ import annotations

from decimal import Decimal

import pytest

from aetherdialect._contracts_core import SessionStep
from aetherdialect._federation_execute import (
    _coordinator_result_to_dataframe,
    _execute_coordinator_sql_with_timeout,
)


@pytest.mark.fast
def test_decimal_survives_coordinator_to_session_step() -> None:
    """Coordinator materialization keeps Decimal values through SessionStep.data."""
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT CAST('19.9900' AS DECIMAL(19, 4)) AS amount")
    amount = Decimal("19.9900")
    raw = _execute_coordinator_sql_with_timeout(conn, "SELECT amount FROM t", {}, timeout_ms=None)
    result = _coordinator_result_to_dataframe(raw)
    assert len(result) == 1
    cell = result.iloc[0, 0]
    assert isinstance(cell, Decimal)
    assert cell == amount
    step = SessionStep(done=True, prompt=None, kind="ok", data=result)
    step_cell = step.data.iloc[0, 0]
    assert isinstance(step_cell, Decimal)
    assert step_cell == amount
