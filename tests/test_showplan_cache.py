"""SQL Server SHOWPLAN row-estimate cache must stay bounded with LRU eviction."""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import SQLSERVER_SHOWPLAN_ROW_CACHE_MAX
from aetherdialect._dialect_sqlglot_engines import SQLServerDialect


def _mock_showplan_engine(finalized_sql: str) -> MagicMock:
    engine = MagicMock()
    raw = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = [("Scan", "EstimateRows=10", f"StmtText={finalized_sql}")]
    raw.cursor.return_value = cursor
    engine.raw_connection.return_value = raw
    return engine


@pytest.mark.fast
def test_cache_evicts_at_cap() -> None:
    cap = 3
    assert cap < SQLSERVER_SHOWPLAN_ROW_CACHE_MAX
    dialect = object.__new__(SQLServerDialect)
    dialect._showplan_row_cache = OrderedDict()
    dialect._explain_disabled = False
    dialect.engine = _mock_showplan_engine("unused")

    with (
        patch("aetherdialect._dialect_sqlglot_engines.SQLSERVER_SHOWPLAN_ROW_CACHE_MAX", cap),
        patch.object(SQLServerDialect, "finalize_render", side_effect=lambda sql, *_a, **_k: sql.strip()),
        patch.object(
            SQLServerDialect,
            "parse_explain_plan",
            return_value=(10.0, None, [], "plan"),
        ),
    ):
        for idx in range(cap + 1):
            ok, _, _ = SQLServerDialect.explain_diagnose(dialect, f"SELECT {idx}")
            assert ok is True

    assert len(dialect._showplan_row_cache) == cap
    assert "SELECT 0" not in dialect._showplan_row_cache
    assert f"SELECT {cap}" in dialect._showplan_row_cache
