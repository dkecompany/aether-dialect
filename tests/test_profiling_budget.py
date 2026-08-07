"""Profiling deep-query budget must gate value-overlap sampling."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_schema import ColumnMetadata
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect
from aetherdialect._schema_catalog import (
    _new_profiling_deep_query_budget,
    _profile_column,
)


def _recording_engine() -> tuple[MagicMock, list[str]]:
    executed: list[str] = []
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    def fake_execute(sql: Any) -> MagicMock:
        executed.append(str(sql))
        result = MagicMock()
        stmt = str(sql)
        if "COUNT(*)" in stmt and "COUNT(DISTINCT" not in stmt and "GROUP BY" not in stmt:
            result.scalar.return_value = 100
            result.fetchone.return_value = (100,)
        elif "COUNT(*)" in stmt and "COUNT(DISTINCT" in stmt:
            result.fetchone.return_value = (100, 5, 0)
        elif "MIN(" in stmt or "MAX(" in stmt:
            result.fetchone.return_value = (1, 9)
        elif "GROUP BY" in stmt and "ORDER BY COUNT" in stmt:
            result.fetchall.return_value = [("alpha",), ("beta",)]
        elif "SELECT MAX(c) FROM" in stmt:
            result.fetchone.return_value = (3,)
        elif "ORDER BY v ASC LIMIT" in stmt:
            result.fetchall.return_value = [("alpha",), ("beta",)]
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = fake_execute
    return engine, executed


def _overlap_profile_sql(statements: list[str]) -> list[str]:
    return [stmt for stmt in statements if "ORDER BY v ASC LIMIT" in stmt]


@pytest.mark.fast
def test_overlap_skipped_when_budget_exhausted() -> None:
    engine, executed = _recording_engine()
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    col = ColumnMetadata(
        name="status",
        data_type="varchar",
        value_type="string",
        distinct_count=5,
    )
    budget = _new_profiling_deep_query_budget(0)
    _profile_column(dialect, engine, col, "events", row_count=100, deep_query_budget=budget)

    assert not _overlap_profile_sql(executed)
    assert col.value_overlap_sample == []
