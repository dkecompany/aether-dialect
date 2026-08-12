"""Result fetch caps: stop during batch fetch and push LIMIT into SQL."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from aetherdialect._config import EngineLimits
from aetherdialect._contracts_base import ResultCapExceededError
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._dialect_sqlglot_helper import ResultBackend
from aetherdialect._pipeline_execute import (
    _fetch_capped_result_rows,
    _push_result_row_limit_sql,
)
from aetherdialect._utils import pop_engine_limits, push_engine_limits


class _CountingBackend:
    """Yields rows in batches while counting how many were requested."""

    def __init__(self, total_rows: int = 100) -> None:
        self.total_rows = total_rows
        self.rows_requested = 0

    def fetch_rows_batched(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        batch_rows: int,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> Iterator[tuple[tuple[Any, ...], ...]]:
        produced = 0
        while produced < self.total_rows:
            chunk_size = min(int(batch_rows), self.total_rows - produced)
            batch = tuple((produced + i,) for i in range(chunk_size))
            self.rows_requested += len(batch)
            produced += len(batch)
            yield batch


@pytest.mark.fast
def test_cap_raises_before_full_materialisation() -> None:
    backend = _CountingBackend(total_rows=100)
    token = push_engine_limits(EngineLimits(max_result_rows=5, result_fetch_batch_rows=2, max_result_bytes=None))
    try:
        with pytest.raises(ResultCapExceededError, match="row cap"):
            _fetch_capped_result_rows(cast(ResultBackend, backend), "SELECT 1", None)
    finally:
        pop_engine_limits(token)
    assert backend.rows_requested < 100
    assert backend.rows_requested <= 6


@pytest.mark.fast
def test_limit_pushed_into_statement() -> None:
    intent = MagicMock(spec=RuntimeIntent)
    intent.limit = None
    sql = _push_result_row_limit_sql("SELECT id FROM orders", intent, "duckdb", max_rows=10)
    assert "LIMIT" in sql.upper()
    assert "11" in sql
    intent.limit = 3
    sql_tight = _push_result_row_limit_sql("SELECT id FROM orders LIMIT 3", intent, "duckdb", max_rows=10)
    assert "LIMIT 3" in sql_tight.upper() or sql_tight.upper().count("LIMIT") >= 1
