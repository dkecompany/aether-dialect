"""Date-only column classification for relative date-window rendering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aetherdialect._sql_gen import _column_is_date_only


@pytest.mark.fast
def test_datetime2_not_date_only() -> None:
    """SQL Server datetime2 carries time-of-day and must not be treated as date-only."""
    column_meta = SimpleNamespace(data_type="datetime2")
    assert _column_is_date_only(column_meta) is False
