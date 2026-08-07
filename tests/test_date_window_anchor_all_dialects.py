"""Relative date-window rendering must honor explicit federation anchors on every dialect."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aetherdialect._dialect import Dialect, DialectRegistry

ANCHOR = datetime(2026, 3, 15, 14, 30, 0, tzinfo=UTC)
ANCHOR_DATE = ANCHOR.date().isoformat()

LIVE_CLOCK_MARKERS = (
    "CURRENT_DATE",
    "CURRENT_TIMESTAMP",
    "GETDATE()",
    "date('now')",
    "datetime('now')",
    "current_date()",
    "current_timestamp()",
)


def _uninit(engine: str) -> Dialect:
    cls = DialectRegistry.get_class(engine)
    return cls.__new__(cls)


def _contains_live_clock(sql: str) -> bool:
    text = sql.lower()
    return any(marker.lower() in text for marker in LIVE_CLOCK_MARKERS)


def _uses_timestamp_clock(sql: str) -> bool:
    upper = sql.upper()
    if " AS DATE" in upper and "TIMESTAMP" not in upper and "DATETIME" not in upper:
        if upper.strip().startswith("DATE '") or "DATE('" in upper:
            return False
    return any(
        token in upper
        for token in (
            "TIMESTAMP",
            "DATETIME",
            "GETDATE()",
            "CURRENT_TIMESTAMP",
            "DATETIME(",
        )
    )


def _uses_date_clock(sql: str) -> bool:
    upper = sql.upper()
    if _uses_timestamp_clock(sql):
        return False
    return any(token in upper for token in (" AS DATE", "DATE '", "DATE(", "CAST(GETDATE() AS DATE)"))


@pytest.mark.fast
@pytest.mark.parametrize("engine", DialectRegistry.get_registered_engines())
def test_anchor_literal_emitted_when_set(engine: str) -> None:
    """Anchored windows must embed the bound instant, not the live session clock."""
    dialect = _uninit(engine)
    lower = dialect.render_date_window("t.col", ">=", "day", 30, anchor=ANCHOR)
    upper = dialect.date_window_upper_bound_sql("day", anchor=ANCHOR)

    assert ANCHOR_DATE in lower, lower
    assert ANCHOR_DATE in upper, upper
    assert not _contains_live_clock(lower), lower
    assert not _contains_live_clock(upper), upper


@pytest.mark.fast
@pytest.mark.parametrize("engine", DialectRegistry.get_registered_engines())
def test_subday_lower_and_upper_same_clock_class(engine: str) -> None:
    """Sub-day units must not mix date-class and timestamp-class anchors within one window."""
    dialect = _uninit(engine)
    lower = dialect.render_date_window("t.col", ">=", "hour", 1, anchor=ANCHOR)
    upper = dialect.date_window_upper_bound_sql("hour", anchor=ANCHOR)

    assert _uses_timestamp_clock(lower), lower
    assert _uses_timestamp_clock(upper), upper
    assert not _uses_date_clock(lower), lower
    assert not _uses_date_clock(upper), upper
