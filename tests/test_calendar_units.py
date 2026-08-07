"""ISO week boundaries and native quarter/half-year calendar rendering."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

import duckdb
from aetherdialect._constants import WEEK_NUMBERING, WEEK_START_DAY
from aetherdialect._dialect import DialectRegistry

_ISO_WEEK_CASES: tuple[tuple[date, date], ...] = (
    (date(2023, 12, 31), date(2023, 12, 25)),
    (date(2024, 1, 1), date(2024, 1, 1)),
    (date(2024, 1, 7), date(2024, 1, 1)),
)

_DUCKDB_EVAL_ENGINES = frozenset({"duckdb", "postgresql", "redshift", "csv"})

_ISO_WEEK_MARKERS: dict[str, tuple[str, ...]] = {
    "mysql": ("WEEKDAY",),
    "mariadb": ("WEEKDAY",),
    "bigquery": ("MONDAY",),
    "sqlserver": ("% 7",),
    "sqlite": ("strftime", "%w"),
    "databricks": ("DATE_TRUNC", "WEEK"),
    "snowflake": ("DATE_TRUNC",),
}


def _uninit_dialect(engine: str):
    cls = DialectRegistry.get_class(engine)
    return cls.__new__(cls)


def _iso_week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _eval_date_sql(sql: str) -> date:
    row = duckdb.execute(f"SELECT {sql}").fetchone()
    assert row is not None
    val = row[0]
    if hasattr(val, "date"):
        return val.date()
    return val


def _assert_iso_week_trunc_sql(dialect, engine: str, input_date: date, expected: date) -> None:
    lit = dialect.render_date_literal(input_date)
    sql = dialect.render_date_trunc("week", lit)
    upper = sql.upper()
    if engine in _DUCKDB_EVAL_ENGINES:
        assert _eval_date_sql(sql) == expected
        return
    markers = _ISO_WEEK_MARKERS.get(engine, ("DATE_TRUNC",))
    assert any(marker in upper or marker in sql for marker in markers)
    if engine in {"mysql", "mariadb"}:
        assert "DAYOFWEEK" not in upper


@pytest.mark.fast
def test_week_calendar_constants() -> None:
    assert WEEK_START_DAY == "monday"
    assert WEEK_NUMBERING == "iso"


@pytest.mark.fast
@pytest.mark.parametrize("engine", DialectRegistry.get_registered_engines())
def test_week_boundaries_are_iso_in_every_dialect(engine: str) -> None:
    dialect = _uninit_dialect(engine)
    for input_date, expected in _ISO_WEEK_CASES:
        assert _iso_week_start(input_date) == expected
        _assert_iso_week_trunc_sql(dialect, engine, input_date, expected)
