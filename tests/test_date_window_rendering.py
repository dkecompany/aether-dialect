"""Relative date-window rendering: per-dialect truncation, clock consistency, and refusals."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import MulGroup, NormalizedExpr, WhereParam
from aetherdialect._contracts_core import SubdayDateWindowOnDateColumnError
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import Dialect, DialectRegistry
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._dialect_sqlglot_engines import (
    BigQueryDialect,
    DatabricksDialect,
    MariaDBDialect,
    MySQLDialect,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import _render_predicate_clause


def _uninit(engine: str) -> Dialect:
    cls = DialectRegistry.get_class(engine)
    return cls.__new__(cls)


def _pg() -> PostgresDialect:
    return PostgresDialect.__new__(PostgresDialect)


def _mysql() -> MySQLDialect:
    return MySQLDialect.__new__(MySQLDialect)


def _mariadb() -> MariaDBDialect:
    return MariaDBDialect.__new__(MariaDBDialect)


def _dbr() -> DatabricksDialect:
    return DatabricksDialect.__new__(DatabricksDialect)


def _bq() -> BigQueryDialect:
    return BigQueryDialect.__new__(BigQueryDialect)


def _date_schema() -> SchemaGraph:
    tables = {
        "events": TableMetadata(
            name="events",
            columns={
                "event_date": ColumnMetadata(name="event_date", data_type="date", value_type="date"),
                "event_ts": ColumnMetadata(name="event_ts", data_type="timestamp", value_type="date"),
            },
            primary_key=["event_date"],
            foreign_keys=[],
        )
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


PERIOD_TO_DATE_FRAGMENTS: list[tuple[str, str, tuple[str, ...]]] = [
    ("mysql", "month", ("DATE_FORMAT", "%Y-%m-01")),
    ("mysql", "year", ("DATE_FORMAT", "%Y-01-01")),
    ("mysql", "week", ("WEEKDAY",)),
    ("mariadb", "month", ("DATE_FORMAT", "%Y-%m-01")),
    ("mariadb", "week", ("WEEKDAY",)),
    ("databricks", "week", ("DATE_TRUNC", "WEEK")),
    ("databricks", "month", ("DATE_TRUNC", "MONTH")),
    ("postgresql", "month", ("DATE_TRUNC", "MONTH", "CURRENT_DATE")),
    ("bigquery", "day", ("DATE_TRUNC", "CURRENT_DATE")),
]


@pytest.mark.fast
@pytest.mark.parametrize("engine, unit, fragments", PERIOD_TO_DATE_FRAGMENTS)
def test_period_to_date_sql_per_dialect(engine: str, unit: str, fragments: tuple[str, ...]) -> None:
    """Each dialect truncates period-to-date anchors with engine-native SQL."""
    sql = _uninit(engine).render_date_window("t.col", ">=", unit, 0)
    upper = sql.upper()
    for fragment in fragments:
        assert fragment.upper() in upper
    if engine in ("mysql", "mariadb"):
        assert "DATE_TRUNC" not in upper


@pytest.mark.fast
@pytest.mark.parametrize(
    "engine, unit",
    [
        ("postgresql", "day"),
        ("postgresql", "hour"),
        ("mysql", "day"),
        ("mysql", "hour"),
        ("databricks", "day"),
        ("databricks", "hour"),
        ("bigquery", "day"),
        ("bigquery", "hour"),
    ],
)
def test_window_uses_single_clock_source(engine: str, unit: str) -> None:
    """Lower and upper window bounds must share one clock (date vs timestamp)."""
    d = _uninit(engine)
    lower = d.render_date_window("t.col", ">=", unit, 0)
    upper_anchor = d.date_window_upper_bound_sql(unit)
    inclusive_upper = d.render_date_window_inclusive_upper("t.col", unit)
    uses_ts = Dialect.relative_window_uses_timestamp(unit)

    def _is_timestamp_clock(text: str) -> bool:
        t = text.upper()
        return "CURRENT_TIMESTAMP" in t or "GETDATE()" in t or "DATETIME(" in t or "current_timestamp" in text

    def _is_date_clock(text: str) -> bool:
        t = text.upper()
        return "CURRENT_DATE" in t or "date(" in text.lower() or "CAST(GETDATE() AS DATE)" in t

    if uses_ts:
        assert _is_timestamp_clock(lower) or _is_timestamp_clock(upper_anchor)
        assert _is_timestamp_clock(upper_anchor)
        assert _is_timestamp_clock(inclusive_upper)
        assert not _is_date_clock(upper_anchor)
    else:
        assert _is_date_clock(lower) or _is_date_clock(upper_anchor)
        assert _is_date_clock(upper_anchor)
        assert not _is_timestamp_clock(upper_anchor)


@pytest.mark.fast
def test_subday_window_on_date_column_refuses() -> None:
    """Hour-level windows on date-only columns refuse rather than truncate."""
    schema = _date_schema()
    pred = WhereParam(
        left_expr=NormalizedExpr(
            add_groups=[MulGroup(multiply=["events.event_date"])],
            sub_groups=[],
        ),
        value_type="date_window",
        raw_value={"unit": "hour", "amount": 1},
    )
    with pytest.raises(SubdayDateWindowOnDateColumnError) as exc_info:
        _render_predicate_clause(pred, _pg(), schema=schema)
    assert "events.event_date" in exc_info.value.column
    assert "day" in exc_info.value.message_for_caller.lower()


@pytest.mark.fast
def test_subday_window_on_timestamp_column_allows_bigquery_without_date_cast() -> None:
    """Sub-day windows on timestamp columns compare timestamps without DATE()."""
    schema = _date_schema()
    pred = WhereParam(
        left_expr=NormalizedExpr(
            add_groups=[MulGroup(multiply=["events.event_ts"])],
            sub_groups=[],
        ),
        value_type="date_window",
        raw_value={"unit": "hour", "amount": 1},
    )
    sql = _render_predicate_clause(pred, _bq(), schema=schema)
    assert "DATE(events.event_ts)" not in sql
    assert "CURRENT_TIMESTAMP" in sql.upper()


@pytest.mark.fast
def test_databricks_week_offset_uses_trunc_not_day_product() -> None:
    """Databricks week offsets use week truncation, not amount * 7 day subtraction."""
    sql = _dbr().render_date_window("col", ">=", "week", 2)
    upper = sql.upper()
    assert "DATE_TRUNC" in upper and "WEEK" in upper
    assert "14" not in sql
    assert " * 7" not in sql


@pytest.mark.fast
def test_databricks_month_diff_uses_months_between() -> None:
    """Databricks month date_diff uses MONTHS_BETWEEN, not fixed day counts."""
    sql = _dbr().render_date_diff(
        "a - b",
        ">=",
        "month",
        3,
        minuend_sql="a",
        subtrahend_sql="b",
    )
    assert "MONTHS_BETWEEN" in sql.upper()
    assert "90" not in sql
