"""Absolute date-window bounds accept ISO 8601 only and render per dialect."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    MulGroup,
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._dialect import DialectRegistry
from aetherdialect._sql_gen import _render_date_window_predicate, build_deterministic_sql


def _uninit_dialect(engine: str):
    cls = DialectRegistry.get_class(engine)
    return cls.__new__(cls)


def _pg_render():
    from aetherdialect._dialect_postgres import PostgresDialect

    return PostgresDialect.__new__(PostgresDialect)


@pytest.mark.fast
def test_ambiguous_date_refused() -> None:
    from aetherdialect._contracts_base import AmbiguousDateLiteralError

    pred = WhereParam(
        left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t.d"])], sub_groups=[]),
        value_type="date_window",
        raw_value={"start": "01/02/2020", "end": "2020-12-31"},
    )
    dialect = _uninit_dialect("postgresql")
    with pytest.raises(AmbiguousDateLiteralError, match="ISO"):
        _render_date_window_predicate(pred, "t.d", dialect)


@pytest.mark.fast
@pytest.mark.parametrize("engine", DialectRegistry.get_registered_engines())
def test_iso_date_rendered_per_dialect(engine: str) -> None:
    dialect = _uninit_dialect(engine)
    pred = WhereParam(
        left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t.d"])], sub_groups=[]),
        value_type="date_window",
        raw_value={"start": "2020-01-01", "end": "2020-12-31"},
    )
    parts = _render_date_window_predicate(pred, "t.d", dialect)
    assert len(parts) == 2
    combined = " ".join(parts)
    assert "2020-01-01" in combined
    assert "2020-12-31" in combined
    assert "01/02/2020" not in combined
    hook_sql = dialect.render_date_literal(__import__("datetime").date(2020, 1, 1))
    assert hook_sql in combined or hook_sql.replace(" ", "") in combined.replace(" ", "")


@pytest.mark.fast
def test_iso_date_window_in_deterministic_sql() -> None:
    intent = RuntimeIntent(
        tables=["t1"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t1.d"])], sub_groups=[]))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr(add_groups=[MulGroup(multiply=["t1.d"])], sub_groups=[]),
                    value_type="date_window",
                    raw_value={"start": "2020-01-01", "end": "2020-12-31"},
                ),
            ]
        ),
        having=None,
    )
    sql = build_deterministic_sql(intent, dialect=_pg_render())
    assert "2020-01-01" in sql
    assert "2020-12-31" in sql
