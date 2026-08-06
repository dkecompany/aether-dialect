"""Explicit null placement in ORDER BY and window ORDER BY."""

from __future__ import annotations

import pytest

from aetherdialect._constants import DEFAULT_NULL_ORDERING_ASC, DEFAULT_NULL_ORDERING_DESC
from aetherdialect._contracts_base import NormalizedExpr, OrderByCol
from aetherdialect._contracts_schema import WindowSpec
from aetherdialect._dialect import DialectRegistry
from aetherdialect._sql_gen import _render_window_over_sql, render_order_by_sql

_IS_NULL_REWRITE_ENGINES = frozenset({"mysql", "mariadb", "sqlserver"})


def _uninit_dialect(engine: str):
    cls = DialectRegistry.get_class(engine)
    return cls.__new__(cls)


def _assert_explicit_null_placement(sql: str, *, direction: str, engine: str) -> None:
    dir_up = direction.upper()
    if engine in _IS_NULL_REWRITE_ENGINES:
        assert "IS NULL" in sql.upper()
        if dir_up == "ASC":
            assert DEFAULT_NULL_ORDERING_ASC == "last"
            assert "IS NULL) ASC" in sql.replace(" ", "").upper() or "IS NULL) ASC," in sql.upper()
        else:
            assert DEFAULT_NULL_ORDERING_DESC == "first"
            assert "IS NULL) DESC" in sql.replace(" ", "").upper() or "IS NULL) DESC," in sql.upper()
    elif dir_up == "ASC":
        assert "NULLS LAST" in sql.upper()
    else:
        assert "NULLS FIRST" in sql.upper()


@pytest.mark.fast
@pytest.mark.parametrize("engine", DialectRegistry.get_registered_engines())
def test_null_placement_explicit_in_every_dialect(engine: str) -> None:
    dialect = _uninit_dialect(engine)
    asc_cols = [OrderByCol(expr=NormalizedExpr.from_column("t.n"), direction="ASC")]
    desc_cols = [OrderByCol(expr=NormalizedExpr.from_column("t.n"), direction="DESC")]

    _assert_explicit_null_placement(render_order_by_sql(asc_cols, dialect), direction="ASC", engine=engine)
    _assert_explicit_null_placement(render_order_by_sql(desc_cols, dialect), direction="DESC", engine=engine)


@pytest.mark.fast
@pytest.mark.parametrize("engine", DialectRegistry.get_registered_engines())
def test_window_order_by_carries_null_placement(engine: str) -> None:
    dialect = _uninit_dialect(engine)
    cols = [OrderByCol(expr=NormalizedExpr.from_column("t.n"), direction="DESC")]
    outer = render_order_by_sql(cols, dialect)
    ws = WindowSpec(function="row_number", partition_by=[], order_by=cols, frame_kind="none")
    window = _render_window_over_sql(ws, dialect)

    _assert_explicit_null_placement(outer, direction="DESC", engine=engine)
    assert f"ORDER BY {outer}" in window
