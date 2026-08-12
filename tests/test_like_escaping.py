"""LIKE/ILIKE wildcard escaping for literal string matching."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    NormalizedExpr,
    WhereParam,
)
from aetherdialect._dialect import DialectRegistry
from aetherdialect._sql_gen import render_predicate_clause
from aetherdialect._utils import escape_like_wildcards

_LIKE_ESCAPE_SUFFIX = "ESCAPE"


def _uninit_dialect(engine: str):
    cls = DialectRegistry.get_class(engine)
    return cls.__new__(cls)


def _ilike_predicate(*, param_key: str | None = "p", raw_value: str | None = None) -> WhereParam:
    return WhereParam(
        left_expr=NormalizedExpr.from_column("t.name"),
        op="ilike",
        value_type="string",
        param_key=param_key or "",
        raw_value=raw_value,
    )


@pytest.mark.fast
def test_percent_in_value_is_literal() -> None:
    assert escape_like_wildcards("50%") == "50/%"
    assert escape_like_wildcards("a_b") == "a/_b"
    assert escape_like_wildcards("back/slash") == "back//slash"

    dialect = _uninit_dialect("postgresql")
    sql = render_predicate_clause(
        _ilike_predicate(param_key="", raw_value="50%"),
        dialect,
    )
    assert _LIKE_ESCAPE_SUFFIX in sql.upper()
    assert "50/%" in sql

    bound_sql = render_predicate_clause(
        _ilike_predicate(param_key="p"),
        dialect,
        param_values={"p": escape_like_wildcards("50%")},
    )
    assert _LIKE_ESCAPE_SUFFIX in bound_sql.upper()
    assert ":p" in bound_sql
    assert "ESCAPE '/'" in sql or "ESCAPE '/'" in bound_sql


@pytest.mark.fast
@pytest.mark.parametrize("engine", DialectRegistry.get_registered_engines())
def test_escape_clause_present_in_every_dialect(engine: str) -> None:
    dialect = _uninit_dialect(engine)
    pred = _ilike_predicate()
    sql = render_predicate_clause(pred, dialect, param_values={"p": "needle"})
    assert _LIKE_ESCAPE_SUFFIX in sql.upper()

    if dialect.supports_ilike:
        assert "ILIKE" in sql.upper()
    else:
        assert "LIKE" in sql.upper()
        assert "LOWER" in sql.upper()
