"""Dialect.sql_has_aggregate must agree with AGGREGATE_FUNCTION_NAMES / IR classification."""

from __future__ import annotations

import pytest

from aetherdialect._constants import AGGREGATE_FUNCTION_NAMES
from aetherdialect._dialect import Dialect
from aetherdialect._intent_expr import parse_expr_string


@pytest.mark.fast
@pytest.mark.parametrize(
    "sql_fn",
    sorted(AGGREGATE_FUNCTION_NAMES),
)
def test_sql_has_aggregate_matches_authoritative_names(sql_fn: str) -> None:
    sql = f"SELECT {sql_fn.upper()}(x) FROM t"
    assert Dialect.sql_has_aggregate(sql, sqlglot_dialect="postgres") is True


@pytest.mark.fast
@pytest.mark.parametrize(
    ("sql_fn", "ir_agg"),
    [
        ("stddev", "stddev"),
        ("stddev_pop", "stddev"),
        ("stddev_samp", "stddev"),
        ("variance", "variance"),
        ("var_pop", "variance"),
        ("var_samp", "variance"),
        ("median", "median"),
    ],
)
def test_sql_has_aggregate_statistical_variants_match_ir(sql_fn: str, ir_agg: str) -> None:
    sql = f"SELECT {sql_fn}(col) FROM t"
    assert Dialect.sql_has_aggregate(sql, sqlglot_dialect="postgres") is True
    expr = parse_expr_string(f"{sql_fn}(col)")
    assert any(g.agg_func == ir_agg for g in (expr.add_groups or []) if g.agg_func)


@pytest.mark.fast
def test_sql_has_aggregate_rejects_non_aggregates() -> None:
    assert Dialect.sql_has_aggregate("SELECT x FROM t", sqlglot_dialect="postgres") is False
    assert Dialect.sql_has_aggregate("SELECT LOWER(x) FROM t", sqlglot_dialect="postgres") is False
    assert Dialect.sql_has_aggregate("SELECT COALESCE(x, 0) FROM t", sqlglot_dialect="postgres") is False
