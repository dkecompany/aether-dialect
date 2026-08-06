"""Semantic/where-segment edges use comma-FROM + WHERE, not CROSS JOIN."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import JoinEdge
from aetherdialect._dialect_sqlglot_engines import DatabricksDialect
from aetherdialect._dialect_sqlglot_helper import SqlglotParseMixin


def _dbr() -> DatabricksDialect:
    return DatabricksDialect.__new__(DatabricksDialect)


def _edge() -> JoinEdge:
    return JoinEdge(table="b", alias=None, kind="INNER", on_terms=(("a", "x", "b", "x"),))


@pytest.mark.fast
def test_semantic_edge_uses_comma_from_where() -> None:
    dx = _dbr()
    parsed = dx.parse_select("SELECT 1 FROM a")
    carriers = dx.ordered_join_carrier_froms(parsed)
    assert carriers
    assert dx.attach_extra_from_and_where(parsed, carriers[0], ["b"], [_edge()]) is True
    out = dx.emit_sql(parsed).lower()
    assert " from a, b" in out or " from a,b" in out
    assert "a`.`x`" in out or "a.x" in out
    assert "b`.`x`" in out or "b.x" in out


@pytest.mark.fast
def test_no_cross_join_for_semantic_edge() -> None:
    dx = _dbr()
    parsed = dx.parse_select("SELECT 1 FROM a")
    carriers = dx.ordered_join_carrier_froms(parsed)
    assert carriers
    assert dx.attach_extra_from_and_where(parsed, carriers[0], ["b"], [_edge()]) is True
    out = dx.emit_sql(parsed).lower()
    assert "cross join" not in out


@pytest.mark.fast
def test_validate_sql_accepts_semantic_multi_table() -> None:
    dx = _dbr()
    parsed = dx.parse_select("SELECT 1 FROM a")
    carriers = dx.ordered_join_carrier_froms(parsed)
    assert carriers
    assert dx.attach_extra_from_and_where(parsed, carriers[0], ["b"], [_edge()]) is True
    sql = dx.emit_sql(parsed)
    assert "cross join" not in sql.lower()
    ok, reason = SqlglotParseMixin.ast_structural_valid_sqlglot(
        sql, sqlglot_dialect=dx.sqlglot_dialect or "databricks", scalar_cte_names=frozenset()
    )
    assert ok is True, reason
