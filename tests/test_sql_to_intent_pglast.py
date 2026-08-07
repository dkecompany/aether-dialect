"""Postgres pglast extraction tests."""

from __future__ import annotations

import pytest

from aetherdialect._constants import SQL_TO_INTENT_LIMIT_OFFSET_PARAM_KEY
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._sql_to_intent import convert_sql_to_intent


def _pg() -> PostgresDialect:
    return PostgresDialect.__new__(PostgresDialect)


pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("pglast") is None,
    reason="pglast not installed",
)


def test_pglast_exists_sublink_lift(schema_graph: SchemaGraph) -> None:
    dialect = _pg()
    sql = (
        "SELECT customer_id FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)"
    )
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None, cr.failure_detail
    assert cr.intent is not None
    assert (cr.intent.where.leaves() if cr.intent.where else []) or cr.intent.cte_steps


def test_pglast_from_subquery_lift(schema_graph: SchemaGraph) -> None:
    dialect = _pg()
    sql = "SELECT sq.customer_id FROM (SELECT customer_id FROM customers) sq"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None, cr.failure_detail
    assert cr.intent is not None
    assert cr.intent.cte_steps


def test_pglast_scalar_sublink(schema_graph: SchemaGraph) -> None:
    dialect = _pg()
    sql = "SELECT customer_id FROM customers WHERE customer_id = (SELECT MAX(customer_id) FROM customers)"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None, cr.failure_detail
    assert cr.intent is not None
    assert cr.intent.cte_steps or (cr.intent.where.leaves() if cr.intent.where else [])


def test_pglast_in_sublink(schema_graph: SchemaGraph) -> None:
    dialect = _pg()
    sql = "SELECT customer_id FROM customers WHERE customer_id IN (SELECT customer_id FROM orders)"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.where.leaves() if cr.intent.where else []


def test_pglast_coalesce_projection(schema_graph: SchemaGraph) -> None:
    dialect = _pg()
    sql = "SELECT COALESCE(name, email) FROM customers"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    expr = cr.intent.select_cols[0].expr
    has_coalesce = (
        expr.scalar_func == "coalesce"
        or any(g.scalar_func == "coalesce" for g in (expr.add_groups or []))
        or "coalesce" in (expr.raw_sql or "").lower()
    )
    assert has_coalesce, "COALESCE not mapped to scalar_func in pglast extract"


def test_pglast_order_by_ordinal(schema_graph: SchemaGraph) -> None:
    dialect = _pg()
    sql = "SELECT customer_id, name FROM customers ORDER BY 2"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.order_by_cols


def test_pglast_limit_offset(schema_graph: SchemaGraph) -> None:
    dialect = _pg()
    sql = "SELECT customer_id FROM customers LIMIT 5 OFFSET 10"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.limit == 5
    assert cr.intent.param_values.get(SQL_TO_INTENT_LIMIT_OFFSET_PARAM_KEY) == 10
