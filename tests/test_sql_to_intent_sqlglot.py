"""Sqlglot-native SQL import tests (non-Postgres backends)."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._dialect_sqlglot_engines import (
    DuckDBDialect,
    MySQLDialect,
    SQLiteDialect,
)
from aetherdialect._sql_to_intent import convert_sql_to_intent
from aetherdialect._sql_to_intent_sqlglot import convert_sql_via_sqlglot


def _shell(cls: type) -> object:
    return cls.__new__(cls)


@pytest.mark.parametrize(
    "dialect_cls",
    (DuckDBDialect, MySQLDialect, SQLiteDialect),
)
def test_sqlglot_plain_projection(schema_graph: SchemaGraph, dialect_cls: type) -> None:
    dialect = _shell(dialect_cls)
    sql = "SELECT customer_id FROM customers"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert "customers" in cr.intent.tables


@pytest.mark.parametrize(
    "dialect_cls",
    (DuckDBDialect, SQLiteDialect),
)
def test_sqlglot_with_cte(schema_graph: SchemaGraph, dialect_cls: type) -> None:
    dialect = _shell(dialect_cls)
    sql = "WITH c AS (SELECT customer_id FROM customers) SELECT customer_id FROM c"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.cte_steps
    assert "customers" in cr.intent.tables


def test_sqlglot_is_null_filter(schema_graph: SchemaGraph) -> None:
    dialect = _shell(DuckDBDialect)
    sql = "SELECT customer_id FROM customers WHERE email IS NULL"
    rt = convert_sql_via_sqlglot(sql, schema_graph, dialect)
    assert rt.filters_param
    assert rt.filters_param[0].op == "is null"


def test_sqlglot_count_aggregate(schema_graph: SchemaGraph) -> None:
    dialect = _shell(MySQLDialect)
    sql = "SELECT COUNT(*) FROM customers"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.select_cols[0].expr.add_groups[0].agg_func == "count"


def test_sqlglot_in_list(schema_graph: SchemaGraph) -> None:
    dialect = _shell(SQLiteDialect)
    sql = "SELECT customer_id FROM customers WHERE customer_id IN (1, 2, 3)"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.filters_param[0].op == "in"


def test_sqlglot_inner_join(schema_graph: SchemaGraph) -> None:
    dialect = _shell(DuckDBDialect)
    sql = "SELECT c.customer_id FROM customers c INNER JOIN orders o ON c.customer_id = o.customer_id"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert set(cr.intent.tables) >= {"customers", "orders"}


def test_sqlglot_group_by_having(schema_graph: SchemaGraph) -> None:
    dialect = _shell(MySQLDialect)
    sql = "SELECT customer_id, COUNT(*) AS n FROM orders GROUP BY customer_id HAVING COUNT(*) > 1"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.group_by_cols
    assert cr.intent.having_param


def test_sqlglot_limit(schema_graph: SchemaGraph) -> None:
    dialect = _shell(SQLiteDialect)
    sql = "SELECT customer_id FROM customers LIMIT 10"
    cr = convert_sql_to_intent(sql, schema_graph, dialect, verify_via_execute=False)
    assert cr.failure_code is None and cr.intent is not None
    assert cr.intent.limit == 10
