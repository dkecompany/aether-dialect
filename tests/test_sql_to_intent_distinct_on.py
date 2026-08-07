"""SQL import maps PostgreSQL ``DISTINCT ON`` onto ``distinct_on``."""

from __future__ import annotations

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._dialect_sqlglot_engines import DatabricksDialect
from aetherdialect._sql_to_intent import _convert_postgres, _convert_sqlglot


def _pg() -> PostgresDialect:
    return PostgresDialect.__new__(PostgresDialect)


def _databricks() -> DatabricksDialect:
    return DatabricksDialect.__new__(DatabricksDialect)


def test_distinct_on_round_trips_to_runtime_intent(schema_graph) -> None:
    """``DISTINCT ON`` partition columns populate ``distinct_on``, not ``distinct_select_index``."""
    sql = (
        "SELECT DISTINCT ON (customers.id) customers.id, customers.name "
        "FROM customers "
        "ORDER BY customers.id, customers.balance DESC"
    )
    pg_intent = _convert_postgres(sql, schema_graph, _pg())
    assert pg_intent.distinct_select_index == -1
    assert pg_intent.distinct_on == [NormalizedExpr.from_column("customers.id")]

    db_intent = _convert_sqlglot(sql, schema_graph, _databricks())
    assert db_intent.distinct_select_index == -1
    assert db_intent.distinct_on == [NormalizedExpr.from_column("customers.id")]

    plain = "SELECT DISTINCT customers.id FROM customers"
    plain_intent = _convert_postgres(plain, schema_graph, _pg())
    assert plain_intent.distinct_select_index == 0
    assert plain_intent.distinct_on == []
