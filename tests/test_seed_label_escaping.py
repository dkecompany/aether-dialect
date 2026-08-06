"""Seed-warmup categorical CASE labels escape embedded quotes."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import NormalizedExpr
from aetherdialect._contracts_core import SeedWarmupIntent, SelectCol
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._expansion_ops import _case_categorical_add
from aetherdialect._sql_gen import build_deterministic_sql


def _pg_render() -> PostgresDialect:
    return PostgresDialect.__new__(PostgresDialect)


@pytest.mark.fast
def test_sample_with_quote_does_not_break_sql(schema_graph) -> None:
    """Quoted categorical samples render as valid SQL string literals."""
    intent = SeedWarmupIntent(
        intent_id="quote_seed",
        tables=["customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        param_values={},
        expansion_metadata=None,
        limit=None,
    )
    column_metadata = {
        "customers": {
            "name": {
                "role": "categorical",
                "value_type": "string",
                "sample_values": ["O'Brien"],
            }
        }
    }
    expanded = _case_categorical_add(intent, schema_graph, column_metadata)
    assert expanded
    sql = build_deterministic_sql(
        expanded[0].to_runtime_intent(),
        schema=schema_graph,
        dialect=_pg_render(),
    )
    assert "O''Brien" in sql
    assert "O'Brien" not in sql.replace("O''Brien", "")
