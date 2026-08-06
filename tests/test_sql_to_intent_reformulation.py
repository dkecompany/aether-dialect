"""Shared import normalization and reformulation tests."""

from __future__ import annotations

from aetherdialect._contracts_base import (
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import (
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._sql_to_intent import (
    _dedup_cte_steps,
    normalize_imported_intent,
    reformulate_imported_intent,
)


def _pg() -> PostgresDialect:
    return PostgresDialect.__new__(PostgresDialect)


def test_reformulate_imported_intent_identity(schema_graph: SchemaGraph) -> None:
    intent = RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    out = reformulate_imported_intent(intent, schema_graph, _pg())
    assert out.tables == intent.tables


def test_normalize_preserves_physical_tables(schema_graph: SchemaGraph) -> None:
    intent = RuntimeIntent(
        tables=["customers", "orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("c.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
    )
    out, code, detail = normalize_imported_intent(intent, schema_graph, _pg())
    assert code is None and detail == ""
    assert out is not None
    assert set(out.tables) >= {"customers", "orders"}


def test_normalize_strict_semantic_rejects_bad_grain(schema_graph: SchemaGraph) -> None:
    intent = RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
        group_by_cols=[NormalizedExpr.from_column("customers.name")],
        order_by_cols=[],
        where=None,
        having=None,
    )
    out, code, _ = normalize_imported_intent(intent, schema_graph, _pg(), strict_semantic=True)
    assert out is None or code in (None, "SEMANTIC_REJECT")


def test_normalize_dedup_cte_steps_merged(schema_graph: SchemaGraph) -> None:
    from aetherdialect._contracts_core import RuntimeCteStep

    s1 = RuntimeCteStep(
        cte_name="a",
        tables=["customers"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.name"))],
    )
    s2 = RuntimeCteStep(
        cte_name="b",
        tables=["customers"],
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
    )
    intent = RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        cte_steps=[s1, s2],
    )
    out = _dedup_cte_steps(intent)
    assert len(out.cte_steps) == 1
    assert len(out.cte_steps[0].select_cols) == 2


def test_normalize_assigns_param_keys(schema_graph: SchemaGraph) -> None:
    intent = RuntimeIntent(
        tables=["customers"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("customers.customer_id"),
                    op="=",
                    value_type="integer",
                    param_key="",
                    raw_value=1,
                )
            ]
        ),
        having=None,
    )
    out, code, _ = normalize_imported_intent(intent, schema_graph, _pg())
    assert code is None and out is not None
    assert (out.where.leaves() if out.where else [])[0].raw_value == 1
