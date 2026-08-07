"""Explicit NULL placement on coordinator residual ORDER BY keys."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_core import OrderByCol, ResidualSpec, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    compose_composite_graph,
    parse_federation_manifest,
    plan_federated_intent,
    render_federation_residual_sql,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(table: str, source_id: str) -> SchemaGraph:
    table_meta = TableMetadata(
        name=table,
        columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )
    tables = {table: table_meta}
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


_MANIFEST = {
    "federation_id": "fed_null_order_l33",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [{"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"}],
}


@pytest.mark.fast
def test_residual_render_adds_explicit_null_placement_when_unset() -> None:
    residual = ResidualSpec(
        select_cols=(SelectCol(expr=NormalizedExpr.from_column("left_t.id")),),
        order_by_cols=(OrderByCol(expr=NormalizedExpr.from_column("left_t.id"), direction="ASC"),),
    )
    sql = render_federation_residual_sql("SELECT * FROM joined", residual)
    assert "NULLS LAST" in sql.upper()


@pytest.mark.fast
def test_cross_source_plan_residual_order_keys_have_explicit_nulls() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("left_t", "a"), "b": _graph("right_t", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("left_t.id")),
            SelectCol(expr=NormalizedExpr.from_column("right_t.id")),
        ],
        group_by_cols=[],
        order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("left_t.id"), direction="DESC")],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.residual is not None
    assert plan.residual.order_by_cols
    for obc in plan.residual.order_by_cols:
        assert obc.nulls in ("first", "last")
    assert plan.residual.order_by_cols[0].nulls == "first"
