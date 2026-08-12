"""Cross-source decomposition: member partial aggregates, literal pushdown, HAVING partition."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import (
    HavingParam,
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import plan_federated_intent
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(table: str, source_id: str, *, extra_cols: dict[str, ColumnMetadata] | None = None) -> SchemaGraph:
    columns: dict[str, ColumnMetadata] = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
    }
    if extra_cols:
        columns.update(extra_cols)
    tables = {
        table: TableMetadata(
            name=table,
            columns=columns,
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


def _inner_manifest() -> object:
    return parse_federation_manifest(
        {
            "federation_id": "fed_decomp_l21_l23",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b"},
            "cross_source_joins": [
                {"left": "t_a.id", "right": "t_b.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )


def _left_join_manifest() -> object:
    return parse_federation_manifest(
        {
            "federation_id": "fed_decomp_l22",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [
                {"left": "ta.id", "right": "tb.a_id", "kind": "left", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )


def _left_join_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "ta": TableMetadata(
                name="ta",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="a",
            ),
            "tb": TableMetadata(
                name="tb",
                columns={
                    "a_id": ColumnMetadata(
                        name="a_id",
                        data_type="integer",
                        sensitivity="none",
                        is_unique=True,
                    ),
                    "status": ColumnMetadata(name="status", data_type="text", sensitivity="none"),
                },
                primary_key=[],
                foreign_keys=[],
                source_id="b",
            ),
        },
        join_paths_multi=recompute_join_paths_multi({}),
    )


@pytest.mark.fast
def test_cross_source_scalar_count_member_applies_partial_aggregation() -> None:
    manifest = _inner_manifest()
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "t_a.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    member_a = next(step for step in plan.steps if step.source_id == "a")
    sub = member_a.sub_intent
    assert sub.grain == "grouped"
    assert sub.group_by_cols
    assert any(g.column_ref == "t_a.id" for g in sub.group_by_cols)
    assert any(sc.is_aggregated for sc in (sub.select_cols or []))


@pytest.mark.fast
def test_join_covered_literal_filter_pushes_to_nullable_member() -> None:
    manifest = _left_join_manifest()
    schema = _left_join_schema()
    join_key_filter = WhereParam(
        left_expr=NormalizedExpr.from_column("tb.a_id"),
        op=">",
        value_type="integer",
        raw_value="5",
    )
    intent = RuntimeIntent(
        tables=["ta", "tb"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list([join_key_filter]),
    )
    plan = plan_federated_intent(intent, schema, manifest)
    assert plan.ineligible_reason is None
    nullable_step = next(step for step in plan.steps if step.source_id == "b")
    assert nullable_step.sub_intent.where is not None
    member_leaves = nullable_step.sub_intent.where.leaves() or []
    assert any(fp.op == ">" and fp.left_expr.column_ref == "tb.a_id" for fp in member_leaves)
    residual_leaves = plan.residual.where.leaves() if plan.residual and plan.residual.where else []
    assert not any(fp.left_expr.column_ref == "tb.a_id" and fp.op == ">" for fp in residual_leaves)


@pytest.mark.fast
def test_cross_source_aggregate_single_source_having_stays_on_member() -> None:
    manifest = _inner_manifest()
    composite = compose_composite_graph(
        {"a": _graph("t_a", "a"), "b": _graph("t_b", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="grouped",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "t_a.id"))],
        group_by_cols=[NormalizedExpr.from_column("t_b.id")],
        order_by_cols=[],
        where=None,
        having=PredicateGroup.from_list(
            [
                HavingParam(
                    left_expr=NormalizedExpr.from_column("t_a.id"),
                    op=">",
                    value_type="integer",
                    raw_value="0",
                ),
            ]
        ),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
    member_a = next(step for step in plan.steps if step.source_id == "a")
    assert member_a.sub_intent.having is not None
    member_leaves = member_a.sub_intent.having.leaves() or []
    assert any(hp.op == ">" and hp.left_expr.column_ref == "t_a.id" for hp in member_leaves)
    residual_leaves = plan.residual.having.leaves() if plan.residual and plan.residual.having else []
    assert not any(hp.left_expr.column_ref == "t_a.id" and hp.op == ">" for hp in residual_leaves)
