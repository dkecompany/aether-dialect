"""Plan-time federation capability intersection beyond median and ILIKE."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import WhereParam, predicate_group_from_list
from aetherdialect._contracts_core import RuntimeIntent, SelectCol, WindowRegistryStep
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata, WindowSpec
from aetherdialect._federation import (
    compose_composite_graph,
    intersect_member_dialect_capabilities,
    parse_federation_manifest,
    plan_federated_intent,
)
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(
    table: str,
    source_id: str,
    *,
    extra_columns: dict[str, ColumnMetadata] | None = None,
    has_array: bool = False,
) -> SchemaGraph:
    columns: dict[str, ColumnMetadata] = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
    }
    if has_array:
        columns["tags"] = ColumnMetadata(
            name="tags",
            data_type="text[]",
            sensitivity="none",
            element_type="text",
        )
    if extra_columns:
        columns.update(extra_columns)
    table_meta = TableMetadata(
        name=table,
        columns=columns,
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )
    tables = {table: table_meta}
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


_MANIFEST_PG_SQLITE = {
    "federation_id": "fed_cap_l31",
    "sources": [
        {"source_id": "a", "engine": "postgresql", "role": "owner"},
        {"source_id": "b", "engine": "sqlite", "role": "owner"},
    ],
    "table_namespace": {"ta": "a", "tb": "b"},
    "cross_source_joins": [{"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"}],
}


@pytest.mark.fast
def test_intersect_member_dialect_capabilities_stddev_requires_all_members() -> None:
    caps = intersect_member_dialect_capabilities(
        engine_types_by_source={"a": "postgresql", "b": "sqlite"},
    )
    assert caps["supports_stddev"] is False
    assert caps["supports_variance"] is False


@pytest.mark.fast
def test_intersect_member_dialect_capabilities_array_contains_requires_all_members() -> None:
    caps = intersect_member_dialect_capabilities(
        engine_types_by_source={"a": "postgresql", "b": "csv"},
    )
    assert caps["supports_array_contains"] is False


@pytest.mark.fast
def test_single_source_stddev_refused_when_member_lacks_support() -> None:
    manifest = parse_federation_manifest(_MANIFEST_PG_SQLITE, include_derived_roster=True)
    composite = compose_composite_graph(
        {"a": _graph("ta", "a"), "b": _graph("tb", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["ta"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_agg("stddev", "ta.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason == "stddev is not supported by all federation members"


@pytest.mark.fast
def test_window_frames_refused_when_member_lacks_frame_support() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_cap_l31_frames",
            "sources": [
                {"source_id": "a", "engine": "postgresql", "role": "owner"},
                {"source_id": "b", "engine": "csv", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [{"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"}],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _graph("ta", "a"), "b": _graph("tb", "b")},
        manifest,
    )
    window = WindowRegistryStep(
        registry_id="w01",
        window_spec=WindowSpec(
            function="sum",
            partition_by=[NormalizedExpr.from_column("ta.id")],
            order_by=[],
            argument=NormalizedExpr.from_column("ta.id"),
            frame_kind="rows",
            frame_start="unbounded_preceding",
            frame_end="current_row",
        ),
    )
    intent = RuntimeIntent(
        tables=["ta"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr(column_ref="w01"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        window_registry=[window],
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason == "window frames are not supported by all federation members"


@pytest.mark.fast
def test_contains_where_op_refused_when_member_lacks_array_contains() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_cap_l31_contains",
            "sources": [
                {"source_id": "a", "engine": "postgresql", "role": "owner"},
                {"source_id": "b", "engine": "csv", "role": "owner"},
            ],
            "table_namespace": {"ta": "a", "tb": "b"},
            "cross_source_joins": [{"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"}],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _graph("ta", "a", has_array=True), "b": _graph("tb", "b")},
        manifest,
    )
    intent = RuntimeIntent(
        tables=["ta"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("ta.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=predicate_group_from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("ta.tags"),
                    op="contains",
                    value_type="array",
                    param_key="tag",
                ),
            ]
        ),
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason == "array contains is not supported by all federation members"
