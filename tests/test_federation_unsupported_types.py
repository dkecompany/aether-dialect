"""Unmapped coordinator column types refuse when projected; untouched columns do not block."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import FederationDeclarationError, NormalizedExpr
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._federation_plan import plan_federated_intent, schema_column_duckdb_type
from aetherdialect._schema_graph import recompute_join_paths_multi


def _graph(
    table: str,
    *,
    source_id: str,
    extra_columns: dict[str, ColumnMetadata] | None = None,
) -> SchemaGraph:
    columns: dict[str, ColumnMetadata] = {
        "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
    }
    if extra_columns:
        columns.update(extra_columns)
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


_MANIFEST = {
    "federation_id": "fed_unsupported_types",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [
        {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
    ],
}


@pytest.mark.fast
def test_projected_unmapped_type_refused() -> None:
    assert schema_column_duckdb_type("geometry") is None
    assert schema_column_duckdb_type("") is None
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {
            "a": _graph(
                "left_t",
                source_id="a",
                extra_columns={
                    "shape": ColumnMetadata(name="shape", data_type="geometry", sensitivity="none"),
                },
            ),
            "b": _graph("right_t", source_id="b"),
        },
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.shape"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    with pytest.raises(FederationDeclarationError, match=r"shape.*geometry"):
        plan_federated_intent(intent, composite, manifest)


@pytest.mark.fast
def test_untouched_unmapped_column_does_not_block() -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {
            "a": _graph(
                "left_t",
                source_id="a",
                extra_columns={
                    "shape": ColumnMetadata(name="shape", data_type="geometry", sensitivity="none"),
                    "label": ColumnMetadata(name="label", data_type="text", sensitivity="none"),
                },
            ),
            "b": _graph("right_t", source_id="b"),
        },
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.label"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert plan.ineligible_reason is None
