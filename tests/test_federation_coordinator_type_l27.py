"""Unsupported coordinator column types refuse at plan time with column named."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import FederationDeclarationError
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    _schema_column_duckdb_type,
    compose_composite_graph,
    parse_federation_manifest,
    plan_federated_intent,
)
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
    "federation_id": "fed_type_l27",
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
def test_unknown_schema_type_maps_to_none_not_varchar() -> None:
    assert _schema_column_duckdb_type("jsonb") is None
    assert _schema_column_duckdb_type("integer") == "INTEGER"


@pytest.mark.fast
def test_unknown_coordinator_column_type_refuses_at_plan_time() -> None:
    """Unsupported schema data_type must refuse during planning with the column named."""
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    composite = compose_composite_graph(
        {
            "a": _graph(
                "left_t",
                source_id="a",
                extra_columns={
                    "meta": ColumnMetadata(name="meta", data_type="jsonb", sensitivity="none"),
                },
            ),
            "b": _graph("right_t", source_id="b"),
        },
        manifest,
    )
    intent = RuntimeIntent(
        tables=["left_t", "right_t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.meta"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    with pytest.raises(FederationDeclarationError, match=r"meta"):
        plan_federated_intent(intent, composite, manifest)
