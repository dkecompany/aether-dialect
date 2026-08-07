"""Identifier casing collisions across federation members."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import FederationDeclarationError
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import compose_composite_graph, parse_federation_manifest
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
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{source_id}_{table}",
        effective_structural_hash=f"eff_{source_id}_{table}",
    )


@pytest.mark.fast
def test_compose_raises_on_identifier_casing_collision_across_members() -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_casing_l32",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    members = {"a": _graph("Orders", "a"), "b": _graph("orders", "b")}
    with pytest.raises(FederationDeclarationError, match=r"a\.'Orders'.*b\.'orders'"):
        compose_composite_graph(members, manifest)
