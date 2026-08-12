"""Composite graph notes_sha256 reflects federation notes content."""

from __future__ import annotations

import hashlib

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, FederationMappings, SchemaGraph, TableMetadata
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import parse_federation_manifest
from aetherdialect._schema_graph import recompute_join_paths_multi


def _members() -> dict[str, SchemaGraph]:
    def _graph(table: str, source_id: str) -> SchemaGraph:
        tables = {
            table: TableMetadata(
                name=table,
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id=source_id,
            )
        }
        return SchemaGraph(
            tables=tables,
            join_paths_multi=recompute_join_paths_multi(tables),
            schema_graph_id=f"sg_{source_id}_{table}",
            effective_structural_hash=f"eff_{source_id}_{table}",
        )

    return {"a": _graph("left_t", "a"), "b": _graph("right_t", "b")}


@pytest.mark.fast
def test_compose_sets_composite_notes_sha256_from_federation_notes() -> None:
    notes = "Customers in the west region use storefront pricing."
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_notes",
            "cross_source_joins": [
                {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
            ],
        },
    )
    composite = compose_composite_graph(
        _members(),
        manifest,
        FederationMappings(version="0.2.3"),
        notes_content=notes,
    )
    assert composite.notes_sha256 == hashlib.sha256(notes.encode("utf-8")).hexdigest()
