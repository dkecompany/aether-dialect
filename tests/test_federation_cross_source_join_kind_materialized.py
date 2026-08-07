"""Declared cross-source join kinds are carried onto materialized graph edges."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import InferenceTag
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._federation import (
    _materialize_cross_source_edges,
    parse_federation_manifest,
    parse_federation_mappings,
)
from aetherdialect._schema_graph import recompute_join_paths_multi


def _member_graph(table: str, source_id: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"join_id": ColumnMetadata(name="join_id", data_type="integer", sensitivity="none")},
            primary_key=["join_id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id=f"sg_{table}",
        effective_structural_hash=f"eff_{table}",
    )


@pytest.mark.fast
@pytest.mark.parametrize("kind", ["inner", "left"])
def test_declared_cross_source_join_kind_reaches_materialized_edge(kind: str) -> None:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_kind",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b"},
            "cross_source_joins": [
                {"left": "t_a.join_id", "right": "t_b.join_id", "kind": kind, "logical_key": "join_id"},
            ],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings({"version": "0.2.1", "logical_columns": []})
    edges = _materialize_cross_source_edges(manifest, mappings)
    declared = [edge for edge in edges if edge.inference_tag == InferenceTag.CROSS_SOURCE]
    assert len(declared) == 1
    assert declared[0].join_kind == kind
