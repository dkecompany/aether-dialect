"""Schema graph identity must be reproducible across independent constructions."""

from __future__ import annotations

from copy import deepcopy

import pytest

from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._schema_graph import assign_schema_graph_hashes, recompute_join_paths_multi


@pytest.mark.fast
def test_identical_schemas_produce_identical_ids(schema_graph: SchemaGraph) -> None:
    """Two independently built graphs with the same structure and scope share one identity."""
    ctx = EngineContext()
    notes = "shared-notes-sha"

    first = deepcopy(schema_graph)
    first.notes_sha256 = notes
    assign_schema_graph_hashes(first, ctx, notes)

    second = deepcopy(schema_graph)
    second.notes_sha256 = notes
    assign_schema_graph_hashes(second, ctx, notes)

    assert first.schema_graph_id
    assert first.schema_graph_id == second.schema_graph_id
    assert first.structural_hash == second.structural_hash
    assert first.effective_structural_hash == second.effective_structural_hash

    tables_a = {"only": deepcopy(schema_graph.tables["orders"])}
    tables_a["only"].name = "only"
    graph_a = SchemaGraph(tables=tables_a, join_paths_multi=recompute_join_paths_multi(tables_a))
    graph_a.notes_sha256 = notes
    assign_schema_graph_hashes(graph_a, ctx, notes)

    tables_b = {"only": deepcopy(schema_graph.tables["orders"])}
    tables_b["only"].name = "only"
    graph_b = SchemaGraph(tables=tables_b, join_paths_multi=recompute_join_paths_multi(tables_b))
    graph_b.notes_sha256 = notes
    assign_schema_graph_hashes(graph_b, ctx, notes)

    assert graph_a.schema_graph_id == graph_b.schema_graph_id
