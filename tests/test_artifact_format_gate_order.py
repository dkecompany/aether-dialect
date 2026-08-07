"""Format/package gate must run before hash equality short-circuits."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ArtifactManifest, MigrationTier
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._schema_graph import classify_migration_tier


def _matching_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={},
        join_paths_multi={},
        structural_hash="s",
        profiling_hash="p",
        scope_hash="c",
        effective_structural_hash="e",
        notes_hash="n",
        semantic_edges_hash="se",
        schema_graph_id="sg-1",
    )


@pytest.mark.fast
def test_matching_hashes_still_destructive_when_format_stale() -> None:
    schema = _matching_schema()
    manifest = ArtifactManifest(
        artifact_format_version="0.0.0",
        effective_structural_hash=schema.effective_structural_hash,
        structural_hash=schema.structural_hash,
        profiling_hash=schema.profiling_hash,
        scope_hash=schema.scope_hash,
        notes_hash=schema.notes_hash,
        semantic_edges_hash=schema.semantic_edges_hash,
        schema_graph_id=schema.schema_graph_id,
    )
    assert classify_migration_tier(manifest, schema) == MigrationTier.DESTRUCTIVE
