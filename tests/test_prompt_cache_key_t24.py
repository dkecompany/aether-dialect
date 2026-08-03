"""prompt_cache_key must rotate when profiling or metadata change."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import EngineContext, RoleOwner, SensitivityClassification
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import (
    prompt_cache_schema_scope,
    schema_prompt_cache_id,
)
from aetherdialect._llm_provider import resolve_prompt_cache_key
from aetherdialect._schema_graph import assign_schema_graph_hashes


def _make_graph(
    *,
    graph_id: str = "sg_test000000000001__abcd1234",
    col_role: str | None = None,
    distinct_count: int | None = None,
    sensitivity: SensitivityClassification = SensitivityClassification.NONE,
) -> SchemaGraph:
    col = ColumnMetadata(
        name="id",
        data_type="integer",
        role=col_role,
        distinct_count=distinct_count,
        sensitivity=sensitivity,
        role_owner=RoleOwner.PROFILE if col_role else None,
    )
    table = TableMetadata(
        name="tbl",
        columns={"id": col},
        primary_key=["id"],
        foreign_keys=[],
    )
    return SchemaGraph(
        tables={"tbl": table},
        join_paths_multi={},
        effective_structural_hash="eff_hash",
        schema_graph_id=graph_id,
    )


def _stamp(graph: SchemaGraph) -> None:
    assign_schema_graph_hashes(graph, EngineContext(), graph.notes_sha256 or "")


@pytest.mark.fast
def test_profiling_change_rotates_prompt_cache_id() -> None:
    graph_id = "sg_test000000000001__abcd1234"
    graph_a = _make_graph(graph_id=graph_id, col_role="identifier", distinct_count=100)
    graph_b = _make_graph(graph_id=graph_id, col_role="metric", distinct_count=50)
    _stamp(graph_a)
    _stamp(graph_b)
    cache_a = schema_prompt_cache_id(graph_a)
    cache_b = schema_prompt_cache_id(graph_b)
    assert cache_a is not None
    assert cache_b is not None
    assert cache_a != cache_b
    assert graph_a.schema_graph_id == graph_b.schema_graph_id
    assert graph_a.profiling_hash != graph_b.profiling_hash


@pytest.mark.fast
def test_metadata_change_rotates_prompt_cache_id() -> None:
    graph_id = "sg_test000000000001__abcd1234"
    graph_a = _make_graph(
        graph_id=graph_id,
        col_role="identifier",
        sensitivity=SensitivityClassification.NONE,
    )
    graph_b = _make_graph(
        graph_id=graph_id,
        col_role="identifier",
        sensitivity=SensitivityClassification.HIDDEN,
    )
    _stamp(graph_a)
    _stamp(graph_b)
    cache_a = schema_prompt_cache_id(graph_a)
    cache_b = schema_prompt_cache_id(graph_b)
    assert cache_a is not None
    assert cache_b is not None
    assert cache_a != cache_b
    assert graph_a.schema_graph_id == graph_b.schema_graph_id


@pytest.mark.fast
def test_resolve_prompt_cache_key_includes_profiling_and_metadata() -> None:
    graph = _make_graph(col_role="identifier", distinct_count=42)
    _stamp(graph)
    cache_id = schema_prompt_cache_id(graph)
    assert cache_id is not None
    assert graph.profiling_hash
    parts = cache_id.split(":")
    assert len(parts) >= 3
    assert graph.profiling_hash[:16] in cache_id
    with prompt_cache_schema_scope(cache_id):
        assert resolve_prompt_cache_key("intent") == f"intent:{cache_id}"
