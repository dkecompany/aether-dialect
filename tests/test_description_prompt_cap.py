"""Tests for bounded schema description prompts and description-aware cache keys."""

from __future__ import annotations

import json

from aetherdialect._constants import (
    SCHEMA_DESCRIPTION_PROMPT_COUNT_CAP,
    SCHEMA_DESCRIPTION_PROMPT_MAX_CHARS,
)
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import (
    descriptions_hash_fp,
    prompt_cache_schema_scope,
    schema_prompt_cache_id,
)
from aetherdialect._llm_provider import LLMProvider
from aetherdialect._schema_graph import assign_schema_graph_hashes, tables_descriptions_payload


def _count_descriptions(payload: dict) -> int:
    total = 0
    for table_body in payload.values():
        if not isinstance(table_body, dict):
            continue
        if table_body.get("description"):
            total += 1
        for col_body in (table_body.get("columns") or {}).values():
            if isinstance(col_body, dict) and col_body.get("description"):
                total += 1
    return total


def _make_graph_with_descriptions(
    *,
    graph_id: str = "sg_test000000000001__abcd1234",
    column_count: int = 1,
    description: str = "short description",
) -> SchemaGraph:
    columns: dict[str, ColumnMetadata] = {}
    for i in range(column_count):
        columns[f"col_{i}"] = ColumnMetadata(
            name=f"col_{i}",
            data_type="varchar",
            description=description,
        )
    table = TableMetadata(
        name="tbl",
        columns=columns,
        primary_key=[],
        foreign_keys=[],
        description="table purpose",
    )
    return SchemaGraph(
        tables={"tbl": table},
        join_paths_multi={},
        effective_structural_hash="eff_hash",
        schema_graph_id=graph_id,
    )


def test_long_description_truncated_in_payload() -> None:
    long_desc = "w" * (SCHEMA_DESCRIPTION_PROMPT_MAX_CHARS + 50)
    graph = _make_graph_with_descriptions(description=long_desc)
    payload = json.loads(graph.schema_payload_interpret(owner_master_scope=True))
    emitted = payload["tbl"]["columns"]["col_0"]["description"]
    assert len(emitted) <= SCHEMA_DESCRIPTION_PROMPT_MAX_CHARS
    assert emitted.endswith("...")


def test_description_count_cap_in_payload() -> None:
    graph = _make_graph_with_descriptions(
        column_count=SCHEMA_DESCRIPTION_PROMPT_COUNT_CAP + 20,
        description="column meaning",
    )
    payload = json.loads(graph.schema_payload_interpret(owner_master_scope=True))
    assert _count_descriptions(payload) <= SCHEMA_DESCRIPTION_PROMPT_COUNT_CAP


def _stamp_graph_hashes(graph: SchemaGraph) -> None:
    ctx = EngineContext()
    assign_schema_graph_hashes(graph, ctx, graph.notes_sha256 or "")


def test_description_edit_rotates_prompt_cache_id() -> None:
    graph_id = "sg_test000000000001__abcd1234"
    graph_a = _make_graph_with_descriptions(graph_id=graph_id, description="alpha wording")
    graph_b = _make_graph_with_descriptions(graph_id=graph_id, description="beta wording")
    _stamp_graph_hashes(graph_a)
    _stamp_graph_hashes(graph_b)
    cache_a = schema_prompt_cache_id(graph_a)
    cache_b = schema_prompt_cache_id(graph_b)
    assert cache_a is not None
    assert cache_b is not None
    assert cache_a != cache_b
    assert graph_a.schema_graph_id == graph_b.schema_graph_id


def test_resolve_prompt_cache_key_uses_description_fingerprint() -> None:
    graph = _make_graph_with_descriptions(description="cache probe text")
    _stamp_graph_hashes(graph)
    cache_id = schema_prompt_cache_id(graph)
    assert cache_id is not None
    with prompt_cache_schema_scope(cache_id):
        key = LLMProvider.resolve_prompt_cache_key("intent")
    assert key is not None
    assert len(key) <= 64
    raw = f"intent:{cache_id}"
    if len(raw) <= 64:
        assert key == raw
    else:
        import hashlib

        assert key == f"intent:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def test_descriptions_hash_fp_stable_for_same_content() -> None:
    graph = _make_graph_with_descriptions(description="stable")
    payload = tables_descriptions_payload(graph.tables)
    first = descriptions_hash_fp(payload)
    second = descriptions_hash_fp(tables_descriptions_payload(graph.tables))
    assert first == second
