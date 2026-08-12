"""Model-derived schema classification is cached by content hash for reproducible rebuilds."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import EngineContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import assign_schema_graph_hashes, recompute_join_paths_multi
from aetherdialect._schema_profile import (
    _load_schema_classification_cache,
    _save_schema_classification_cache,
    llm_classification_column_scope,
    llm_classify_schema,
    schema_classification_content_hash,
)


def _graph() -> SchemaGraph:
    table = TableMetadata(
        name="orders",
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", row_count=10),
            "status": ColumnMetadata(name="status", data_type="text", row_count=10),
        },
        primary_key=["id"],
        foreign_keys=[],
    )
    tables = {"orders": table}
    graph = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="eff_hash_orders",
        profiling_hash="profile_hash_orders",
    )
    assign_schema_graph_hashes(graph, EngineContext(), "")
    return graph


def _llm_payload(call_index: int) -> str:
    suffix = "alpha" if call_index == 0 else "beta"
    return json.dumps(
        {
            "orders": {
                "table_role": "fact",
                "description": f"order events {suffix}",
                "columns": {
                    "id": {"role": "identifier", "description": f"order id {suffix}", "sensitivity": None},
                    "status": {"role": "categorical", "description": f"order status {suffix}", "sensitivity": None},
                },
            }
        }
    )


@pytest.mark.fast
def test_llm_classify_schema_reuses_disk_cache_for_unchanged_inputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "schema_graph.json.gz"
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", str(cache_path))
    graph = _graph()
    notes = "domain notes"
    calls = {"count": 0}

    def _llm_chat(*_args, **_kwargs) -> str:
        payload = _llm_payload(calls["count"])
        calls["count"] += 1
        return payload

    with patch("aetherdialect._schema_profile.LLMProvider.chat", side_effect=_llm_chat):
        scope = llm_classification_column_scope(graph)
        cache_holder: list[dict] = []
        first = llm_classify_schema(graph, notes, column_scope=scope, cache_payload_out=cache_holder)
        _save_schema_classification_cache(
            schema_classification_content_hash(graph, notes, scope),
            cache_holder[0],
        )
        second = llm_classify_schema(graph, notes, column_scope=scope)

    assert calls["count"] == 2
    assert first == second
    assert first["orders"][1] == second["orders"][1]


@pytest.mark.fast
def test_schema_classification_cache_round_trips_by_content_hash(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aetherdialect._schema_profile import _normalize_llm_classification_payload

    cache_path = tmp_path / "schema_graph.json.gz"
    monkeypatch.setattr(EngineConfig, "SCHEMA_JSON_PATH", str(cache_path))
    graph = _graph()
    scope = {"orders": frozenset({"id", "status"})}
    content_hash = schema_classification_content_hash(graph, "notes", scope)
    _save_schema_classification_cache(
        content_hash,
        {
            "orders": {
                "table_role": "fact",
                "description": "cached orders",
                "columns": {
                    "id": {"role": "identifier", "description": "cached id", "sensitivity": None},
                    "status": {"role": "categorical", "description": "cached status", "sensitivity": None},
                },
            }
        },
    )
    loaded = _load_schema_classification_cache(content_hash)
    assert loaded is not None
    normalized = _normalize_llm_classification_payload(loaded)
    assert normalized["orders"][1] == "cached orders"
