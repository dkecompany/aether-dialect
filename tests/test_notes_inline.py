"""Inline notes string vs notes_file path: mutual exclusion and Pass B extraction."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aetherdialect._contracts_base import BusinessKnowledgeEntry, ConfigError, EngineContext, SpaceContext
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_catalog import extract_business_knowledge_from_notes


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        effective_structural_hash="h-ni",
        schema_graph_id="g-ni",
    )


@pytest.mark.fast
def test_notes_and_notes_file_both_set_raises(tmp_path) -> None:
    notes_path = tmp_path / "notes.txt"
    notes_path.write_text("ARR means annual recurring revenue.\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="notes"):
        EngineContext(notes="ARR means annual recurring revenue.", notes_file=str(notes_path))
    with pytest.raises(ConfigError, match="notes"):
        SpaceContext(tables=frozenset({"orders"}), notes="inline", notes_file=str(notes_path))


@pytest.mark.fast
def test_notes_string_extracts_bk() -> None:
    schema = _schema()
    ctx = EngineContext(notes="ARR means annual recurring revenue.")
    assert ctx.notes_file is None
    assert ctx.notes == "ARR means annual recurring revenue."
    llm_payload = [{"key": "arr", "kind": "glossary", "text": "ARR means annual recurring revenue."}]
    with patch("aetherdialect._schema_catalog.LLMProvider.chat", return_value=json.dumps(llm_payload)):
        with patch("aetherdialect._schema_catalog.EngineConfig.llm_credentials_configured", return_value=True):
            from aetherdialect._core_utils import notes_content_from_context

            content = notes_content_from_context(ctx)
            entries = extract_business_knowledge_from_notes(content, schema)
    assert any(isinstance(e, BusinessKnowledgeEntry) and e.key == "arr" for e in entries)
