"""Space knowledge: skip merge when one side empty; always scope descriptions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aetherdialect._contracts_base import DomainKnowledgeEntry, SpaceContext, StructuralKnowledgeFact
from aetherdialect._contracts_schema import ColumnMetadata, DescriptionOwner, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps


def _schema(*, notes_owned: bool = False) -> SchemaGraph:
    owner = DescriptionOwner.NOTES if notes_owned else DescriptionOwner.LLM_REFINEMENT
    graph = SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "id": ColumnMetadata(
                        name="id",
                        data_type="integer",
                        description="order id mentions payments elsewhere",
                        description_owner=owner,
                        base_description="order identifier",
                    )
                },
                primary_key=["id"],
                foreign_keys=[],
                description="orders linked to payments and inventory",
                description_owner=owner,
                base_description="order header rows",
            )
        },
        join_paths_multi={},
        effective_structural_hash="h",
        schema_graph_id="g1",
    )
    if notes_owned:
        graph.structural_knowledge = (
            StructuralKnowledgeFact(
                kind="relation",
                text="orders is the sales header",
                referenced_entities=frozenset({"orders"}),
            ),
        )
    return graph


@pytest.mark.fast
def test_resolve_knowledge_skips_merge_when_space_empty() -> None:
    engine = (DomainKnowledgeEntry(key="arr", text="engine arr", kind="glossary"),)
    calls: list[tuple[object, object]] = []

    def _merge(a, b):
        calls.append((a, b))
        return a

    out = MainExecutionOps._resolve_space_over_engine_knowledge(engine, (), merge_both=_merge)
    assert out == engine
    assert calls == []


@pytest.mark.fast
def test_resolve_knowledge_skips_merge_when_engine_empty() -> None:
    space = (DomainKnowledgeEntry(key="nrr", text="space nrr", kind="metric"),)
    calls: list[tuple[object, object]] = []

    def _merge(a, b):
        calls.append((a, b))
        return b

    out = MainExecutionOps._resolve_space_over_engine_knowledge((), space, merge_both=_merge)
    assert out == space
    assert calls == []


@pytest.mark.fast
def test_resolve_knowledge_merges_when_both_present() -> None:
    engine = (DomainKnowledgeEntry(key="arr", text="engine", kind="glossary"),)
    space = (DomainKnowledgeEntry(key="arr", text="space", kind="glossary"),)
    calls: list[tuple[object, object]] = []

    def _merge(a, b):
        calls.append((a, b))
        return space

    out = MainExecutionOps._resolve_space_over_engine_knowledge(engine, space, merge_both=_merge)
    assert out == space
    assert len(calls) == 1


@pytest.mark.fast
def test_no_space_notes_carries_master_and_scopes_with_master_prose(tmp_path: Path) -> None:
    schema = _schema(notes_owned=True)
    schema.structural_knowledge = (
        StructuralKnowledgeFact(
            kind="relation",
            text="orders is the sales header",
            referenced_entities=frozenset({"orders"}),
        ),
    )
    engine_dk = (DomainKnowledgeEntry(key="arr", text="annual recurring revenue", kind="glossary"),)
    snapshot = {
        "tables": ["orders"],
        "columns": ["orders.id"],
        "table_descriptions": {"orders": "orders linked to payments and inventory"},
        "column_meta": {},
    }
    space_ctx = SpaceContext(tables=frozenset({"orders"}))
    classify = {
        "orders": (
            "entity",
            "orders header only",
            {"id": ("identifier", "order id", None)},
        )
    }
    merge_calls = {"n": 0}

    def _boom(*_a, **_k):
        merge_calls["n"] += 1
        raise AssertionError("merge must not run when space notes are absent")

    with patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=True):
        with patch("aetherdialect._main_spaces.EngineConfig.llm_credentials_configured", return_value=True):
            with patch(
                "aetherdialect._main_spaces.merge_domain_knowledge_notes_overlay",
                side_effect=_boom,
            ):
                with patch(
                    "aetherdialect._main_spaces.merge_structural_knowledge_notes_overlay",
                    side_effect=_boom,
                ):
                    with patch(
                        "aetherdialect._main_spaces.llm_enrich_schema_from_structural_knowledge",
                        return_value=classify,
                    ) as enrich:
                        out = MainExecutionOps.enrich_space_snapshot_with_notes(
                            snapshot,
                            schema,
                            space_ctx,
                            engine_domain_knowledge=engine_dk,
                        )
    assert merge_calls["n"] == 0
    assert out["domain_knowledge"] == [
        {"key": "arr", "kind": "glossary", "text": "annual recurring revenue", "referenced_entities": []}
    ]
    assert out["structural_knowledge"] == [
        {"kind": "relation", "text": "orders is the sales header", "referenced_entities": ["orders"]}
    ]
    assert out["table_descriptions"]["orders"] == "orders header only"
    assert enrich.call_args.kwargs.get("prefer_base_descriptions") is False
    assert list(enrich.call_args.args[1]) == list(schema.structural_knowledge)


@pytest.mark.fast
def test_no_space_notes_no_master_notes_uses_base_descriptions(tmp_path: Path) -> None:
    schema = _schema(notes_owned=False)
    schema.structural_knowledge = ()
    snapshot = {
        "tables": ["orders"],
        "columns": ["orders.id"],
        "table_descriptions": {"orders": "orders linked to payments and inventory"},
        "column_meta": {},
    }
    space_ctx = SpaceContext(tables=frozenset({"orders"}))
    classify = {
        "orders": (
            "entity",
            "order header rows",
            {"id": ("identifier", "order identifier", None)},
        )
    }
    with patch("aetherdialect._config.EngineConfig.llm_credentials_configured", return_value=True):
        with patch("aetherdialect._main_spaces.EngineConfig.llm_credentials_configured", return_value=True):
            with patch(
                "aetherdialect._main_spaces.llm_enrich_schema_from_structural_knowledge",
                return_value=classify,
            ) as enrich:
                out = MainExecutionOps.enrich_space_snapshot_with_notes(snapshot, schema, space_ctx)
    assert out["domain_knowledge"] == []
    assert out["structural_knowledge"] == []
    assert enrich.call_args.kwargs.get("prefer_base_descriptions") is True
    assert list(enrich.call_args.args[1]) == []
