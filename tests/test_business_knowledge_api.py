"""Tests for versioned business knowledge on engines and federations."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from aetherdialect import AetherEngine, BusinessKnowledgeEntry
from aetherdialect._core_utils import (
    BusinessKnowledgeHolder,
    business_knowledge_digest,
    business_knowledge_scope,
    empty_business_knowledge_digest,
)
from aetherdialect._contracts_base import ConfigError, EngineContext, LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SensitivityClassification, TableMetadata
from aetherdialect._core_utils import load_runtime_config, prompt_cache_schema_scope
from aetherdialect._intent_process import build_intent_interpret_prompt
from aetherdialect._llm_provider import resolve_prompt_cache_key
from aetherdialect._templates import empty_template_store


def _minimal_engine(**overrides: object) -> AetherEngine:
    llm_exec = load_runtime_config(merged_env=dict(os.environ))
    defaults: dict[str, object] = dict(
        _runtime_config=RuntimeConfig(
            engine="postgresql",
            artifacts_dir="/tmp/aether_bk",
            engine_context=EngineContext(),
            llm_execution=llm_exec,
        ),
        _llm_config=LLMConfig(provider="openai"),
        _schema_graph=_schema_with_hidden_email(),
        _dialect=MagicMock(),
        _artifacts_dir="/tmp/aether_bk",
        _store=empty_template_store("graph-bk-1"),
        _templates={},
        _rejected={},
        _schema_terms=set(),
        _config_file=None,
        _execution_engine=None,
        _audit_sink=None,
        _pipeline_writer_lock=__import__("threading").Lock(),
        _schema_role="owner",
        _consumer_visible_objects=None,
        _schema_stats={"table_count": 1, "total_filterable": 2},
        _construction_phase_callback=None,
        _ask_phase_callback=None,
        _token_provider=None,
    )
    defaults.update(overrides)
    obj = AetherEngine.__new__(AetherEngine)
    for key, value in defaults.items():
        setattr(obj, str(key), value)
    obj._business_knowledge = BusinessKnowledgeHolder()
    return obj


def _schema_with_hidden_email() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "customers": TableMetadata(
                name="customers",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer"),
                    "email": ColumnMetadata(
                        name="email",
                        data_type="text",
                        sensitivity=SensitivityClassification.HIDDEN,
                    ),
                },
                primary_key=["id"],
                foreign_keys=[],
                row_count=10,
            )
        },
        join_paths_multi={},
        effective_structural_hash="struct-bk-1",
        schema_graph_id="sg-bk-1__struct-b1",
    )


@pytest.mark.fast
def test_set_and_read_back_business_knowledge() -> None:
    engine = _minimal_engine()
    entries = (
        BusinessKnowledgeEntry(key="revenue", text="Monthly recurring revenue from paid subscriptions."),
        BusinessKnowledgeEntry(key="fiscal_year", text="Fiscal year starts in July.", kind="convention"),
    )
    version = engine.set_business_knowledge(entries)
    assert version == 1
    assert engine.business_knowledge() == entries
    assert engine.business_knowledge_version() == 1
    assert engine.business_knowledge_digest() == business_knowledge_digest(entries)


@pytest.mark.fast
def test_business_knowledge_digest_changes_on_edit() -> None:
    engine = _minimal_engine()
    first = (BusinessKnowledgeEntry(key="term", text="First definition."),)
    second = (BusinessKnowledgeEntry(key="term", text="Updated definition."),)
    engine.set_business_knowledge(first)
    digest_first = engine.business_knowledge_digest()
    engine.set_business_knowledge(second)
    digest_second = engine.business_knowledge_digest()
    assert digest_first != digest_second
    assert digest_first == business_knowledge_digest(first)
    assert digest_second == business_knowledge_digest(second)


@pytest.mark.fast
def test_business_knowledge_version_increments() -> None:
    engine = _minimal_engine()
    assert engine.business_knowledge_version() == 0
    assert engine.business_knowledge_digest() == empty_business_knowledge_digest()
    v1 = engine.set_business_knowledge((BusinessKnowledgeEntry(key="a", text="Alpha"),))
    v2 = engine.set_business_knowledge((BusinessKnowledgeEntry(key="b", text="Beta"),))
    assert v1 == 1
    assert v2 == 2
    assert engine.business_knowledge_version() == 2


@pytest.mark.fast
def test_hidden_column_reference_refused() -> None:
    engine = _minimal_engine()
    with pytest.raises(ConfigError, match="hidden column"):
        engine.set_business_knowledge(
            (BusinessKnowledgeEntry(key="bad", text="Do not expose customers.email in answers."),)
        )


@pytest.mark.fast
def test_business_context_injected_into_interpret_prompt() -> None:
    entries = (BusinessKnowledgeEntry(key="mrr", text="Monthly recurring revenue."),)
    digest = business_knowledge_digest(entries)
    with business_knowledge_scope(entries, digest):
        prompt = build_intent_interpret_prompt("show revenue", "{}", "", ())
    payload = json.loads(prompt)
    assert payload["business_context"] == [{"key": "mrr", "kind": "glossary", "text": "Monthly recurring revenue."}]


@pytest.mark.fast
def test_schema_graph_id_unchanged_after_business_knowledge_edit() -> None:
    engine = _minimal_engine()
    graph_id_before = engine._schema_graph.schema_graph_id
    engine.set_business_knowledge((BusinessKnowledgeEntry(key="term", text="Business-only glossary entry."),))
    engine.set_business_knowledge(
        (
            BusinessKnowledgeEntry(key="term", text="Revised glossary entry."),
            BusinessKnowledgeEntry(key="alias", text="VIP means top-tier customer."),
        )
    )
    assert engine._schema_graph.schema_graph_id == graph_id_before


@pytest.mark.fast
def test_prompt_cache_key_includes_business_digest() -> None:
    entries = (BusinessKnowledgeEntry(key="term", text="Definition."),)
    digest = business_knowledge_digest(entries)
    with prompt_cache_schema_scope("schema-hash-1"):
        without_business = resolve_prompt_cache_key("intent")
    with prompt_cache_schema_scope("schema-hash-1"):
        with business_knowledge_scope(entries, digest):
            with_business = resolve_prompt_cache_key("intent")
    assert without_business == "intent:schema-hash-1"
    assert with_business == f"intent:schema-hash-1:{digest[:16]}"
