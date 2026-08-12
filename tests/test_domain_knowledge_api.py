"""Tests for domain knowledge on engines (internal holder + public export/apply)."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from aetherdialect import AetherEngine, DomainKnowledgeEntry
from aetherdialect._contracts_base import ConfigError, DomainKnowledgeHolder, EngineContext, SensitivityClassification
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._intent_loop import build_intent_interpret_prompt
from aetherdialect._llm_provider import LLMProvider
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import (
    domain_knowledge_digest,
    domain_knowledge_scope,
    prompt_cache_schema_scope,
)
from aetherdialect._utils_artifacts import load_runtime_config


def _minimal_engine(**overrides: object) -> AetherEngine:
    llm_exec = load_runtime_config(merged_env=dict(os.environ))
    defaults: dict[str, object] = dict(
        _runtime_config=RuntimeConfig(
            engine="postgresql",
            artifacts_dir="/tmp/aether_dk",
            engine_context=EngineContext(),
            llm_execution=llm_exec,
        ),
        _llm_config=LLMConfig(provider="openai"),
        _schema_graph=_schema_with_hidden_email(),
        _dialect=MagicMock(),
        _artifacts_dir="/tmp/aether_dk",
        _store=TemplateOps.empty_template_store("graph-dk-1"),
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
        _phase_callback=None,
        _token_provider=None,
    )
    defaults.update(overrides)
    obj = AetherEngine.__new__(AetherEngine)
    for key, value in defaults.items():
        setattr(obj, str(key), value)
    obj._domain_knowledge = DomainKnowledgeHolder()
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
        effective_structural_hash="struct-dk-1",
        schema_graph_id="sg-dk-1__struct-b1",
    )


@pytest.mark.fast
def test_set_and_read_back_domain_knowledge() -> None:
    engine = _minimal_engine()
    entries = (
        DomainKnowledgeEntry(key="revenue", text="Monthly recurring revenue from paid subscriptions."),
        DomainKnowledgeEntry(key="fiscal_year", text="Fiscal year starts in July.", kind="policy"),
    )
    engine._replace_domain_knowledge(entries)
    assert engine._domain_knowledge_entries() == (
        DomainKnowledgeEntry(key="revenue", text="Monthly recurring revenue from paid subscriptions.", kind="glossary"),
        DomainKnowledgeEntry(key="fiscal_year", text="Fiscal year starts in July.", kind="policy"),
    )
    assert engine._domain_knowledge.digest() == domain_knowledge_digest(entries)


@pytest.mark.fast
def test_domain_knowledge_digest_changes_on_edit() -> None:
    engine = _minimal_engine()
    first = (DomainKnowledgeEntry(key="term", text="First definition."),)
    second = (DomainKnowledgeEntry(key="term", text="Updated definition."),)
    engine._replace_domain_knowledge(first)
    digest_first = engine._domain_knowledge.digest()
    engine._replace_domain_knowledge(second)
    digest_second = engine._domain_knowledge.digest()
    assert digest_first != digest_second
    assert digest_first == domain_knowledge_digest(first)
    assert digest_second == domain_knowledge_digest(second)


@pytest.mark.fast
def test_hidden_column_reference_refused() -> None:
    engine = _minimal_engine()
    with pytest.raises(ConfigError, match="sensitive column"):
        engine._replace_domain_knowledge(
            (DomainKnowledgeEntry(key="bad", text="Do not expose customers.email in answers."),)
        )


@pytest.mark.fast
def test_domain_context_injected_into_interpret_prompt() -> None:
    entries = (DomainKnowledgeEntry(key="mrr", text="Monthly recurring revenue."),)
    digest = domain_knowledge_digest(entries)
    with domain_knowledge_scope(entries, digest):
        prompt = build_intent_interpret_prompt("show revenue", "{}", "", ())
    payload = json.loads(prompt)
    assert payload["domain_context"] == [{"key": "mrr", "kind": "glossary", "text": "Monthly recurring revenue."}]


@pytest.mark.fast
def test_schema_graph_id_unchanged_after_domain_knowledge_edit() -> None:
    engine = _minimal_engine()
    graph_id_before = engine._schema_graph.schema_graph_id
    engine._replace_domain_knowledge((DomainKnowledgeEntry(key="term", text="Domain-only glossary entry."),))
    engine._replace_domain_knowledge(
        (
            DomainKnowledgeEntry(key="term", text="Revised glossary entry."),
            DomainKnowledgeEntry(key="alias", text="VIP means top-tier customer."),
        )
    )
    assert engine._schema_graph.schema_graph_id == graph_id_before


@pytest.mark.fast
def test_prompt_cache_key_includes_domain_digest() -> None:
    entries = (DomainKnowledgeEntry(key="term", text="Definition."),)
    digest = domain_knowledge_digest(entries)
    with prompt_cache_schema_scope("schema-hash-1"):
        without_domain = LLMProvider.resolve_prompt_cache_key("intent")
    with prompt_cache_schema_scope("schema-hash-1"):
        with domain_knowledge_scope(entries, digest):
            with_domain = LLMProvider.resolve_prompt_cache_key("intent")
    assert without_domain == "intent:schema-hash-1"
    assert with_domain == f"intent:schema-hash-1:{digest[:16]}"
