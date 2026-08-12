"""export_knowledge keeps master and named-space layers distinct."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect import AetherEngine, DomainKnowledgeEntry
from aetherdialect._contracts_base import DomainKnowledgeHolder, EngineContext
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import load_runtime_config


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
        effective_structural_hash="h-ekl",
        schema_graph_id="sg-ekl",
    )


def _minimal_engine(tmp_path: Path) -> AetherEngine:
    llm_exec = load_runtime_config(merged_env={})
    obj = AetherEngine.__new__(AetherEngine)
    obj._runtime_config = RuntimeConfig(
        engine="postgresql",
        artifacts_dir=str(tmp_path),
        engine_context=EngineContext(),
        llm_execution=llm_exec,
    )
    obj._llm_config = LLMConfig(provider="openai")
    obj._schema_graph = _schema()
    obj._dialect = MagicMock()
    obj._artifacts_dir = tmp_path
    obj._store = TemplateOps.empty_template_store("graph-ekl")
    obj._templates = {}
    obj._rejected = {}
    obj._schema_terms = set()
    obj._config_file = None
    obj._execution_engine = None
    obj._audit_sink = None
    obj._pipeline_writer_lock = __import__("threading").Lock()
    obj._schema_role = "owner"
    obj._consumer_visible_objects = None
    obj._schema_stats = {"table_count": 1, "total_filterable": 1}
    obj._phase_callback = None
    obj._phase_callback = None
    obj._token_provider = None
    obj._context_name = "master"
    obj._domain_knowledge = DomainKnowledgeHolder()
    return obj


@pytest.mark.fast
def test_engine_and_space_layers_distinct(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    engine_entries = (
        DomainKnowledgeEntry(key="arr", text="engine arr", kind="glossary"),
        DomainKnowledgeEntry(key="fy", text="engine fy", kind="policy"),
    )
    engine._replace_domain_knowledge(engine_entries)
    space_snap: dict[str, Any] = {
        "uid": "analytics",
        "tables": ["orders"],
        "columns": ["orders.id"],
        "table_descriptions": {},
        "column_meta": {},
        "domain_knowledge": [
            {"key": "arr", "kind": "glossary", "text": "space arr", "referenced_entities": []},
            {"key": "nrr", "kind": "metric", "text": "space only", "referenced_entities": []},
        ],
    }
    MainExecutionOps.save_aetherspace_snapshot(str(tmp_path), "analytics", space_snap)
    master = engine.export_knowledge()
    named = engine.export_knowledge(space="analytics")
    assert master["uid"] == "master"
    assert named["uid"] == "analytics"
    assert master["domain_knowledge"] == [
        {"key": "arr", "kind": "glossary", "text": "engine arr", "referenced_entities": []},
        {"key": "fy", "kind": "policy", "text": "engine fy", "referenced_entities": []},
    ]
    assert named["domain_knowledge"] == [
        MainExecutionOps._domain_knowledge_entry_to_dict(e)
        for e in MainExecutionOps.merge_domain_knowledge(
            engine_entries,
            (
                DomainKnowledgeEntry(key="arr", text="space arr", kind="glossary"),
                DomainKnowledgeEntry(key="nrr", text="space only", kind="metric"),
            ),
        )
    ]
    assert named["domain_knowledge"] != master["domain_knowledge"]
