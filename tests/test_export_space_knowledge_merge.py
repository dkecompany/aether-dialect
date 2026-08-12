"""export_knowledge returns the same merged DK as the ask path."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect import DomainKnowledgeEntry
from aetherdialect._contracts_base import DomainKnowledgeHolder, EngineContext
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._main_spaces import MainSpaceOps
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import load_runtime_config
from aetherdialect.aetherdialect import AetherEngine


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
    obj._schema_graph = SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        effective_structural_hash="h",
        schema_graph_id="g1",
    )
    obj._dialect = MagicMock()
    obj._artifacts_dir = tmp_path
    obj._store = TemplateOps.empty_template_store("g1")
    obj._templates = {}
    obj._schema_role = "owner"
    obj._context_name = "master"
    obj._domain_knowledge = DomainKnowledgeHolder()
    return obj


@pytest.mark.fast
def test_export_named_space_matches_runtime_merge(tmp_path: Path) -> None:
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
        "domain_knowledge": [
            {"key": "arr", "kind": "glossary", "text": "space arr", "referenced_entities": []},
            {"key": "nrr", "kind": "metric", "text": "space only", "referenced_entities": []},
        ],
    }
    MainExecutionOps.save_aetherspace_snapshot(str(tmp_path), "analytics", space_snap)
    exported = engine.export_knowledge(space="analytics")
    space_dk = MainSpaceOps.entries_from_snapshot_domain_knowledge(space_snap)
    merged = MainExecutionOps.merge_domain_knowledge(engine_entries, space_dk)
    expected = [MainSpaceOps._domain_knowledge_entry_to_dict(e) for e in merged]
    assert exported["domain_knowledge"] == expected
