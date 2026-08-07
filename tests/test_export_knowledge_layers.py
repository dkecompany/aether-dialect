"""export_knowledge wrapper returns per-level BK for metadata-review read-back."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, BusinessKnowledgeEntry
from aetherdialect._contracts_base import BusinessKnowledgeHolder, EngineContext, LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import load_runtime_config
from aetherdialect._templates import TemplateOps


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
    obj._construction_phase_callback = None
    obj._ask_phase_callback = None
    obj._token_provider = None
    obj._business_knowledge = BusinessKnowledgeHolder()
    return obj


@pytest.mark.fast
def test_engine_and_space_layers_distinct(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    engine_entries = (
        BusinessKnowledgeEntry(key="arr", text="engine arr", kind="glossary"),
        BusinessKnowledgeEntry(key="fy", text="engine fy", kind="policy"),
    )
    engine.set_business_knowledge(engine_entries)
    space_snap: dict[str, Any] = {
        "tables": ["orders"],
        "business_knowledge": [
            {"key": "arr", "kind": "glossary", "text": "space arr"},
            {"key": "nrr", "kind": "metric", "text": "space only"},
        ],
    }
    with patch(
        "aetherdialect.aetherdialect.list_saved_aetherspace_names",
        return_value=("analytics",),
    ):
        with patch(
            "aetherdialect.aetherdialect.load_aetherspace_snapshot",
            return_value=space_snap,
        ):
            layers = engine.export_knowledge()
    assert layers["format_version"] == "0.2.1"
    assert layers["engine"]["business_knowledge"] == [
        {"key": "arr", "kind": "glossary", "text": "engine arr"},
        {"key": "fy", "kind": "policy", "text": "engine fy"},
    ]
    assert layers["spaces"]["analytics"]["business_knowledge"] == [
        {"key": "arr", "kind": "glossary", "text": "space arr"},
        {"key": "nrr", "kind": "metric", "text": "space only"},
    ]
    assert layers["spaces"]["analytics"]["business_knowledge"] != layers["engine"]["business_knowledge"]
