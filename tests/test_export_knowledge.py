"""export_space_knowledge and export_knowledge (wrapper) shapes."""

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
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", role="identifier", description="pk"),
                    "amount": ColumnMetadata(name="amount", data_type="numeric", role="measure", description="money"),
                },
                primary_key=["id"],
                foreign_keys=[],
                description="customer orders",
            )
        },
        join_paths_multi={},
        effective_structural_hash="h-ek",
        schema_graph_id="sg-ek__h",
    )


def _minimal_engine(tmp_path: Path, **overrides: object) -> AetherEngine:
    llm_exec = load_runtime_config(merged_env={})
    defaults: dict[str, object] = dict(
        _runtime_config=RuntimeConfig(
            engine="postgresql",
            artifacts_dir=str(tmp_path),
            engine_context=EngineContext(),
            llm_execution=llm_exec,
        ),
        _llm_config=LLMConfig(provider="openai"),
        _schema_graph=_schema(),
        _dialect=MagicMock(),
        _artifacts_dir=tmp_path,
        _store=TemplateOps.empty_template_store("graph-ek"),
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


@pytest.mark.fast
def test_master_json_shape(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    entries = (BusinessKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="glossary"),)
    engine.set_business_knowledge(entries)
    payload = engine.export_space_knowledge()
    assert payload["format_version"] == "0.2.1"
    assert payload["scope"] == "master"
    assert payload["business_knowledge_version"] == 1
    assert payload["business_knowledge"] == [
        {"key": "arr", "kind": "glossary", "text": "Annual recurring revenue."},
    ]
    assert "tables" not in payload


@pytest.mark.fast
def test_space_overlay_differs(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    engine.set_business_knowledge((BusinessKnowledgeEntry(key="arr", text="engine arr", kind="glossary"),))
    space_snap: dict[str, Any] = {
        "tables": ["orders"],
        "columns": ["orders.id", "orders.amount"],
        "business_knowledge": [
            {"key": "arr", "kind": "glossary", "text": "space arr wins"},
            {"key": "nrr", "kind": "metric", "text": "net recurring"},
        ],
    }
    with patch(
        "aetherdialect.aetherdialect.load_aetherspace_snapshot",
        return_value=space_snap,
    ):
        master = engine.export_space_knowledge()
        space = engine.export_space_knowledge(space="analytics")
    assert master["scope"] == "master"
    assert space["scope"] == "analytics"
    assert master["business_knowledge"][0]["text"] == "engine arr"
    by_key = {e["key"]: e["text"] for e in space["business_knowledge"]}
    assert by_key["arr"] == "space arr wins"
    assert by_key["nrr"] == "net recurring"


@pytest.mark.fast
def test_digest_changes_when_bk_changes(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    engine.set_business_knowledge((BusinessKnowledgeEntry(key="t", text="one"),))
    v1 = engine.export_space_knowledge()["business_knowledge_version"]
    engine.set_business_knowledge((BusinessKnowledgeEntry(key="t", text="two"),))
    v2 = engine.export_space_knowledge()["business_knowledge_version"]
    assert v1 == 1
    assert v2 == 2


@pytest.mark.fast
def test_export_metadata_has_tables(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    meta = engine.export_metadata()
    assert meta["format_version"] == "0.2.1"
    assert meta["table_count"] == 1
    assert meta["tables"][0]["name"] == "orders"
