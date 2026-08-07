"""Schema overrides export/apply include business_knowledge and post- apply refine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, BusinessKnowledgeEntry
from aetherdialect._constants import SCHEMA_OVERRIDES_VERSION
from aetherdialect._contracts_base import BusinessKnowledgeHolder, EngineContext, LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import load_runtime_config
from aetherdialect._schema_overrides import (
    apply_overrides_and_persist,
    dump_schema_overrides_dict,
    dump_schema_overrides_to_path,
)
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
        effective_structural_hash="h-ov-bk",
        schema_graph_id="g-ov-bk",
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
        _store=TemplateOps.empty_template_store("graph-ov-bk"),
        _templates={},
        _rejected={},
        _schema_terms=set(),
        _config_file=None,
        _execution_engine=None,
        _audit_sink=None,
        _pipeline_writer_lock=__import__("threading").Lock(),
        _schema_role="owner",
        _consumer_visible_objects=None,
        _schema_stats={"table_count": 1, "total_filterable": 1},
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
def test_export_contains_bk(tmp_path: Path) -> None:
    entries = (
        BusinessKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="glossary"),
        BusinessKnowledgeEntry(key="fy", text="Fiscal year starts July.", kind="policy"),
    )
    engine = _minimal_engine(tmp_path)
    engine.set_business_knowledge(entries)
    with patch.object(AetherEngine, "_require_owner", return_value=None):
        path = engine.export_overrides()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert "business_knowledge" in doc
    assert doc["business_knowledge"] == {
        "entries": [
            {"key": "arr", "kind": "glossary", "text": "Annual recurring revenue."},
            {"key": "fy", "kind": "policy", "text": "Fiscal year starts July."},
        ]
    }
    dumped = dump_schema_overrides_dict(_schema(), business_knowledge=entries)
    assert dumped["business_knowledge"]["entries"][0]["key"] == "arr"


@pytest.mark.fast
def test_apply_replaces_bk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aetherdialect._config.EngineConfig.llm_credentials_configured",
        lambda: False,
    )
    engine = _minimal_engine(tmp_path)
    engine.set_business_knowledge((BusinessKnowledgeEntry(key="old", text="replace me", kind="glossary"),))
    doc: dict[str, Any] = {
        "version": SCHEMA_OVERRIDES_VERSION,
        "tables": {},
        "foreign_keys_add": [],
        "foreign_keys_remove": [],
        "primary_keys_add": [],
        "primary_keys_remove": [],
        "business_knowledge": {
            "entries": [
                {"key": "nrr", "kind": "metric", "text": "Net recurring revenue."},
            ]
        },
    }
    overrides_path = tmp_path / "schema_overrides.json"
    overrides_path.write_text(json.dumps(doc), encoding="utf-8")
    schema_json = tmp_path / "schema_graph.json.gz"
    schema_json.write_bytes(b"")

    def _fake_persist(*_a: Any, **_k: Any) -> None:
        return None

    with (
        patch("aetherdialect._schema_overrides.save_schema_to_cache", _fake_persist),
        patch("aetherdialect._schema_overrides._write_overrides_sidecar_payload", _fake_persist),
        patch("aetherdialect._schema_overrides.load_overrides_sidecar", return_value={}),
        patch.object(AetherEngine, "_require_owner", return_value=None),
        patch.object(AetherEngine, "_audit_emit", return_value=None),
        patch("aetherdialect.aetherdialect._print_override_summary", return_value=None),
        patch(
            "aetherdialect.aetherdialect.apply_overrides_and_persist",
            wraps=apply_overrides_and_persist,
        ),
    ):
        # Prefer exercising the real apply path via engine after wiring report BK.
        report = apply_overrides_and_persist(
            engine._schema_graph,
            overrides_path,
            schema_json_path=str(schema_json),
            dialect=None,
        )
        assert report.business_knowledge_entries is not None
        engine.set_business_knowledge(report.business_knowledge_entries)
    keys = {e.key: e.text for e in engine.business_knowledge()}
    assert "old" not in keys
    assert keys["nrr"] == "Net recurring revenue."


@pytest.mark.fast
def test_refine_runs_after_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aetherdialect._config.EngineConfig.llm_credentials_configured",
        lambda: True,
    )
    doc: dict[str, Any] = {
        "version": SCHEMA_OVERRIDES_VERSION,
        "tables": {},
        "foreign_keys_add": [],
        "foreign_keys_remove": [],
        "primary_keys_add": [],
        "primary_keys_remove": [],
        "business_knowledge": {
            "entries": [
                {"key": "arr", "kind": "glossary", "text": "arr means annual recurring revenue"},
            ]
        },
    }
    overrides_path = tmp_path / "schema_overrides.json"
    overrides_path.write_text(json.dumps(doc), encoding="utf-8")
    schema_json = tmp_path / "schema_graph.json.gz"
    schema_json.write_bytes(b"")
    refined_payload = json.dumps(
        {
            "entries": [
                {
                    "key": "arr",
                    "kind": "glossary",
                    "text": "ARR means annual recurring revenue.",
                }
            ]
        }
    )
    with (
        patch("aetherdialect._schema_overrides.save_schema_to_cache"),
        patch("aetherdialect._schema_overrides._write_overrides_sidecar_payload"),
        patch("aetherdialect._schema_overrides.load_overrides_sidecar", return_value={}),
        patch(
            "aetherdialect._schema_overrides.LLMProvider.chat",
            return_value=refined_payload,
        ) as chat_mock,
    ):
        report = apply_overrides_and_persist(
            _schema(),
            overrides_path,
            schema_json_path=str(schema_json),
            dialect=None,
        )
    assert chat_mock.called
    assert report.business_knowledge_entries is not None
    assert len(report.business_knowledge_entries) == 1
    assert report.business_knowledge_entries[0].text == "ARR means annual recurring revenue."
    assert report.business_knowledge_refined == 1


@pytest.mark.fast
def test_dump_to_path_includes_bk_section(tmp_path: Path) -> None:
    entries = (BusinessKnowledgeEntry(key="x", text="X definition.", kind="glossary"),)
    path = dump_schema_overrides_to_path(_schema(), tmp_path / "schema_overrides.json", business_knowledge=entries)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["business_knowledge"]["entries"][0]["key"] == "x"
