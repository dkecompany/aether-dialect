"""Structure documents reject prose; domain knowledge belongs on space knowledge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, DomainKnowledgeEntry
from aetherdialect._constants import STRUCTURE_DOCUMENT_VERSION, STRUCTURE_PROSE_REDIRECT_HINT
from aetherdialect._contracts_base import DomainKnowledgeHolder, EngineContext
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_finalize import (
    _validate_structure_edits,
    dump_structure_edits,
    dump_structure_to_path,
)
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
        effective_structural_hash="h-ov-dk",
        schema_graph_id="g-ov-dk",
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
        _store=TemplateOps.empty_template_store("graph-ov-dk"),
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
        _phase_callback=None,
        _token_provider=None,
    )
    defaults.update(overrides)
    obj = AetherEngine.__new__(AetherEngine)
    for key, value in defaults.items():
        setattr(obj, str(key), value)
    obj._domain_knowledge = DomainKnowledgeHolder()
    return obj


@pytest.mark.fast
def test_export_omits_dk_and_descriptions(tmp_path: Path) -> None:
    entries = (
        DomainKnowledgeEntry(key="arr", text="Annual recurring revenue.", kind="glossary"),
        DomainKnowledgeEntry(key="fy", text="Fiscal year starts July.", kind="policy"),
    )
    engine = _minimal_engine(tmp_path)
    engine._replace_domain_knowledge(entries)
    with patch.object(AetherEngine, "_require_owner", return_value=None):
        doc = engine.export_structure()
    assert isinstance(doc, dict)
    assert "domain_knowledge" not in doc
    assert "description" not in json.dumps(doc.get("tables") or {})
    dumped = dump_structure_edits(_schema())
    assert "domain_knowledge" not in dumped


@pytest.mark.fast
def test_validate_rejects_domain_knowledge() -> None:
    doc: dict[str, Any] = {
        "version": STRUCTURE_DOCUMENT_VERSION,
        "tables": {},
        "foreign_keys_add": [],
        "foreign_keys_remove": [],
        "primary_keys_add": [],
        "primary_keys_remove": [],
        "domain_knowledge": {"entries": [{"key": "nrr", "kind": "metric", "text": "Net"}]},
    }
    with pytest.raises(ValueError, match="domain_knowledge") as exc:
        _validate_structure_edits(doc)
    assert STRUCTURE_PROSE_REDIRECT_HINT in str(exc.value)


@pytest.mark.fast
def test_validate_rejects_description() -> None:
    doc: dict[str, Any] = {
        "version": STRUCTURE_DOCUMENT_VERSION,
        "tables": {"orders": {"description": "orders table"}},
        "foreign_keys_add": [],
        "foreign_keys_remove": [],
        "primary_keys_add": [],
        "primary_keys_remove": [],
    }
    with pytest.raises(ValueError, match="description") as exc:
        _validate_structure_edits(doc)
    assert STRUCTURE_PROSE_REDIRECT_HINT in str(exc.value)


@pytest.mark.fast
def test_dump_to_path_is_structural_only(tmp_path: Path) -> None:
    path = dump_structure_to_path(_schema(), tmp_path / "schema_structure.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["version"] == STRUCTURE_DOCUMENT_VERSION
    assert "domain_knowledge" not in doc
