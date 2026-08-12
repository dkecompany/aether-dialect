"""Hermetic notes lifecycle matrix — domain knowledge artifact and space notes hooks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aetherdialect import DomainKnowledgeEntry
from aetherdialect._contracts_base import DomainKnowledgeHolder, EngineContext, SpaceContext
from aetherdialect._contracts_core import RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_spaces import MainSpaceOps
from aetherdialect._utils import (
    delete_domain_knowledge_artifact,
    domain_knowledge_artifact_path,
    knowledge_scope_fingerprint,
    load_domain_knowledge_artifact,
)
from aetherdialect._utils_artifacts import save_domain_knowledge_artifact
from aetherdialect.aetherdialect import AetherEngine


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
        effective_structural_hash="notes-lifecycle",
        schema_graph_id="notes-lifecycle",
    )


def _minimal_engine(tmp_path: Path, *, notes_file: str | None = None) -> SimpleNamespace:
    notes_path = None
    if notes_file is not None:
        notes_path = tmp_path / notes_file
        notes_path.write_text("ARR means annual recurring revenue.\n", encoding="utf-8")
    llm_exec = __import__("aetherdialect._utils_artifacts", fromlist=["load_runtime_config"]).load_runtime_config(
        merged_env={}
    )
    obj = SimpleNamespace(
        _runtime_config=RuntimeConfig(
            engine="postgresql",
            artifacts_dir=str(tmp_path),
            engine_context=EngineContext(notes_file=str(notes_path) if notes_path else None),
            llm_execution=llm_exec,
        ),
        _schema_graph=_schema(),
        _artifacts_dir=tmp_path,
        _domain_knowledge=DomainKnowledgeHolder(),
        _audit_emit=lambda *_a, **_k: None,
    )
    obj._schema_graph.notes_sha256 = ""
    obj._ingest_notes_domain_knowledge = AetherEngine._ingest_notes_domain_knowledge.__get__(obj, AetherEngine)
    obj._load_persisted_domain_knowledge = AetherEngine._load_persisted_domain_knowledge.__get__(obj, AetherEngine)
    obj._replace_domain_knowledge = AetherEngine._replace_domain_knowledge.__get__(obj, AetherEngine)
    obj._persist_domain_knowledge = AetherEngine._persist_domain_knowledge.__get__(obj, AetherEngine)
    obj._clear_notes_domain_knowledge = AetherEngine._clear_notes_domain_knowledge.__get__(obj, AetherEngine)
    obj._domain_knowledge_entries = AetherEngine._domain_knowledge_entries.__get__(obj, AetherEngine)
    return obj


@pytest.mark.fast
def test_none_to_none_writes_no_domain_knowledge_artifact(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    engine._ingest_notes_domain_knowledge()
    assert engine._domain_knowledge_entries() == ()
    assert not domain_knowledge_artifact_path(tmp_path).is_file()


@pytest.mark.fast
def test_none_to_provided_extracts_and_persists_domain_knowledge(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path, notes_file="notes.txt")
    engine._schema_graph.notes_sha256 = hashlib.sha256(b"ARR means annual recurring revenue.\n").hexdigest()
    entries = (DomainKnowledgeEntry(key="arr", text="annual recurring revenue", kind="glossary"),)
    with patch("aetherdialect.aetherdialect.extract_domain_knowledge_from_notes", return_value=entries):
        engine._ingest_notes_domain_knowledge()
    assert engine._domain_knowledge_entries() == entries
    assert domain_knowledge_artifact_path(tmp_path).is_file()
    loaded = load_domain_knowledge_artifact(tmp_path, engine._schema_graph)
    assert loaded == entries


@pytest.mark.fast
def test_provided_same_loads_artifact_without_reextract(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path, notes_file="notes.txt")
    notes_sha = hashlib.sha256(b"ARR means annual recurring revenue.\n").hexdigest()
    engine._schema_graph.notes_sha256 = notes_sha
    entries = (DomainKnowledgeEntry(key="arr", text="annual recurring revenue", kind="glossary"),)
    save_domain_knowledge_artifact(
        tmp_path,
        entries,
        notes_sha256=notes_sha,
        scope_fingerprint=knowledge_scope_fingerprint(engine._schema_graph),
    )
    with patch("aetherdialect.aetherdialect.extract_domain_knowledge_from_notes") as extract:
        assert engine._load_persisted_domain_knowledge() is True
        extract.assert_not_called()
    assert engine._domain_knowledge_entries() == entries


@pytest.mark.fast
def test_notes_hash_drift_refresh_hook_exists_in_schema_finalize() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "src" / "aetherdialect" / "_schema_finalize.py").read_text(encoding="utf-8")
    assert "notes_refresh_only" in source
    assert "incoming_notes_hash" in source


@pytest.mark.fast
def test_provided_to_removed_clears_memory_and_deletes_stale_artifact(tmp_path: Path) -> None:
    engine = _minimal_engine(tmp_path)
    notes_sha = hashlib.sha256(b"ARR means annual recurring revenue.\n").hexdigest()
    engine._schema_graph.notes_sha256 = notes_sha
    entries = (DomainKnowledgeEntry(key="arr", text="annual recurring revenue", kind="glossary"),)
    save_domain_knowledge_artifact(tmp_path, entries, notes_sha256=notes_sha)
    engine._replace_domain_knowledge(entries)
    engine._runtime_config = RuntimeConfig(
        engine="postgresql",
        artifacts_dir=str(tmp_path),
        engine_context=EngineContext(),
        llm_execution=engine._runtime_config.llm_execution,
    )
    engine._ingest_notes_domain_knowledge()
    assert engine._domain_knowledge_entries() == ()
    assert not domain_knowledge_artifact_path(tmp_path).is_file()


@pytest.mark.fast
def test_delete_domain_knowledge_artifact_is_idempotent(tmp_path: Path) -> None:
    entries = (DomainKnowledgeEntry(key="arr", text="annual recurring revenue", kind="glossary"),)
    save_domain_knowledge_artifact(tmp_path, entries, notes_sha256="abc")
    assert delete_domain_knowledge_artifact(tmp_path) is True
    assert delete_domain_knowledge_artifact(tmp_path) is False


@pytest.mark.fast
def test_space_notes_added_stamps_notes_hash() -> None:
    schema = _schema()
    snapshot = {"tables": ["orders"], "columns": ["orders.id"]}
    space_ctx = SpaceContext(tables=frozenset({"orders"}))
    with patch("aetherdialect._main_spaces.EngineConfig.llm_credentials_configured", return_value=False):
        out = MainSpaceOps.enrich_space_snapshot_with_notes(
            snapshot,
            schema,
            space_ctx,
            notes="left_t.id is the join key.\n",
        )
    assert out["notes_hash"] == hashlib.sha256(b"left_t.id is the join key.\n").hexdigest()


@pytest.mark.fast
def test_restricted_columns_omit_distinct_count_from_ground_samples() -> None:
    from aetherdialect._constants_runtime import GROUND_FIELDS, SCHEMA_FIELD_SAMPLES
    from aetherdialect._contracts_base import SensitivityClassification

    graph = SchemaGraph(
        tables={
            "staff": TableMetadata(
                name="staff",
                columns={
                    "email": ColumnMetadata(
                        name="email",
                        data_type="varchar",
                        value_type="string",
                        distinct_count=42,
                        sensitivity=SensitivityClassification.RESTRICTED,
                    )
                },
                primary_key=["email"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        effective_structural_hash="restricted-samples",
        schema_graph_id="restricted-samples",
    )
    payload = __import__("json").loads(graph.schema_payload_json(GROUND_FIELDS, owner_master_scope=True))
    samples = payload["staff"]["columns"]["email"].get(SCHEMA_FIELD_SAMPLES, {})
    assert "distinct_count" not in samples


@pytest.mark.fast
def test_space_notes_isolated_from_master_domain_knowledge() -> None:
    schema = _schema()
    engine_dk = (DomainKnowledgeEntry(key="arr", text="engine annual recurring revenue", kind="glossary"),)
    space_dk = (DomainKnowledgeEntry(key="nrr", text="space net retention", kind="metric"),)
    snapshot = {"tables": ["orders"], "columns": ["orders.id"]}
    space_ctx = SpaceContext(tables=frozenset({"orders"}), notes="space-only notes.\n")
    with patch("aetherdialect._main_spaces.EngineConfig.llm_credentials_configured", return_value=False):
        with patch(
            "aetherdialect._main_spaces.MainSpaceOps.merge_overlay_knowledge_layers",
            return_value=(space_dk, (), True),
        ):
            out = MainSpaceOps.enrich_space_snapshot_with_notes(
                snapshot,
                schema,
                space_ctx,
                engine_domain_knowledge=engine_dk,
            )
    keys = {item["key"] for item in out["domain_knowledge"]}
    assert keys == {"nrr"}
    assert "arr" not in keys
