"""AetherSpace uid identity: duplicate names, list descriptors, session resolve."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aetherdialect import AetherEngine
from aetherdialect._contracts_base import ConfigError, EngineContext, SchemaRole, SpaceContext
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import load_runtime_config


def _sample_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "film": TableMetadata(
                name="film",
                columns={
                    "film_id": ColumnMetadata(name="film_id", data_type="integer"),
                    "title": ColumnMetadata(name="title", data_type="text"),
                },
                primary_key=["film_id"],
                foreign_keys=[],
            ),
            "customer": TableMetadata(
                name="customer",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="eff_uid",
        schema_graph_id="sg_uid__h",
    )


def _engine(tmp_path: Path) -> AetherEngine:
    schema = _sample_schema()
    ctx = EngineContext(allow_objects=frozenset())
    llm_exec = load_runtime_config(merged_env={})
    runtime = RuntimeConfig(
        engine="postgresql",
        artifacts_dir=str(tmp_path),
        engine_context=ctx,
        execution_context=ctx,
        llm_execution=llm_exec,
    )
    obj = AetherEngine.__new__(AetherEngine)
    obj._runtime_config = runtime
    obj._llm_config = LLMConfig(provider="openai")
    obj._schema_graph = schema
    obj._dialect = MagicMock()
    obj._artifacts_dir = tmp_path
    obj._store = TemplateOps.empty_template_store("sg_uid__h")
    obj._templates = {}
    obj._rejected = {}
    obj._schema_terms = set()
    obj._pipeline_writer_lock = threading.Lock()
    obj._schema_role = SchemaRole.OWNER
    obj._consumer_visible_objects = None
    obj._context_name = "master"
    obj._closed = False
    obj._sandbox_closed = False
    obj._sandbox_mode = False
    return obj


@pytest.mark.fast
def test_duplicate_display_names_mint_distinct_uids(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    a = engine.aetherspace("analytics", SpaceContext(tables=frozenset({"film"})))
    b = engine.aetherspace("analytics", SpaceContext(tables=frozenset({"customer"})))
    assert a.name == "analytics"
    assert b.name == "analytics"
    assert a.uid != b.uid
    assert a.uid.startswith("S")
    assert b.uid.startswith("S")
    listed = engine.list_aetherspaces()
    names = [s.name for s in listed if s.uid != "master"]
    assert names.count("analytics") == 2
    uids = {s.uid for s in listed}
    assert a.uid in uids and b.uid in uids


@pytest.mark.fast
def test_session_by_uid_and_ambiguous_name(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    a = engine.aetherspace("analytics", SpaceContext(tables=frozenset({"film"})))
    engine.aetherspace("analytics", SpaceContext(tables=frozenset({"customer"})))
    with engine.session(mode="reader", space=a.uid) as session:
        assert session.space_name == a.uid
    with pytest.raises(ConfigError, match="ambiguous aetherspace name"):
        with engine.session(mode="reader", space="analytics"):
            pass
    unique = engine.aetherspace("solo", SpaceContext(tables=frozenset({"film"})))
    with engine.session(mode="reader", space="solo") as session:
        assert session.space_name == unique.uid


@pytest.mark.fast
def test_snapshot_without_uid_is_rejected(tmp_path: Path) -> None:
    engine_dir = str(tmp_path / "conn")
    spaces = Path(engine_dir) / "aetherspaces"
    spaces.mkdir(parents=True)
    payload = {
        "version": "0.2.3",
        "tables": ["film"],
        "columns": ["film.film_id"],
        "deny_objects": [],
        "deny_columns": [],
        "table_descriptions": {},
        "column_meta": {},
        "notes": None,
        "notes_hash": "",
    }
    (spaces / "films_only.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing uid"):
        MainExecutionOps.ensure_aetherspace_catalog_upgraded(engine_dir)


@pytest.mark.fast
def test_export_knowledge_keys_by_uid(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    a = engine.aetherspace("analytics", SpaceContext(tables=frozenset({"film"})))
    payload = engine.export_knowledge(a.uid)
    assert payload["uid"] == a.uid
    assert set(payload) == {
        "uid",
        "domain_knowledge",
        "table_descriptions",
        "column_descriptions",
    }


@pytest.mark.fast
def test_update_by_uid_renames_label(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    created = engine.aetherspace("analytics", SpaceContext(tables=frozenset({"film"})))
    updated = engine.aetherspace(
        "reports",
        SpaceContext(tables=frozenset({"film"})),
        uid=created.uid,
    )
    assert updated.uid == created.uid
    assert updated.name == "reports"
    assert engine.aetherspace(uid=created.uid).name == "reports"
