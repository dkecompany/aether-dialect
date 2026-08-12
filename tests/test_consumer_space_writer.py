"""Consumer writer sessions persist learning into the active space partition only."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aetherdialect import AetherEngine
from aetherdialect._contracts_base import EngineContext, OwnerOnlyOperationError, SchemaRole, SpaceContext
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._main_session import PipelineSession
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
        effective_structural_hash="eff_consumer_writer",
        schema_graph_id="sg_consumer_writer__h",
    )


def _engine(tmp_path: Path, *, role: SchemaRole = SchemaRole.OWNER) -> AetherEngine:
    schema = _sample_schema()
    ctx = EngineContext(allow_objects=frozenset({"film", "customer"}))
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
    obj._store = TemplateOps.empty_template_store("sg_consumer_writer__h")
    obj._templates = {}
    obj._rejected = {}
    obj._schema_terms = set()
    obj._pipeline_writer_lock = threading.Lock()
    obj._schema_role = role
    obj._consumer_visible_objects = frozenset({"film", "customer"}) if role == SchemaRole.CONSUMER else None
    obj._context_name = "master"
    obj._closed = False
    obj._sandbox_closed = False
    obj._sandbox_mode = False
    obj._credential_default_space_uid = "master"
    return obj


@pytest.mark.fast
def test_consumer_writer_session_opens(tmp_path: Path) -> None:
    owner = _engine(tmp_path, role=SchemaRole.OWNER)
    space = owner.aetherspace("analytics", SpaceContext(tables=frozenset({"film"})))
    consumer = _engine(tmp_path, role=SchemaRole.CONSUMER)
    with consumer.session(mode="writer", space=space.uid) as session:
        assert isinstance(session, PipelineSession)
        assert session._session_mode == "writer"
        assert session.space_name == space.uid


@pytest.mark.fast
def test_consumer_cannot_define_aetherspace(tmp_path: Path) -> None:
    consumer = _engine(tmp_path, role=SchemaRole.CONSUMER)
    with pytest.raises(OwnerOnlyOperationError, match="aetherspace"):
        consumer.aetherspace("analytics", SpaceContext(tables=frozenset({"film"})))


@pytest.mark.fast
def test_consumer_cannot_apply_structure(tmp_path: Path) -> None:
    consumer = _engine(tmp_path, role=SchemaRole.CONSUMER)
    with pytest.raises(OwnerOnlyOperationError):
        consumer.apply_structure({})
