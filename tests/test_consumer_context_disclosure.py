"""Owner-only gates for list_contexts and export_context (consumers may not call them)."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_base import EngineContext, OwnerOnlyOperationError, SchemaRole
from aetherdialect._contracts_core import LLMConfig, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import load_runtime_config
from aetherdialect.aetherdialect import AetherEngine


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer"),
                    "secret": ColumnMetadata(name="secret", data_type="text"),
                },
                primary_key=["id"],
                foreign_keys=[],
            ),
            "staff": TableMetadata(
                name="staff",
                columns={"id": ColumnMetadata(name="id", data_type="integer")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h",
        schema_graph_id="g1",
    )


def _consumer_engine(tmp_path: Path) -> AetherEngine:
    llm_exec = load_runtime_config(merged_env={})
    engine = AetherEngine.__new__(AetherEngine)
    engine._runtime_config = RuntimeConfig(
        engine="postgresql",
        artifacts_dir=str(tmp_path),
        engine_context=EngineContext(deny_columns=frozenset({"orders.secret"})),
        llm_execution=llm_exec,
    )
    engine._llm_config = LLMConfig(provider="openai")
    engine._schema_graph = _schema()
    engine._schema_role = SchemaRole.CONSUMER
    engine._consumer_visible_objects = frozenset({"orders", "orders.id"})
    engine._artifacts_dir = tmp_path
    engine._context_name = "master"
    engine._pipeline_writer_lock = threading.Lock()
    engine._dialect = MagicMock()
    engine._store = TemplateOps.empty_template_store("g1")
    engine._templates = {}
    return engine


@pytest.mark.fast
def test_consumer_list_contexts_is_owner_only(tmp_path: Path) -> None:
    engine = _consumer_engine(tmp_path)
    with pytest.raises(OwnerOnlyOperationError, match="list_contexts"):
        engine.list_contexts()


@pytest.mark.fast
def test_consumer_export_context_is_owner_only(tmp_path: Path) -> None:
    engine = _consumer_engine(tmp_path)
    with pytest.raises(OwnerOnlyOperationError, match="export_context"):
        engine.export_context("master")
