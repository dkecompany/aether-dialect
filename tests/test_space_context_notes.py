"""Unit tests for SpaceContext.notes_file parallelism (no sandbox corpus required)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherFederation
from aetherdialect._constants import AETHERSPACE_ARTIFACT_VERSION
from aetherdialect._contracts_base import (
    AetherFederationInitResult,
    ConfigError,
    EngineContext,
    FederationContext,
    FederationMappings,
    LLMConfig,
    RuntimeConfig,
    SpaceContext,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import load_runtime_config
from aetherdialect._federation import compose_composite_graph, parse_federation_manifest, parse_federation_mappings
from aetherdialect._main_execution import (
    load_aetherspace_snapshot,
    save_aetherspace_snapshot,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import empty_template_store


@pytest.mark.fast
def test_blank_notes_file_raises_config_error_on_all_three_contexts() -> None:
    with pytest.raises(ConfigError, match="notes_file must be omitted or a non-empty path"):
        SpaceContext(notes_file="   ")
    with pytest.raises(ConfigError, match="notes_file must be omitted or a non-empty path"):
        EngineContext(notes_file="")
    with pytest.raises(ConfigError, match="notes_file must be omitted or a non-empty path"):
        FederationContext(notes_file="\t")


@pytest.mark.fast
def test_old_version_aetherspace_snapshot_raises_version_mismatch(tmp_path: Path) -> None:
    engine_dir = str(tmp_path)
    assert load_aetherspace_snapshot(engine_dir, "missing") is None
    stale = {
        "version": AETHERSPACE_ARTIFACT_VERSION - 1,
        "tables": ["film"],
        "columns": ["film.film_id"],
        "notes": None,
    }
    save_aetherspace_snapshot(engine_dir, "stale", stale)
    with pytest.raises(ConfigError, match=r"version .*2") as exc_info:
        load_aetherspace_snapshot(engine_dir, "stale")
    msg = str(exc_info.value)
    assert str(AETHERSPACE_ARTIFACT_VERSION) in msg
    assert "Delete" in msg


@pytest.mark.fast
def test_federation_aetherspace_accepts_notes_file_on_space_context(tmp_path: Path) -> None:
    notes = tmp_path / "fed_space_notes.txt"
    notes.write_text("left_t.id is the join key.\n", encoding="utf-8")
    arts = str(tmp_path / "fed_arts")
    os.makedirs(arts, exist_ok=True)

    def _graph(table: str, source_id: str) -> SchemaGraph:
        tables = {
            table: TableMetadata(
                name=table,
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id=source_id,
            )
        }
        return SchemaGraph(
            tables=tables,
            join_paths_multi=recompute_join_paths_multi(tables),
            schema_graph_id=f"sg_{source_id}_{table}",
            effective_structural_hash=f"eff_{source_id}_{table}",
        )

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_notes",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"left_t": "a", "right_t": "b"},
            "cross_source_joins": [
                {"left": "left_t.id", "right": "right_t.id", "kind": "inner", "logical_key": "id"},
            ],
        },
        include_derived_roster=True,
    )
    composite = compose_composite_graph(
        {"a": _graph("left_t", "a"), "b": _graph("right_t", "b")},
        manifest,
    )
    store = empty_template_store(str(composite.schema_graph_id))
    llm_exec = load_runtime_config(merged_env={})
    member = MagicMock()
    member.dialect = "duckdb"
    member._runtime_config = RuntimeConfig(
        engine="duckdb",
        artifacts_dir=arts,
        engine_context=MagicMock(),
        llm_execution=llm_exec,
    )
    member._llm_config = LLMConfig(provider="openai")
    member._schema_graph = _graph("left_t", "a")
    member._artifacts_dir = arts
    member._dialect = MagicMock()
    member._execution_engine = None
    member._native_connection = None
    member._context_name = "master"
    member._schema_role = "owner"
    member._engine_identity = None
    bundle = AetherFederationInitResult(
        runtime_config=RuntimeConfig(
            engine="duckdb",
            artifacts_dir=arts,
            engine_context=MagicMock(),
            llm_execution=llm_exec,
        ),
        llm_config=LLMConfig(provider="openai"),
        schema_graph=composite,
        dialect=MagicMock(),
        artifacts_dir=arts,
        store=store,
        templates={},
        rejected={},
        schema_terms=set(),
        schema_stats={"table_count": 2},
        federation_manifest=manifest,
        federation_mappings=FederationMappings(version=2),
        federation_member_graphs={"a": _graph("left_t", "a"), "b": _graph("right_t", "b")},
        federation_storage_dir=arts,
        federation_source_runtimes={
            "a": MagicMock(source_id="a", dialect=MagicMock(), sqlalchemy_engine=None, native_connection=None),
            "b": MagicMock(source_id="b", dialect=MagicMock(), sqlalchemy_engine=None, native_connection=None),
        },
        members={"conn_a": member, "conn_b": member},
    )
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed = AetherFederation(
            "fed_notes",
            members={"conn_a": member, "conn_b": member},
            declaration_file="/tmp/aether_fed_notes_declaration.json",
        )
    desc = fed.aetherspace(
        "left_only",
        SpaceContext(tables=frozenset({"left_t"}), notes_file=str(notes)),
    )
    assert desc.notes is not None
    assert "join key" in desc.notes
    snap = load_aetherspace_snapshot(arts, "left_only")
    assert snap is not None
    assert snap["notes_hash"] == hashlib.sha256(b"left_t.id is the join key.\n").hexdigest()
    assert snap["version"] == AETHERSPACE_ARTIFACT_VERSION


@pytest.mark.fast
def test_federation_aetherspace_definition_rejects_collapsed_member_table(tmp_path: Path) -> None:
    arts = str(tmp_path / "fed_arts")
    os.makedirs(arts, exist_ok=True)

    def _graph(table: str, source_id: str) -> SchemaGraph:
        tables = {
            table: TableMetadata(
                name=table,
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id=source_id,
            )
        }
        return SchemaGraph(
            tables=tables,
            join_paths_multi=recompute_join_paths_multi(tables),
            schema_graph_id=f"sg_{source_id}_{table}",
            effective_structural_hash=f"eff_{source_id}_{table}",
        )

    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_collapsed_space",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"payment_a": "a", "payment_b": "b"},
            "cross_source_joins": [],
        },
        include_derived_roster=True,
    )
    mappings = parse_federation_mappings(
        {
            "version": 2,
            "logical_tables": [
                {
                    "logical": "payment",
                    "semantics": "union",
                    "members": [
                        {"source": "a", "table": "payment_a", "columns": {"id": "id"}},
                        {"source": "b", "table": "payment_b", "columns": {"id": "id"}},
                    ],
                },
            ],
            "logical_columns": [],
        }
    )
    composite = compose_composite_graph(
        {"a": _graph("payment_a", "a"), "b": _graph("payment_b", "b")},
        manifest,
        mappings,
    )
    store = empty_template_store(str(composite.schema_graph_id))
    llm_exec = load_runtime_config(merged_env={})
    member = MagicMock()
    member.dialect = "duckdb"
    member._runtime_config = RuntimeConfig(
        engine="duckdb",
        artifacts_dir=arts,
        engine_context=MagicMock(),
        llm_execution=llm_exec,
    )
    member._llm_config = LLMConfig(provider="openai")
    member._schema_graph = _graph("payment_a", "a")
    member._artifacts_dir = arts
    member._dialect = MagicMock()
    member._execution_engine = None
    member._native_connection = None
    member._context_name = "master"
    member._schema_role = "owner"
    member._engine_identity = None
    bundle = AetherFederationInitResult(
        runtime_config=RuntimeConfig(
            engine="duckdb",
            artifacts_dir=arts,
            engine_context=MagicMock(),
            llm_execution=llm_exec,
        ),
        llm_config=LLMConfig(provider="openai"),
        schema_graph=composite,
        dialect=MagicMock(),
        artifacts_dir=arts,
        store=store,
        templates={},
        rejected={},
        schema_terms=set(),
        schema_stats={"table_count": 1},
        federation_manifest=manifest,
        federation_mappings=mappings,
        federation_member_graphs={"a": _graph("payment_a", "a"), "b": _graph("payment_b", "b")},
        federation_storage_dir=arts,
        federation_source_runtimes={
            "a": MagicMock(source_id="a", dialect=MagicMock(), sqlalchemy_engine=None, native_connection=None),
            "b": MagicMock(source_id="b", dialect=MagicMock(), sqlalchemy_engine=None, native_connection=None),
        },
        members={"conn_a": member, "conn_b": member},
    )
    with patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=bundle):
        fed = AetherFederation(
            "fed_collapsed_space",
            members={"conn_a": member, "conn_b": member},
            declaration_file="/tmp/aether_fed_collapsed_declaration.json",
        )
    with pytest.raises(ConfigError, match="SpaceContext names collapsed member table 'payment_a'"):
        fed.aetherspace("payments", SpaceContext(deny_objects=frozenset({"payment_a"})))
