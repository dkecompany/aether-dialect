"""Artifact growth diagnostics emitted during refresh."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineLimits
from aetherdialect._contracts_base import EngineContext, RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import load_runtime_config, write_artifact_manifest
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._schema_overrides import save_schema_to_cache


def _table(name: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={"id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True)},
        primary_key=["id"],
        foreign_keys=[],
    )


def _mock_engine(schema: SchemaGraph, artifacts_dir: str, *, limits: EngineLimits) -> MagicMock:
    engine = MagicMock()
    engine._closed = False
    engine._schema_role = "owner"
    engine._schema_graph = schema
    engine._dialect = MagicMock()
    engine._artifacts_dir = Path(artifacts_dir)
    engine._runtime_config = RuntimeConfig(
        engine="duckdb",
        artifacts_dir=artifacts_dir,
        engine_context=EngineContext(),
        llm_execution=load_runtime_config(merged_env={}),
    )
    engine._trust_bundled_baseline = False
    engine._limits = limits
    engine._context_name = "master"
    return engine


@pytest.mark.fast
def test_growth_reported_on_refresh(tmp_path: Path) -> None:
    tables = {"t": _table("t")}
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="sg_growth",
        structural_hash="s",
        profiling_hash="p",
        scope_hash="sc",
        effective_structural_hash="eff",
    )
    artifacts_dir = str(tmp_path)
    save_schema_to_cache(schema, str(tmp_path / "schema_graph.json.gz"))
    write_artifact_manifest(
        artifacts_dir,
        structural_hash="s",
        profiling_hash="p",
        scope_hash="sc",
        effective_structural_hash="eff",
        schema_graph_id=schema.schema_graph_id,
    )
    engine = _mock_engine(schema, artifacts_dir, limits=EngineLimits())
    with (
        patch("aetherdialect._main_execution.assign_schema_graph_hashes"),
        patch("aetherdialect._main_execution.build_schema_graph_with_diff", return_value=(schema, None)),
        patch("aetherdialect._templates.TemplateOps.load_template_store", return_value=MagicMock(partition_map={})),
        patch("aetherdialect._templates.TemplateOps.store_to_templates", return_value={}),
        patch("aetherdialect._templates.TemplateOps.reconcile_template_store") as reconcile_mock,
        patch("aetherdialect._templates.TemplateOps.collect_orphaned_migration_checkpoints", return_value=[]),
        patch("aetherdialect._templates.TemplateOps.collect_expired_template_orphans", return_value=(0, 0)),
    ):
        reconcile_mock.return_value = MagicMock(dropped_template_ids=())
        report = MainExecutionOps.refresh_aether_engine(engine, reflect=True)

    growth = [d for d in report.diagnostics if d.code == "ARTIFACT_GROWTH"]
    assert len(growth) == 1
    detail = dict(growth[0].details)
    assert "artifact_bytes" in detail
    assert "template_count" in detail
    assert "feedback_shard_count" in detail
    assert "orphan_count" in detail


@pytest.mark.fast
def test_no_warning_when_limit_is_unset(tmp_path: Path) -> None:
    tables = {"t": _table("t")}
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="sg_growth_unset",
        structural_hash="s",
        profiling_hash="p",
        scope_hash="sc",
        effective_structural_hash="eff",
    )
    artifacts_dir = str(tmp_path)
    save_schema_to_cache(schema, str(tmp_path / "schema_graph.json.gz"))
    write_artifact_manifest(
        artifacts_dir,
        structural_hash="s",
        profiling_hash="p",
        scope_hash="sc",
        effective_structural_hash="eff",
        schema_graph_id=schema.schema_graph_id,
    )
    limits = EngineLimits(template_store_max_count=None, template_store_max_disk_bytes=None)
    engine = _mock_engine(schema, artifacts_dir, limits=limits)
    with (
        patch("aetherdialect._main_execution.assign_schema_graph_hashes"),
        patch("aetherdialect._main_execution.build_schema_graph_with_diff", return_value=(schema, None)),
        patch(
            "aetherdialect._templates.TemplateOps.load_template_store", return_value=MagicMock(partition_map={"T1": 0})
        ),
        patch("aetherdialect._templates.TemplateOps.store_to_templates", return_value={}),
        patch("aetherdialect._templates.TemplateOps.reconcile_template_store") as reconcile_mock,
        patch("aetherdialect._templates.TemplateOps.collect_orphaned_migration_checkpoints", return_value=[]),
        patch("aetherdialect._templates.TemplateOps.collect_expired_template_orphans", return_value=(0, 0)),
    ):
        reconcile_mock.return_value = MagicMock(dropped_template_ids=())
        report = MainExecutionOps.refresh_aether_engine(engine, reflect=True)

    assert not any(d.code == "ARTIFACT_LIMIT_NEAR" for d in report.diagnostics)
