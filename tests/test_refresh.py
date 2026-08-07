"""AetherEngine.refresh and AetherFederation.refresh lifecycle coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import AetherEngine, AetherFederation
from aetherdialect._contracts_base import (
    EngineContext,
    LLMConfig,
    MigrationPendingError,
    MigrationTier,
    RefreshReport,
    RuntimeConfig,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import load_runtime_config, write_artifact_manifest
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_graph import diff_schemas, recompute_join_paths_multi
from aetherdialect._schema_overrides import save_schema_to_cache
from tests.federation_helpers import write_federation_declaration_file


def _col(name: str, *, pk: bool = False) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", is_primary_key=pk)


def _table(name: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={"id": _col("id", pk=True)},
        primary_key=["id"],
        foreign_keys=[],
    )


def _engine_bundle(*, schema: SchemaGraph) -> MagicMock:
    bundle = MagicMock()
    bundle.dialect = MagicMock()
    bundle.schema_graph = schema
    bundle.artifacts_dir = "unused"
    bundle.store = MagicMock()
    bundle.templates = {}
    bundle.rejected = {}
    bundle.schema_terms = set()
    bundle.schema_stats = {}
    bundle.schema_role = "owner"
    bundle.consumer_visible_objects = None
    bundle.context_name = "master"
    bundle.data_quality_report = None
    bundle.engine_identity = None
    llm_exec = load_runtime_config(merged_env={})
    bundle.runtime_config = RuntimeConfig(
        engine="duckdb",
        artifacts_dir="unused",
        engine_context=EngineContext(),
        llm_execution=llm_exec,
    )
    bundle.llm_config = LLMConfig(provider="openai")
    return bundle


def _mock_engine(schema: SchemaGraph, artifacts_dir: str) -> MagicMock:
    engine = MagicMock(spec=AetherEngine)
    engine._closed = False
    engine._schema_role = "owner"
    engine._sandbox_closed = False
    engine._sandbox_mode = False
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
    engine._limits = MagicMock()
    engine._limits.template_store_max_count = None
    engine._limits.template_store_max_disk_bytes = None
    engine._context_name = "master"
    engine._pipeline_writer_lock = MagicMock()
    engine._pipeline_writer_lock.locked.return_value = False
    engine._pipeline_writer_lock.__enter__ = MagicMock(return_value=None)
    engine._pipeline_writer_lock.__exit__ = MagicMock(return_value=None)
    return engine


@pytest.mark.fast
def test_refresh_detects_drift_like_construction(tmp_path: Path) -> None:
    old_tables = {"items": _table("items")}
    old = SchemaGraph(
        tables=old_tables,
        join_paths_multi=recompute_join_paths_multi(old_tables),
        schema_graph_id="sg_refresh_drift",
        structural_hash="old_struct",
        profiling_hash="profile",
        scope_hash="scope",
        effective_structural_hash="old_eff",
    )
    new_tables = {"items": _table("items"), "extras": _table("extras")}
    new = SchemaGraph(
        tables=new_tables,
        join_paths_multi=recompute_join_paths_multi(new_tables),
        schema_graph_id="sg_refresh_drift",
        structural_hash="new_struct",
        profiling_hash="profile",
        scope_hash="scope",
        effective_structural_hash="new_eff",
    )
    artifacts_dir = str(tmp_path)
    schema_path = tmp_path / "schema_graph.json.gz"
    save_schema_to_cache(old, str(schema_path))
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=old.structural_hash,
        profiling_hash=old.profiling_hash,
        scope_hash=old.scope_hash,
        effective_structural_hash=old.effective_structural_hash,
        schema_graph_id=old.schema_graph_id,
    )

    def _build_graph(*_args, **_kwargs):
        return new, diff_schemas(old, new)

    engine = _mock_engine(old, artifacts_dir)
    with (
        patch("aetherdialect._main_execution.build_schema_graph_with_diff", side_effect=_build_graph),
        patch("aetherdialect._templates.TemplateOps.load_template_store", return_value=MagicMock()),
        patch("aetherdialect._templates.TemplateOps.store_to_templates", return_value={}),
        patch("aetherdialect._templates.TemplateOps.reconcile_template_store") as reconcile_mock,
        patch("aetherdialect._templates.TemplateOps.collect_orphaned_migration_checkpoints", return_value=[]),
        patch("aetherdialect._templates.TemplateOps.collect_expired_template_orphans", return_value=(0, 0)),
        patch.object(MainExecutionOps, "_emit_artifact_growth_diagnostics", return_value=[]),
    ):
        reconcile_mock.return_value = MagicMock(dropped_template_ids=())
        report = MainExecutionOps.refresh_aether_engine(engine, reflect=True)

    assert isinstance(report, RefreshReport)
    assert report.migration_tier == MigrationTier.ADDITIVE
    assert report.schema_changed is True
    assert "extras" in report.objects_added


@pytest.mark.fast
def test_refresh_raises_migration_pending_like_construction(tmp_path: Path) -> None:
    old_tables = {"items": _table("items")}
    old = SchemaGraph(
        tables=old_tables,
        join_paths_multi=recompute_join_paths_multi(old_tables),
        schema_graph_id="sg_refresh_remap",
        structural_hash="old_struct",
        profiling_hash="profile",
        scope_hash="scope",
        effective_structural_hash="old_eff",
    )
    new_tables = {"products": _table("products")}
    new = SchemaGraph(
        tables=new_tables,
        join_paths_multi=recompute_join_paths_multi(new_tables),
        schema_graph_id="sg_refresh_remap_new",
        structural_hash="new_struct",
        profiling_hash="profile",
        scope_hash="scope",
        effective_structural_hash="new_eff",
    )
    artifacts_dir = str(tmp_path)
    save_schema_to_cache(old, str(tmp_path / "schema_graph.json.gz"))
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=old.structural_hash,
        profiling_hash=old.profiling_hash,
        scope_hash=old.scope_hash,
        effective_structural_hash=old.effective_structural_hash,
        schema_graph_id=old.schema_graph_id,
    )

    def _build_graph(*_args, **_kwargs):
        return new, diff_schemas(old, new)

    engine = _mock_engine(old, artifacts_dir)
    with (
        patch("aetherdialect._main_execution.build_schema_graph_with_diff", side_effect=_build_graph),
        patch("aetherdialect._templates.TemplateOps.collect_orphaned_migration_checkpoints", return_value=[]),
        pytest.raises(MigrationPendingError, match="Schema migration required"),
    ):
        MainExecutionOps.refresh_aether_engine(engine, reflect=True)


@pytest.mark.fast
def test_refresh_reports_reclaimed_bytes(tmp_path: Path) -> None:
    old_tables = {"t": _table("t")}
    schema = SchemaGraph(
        tables=old_tables,
        join_paths_multi=recompute_join_paths_multi(old_tables),
        schema_graph_id="sg_refresh_reclaim",
        structural_hash="s",
        profiling_hash="p",
        scope_hash="sc",
        effective_structural_hash="eff",
    )
    artifacts_dir = str(tmp_path)
    save_schema_to_cache(schema, str(tmp_path / "schema_graph.json.gz"))
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=schema.structural_hash,
        profiling_hash=schema.profiling_hash,
        scope_hash=schema.scope_hash,
        effective_structural_hash=schema.effective_structural_hash,
        schema_graph_id=schema.schema_graph_id,
    )
    engine = _mock_engine(schema, artifacts_dir)
    with (
        patch("aetherdialect._main_execution.assign_schema_graph_hashes"),
        patch("aetherdialect._main_execution.build_schema_graph_with_diff", return_value=(schema, None)),
        patch("aetherdialect._templates.TemplateOps.load_template_store", return_value=MagicMock()),
        patch("aetherdialect._templates.TemplateOps.store_to_templates", return_value={}),
        patch("aetherdialect._templates.TemplateOps.reconcile_template_store") as reconcile_mock,
        patch("aetherdialect._templates.TemplateOps.collect_orphaned_migration_checkpoints", return_value=[]),
        patch("aetherdialect._templates.TemplateOps.collect_expired_template_orphans", return_value=(2, 4096)),
        patch.object(MainExecutionOps, "_emit_artifact_growth_diagnostics", return_value=[]),
    ):
        reconcile_mock.return_value = MagicMock(dropped_template_ids=())
        report = MainExecutionOps.refresh_aether_engine(engine, reflect=True)

    assert report.orphans_removed == 2
    assert report.bytes_reclaimed == 4096


@pytest.mark.fast
def test_refresh_without_reflect_does_artifact_work_only(tmp_path: Path) -> None:
    old_tables = {"t": _table("t")}
    schema = SchemaGraph(
        tables=old_tables,
        join_paths_multi=recompute_join_paths_multi(old_tables),
        schema_graph_id="sg_refresh_no_reflect",
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
    engine = _mock_engine(schema, artifacts_dir)
    build_calls: list[bool] = []
    with (
        patch("aetherdialect._main_execution.assign_schema_graph_hashes"),
        patch(
            "aetherdialect._main_execution.build_schema_graph_with_diff",
            side_effect=lambda *_a, **_k: (build_calls.append(True), (schema, None))[1],
        ),
        patch("aetherdialect._schema_overrides.finalize_with_overrides"),
        patch("aetherdialect._templates.TemplateOps.load_template_store", return_value=MagicMock()),
        patch("aetherdialect._templates.TemplateOps.store_to_templates", return_value={}),
        patch("aetherdialect._templates.TemplateOps.reconcile_template_store") as reconcile_mock,
        patch("aetherdialect._templates.TemplateOps.collect_orphaned_migration_checkpoints", return_value=[]),
        patch("aetherdialect._templates.TemplateOps.collect_expired_template_orphans", return_value=(0, 0)),
        patch.object(MainExecutionOps, "_emit_artifact_growth_diagnostics", return_value=[]) as growth_mock,
    ):
        reconcile_mock.return_value = MagicMock(dropped_template_ids=())
        report = MainExecutionOps.refresh_aether_engine(engine, reflect=False)

    assert build_calls == []
    growth_mock.assert_called_once()
    assert report.migration_tier == MigrationTier.NO_CHANGE


@pytest.mark.fast
def test_federation_refresh_covers_members_and_composite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    member_report = RefreshReport(
        migration_tier=MigrationTier.NO_CHANGE,
        schema_changed=False,
        objects_added=(),
        objects_removed=(),
        templates_invalidated=0,
        orphans_removed=0,
        bytes_reclaimed=0,
        diagnostics=(),
    )
    members = {
        "alpha": _mock_engine(
            SchemaGraph(
                tables={"a": _table("a")},
                join_paths_multi={},
                schema_graph_id="sg_alpha",
                effective_structural_hash="eff_alpha",
            ),
            str(tmp_path / "alpha"),
        ),
        "beta": _mock_engine(
            SchemaGraph(
                tables={"b": _table("b")},
                join_paths_multi={},
                schema_graph_id="sg_beta",
                effective_structural_hash="eff_beta",
            ),
            str(tmp_path / "beta"),
        ),
    }
    members["alpha"].refresh = MagicMock(return_value=member_report)
    members["beta"].refresh = MagicMock(return_value=member_report)
    declaration_path = write_federation_declaration_file(
        tmp_path,
        {
            "federation_id": "fed_refresh",
            "cross_source_joins": [
                {"left": "a.id", "right": "b.id", "kind": "inner", "logical_key": "id"},
            ],
        },
    )
    fed_bundle = MagicMock()
    fed_bundle.dialect = MagicMock()
    fed_bundle.schema_graph = MagicMock()
    fed_bundle.artifacts_dir = str(tmp_path / "fed")
    fed_bundle.store = MagicMock()
    fed_bundle.templates = {}
    fed_bundle.rejected = {}
    fed_bundle.schema_terms = set()
    fed_bundle.schema_stats = {}
    fed_bundle.schema_role = "owner"
    fed_bundle.consumer_visible_objects = None
    fed_bundle.context_name = "master"
    fed_bundle.data_quality_report = None
    fed_bundle.federation_manifest = MagicMock()
    fed_bundle.federation_mappings = MagicMock()
    fed_bundle.federation_member_graphs = {
        "alpha": members["alpha"]._schema_graph,
        "beta": members["beta"]._schema_graph,
    }
    fed_bundle.federation_storage_dir = str(tmp_path / "fed")
    fed_bundle.federation_source_runtimes = {}
    fed_bundle.federation_mapping_suggestions = ()
    fed_bundle.federation_dialects_by_source = {}
    fed_bundle.members = members
    fed_bundle.engine_identity = None
    fed_bundle.runtime_config = RuntimeConfig(
        engine="duckdb",
        artifacts_dir=str(tmp_path / "fed"),
        engine_context=EngineContext(),
        llm_execution=load_runtime_config(merged_env={}),
    )
    fed_bundle.llm_config = LLMConfig(provider="openai")

    with (
        patch.object(AetherEngine, "_initialize_engine_bundle", return_value=_engine_bundle(schema=MagicMock())),
        patch("aetherdialect.aetherdialect.initialize_aether_federation", return_value=fed_bundle),
    ):
        fed = AetherFederation(
            "fed_refresh",
            members=members,
            declaration_file=str(declaration_path),
            artifacts_dir=str(tmp_path),
        )

    with (
        patch.object(AetherFederation, "_recompose"),
        patch.object(AetherFederation, "_replay_composite_overrides"),
        patch("aetherdialect.aetherdialect.prune_federation_plan_templates_on_drift") as prune_mock,
    ):
        report = fed.refresh()

    members["alpha"].refresh.assert_called_once_with(reflect=True)
    members["beta"].refresh.assert_called_once_with(reflect=True)
    prune_mock.assert_called_once()
    assert isinstance(report, RefreshReport)
