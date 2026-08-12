"""Engine identity isolation across coexisting AetherEngine instances."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._constants import MIGRATION_MAP_ACTION_REMAP
from aetherdialect._contracts_base import (
    EngineContext,
    EngineIdentity,
    MigrationTier,
    SchemaAccessError,
    SchemaMigrationMap,
    SchemaMigrationMapEntry,
)
from aetherdialect._contracts_core import RuntimeConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import Dialect, DialectRegistry
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._main_init import MainInitOps
from aetherdialect._schema_graph import diff_schemas, recompute_join_paths_multi
from aetherdialect._schema_reflect import reflect_schema_graph_for_context, save_schema_to_cache
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import (
    active_engine_identity,
    pop_engine_identity,
    push_engine_identity,
)
from aetherdialect._utils_artifacts import load_runtime_config, write_artifact_manifest


@pytest.mark.fast
def test_active_engine_identity_requires_pushed_context(unbound_engine_identity: None) -> None:
    with pytest.raises(RuntimeError, match="no active engine identity"):
        active_engine_identity()


@pytest.mark.fast
def test_sqlglot_dialect_tracks_engine_type() -> None:
    assert DialectRegistry.sqlglot_dialect_for_engine("duckdb") == "duckdb"
    assert DialectRegistry.sqlglot_dialect_for_engine("postgresql") == "postgres"


@pytest.mark.fast
def test_active_sqlglot_dialect_follows_identity_context() -> None:
    duck = EngineIdentity("duckdb", EngineConfig.RUNTIME)
    pg = EngineIdentity("postgresql", EngineConfig.RUNTIME)
    token_duck = push_engine_identity(duck)
    assert Dialect.active_sqlglot_dialect() == "duckdb"
    token_pg = push_engine_identity(pg)
    assert Dialect.active_sqlglot_dialect() == "postgres"
    pop_engine_identity(token_pg)
    assert Dialect.active_sqlglot_dialect() == "duckdb"
    pop_engine_identity(token_duck)


def _col(name: str, *, pk: bool = False) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", is_primary_key=pk)


def _table(name: str) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns={"id": _col("id", pk=True)},
        primary_key=["id"],
        foreign_keys=[],
    )


def _mock_refresh_engine(schema: SchemaGraph, artifacts_dir: str) -> MagicMock:
    engine = MagicMock()
    engine._closed = False
    engine._schema_role = "owner"
    engine._sandbox_closed = False
    engine._sandbox_mode = False
    engine._schema_graph = schema
    engine._dialect = MagicMock()
    engine._dialect.name = "duckdb"
    engine._engine_identity = None
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
def test_reflect_schema_graph_for_context_requires_active_identity(unbound_engine_identity: None) -> None:
    dialect = MagicMock()
    ctx = EngineContext()
    with pytest.raises(RuntimeError, match="no active engine identity"):
        reflect_schema_graph_for_context(dialect, ctx)


@pytest.mark.fast
def test_reflect_schema_graph_for_context_does_not_mask_missing_identity(
    unbound_engine_identity: None,
) -> None:
    dialect = MagicMock()
    dialect.reflect_schema_graph.side_effect = RuntimeError(
        "no active engine identity; bind one with push_engine_identity before calling active_engine_identity"
    )
    ctx = EngineContext()
    with pytest.raises(RuntimeError, match="no active engine identity") as exc_info:
        reflect_schema_graph_for_context(dialect, ctx)
    assert not isinstance(exc_info.value, SchemaAccessError)


@pytest.mark.fast
def test_refresh_initial_rebuild_binds_engine_identity(tmp_path: Path, unbound_engine_identity: None) -> None:
    old_tables = {"items": _table("items")}
    old = SchemaGraph(
        tables=old_tables,
        join_paths_multi=recompute_join_paths_multi(old_tables),
        schema_graph_id="sg_refresh_identity",
        structural_hash="old_struct",
        profiling_hash="profile",
        scope_hash="scope",
        effective_structural_hash="old_eff",
    )
    new_tables = {"items": _table("items"), "extras": _table("extras")}
    new = SchemaGraph(
        tables=new_tables,
        join_paths_multi=recompute_join_paths_multi(new_tables),
        schema_graph_id="sg_refresh_identity",
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
    seen: dict[str, EngineIdentity] = {}

    def _build_graph(*_args, **_kwargs):
        seen["identity"] = active_engine_identity()
        return new, diff_schemas(old, new)

    engine = _mock_refresh_engine(old, artifacts_dir)
    with (
        patch("aetherdialect._main_init.build_schema_graph_with_diff", side_effect=_build_graph),
        patch("aetherdialect._templates_ops.TemplateOps.load_template_store", return_value=MagicMock()),
        patch("aetherdialect._templates_ops.TemplateOps.store_to_templates", return_value={}),
        patch("aetherdialect._templates_ops.TemplateOps.reconcile_template_store") as reconcile_mock,
        patch("aetherdialect._templates_ops.TemplateOps.collect_orphaned_migration_checkpoints", return_value=[]),
        patch("aetherdialect._templates_ops.TemplateOps.collect_expired_template_orphans", return_value=(0, 0)),
        patch.object(MainInitOps, "_emit_artifact_growth_diagnostics", return_value=[]),
    ):
        reconcile_mock.return_value = MagicMock(dropped_template_ids=())
        MainExecutionOps.refresh_aether_engine(engine, reflect=True)

    assert seen["identity"].engine_type == "duckdb"
    assert seen["identity"].runtime_config is engine._runtime_config


@pytest.mark.fast
def test_refresh_post_migration_map_rebuild_binds_engine_identity(
    tmp_path: Path,
    unbound_engine_identity: None,
) -> None:
    owner_tables = {"items": _table("items")}
    owner = SchemaGraph(
        tables=owner_tables,
        join_paths_multi=recompute_join_paths_multi(owner_tables),
        schema_graph_id="sg_refresh_map_identity",
        structural_hash="owner_struct",
        profiling_hash="profile_1",
        scope_hash="scope_1",
        effective_structural_hash="owner_eff",
    )
    live_tables = {"products": _table("products")}
    live = SchemaGraph(
        tables=live_tables,
        join_paths_multi=recompute_join_paths_multi(live_tables),
        schema_graph_id="sg_refresh_map_identity",
        structural_hash="live_struct",
        profiling_hash="profile_1",
        scope_hash="scope_1",
        effective_structural_hash="live_eff",
    )
    artifacts_dir = str(tmp_path)
    schema_path = tmp_path / "schema_graph.json.gz"
    save_schema_to_cache(owner, str(schema_path))
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=owner.structural_hash,
        profiling_hash=owner.profiling_hash,
        scope_hash=owner.scope_hash,
        effective_structural_hash=owner.effective_structural_hash,
        schema_graph_id=owner.schema_graph_id,
        last_migration_tier=MigrationTier.SOFT_REFRESH.value,
        last_action="seed",
    )
    map_obj = SchemaMigrationMap(
        version=1,
        action=MIGRATION_MAP_ACTION_REMAP,
        table_renames=(SchemaMigrationMapEntry(entry_type="table", from_name="items", to_name="products"),),
        column_renames=(),
        dropped_tables=(),
        dropped_columns=(),
        added_tables=(),
        added_columns=(),
    )
    (tmp_path / "schema_migration_map.json").write_text(
        json.dumps(
            {
                "version": map_obj.version,
                "action": map_obj.action,
                "table_renames": [{"entry_type": "table", "from_name": "items", "to_name": "products"}],
                "column_renames": [],
                "dropped_tables": [],
                "dropped_columns": [],
                "added_tables": [],
                "added_columns": [],
            }
        ),
        encoding="utf-8",
    )
    rebuild_calls: list[bool] = []
    seen_identities: list[EngineIdentity] = []

    def _build_graph(*_args, **kwargs):
        rebuild_calls.append(bool(kwargs.get("force_live_schema_reflect")))
        seen_identities.append(active_engine_identity())
        return live, diff_schemas(owner, live)

    engine = _mock_refresh_engine(owner, artifacts_dir)
    with (
        patch("aetherdialect._main_init.build_schema_graph_with_diff", side_effect=_build_graph),
        patch("aetherdialect._main_init.classify_migration_tier", return_value=MigrationTier.NO_CHANGE),
        patch("aetherdialect._templates_ops.TemplateOps.load_template_store", return_value=MagicMock()),
        patch("aetherdialect._templates_ops.TemplateOps.store_to_templates", return_value={}),
        patch("aetherdialect._templates_ops.TemplateOps.reconcile_template_store") as reconcile_mock,
        patch("aetherdialect._templates_ops.TemplateOps.collect_orphaned_migration_checkpoints", return_value=[]),
        patch("aetherdialect._templates_ops.TemplateOps.collect_expired_template_orphans", return_value=(0, 0)),
        patch(
            "aetherdialect._templates_ops.TemplateOps.apply_schema_migration_map",
            wraps=TemplateOps.apply_schema_migration_map,
        ),
        patch.object(MainInitOps, "_emit_artifact_growth_diagnostics", return_value=[]),
    ):
        reconcile_mock.return_value = MagicMock(dropped_template_ids=())
        MainExecutionOps.refresh_aether_engine(engine, reflect=True)

    assert len(rebuild_calls) >= 2
    assert rebuild_calls[0] is True
    assert rebuild_calls[-1] is True
    assert all(identity.engine_type == "duckdb" for identity in seen_identities)
    assert all(identity.runtime_config is engine._runtime_config for identity in seen_identities)
