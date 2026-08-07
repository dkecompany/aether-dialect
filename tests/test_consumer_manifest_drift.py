"""Consumer drift detection for single-engine init and reader reload."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import ConfigError, EngineContext, MigrationTier
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import read_artifact_manifest, write_artifact_manifest
from aetherdialect._main_execution import (
    MainExecutionOps,
)
from aetherdialect._schema_graph import (
    classify_migration_tier,
    diff_schemas,
    recompute_join_paths_multi,
)
from aetherdialect._schema_overrides import save_schema_to_cache
from aetherdialect._templates import TemplateOps


def _graph(*, structural_hash: str, profiling_hash: str = "profile_1") -> SchemaGraph:
    table = TableMetadata(
        name="t",
        columns={"id": ColumnMetadata(name="id", data_type="integer")},
        primary_key=["id"],
        foreign_keys=[],
    )
    tables = {"t": table}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="sg_same_id",
        effective_structural_hash=f"eff_{structural_hash}",
        structural_hash=structural_hash,
        profiling_hash=profiling_hash,
        scope_hash="scope_1",
    )


def _drift_owner_live_pair() -> tuple[SchemaGraph, SchemaGraph]:
    def _col(name: str, *, pk: bool = False) -> ColumnMetadata:
        return ColumnMetadata(name=name, data_type="integer", is_primary_key=pk)

    def _table(name: str) -> TableMetadata:
        return TableMetadata(
            name=name,
            columns={"id": _col("id", pk=True), "name": _col("name")},
            primary_key=["id"],
            foreign_keys=[],
        )

    owner_tables = {"items": _table("items")}
    owner = SchemaGraph(
        tables=owner_tables,
        join_paths_multi=recompute_join_paths_multi(owner_tables),
        schema_graph_id="sg_consumer_drift_remap",
        structural_hash="owner_struct",
        profiling_hash="profile_1",
        scope_hash="scope_1",
        effective_structural_hash="owner_eff",
    )
    live_tables = {"products": _table("products")}
    live = SchemaGraph(
        tables=live_tables,
        join_paths_multi=recompute_join_paths_multi(live_tables),
        schema_graph_id="sg_consumer_drift_remap",
        structural_hash="live_struct",
        profiling_hash="profile_1",
        scope_hash="scope_1",
        effective_structural_hash="live_eff",
    )
    return owner, live


@pytest.mark.fast
def test_consumer_init_refuses_remap_tier(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    owner, live = _drift_owner_live_pair()
    schema_diff = diff_schemas(owner, live)
    assert schema_diff.table_renames == (("items", "products"),)

    artifacts_dir = str(tmp_path)
    save_schema_to_cache(owner, str(tmp_path / "schema_graph.json.gz"))
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=owner.structural_hash,
        profiling_hash=owner.profiling_hash,
        scope_hash=owner.scope_hash,
        effective_structural_hash=owner.effective_structural_hash,
        schema_graph_id=owner.schema_graph_id,
    )
    stored = read_artifact_manifest(artifacts_dir)
    assert (
        classify_migration_tier(
            stored,
            live,
            previous_schema=owner,
            schema_diff=schema_diff,
        )
        == MigrationTier.REMAP
    )

    MainExecutionOps.write_schema_context_cache(artifacts_dir, EngineContext())

    monkeypatch.setenv("AETHERDIALECT_ENGINE", "duckdb")
    monkeypatch.setenv("DUCKDB_DATABASE", ":memory:")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with (
        patch("aetherdialect._main_execution.MainExecutionOps.compute_engine_storage_dir", return_value=artifacts_dir),
        patch("aetherdialect._main_execution.DialectRegistry.get", return_value=MagicMock()),
        patch(
            "aetherdialect._main_execution.build_schema_graph_with_diff",
            return_value=(live, schema_diff),
        ),
    ):
        with pytest.raises(ConfigError, match="Schema has drifted since artifacts were published"):
            MainExecutionOps.initialize_aether_engine(
                artifacts_dir=artifacts_dir,
                schema_role="consumer",
                execution_engine=MagicMock(),
                log_sink=lambda _msg: None,
            )


@pytest.mark.fast
def test_reload_reader_learning_detects_fingerprint_drift(tmp_path) -> None:
    stored_graph = _graph(structural_hash="stored_struct")
    live_graph = _graph(structural_hash="live_struct")
    artifacts_dir = str(tmp_path)
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=stored_graph.structural_hash,
        profiling_hash=stored_graph.profiling_hash,
        scope_hash=stored_graph.scope_hash,
        effective_structural_hash=stored_graph.effective_structural_hash,
        schema_graph_id=stored_graph.schema_graph_id,
    )
    store = TemplateOps.empty_template_store_for_space(live_graph.schema_graph_id, artifacts_dir=artifacts_dir)
    TemplateOps.save_template_store(store)

    owner = SimpleNamespace(
        _artifacts_dir=artifacts_dir,
        _schema_graph=live_graph,
        _store=None,
        _templates=None,
        _dialect=None,
    )
    with patch("aetherdialect._main_execution.finalize_with_overrides"):
        MainExecutionOps._reload_reader_learning_if_manifest_drift(owner)

    assert owner._store is not None
    assert owner._templates == TemplateOps.store_to_templates(owner._store)


@pytest.mark.fast
def test_reload_reader_learning_no_op_when_fingerprints_match(tmp_path) -> None:
    graph = _graph(structural_hash="same_struct")
    artifacts_dir = str(tmp_path)
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=graph.structural_hash,
        profiling_hash=graph.profiling_hash,
        scope_hash=graph.scope_hash,
        effective_structural_hash=graph.effective_structural_hash,
        schema_graph_id=graph.schema_graph_id,
    )
    sentinel_store = {"sentinel": object()}
    owner = SimpleNamespace(
        _artifacts_dir=artifacts_dir,
        _schema_graph=graph,
        _store=sentinel_store,
        _templates={"keep": object()},
        _dialect=None,
    )
    MainExecutionOps._reload_reader_learning_if_manifest_drift(owner)
    assert owner._store is sentinel_store
