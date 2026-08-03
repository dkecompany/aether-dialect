"""Consumer drift detection for single-engine init and reader reload."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import write_artifact_manifest
from aetherdialect._main_execution import _reload_reader_learning_if_manifest_drift
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates import empty_template_store_for_space, save_template_store, store_to_templates


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


@pytest.mark.fast
def test_consumer_init_refuses_remap_tier() -> None:
    import inspect

    from aetherdialect import _main_execution

    source = inspect.getsource(_main_execution.initialize_aether_engine)
    assert "Schema has drifted since artifacts were published" in source
    assert "tier_preview = MigrationTier.NO_CHANGE" not in source


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
    store = empty_template_store_for_space(live_graph.schema_graph_id, artifacts_dir=artifacts_dir)
    save_template_store(store)

    owner = SimpleNamespace(
        _artifacts_dir=artifacts_dir,
        _schema_graph=live_graph,
        _store=None,
        _templates=None,
        _dialect=None,
    )
    with patch("aetherdialect._main_execution.finalize_with_overrides"):
        _reload_reader_learning_if_manifest_drift(owner)

    assert owner._store is not None
    assert owner._templates == store_to_templates(owner._store)


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
    _reload_reader_learning_if_manifest_drift(owner)
    assert owner._store is sentinel_store
