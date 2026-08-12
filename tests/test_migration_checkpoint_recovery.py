"""Incomplete migration checkpoints restore on engine construction."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect import _main_execution
from aetherdialect._config import EngineConfig
from aetherdialect._constants import TEMPLATE_STORE_SEGMENT
from aetherdialect._contracts_base import EngineContext, MigrationTier, NormalizedExpr
from aetherdialect._contracts_core import ConcreteIntent, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    ColumnRole,
    SchemaGraph,
    SQLShape,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import read_artifact_manifest, write_artifact_manifest


def _col(name: str, *, pk: bool = False) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="integer",
        value_type="integer",
        is_primary_key=pk,
        role=ColumnRole.IDENTIFIER.value if pk else ColumnRole.NUMERIC_MEASURE.value,
    )


def _schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={"order_id": _col("order_id", pk=True), "amount": _col("amount")},
                foreign_keys=[],
                primary_key="order_id",
            )
        },
        join_paths_multi={},
        structural_hash="s_pre",
        profiling_hash="p_pre",
        scope_hash="sc_pre",
        effective_structural_hash="eff_pre",
    )


def _make_template(tid: str) -> Template:
    intent = ConcreteIntent(
        intent_id=f"intent_{tid}",
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        column_map={"amount": "orders"},
    )
    return Template(
        id=tid,
        effective_structural_hash="eff_pre",
        intent_signature=intent,
        intent_key=f"key_{tid}",
        tables_used=["orders"],
        sql_param="SELECT amount FROM orders",
        sql_fp=f"fp_{tid}",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig=f"sig_{tid}",
        value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["q"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
    )


def _seed_store(artifacts_dir: str, schema: SchemaGraph, templates: dict[str, Template]) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir, "master")
    os.makedirs(store_dir, exist_ok=True)
    prev = EngineConfig.TEMPLATE_STORE_DIR
    EngineConfig.TEMPLATE_STORE_DIR = store_dir
    try:
        store = TemplateOps.empty_template_store("eff_pre")
        TemplateOps.templates_to_store(store, templates)
        TemplateOps.save_template_store(store)
    finally:
        EngineConfig.TEMPLATE_STORE_DIR = prev
    write_artifact_manifest(
        artifacts_dir,
        structural_hash="s_pre",
        profiling_hash="p_pre",
        scope_hash=schema.scope_hash,
        effective_structural_hash="eff_pre",
        schema_graph_id="sg_pre_migration",
        last_migration_tier=MigrationTier.SOFT_REFRESH.value,
        last_action="seed",
    )


@pytest.mark.fast
def test_incomplete_checkpoint_restored_on_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts_dir = str(tmp_path / "artifacts")
    schema = _schema()
    _seed_store(artifacts_dir, schema, {"T0001": _make_template("T0001")})
    schema_json_path = Path(artifacts_dir) / "schema_graph.json.gz"
    schema_json_path.write_bytes(b"pre-migration-schema")

    checkpoint = TemplateOps._migration_map_checkpoint_begin(artifacts_dir, schema_json_path=schema_json_path)
    assert checkpoint is not None

    write_artifact_manifest(
        artifacts_dir,
        structural_hash="s_live",
        profiling_hash="p_live",
        scope_hash="sc_live",
        effective_structural_hash="eff_live",
        schema_graph_id="sg_after_partial_migration",
        last_migration_tier=MigrationTier.REMAP.value,
        last_action="partial_migration",
    )
    shutil.rmtree(os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT))
    os.makedirs(os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT))
    schema_json_path.write_bytes(b"corrupted-schema")

    monkeypatch.setenv("AETHERDIALECT_ENGINE", "duckdb")
    monkeypatch.setenv("DUCKDB_DATABASE", ":memory:")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    init_error = RuntimeError("stop-after-checkpoint-restore")

    with (
        patch(
            "aetherdialect._main_init.MainInitOps.compute_engine_storage_dir",
            return_value=artifacts_dir,
        ),
        patch("aetherdialect._dialect.DialectRegistry.get_dialect", return_value=MagicMock()),
        patch(
            "aetherdialect._main_init.build_schema_graph_with_diff",
            side_effect=init_error,
        ),
    ):
        with pytest.raises(RuntimeError, match="stop-after-checkpoint-restore"):
            _main_execution.MainExecutionOps.initialize_aether_engine(
                EngineContext(),
                artifacts_dir=artifacts_dir,
                schema_role="owner",
                execution_engine=MagicMock(),
                log_sink=lambda _msg: None,
            )

    assert not os.path.isdir(checkpoint)
    store = TemplateOps._load_partitioned_view_unlocked(
        TemplateOps.template_store_dir_for_space(artifacts_dir, "master")
    )
    assert store is not None
    assert "T0001" in store.partition_map
    assert schema_json_path.read_bytes() == b"pre-migration-schema"
    manifest = read_artifact_manifest(artifacts_dir)
    assert manifest is not None
    assert manifest.schema_graph_id == "sg_pre_migration"
    assert manifest.structural_hash == "s_pre"
