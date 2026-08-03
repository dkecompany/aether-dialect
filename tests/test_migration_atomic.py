"""Migration map atomic write and locked apply."""

from __future__ import annotations

import inspect
import json
import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._constants import MIGRATION_MAP_ACTION_REMAP
from aetherdialect._contracts_base import ColumnRole, MigrationTier, SchemaMigrationMap, SchemaMigrationMapEntry
from aetherdialect._contracts_core import ConcreteIntent, NormalizedExpr, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    SchemaDiff,
    SchemaGraph,
    SQLShape,
    TableDiff,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._core_utils import write_artifact_manifest
from aetherdialect._templates import (
    apply_schema_migration_map,
    empty_template_store,
    export_schema_migration_map_skeleton,
    save_template_store,
    template_store_dir_for_space,
    templates_to_store,
)


def _col(name: str, dt: str = "integer", *, pk: bool = False) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type=dt,
        value_type=dt,
        is_primary_key=pk,
        role=ColumnRole.IDENTIFIER.value if pk else ColumnRole.NUMERIC_MEASURE.value,
    )


def _table(name: str, cols: dict[str, ColumnMetadata]) -> TableMetadata:
    pk = next((c.name for c in cols.values() if c.is_primary_key), "")
    return TableMetadata(name=name, columns=cols, foreign_keys=[], primary_key=pk)


def _schema(tables: dict[str, TableMetadata], *, effective: str = "eff_new") -> SchemaGraph:
    return SchemaGraph(
        tables=tables,
        join_paths_multi={},
        structural_hash="s_new",
        profiling_hash="p_new",
        scope_hash="sc_new",
        effective_structural_hash=effective,
    )


def _make_template(tid: str, table: str, column: str) -> Template:
    intent = ConcreteIntent(
        intent_id=f"intent_{tid}",
        tables=[table],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{table}.{column}"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        column_map={column: table},
    )
    return Template(
        id=tid,
        effective_structural_hash="eff_old",
        intent_signature=intent,
        intent_key=f"key_{tid}",
        tables_used=[table],
        sql_param=f"SELECT {column} FROM {table}",
        sql_fp=f"fp_{tid}",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig=f"sig_{tid}",
        value_history=ValueHistory(param_values=[{}], questions=["q"], natural_language=["q"]),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
    )


def _seed_store(artifacts_dir: str, schema: SchemaGraph, templates: dict[str, Template]) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    store_dir = template_store_dir_for_space(artifacts_dir, "master")
    os.makedirs(store_dir, exist_ok=True)
    prev = EngineConfig.TEMPLATE_STORE_DIR
    EngineConfig.TEMPLATE_STORE_DIR = store_dir
    try:
        store = empty_template_store("eff_old")
        templates_to_store(store, templates)
        save_template_store(store)
    finally:
        EngineConfig.TEMPLATE_STORE_DIR = prev
    write_artifact_manifest(
        artifacts_dir,
        structural_hash="s_old",
        profiling_hash="p_old",
        scope_hash=schema.scope_hash,
        effective_structural_hash="eff_old",
        last_migration_tier=MigrationTier.SOFT_REFRESH.value,
        last_action="seed",
    )


@pytest.mark.fast
def test_skeleton_export_uses_tmp_and_replace_not_plain_open(tmp_path) -> None:
    source = inspect.getsource(export_schema_migration_map_skeleton)
    assert "os.replace" in source
    assert '.open(path, "w")' not in source
    assert 'open(path, "w")' not in source

    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def _capture_replace(src: str, dst: str) -> None:
        replaced.append((src, dst))
        real_replace(src, dst)

    diff = SchemaDiff(per_table={"orders": TableDiff(dropped_columns=("amount",))})
    with patch("aetherdialect._templates.os.replace", side_effect=_capture_replace):
        path = export_schema_migration_map_skeleton(
            tmp_path,
            tier=MigrationTier.DESTRUCTIVE,
            schema_diff=diff,
            rename_plan=None,
        )
    assert path.is_file()
    assert replaced
    tmp_src, final_dst = replaced[-1]
    assert str(tmp_src).endswith(".tmp")
    assert str(final_dst) == str(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["action"] == MIGRATION_MAP_ACTION_REMAP or payload["action"] == "destructive"


@pytest.mark.fast
def test_apply_schema_migration_map_holds_lock_for_remap_surgery_stamp(tmp_path) -> None:
    schema = _schema({"orders": _table("orders", {"order_id": _col("order_id", pk=True), "amount": _col("amount")})})
    t = _make_template("T0001", "orders", "amount")
    _seed_store(str(tmp_path), schema, {"T0001": t})
    map_obj = SchemaMigrationMap(
        version=1,
        action=MIGRATION_MAP_ACTION_REMAP,
        table_renames=(SchemaMigrationMapEntry(entry_type="table", from_name="orders", to_name="sales_orders"),),
        column_renames=(),
        dropped_tables=(),
        dropped_columns=(SchemaMigrationMapEntry(entry_type="dropped_column", table="orders", from_name="amount"),),
        added_tables=(),
        added_columns=(),
    )

    lock_depth = 0
    phases: list[str] = []

    @contextmanager
    def _tracking_lock(artifacts_dir: str, timeout: float = 30.0):
        nonlocal lock_depth
        lock_depth += 1
        try:
            yield
        finally:
            lock_depth -= 1

    def _remap_track(*args, **kwargs):
        phases.append(f"remap_depth_{lock_depth}")
        return (0, 0)

    def _surgery_track(*args, **kwargs):
        phases.append(f"surgery_depth_{lock_depth}")
        return 1

    def _stamp_track(*args, **kwargs):
        phases.append(f"stamp_depth_{lock_depth}")

    with (
        patch("aetherdialect._templates.artifact_lock", side_effect=_tracking_lock),
        patch("aetherdialect._templates._apply_schema_rename_migration_to_store", side_effect=_remap_track),
        patch("aetherdialect._templates.surgical_invalidate_templates_by_diff", side_effect=_surgery_track),
        patch("aetherdialect._templates._stamp_manifest", side_effect=_stamp_track),
        patch("aetherdialect._templates.apply_structural_migration_from_map"),
        patch("aetherdialect._templates.migrate_sidecar_for_diff"),
    ):
        apply_schema_migration_map(map_obj, str(tmp_path), schema, tmp_path / "schema.json.gz")

    assert phases == ["remap_depth_1", "surgery_depth_1", "stamp_depth_1"]
