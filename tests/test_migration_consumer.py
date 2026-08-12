"""Consumer migration path must not mutate owner artifacts when allow_destructive=False."""

from __future__ import annotations

import os

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import MigrationTier, NormalizedExpr
from aetherdialect._contracts_core import ConcreteIntent, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    ColumnRole,
    SchemaDiff,
    SchemaGraph,
    SQLShape,
    TableDiff,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils_artifacts import write_artifact_manifest


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
    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir, "master")
    os.makedirs(store_dir, exist_ok=True)
    prev = EngineConfig.TEMPLATE_STORE_DIR
    EngineConfig.TEMPLATE_STORE_DIR = store_dir
    try:
        store = TemplateOps.empty_template_store("eff_old")
        TemplateOps.templates_to_store(store, templates)
        TemplateOps.save_template_store(store)
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


def _reload_store(artifacts_dir: str):
    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir, "master")
    return TemplateOps._load_partitioned_view_unlocked(store_dir)


@pytest.mark.fast
def test_allow_destructive_false_blocks_surgical_drop(tmp_path) -> None:
    schema = _schema({"orders": _table("orders", {"order_id": _col("order_id", pk=True), "amount": _col("amount")})})
    t = _make_template("T0001", "orders", "amount")
    _seed_store(str(tmp_path), schema, {"T0001": t})
    diff = SchemaDiff(per_table={"orders": TableDiff(dropped_columns=("amount",))})

    report = TemplateOps.apply_migration_policy(str(tmp_path), schema, schema_diff=diff, allow_destructive=False)

    assert report.tier == MigrationTier.NO_CHANGE
    assert report.surgically_invalidated == 0
    store = _reload_store(str(tmp_path))
    assert store is not None
    assert "T0001" in store.partition_map


@pytest.mark.fast
def test_allow_destructive_false_blocks_table_drop(tmp_path) -> None:
    schema = _schema(
        {
            "orders": _table("orders", {"order_id": _col("order_id", pk=True)}),
            "customers": _table("customers", {"customer_id": _col("customer_id", pk=True)}),
        },
    )
    t_orders = _make_template("T0001", "orders", "order_id")
    t_customers = _make_template("T0002", "customers", "customer_id")
    _seed_store(str(tmp_path), schema, {"T0001": t_orders, "T0002": t_customers})
    diff = SchemaDiff(dropped_tables=("orders",))

    report = TemplateOps.apply_migration_policy(str(tmp_path), schema, schema_diff=diff, allow_destructive=False)

    assert report.tier == MigrationTier.NO_CHANGE
    assert report.surgically_invalidated == 0
    store = _reload_store(str(tmp_path))
    assert store is not None
    assert "T0001" in store.partition_map
    assert "T0002" in store.partition_map
