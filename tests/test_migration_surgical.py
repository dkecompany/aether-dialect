"""Surgical invalidation must cover FK/PK/redeclared column changes."""

from __future__ import annotations

import os

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import MigrationTier, NormalizedExpr
from aetherdialect._contracts_core import ConcreteIntent, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    ColumnRole,
    FKEdge,
    SchemaDiff,
    SchemaGraph,
    SQLShape,
    TableDiff,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._schema_graph import diff_schemas
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


def _table(name: str, cols: dict[str, ColumnMetadata], **kwargs) -> TableMetadata:
    pk = next((c.name for c in cols.values() if c.is_primary_key), "")
    return TableMetadata(name=name, columns=cols, foreign_keys=[], primary_key=pk, **kwargs)


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
def test_redeclared_column_is_surgical_target() -> None:
    diff = SchemaDiff(per_table={"orders": TableDiff(redeclared_columns=(("amount", "integer", "bigint"),))})
    _tables, cols = TemplateOps._surgical_invalidation_targets(diff)
    assert ("orders", "amount") in cols


@pytest.mark.fast
def test_redeclared_column_soft_refresh_invalidates_template(tmp_path) -> None:
    schema = _schema({"orders": _table("orders", {"order_id": _col("order_id", pk=True), "amount": _col("amount")})})
    t = _make_template("T0001", "orders", "amount")
    _seed_store(str(tmp_path), schema, {"T0001": t})
    diff = SchemaDiff(per_table={"orders": TableDiff(redeclared_columns=(("amount", "integer", "bigint"),))})

    report = TemplateOps.apply_migration_policy(str(tmp_path), schema, schema_diff=diff)

    assert report.tier == MigrationTier.SOFT_REFRESH
    assert report.surgically_invalidated == 1
    store = _reload_store(str(tmp_path))
    assert store is not None
    assert "T0001" not in store.partition_map


@pytest.mark.fast
def test_pk_changed_soft_refresh_invalidates_template(tmp_path) -> None:
    old = _schema({"orders": _table("orders", {"order_id": _col("order_id", pk=True), "amount": _col("amount")})})
    new = _schema(
        {
            "orders": _table(
                "orders",
                {
                    "order_id": _col("order_id", pk=True),
                    "amount": _col("amount", pk=True),
                },
            ),
        },
        effective="eff_new",
    )
    t = _make_template("T0001", "orders", "amount")
    _seed_store(str(tmp_path), new, {"T0001": t})
    diff = diff_schemas(old, new)
    assert diff.per_table["orders"].pk_changed is True

    report = TemplateOps.apply_migration_policy(str(tmp_path), new, schema_diff=diff)

    assert report.tier == MigrationTier.SOFT_REFRESH
    assert report.surgically_invalidated == 1
    store = _reload_store(str(tmp_path))
    assert store is not None
    assert "T0001" not in store.partition_map


@pytest.mark.fast
def test_fk_changed_soft_refresh_invalidates_template(tmp_path) -> None:
    fk = FKEdge(src_table="orders", src_cols=["customer_id"], dst_table="customers", dst_cols=["customer_id"])
    old = _schema(
        {
            "orders": _table("orders", {"order_id": _col("order_id", pk=True), "customer_id": _col("customer_id")}),
            "customers": _table("customers", {"customer_id": _col("customer_id", pk=True)}),
        },
    )
    old.tables["orders"].foreign_keys = [fk]
    new = _schema(
        {
            "orders": _table("orders", {"order_id": _col("order_id", pk=True), "customer_id": _col("customer_id")}),
            "customers": _table("customers", {"customer_id": _col("customer_id", pk=True)}),
        },
        effective="eff_new",
    )
    diff = diff_schemas(old, new)
    assert diff.per_table["orders"].fk_changed is True
    t = _make_template("T0001", "orders", "order_id")
    _seed_store(str(tmp_path), new, {"T0001": t})

    report = TemplateOps.apply_migration_policy(str(tmp_path), new, schema_diff=diff)

    assert report.tier == MigrationTier.SOFT_REFRESH
    assert report.surgically_invalidated == 1
    store = _reload_store(str(tmp_path))
    assert store is not None
    assert "T0001" not in store.partition_map
