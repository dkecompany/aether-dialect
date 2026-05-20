"""End-to-end tests for the diff-driven migration policy path."""

from __future__ import annotations

import os

import pytest

from aetherdialect._config import (
    TEMPLATE_QUESTION_TOKEN_INDEX_KEY,
    TEMPLATE_STORE_HEADER_FILENAME,
    TEMPLATE_STORE_SEGMENT,
    EngineConfig,
)
from aetherdialect._contracts_base import (
    ColumnMetadata,
    ColumnRole,
    MigrationTier,
    SchemaGraph,
    SQLShape,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._contracts_core import (
    ConcreteIntent,
    NormalizedExpr,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._core_utils import (
    read_gzip_json,
    write_artifact_manifest,
    write_gzip_json_atomic,
)
from aetherdialect._schema import SchemaDiff, TableDiff
from aetherdialect._templates import (
    TemplateStoreView,
    apply_migration_policy,
    empty_template_store,
    save_template_store,
    surgical_invalidate_templates_by_diff,
    templates_to_store,
    _load_partitioned_view_unlocked,
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


def _schema(
    tables: dict[str, TableMetadata],
    *,
    structural: str = "s_new",
    profiling: str = "p_new",
    scope: str = "sc_new",
    effective: str = "eff_new",
) -> SchemaGraph:
    return SchemaGraph(
        tables=tables,
        join_paths_multi={},
        structural_hash=structural,
        profiling_hash=profiling,
        scope_hash=scope,
        effective_structural_hash=effective,
    )


def _two_table_schema(
    *,
    structural: str = "s_new",
    profiling: str = "p_new",
    scope: str = "sc_new",
    effective: str = "eff_new",
) -> SchemaGraph:
    return _schema(
        {
            "orders": _table(
                "orders",
                {
                    "order_id": _col("order_id", "integer", pk=True),
                    "amount": _col("amount", "integer"),
                },
            ),
            "customers": _table(
                "customers",
                {
                    "customer_id": _col("customer_id", "integer", pk=True),
                    "name": _col("name", "varchar"),
                },
            ),
        },
        structural=structural,
        profiling=profiling,
        scope=scope,
        effective=effective,
    )


def _make_template(
    tid: str,
    table: str,
    column: str,
) -> Template:
    intent = ConcreteIntent(
        intent_id=f"intent_{tid}",
        tables=[table],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{table}.{column}"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
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


def _seed_store(
    artifacts_dir: str,
    schema: SchemaGraph,
    templates: dict[str, Template],
    *,
    stale_eff: str = "eff_old",
) -> str:
    """Write a partitioned template store + a *stale* manifest so apply_migration_policy proceeds."""

    os.makedirs(artifacts_dir, exist_ok=True)
    store_dir = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
    os.makedirs(store_dir, exist_ok=True)
    prev = EngineConfig.TEMPLATE_STORE_DIR
    EngineConfig.TEMPLATE_STORE_DIR = store_dir
    try:
        store = empty_template_store(stale_eff)
        templates_to_store(store, templates)
        save_template_store(store)
    finally:
        EngineConfig.TEMPLATE_STORE_DIR = prev
    write_artifact_manifest(
        artifacts_dir,
        structural_hash="s_old",
        profiling_hash="p_old",
        scope_hash=schema.scope_hash,
        effective_structural_hash=stale_eff,
        last_migration_tier=MigrationTier.SOFT_REFRESH.value,
        last_action="seed",
    )
    return store_dir


def _reload_store(artifacts_dir: str):
    """Load the on-disk partitioned store for assertions (ignores hash reconciliation)."""

    store_dir = os.path.join(artifacts_dir, TEMPLATE_STORE_SEGMENT)
    prev = EngineConfig.TEMPLATE_STORE_DIR
    EngineConfig.TEMPLATE_STORE_DIR = store_dir
    try:
        raw = _load_partitioned_view_unlocked(store_dir)
        if raw is None:
            return TemplateStoreView.empty(store_dir, "eff_placeholder")
        return raw
    finally:
        EngineConfig.TEMPLATE_STORE_DIR = prev


class TestSurgicalInvalidateDirect:
    """Direct invocations of ``surgical_invalidate_templates_by_diff``."""

    def test_dropped_column_removes_only_matching(self, tmp_path) -> None:
        schema = _two_table_schema()
        t_keep = _make_template("T0001", "orders", "order_id")
        t_drop = _make_template("T0002", "orders", "amount")
        _seed_store(str(tmp_path), schema, {"T0001": t_keep, "T0002": t_drop})
        diff = SchemaDiff(per_table={"orders": TableDiff(dropped_columns=("amount",))})

        deleted = surgical_invalidate_templates_by_diff(str(tmp_path), schema, diff)

        assert deleted == 1
        store = _reload_store(str(tmp_path))
        assert "T0001" in store.partition_map
        assert "T0002" not in store.partition_map

    def test_surgical_invalidate_rebuilds_question_token_index(self, tmp_path) -> None:
        """Stale token-index rows for deleted templates are dropped when the store is rewritten."""
        schema = _two_table_schema()
        t_keep = _make_template("T0001", "orders", "order_id")
        t_drop = _make_template("T0002", "orders", "amount")
        store_dir = _seed_store(str(tmp_path), schema, {"T0001": t_keep, "T0002": t_drop})
        hdr = os.path.join(store_dir, TEMPLATE_STORE_HEADER_FILENAME)
        poisoned = read_gzip_json(hdr)
        assert isinstance(poisoned, dict)
        poisoned[TEMPLATE_QUESTION_TOKEN_INDEX_KEY] = {"stale_fp": [["T0002", "0"]]}
        write_gzip_json_atomic(hdr, poisoned, sort_keys=True)

        diff = SchemaDiff(per_table={"orders": TableDiff(dropped_columns=("amount",))})
        deleted = surgical_invalidate_templates_by_diff(str(tmp_path), schema, diff)

        assert deleted == 1
        store = _reload_store(str(tmp_path))
        assert "stale_fp" not in store.get(TEMPLATE_QUESTION_TOKEN_INDEX_KEY, {})

    def test_dropped_table_removes_all_referencing(self, tmp_path) -> None:
        schema = _two_table_schema()
        t1 = _make_template("T0001", "orders", "order_id")
        t2 = _make_template("T0002", "customers", "customer_id")
        _seed_store(str(tmp_path), schema, {"T0001": t1, "T0002": t2})
        diff = SchemaDiff(dropped_tables=("orders",))

        deleted = surgical_invalidate_templates_by_diff(str(tmp_path), schema, diff)

        assert deleted == 1
        store = _reload_store(str(tmp_path))
        assert "T0001" not in store.partition_map
        assert "T0002" in store.partition_map

    def test_value_type_change_invalidates(self, tmp_path) -> None:
        schema = _two_table_schema()
        t = _make_template("T0001", "orders", "amount")
        _seed_store(str(tmp_path), schema, {"T0001": t})
        diff = SchemaDiff(
            per_table={
                "orders": TableDiff(
                    value_type_changed_columns=(("amount", "integer", "string"),),
                ),
            },
        )

        deleted = surgical_invalidate_templates_by_diff(str(tmp_path), schema, diff)

        assert deleted == 1
        store = _reload_store(str(tmp_path))
        assert "T0001" not in store.partition_map

    def test_empty_diff_is_noop(self, tmp_path) -> None:
        schema = _two_table_schema()
        t = _make_template("T0001", "orders", "amount")
        _seed_store(str(tmp_path), schema, {"T0001": t})

        deleted = surgical_invalidate_templates_by_diff(str(tmp_path), schema, SchemaDiff())

        assert deleted == 0
        store = _reload_store(str(tmp_path))
        assert "T0001" in store.partition_map

    def test_missing_store_returns_zero(self, tmp_path) -> None:
        schema = _two_table_schema()
        diff = SchemaDiff(dropped_tables=("orders",))

        deleted = surgical_invalidate_templates_by_diff(str(tmp_path), schema, diff)
        assert deleted == 0


class TestApplyMigrationPolicyDiffDriven:
    """``apply_migration_policy`` consumes ``schema_diff`` directly."""

    def test_dropped_column_soft_refresh_with_surgery(self, tmp_path) -> None:
        schema = _two_table_schema()
        t_keep = _make_template("T0001", "orders", "order_id")
        t_drop = _make_template("T0002", "orders", "amount")
        _seed_store(str(tmp_path), schema, {"T0001": t_keep, "T0002": t_drop})
        diff = SchemaDiff(per_table={"orders": TableDiff(dropped_columns=("amount",))})

        report = apply_migration_policy(str(tmp_path), schema, schema_diff=diff)

        assert report.tier == MigrationTier.SOFT_REFRESH
        assert report.surgically_invalidated == 1
        assert report.remapped_templates == 0
        assert report.renamed_tables == ()
        assert report.dropped_tables == ()

    def test_dropped_table_records_in_report(self, tmp_path) -> None:
        schema = _two_table_schema()
        t = _make_template("T0001", "orders", "order_id")
        _seed_store(str(tmp_path), schema, {"T0001": t})
        diff = SchemaDiff(dropped_tables=("orders",))

        report = apply_migration_policy(str(tmp_path), schema, schema_diff=diff)

        assert report.tier == MigrationTier.SOFT_REFRESH
        assert report.dropped_tables == ("orders",)
        assert report.surgically_invalidated == 1

    def test_value_type_change_recorded(self, tmp_path) -> None:
        schema = _two_table_schema()
        t = _make_template("T0001", "orders", "amount")
        _seed_store(str(tmp_path), schema, {"T0001": t})
        diff = SchemaDiff(
            per_table={
                "orders": TableDiff(
                    value_type_changed_columns=(("amount", "integer", "string"),),
                ),
            },
        )

        report = apply_migration_policy(str(tmp_path), schema, schema_diff=diff)

        assert report.tier == MigrationTier.SOFT_REFRESH
        assert ("amount", "integer", "string") in report.value_type_changed_columns
        assert report.surgically_invalidated == 1

    def test_remap_via_diff_with_added_columns(self, tmp_path) -> None:
        """
        A diff that mixes a table rename with an added column must still REMAP.

        ``try_rename_migration_plan`` would refuse this (column counts differ); the diff-driven path takes the ``table_renames`` directly.
        """

        new_schema = _schema(
            {
                "sales_orders": _table(
                    "sales_orders",
                    {
                        "order_id": _col("order_id", "integer", pk=True),
                        "amount": _col("amount", "integer"),
                        "tax": _col("tax", "integer"),
                    },
                ),
                "customers": _table(
                    "customers",
                    {"customer_id": _col("customer_id", "integer", pk=True)},
                ),
            },
        )
        t = _make_template("T0001", "orders", "order_id")
        _seed_store(str(tmp_path), new_schema, {"T0001": t})
        diff = SchemaDiff(
            table_renames=(("orders", "sales_orders"),),
            per_table={"sales_orders": TableDiff(added_columns=("tax",))},
        )

        report = apply_migration_policy(str(tmp_path), new_schema, schema_diff=diff)

        assert report.tier == MigrationTier.REMAP
        assert report.renamed_tables == (("orders", "sales_orders"),)
        assert report.remapped_templates == 1
        assert report.surgically_invalidated == 0

    def test_combo_rename_and_drop_yields_remap_and_surgical(self, tmp_path) -> None:
        new_schema = _schema(
            {
                "sales_orders": _table(
                    "sales_orders",
                    {"order_id": _col("order_id", "integer", pk=True)},
                ),
                "customers": _table(
                    "customers",
                    {
                        "customer_id": _col("customer_id", "integer", pk=True),
                    },
                ),
            },
        )
        t_remap = _make_template("T0001", "orders", "order_id")
        t_drop = _make_template("T0002", "customers", "name")
        _seed_store(str(tmp_path), new_schema, {"T0001": t_remap, "T0002": t_drop})
        diff = SchemaDiff(
            table_renames=(("orders", "sales_orders"),),
            per_table={"customers": TableDiff(dropped_columns=("name",))},
        )

        report = apply_migration_policy(str(tmp_path), new_schema, schema_diff=diff)

        assert report.tier == MigrationTier.REMAP
        assert report.remapped_templates >= 1
        assert report.surgically_invalidated == 1

    def test_column_rename_within_stable_table(self, tmp_path) -> None:
        new_schema = _schema(
            {
                "orders": _table(
                    "orders",
                    {
                        "order_id": _col("order_id", "integer", pk=True),
                        "total": _col("total", "integer"),
                    },
                ),
                "customers": _table(
                    "customers",
                    {"customer_id": _col("customer_id", "integer", pk=True)},
                ),
            },
        )
        t = _make_template("T0001", "orders", "amount")
        _seed_store(str(tmp_path), new_schema, {"T0001": t})
        diff = SchemaDiff(
            per_table={"orders": TableDiff(renamed_columns=(("amount", "total"),))},
        )

        report = apply_migration_policy(str(tmp_path), new_schema, schema_diff=diff)

        assert report.tier == MigrationTier.REMAP
        assert report.renamed_columns == (("orders", "amount", "total"),)
        assert report.remapped_templates == 1


class TestBackwardsCompatNoDiff:
    """When ``schema_diff`` is None the legacy ``try_rename_migration_plan`` path runs."""

    def test_no_diff_soft_refresh_when_only_eff_hash_changed(self, tmp_path) -> None:
        schema = _two_table_schema()
        t = _make_template("T0001", "orders", "order_id")
        _seed_store(str(tmp_path), schema, {"T0001": t})

        report = apply_migration_policy(str(tmp_path), schema)

        assert report.surgically_invalidated == 0
        assert report.dropped_tables == ()
        assert report.value_type_changed_columns == ()
