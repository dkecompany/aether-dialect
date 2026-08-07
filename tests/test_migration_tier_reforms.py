"""Migration tier reforms: additive tier, diff visibility, rename confidence, atomic map apply, auxiliary artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._constants import (
    AETHERSPACE_ARTIFACT_VERSION,
    ARTIFACT_FORMAT_VERSION,
    MIGRATION_MAP_ACTION_DESTRUCTIVE,
    MIGRATION_MAP_ACTION_REMAP,
    SEED_WARMUP_CACHE_ZIP,
    WRITE_QUEUE_FILENAME,
)
from aetherdialect._contracts_base import (
    ArtifactManifest,
    ColumnRole,
    MigrationReport,
    MigrationTier,
    SchemaMigrationMap,
    SchemaMigrationMapEntry,
)
from aetherdialect._contracts_core import ConcreteIntent, NormalizedExpr, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    FKEdge,
    SchemaDiff,
    SchemaGraph,
    SQLShape,
    TableDiff,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._core_utils import (
    assess_rename_migration_plan,
    refresh_migration_simulation_caches,
    structural_hash_fp,
    try_rename_migration_plan,
    wipe_versioned_artifacts,
    write_artifact_manifest,
)
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_graph import (
    classify_migration_tier,
    diff_schemas,
    schema_diff_is_additive_only,
    tables_structural_payload,
)
from aetherdialect._templates import (
    TemplateOps,
)


def _col(
    name: str,
    dt: str = "integer",
    *,
    pk: bool = False,
    nullable: bool = True,
    unique: bool = False,
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type=dt,
        value_type=dt,
        is_primary_key=pk,
        is_nullable=nullable,
        is_unique=unique,
        role=ColumnRole.IDENTIFIER.value if pk else ColumnRole.NUMERIC_MEASURE.value,
    )


def _table(name: str, cols: dict[str, ColumnMetadata], **kwargs) -> TableMetadata:
    pk = next((c.name for c in cols.values() if c.is_primary_key), "")
    return TableMetadata(name=name, columns=cols, foreign_keys=[], primary_key=pk, **kwargs)


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


def _seed_store(
    artifacts_dir: str,
    schema: SchemaGraph,
    templates: dict[str, Template],
    *,
    stale_eff: str = "eff_old",
    structural: str = "s_old",
) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    store_dir = TemplateOps.template_store_dir_for_space(artifacts_dir, "master")
    os.makedirs(store_dir, exist_ok=True)
    prev = EngineConfig.TEMPLATE_STORE_DIR
    EngineConfig.TEMPLATE_STORE_DIR = store_dir
    try:
        store = TemplateOps.empty_template_store(stale_eff)
        TemplateOps.templates_to_store(store, templates)
        TemplateOps.save_template_store(store)
    finally:
        EngineConfig.TEMPLATE_STORE_DIR = prev
    write_artifact_manifest(
        artifacts_dir,
        structural_hash=structural,
        profiling_hash="p_old",
        scope_hash=schema.scope_hash,
        effective_structural_hash=stale_eff,
        last_migration_tier=MigrationTier.SOFT_REFRESH.value,
        last_action="seed",
    )


class TestAdditiveTiers:
    def test_classify_pure_column_add_is_additive(self) -> None:
        old = _schema(
            {
                "orders": _table("orders", {"order_id": _col("order_id", pk=True), "amount": _col("amount")}),
            },
            structural="s_old",
            effective="eff_old",
        )
        new = _schema(
            {
                "orders": _table(
                    "orders",
                    {
                        "order_id": _col("order_id", pk=True),
                        "amount": _col("amount"),
                        "note": _col("note", "varchar"),
                    },
                ),
            },
            structural="s_new",
            effective="eff_new",
        )
        manifest = ArtifactManifest(
            artifact_format_version=ARTIFACT_FORMAT_VERSION,
            effective_structural_hash="eff_old",
            structural_hash="s_old",
            profiling_hash="p0",
            scope_hash="sc",
            notes_hash="n",
            semantic_edges_hash="s",
        )
        diff = diff_schemas(old, new)
        assert schema_diff_is_additive_only(diff)
        assert classify_migration_tier(manifest, new, previous_schema=old, schema_diff=diff) == MigrationTier.ADDITIVE

    def test_additive_policy_preserves_templates(self, tmp_path) -> None:
        old_schema = _schema(
            {"orders": _table("orders", {"order_id": _col("order_id", pk=True), "amount": _col("amount")})},
            structural="s_old",
            effective="eff_old",
            scope="sc",
        )
        new_schema = _schema(
            {
                "orders": _table(
                    "orders",
                    {
                        "order_id": _col("order_id", pk=True),
                        "amount": _col("amount"),
                        "note": _col("note", "varchar"),
                    },
                ),
            },
            structural="s_new",
            effective="eff_new",
            scope="sc",
        )
        t = _make_template("T0001", "orders", "amount")
        _seed_store(str(tmp_path), new_schema, {"T0001": t}, structural="s_old")
        diff = diff_schemas(old_schema, new_schema)
        report = TemplateOps.apply_migration_policy(str(tmp_path), new_schema, schema_diff=diff)
        assert report.tier == MigrationTier.ADDITIVE
        assert report.added_columns == (("orders", "note"),)
        assert report.added_tables == ()
        store = TemplateOps._load_partitioned_view_unlocked(
            TemplateOps.template_store_dir_for_space(str(tmp_path), "master")
        )
        assert store is not None
        assert "T0001" in store.partition_map

    def test_additive_skeleton_not_destructive(self, tmp_path) -> None:
        diff = SchemaDiff(per_table={"orders": TableDiff(added_columns=("note",))})
        path = TemplateOps.export_schema_migration_map_skeleton(
            tmp_path, tier=MigrationTier.ADDITIVE, schema_diff=diff, rename_plan=None
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["action"] != MIGRATION_MAP_ACTION_DESTRUCTIVE

    def test_additive_print_header_reports_additions(self) -> None:
        report = MigrationReport(
            tier=MigrationTier.ADDITIVE,
            added_tables=("customers",),
            added_columns=(("orders", "note"),),
        )
        lines: list[str] = []
        MainExecutionOps._print_migration_applied(report, lines.append)
        assert lines
        assert "Schema expanded" in lines[0]
        assert any("Added 1 table" in line for line in lines)
        assert any("Added 1 column" in line for line in lines)


class TestDiffVisibility:
    def test_diff_detects_nullability_change(self) -> None:
        old = _schema({"t": _table("t", {"a": _col("a", nullable=True)})})
        new = _schema({"t": _table("t", {"a": replace(_col("a", nullable=True), is_nullable=False)})})
        diff = diff_schemas(old, new)
        assert diff.per_table["t"].nullability_changed_columns == ("a",)

    def test_diff_detects_uniqueness_change(self) -> None:
        old = _schema({"t": _table("t", {"a": _col("a", unique=False)})})
        new = _schema({"t": _table("t", {"a": replace(_col("a", unique=False), is_unique=True)})})
        diff = diff_schemas(old, new)
        assert diff.per_table["t"].uniqueness_changed_columns == ("a",)

    def test_structural_hash_includes_index_changes(self) -> None:
        t1 = _table("t", {"a": _col("a")}, indexed_columns=["a"])
        t2 = _table("t", {"a": _col("a")}, indexed_columns=[])
        h1 = structural_hash_fp(tables_structural_payload({"t": t1}))
        h2 = structural_hash_fp(tables_structural_payload({"t": t2}))
        assert h1 != h2

    def test_diff_detects_view_definition_change(self) -> None:
        old = _schema({"v": _table("v", {"a": _col("a")}, kind="view", view_definition="SELECT 1")})
        new = _schema({"v": _table("v", {"a": _col("a")}, kind="view", view_definition="SELECT 2")})
        diff = diff_schemas(old, new)
        assert diff.per_table["v"].view_definition_changed is True

    def test_fk_change_tombstones_table(self, tmp_path) -> None:
        fk = FKEdge(src_table="orders", src_cols=["customer_id"], dst_table="customers", dst_cols=["customer_id"])
        old = _schema(
            {
                "orders": _table("orders", {"order_id": _col("order_id", pk=True), "customer_id": _col("customer_id")}),
                "customers": _table("customers", {"customer_id": _col("customer_id", pk=True)}),
            }
        )
        old.tables["orders"].foreign_keys = [fk]
        new = _schema(
            {
                "orders": _table("orders", {"order_id": _col("order_id", pk=True), "customer_id": _col("customer_id")}),
                "customers": _table("customers", {"customer_id": _col("customer_id", pk=True)}),
            }
        )
        diff = diff_schemas(old, new)
        assert diff.per_table["orders"].fk_changed is True
        t = _make_template("T0001", "orders", "order_id")
        _seed_store(str(tmp_path), new, {"T0001": t})
        deleted = TemplateOps.surgical_invalidate_templates_by_diff(str(tmp_path), new, diff)
        assert deleted == 1

    def test_profiling_drift_above_overlap_is_soft_refresh(self) -> None:
        manifest = ArtifactManifest(
            artifact_format_version=ARTIFACT_FORMAT_VERSION,
            effective_structural_hash="e",
            structural_hash="t",
            profiling_hash="p0",
            scope_hash="c",
            notes_hash="n",
            semantic_edges_hash="s",
        )
        sch = MagicMock()
        sch.effective_structural_hash = "e"
        sch.structural_hash = "t"
        sch.profiling_hash = "p1"
        sch.scope_hash = "c"
        sch.notes_hash = "n"
        sch.semantic_edges_hash = "s"
        prev = MagicMock()
        with patch("aetherdialect._schema_graph.profiling_value_overlap", return_value=0.5):
            assert classify_migration_tier(manifest, sch, previous_schema=prev) == MigrationTier.SOFT_REFRESH

    def test_profiling_drift_below_overlap_is_destructive(self) -> None:
        manifest = ArtifactManifest(
            artifact_format_version=ARTIFACT_FORMAT_VERSION,
            effective_structural_hash="e",
            structural_hash="t",
            profiling_hash="p0",
            scope_hash="c",
            notes_hash="n",
            semantic_edges_hash="s",
        )
        sch = MagicMock()
        sch.effective_structural_hash = "e"
        sch.structural_hash = "t"
        sch.profiling_hash = "p1"
        sch.scope_hash = "c"
        sch.notes_hash = "n"
        sch.semantic_edges_hash = "s"
        prev = MagicMock()
        with patch("aetherdialect._schema_graph.profiling_value_overlap", return_value=0.01):
            assert classify_migration_tier(manifest, sch, previous_schema=prev) == MigrationTier.DESTRUCTIVE


class TestRenameConfidence:
    def test_ambiguous_rename_refuses(self) -> None:
        old = _schema(
            {
                "a1": _table("a1", {"x": _col("x"), "y": _col("y")}),
                "a2": _table("a2", {"x": _col("x"), "y": _col("y")}),
            }
        )
        new = _schema(
            {
                "b1": _table("b1", {"x": _col("x"), "y": _col("y")}),
                "b2": _table("b2", {"x": _col("x"), "y": _col("y")}),
            }
        )
        assert try_rename_migration_plan(old, new) is None
        assert assess_rename_migration_plan(old, new) is None

    def test_confident_rename_reports_confidence(self) -> None:
        old = _schema({"old_t": _table("old_t", {"id": _col("id", pk=True), "amt": _col("amt")})})
        new = _schema({"new_t": _table("new_t", {"id": _col("id", pk=True), "amt": _col("amt")})})
        assessment = assess_rename_migration_plan(old, new)
        assert assessment is not None
        assert assessment.confidence == 1.0
        assert assessment.plan[0] == (("old_t", "new_t"),)

    def test_skeleton_includes_rename_confidence(self, tmp_path) -> None:
        old = _schema({"old_t": _table("old_t", {"id": _col("id", pk=True)})})
        new = _schema({"new_t": _table("new_t", {"id": _col("id", pk=True)})})
        plan = try_rename_migration_plan(old, new)
        path = TemplateOps.export_schema_migration_map_skeleton(
            tmp_path,
            tier=MigrationTier.REMAP,
            schema_diff=None,
            rename_plan=plan,
            previous_schema=old,
            schema=new,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["rename_confidence"] == 1.0


class TestAtomicMapApply:
    def test_rollback_on_mid_sequence_failure(self, tmp_path, monkeypatch) -> None:
        schema = _schema({"orders": _table("orders", {"order_id": _col("order_id", pk=True)})})
        t = _make_template("T0001", "orders", "order_id")
        _seed_store(str(tmp_path), schema, {"T0001": t})
        map_obj = SchemaMigrationMap(
            version=1,
            action=MIGRATION_MAP_ACTION_REMAP,
            table_renames=(SchemaMigrationMapEntry(entry_type="table", from_name="orders", to_name="sales_orders"),),
            column_renames=(),
            dropped_tables=(),
            dropped_columns=(),
            added_tables=(),
            added_columns=(),
        )

        def _boom(*_a, **_k):
            raise RuntimeError("boom")

        monkeypatch.setattr("aetherdialect._templates.migrate_sidecar_for_diff", _boom)
        with pytest.raises(RuntimeError, match="boom"):
            TemplateOps.apply_schema_migration_map(map_obj, str(tmp_path), schema, tmp_path / "schema.json.gz")
        store = TemplateOps._load_partitioned_view_unlocked(
            TemplateOps.template_store_dir_for_space(str(tmp_path), "master")
        )
        assert store is not None
        assert "T0001" in store.partition_map

    def test_rollback_restores_sidecar_and_aetherspace(self, tmp_path, monkeypatch) -> None:
        assert callable(MainExecutionOps.apply_structural_migration_to_persisted_scopes)

        schema = _schema(
            {"orders": _table("orders", {"order_id": _col("order_id", pk=True), "amount": _col("amount")})}
        )
        t = _make_template("T0001", "orders", "order_id")
        _seed_store(str(tmp_path), schema, {"T0001": t})
        schema_path = tmp_path / "schema.json.gz"
        schema_path.write_bytes(b"original-sidecar")
        MainExecutionOps.save_aetherspace_snapshot(
            str(tmp_path),
            "sales",
            {
                "version": AETHERSPACE_ARTIFACT_VERSION,
                "tables": ["orders"],
                "columns": ["orders.amount"],
                "deny_objects": [],
                "deny_columns": [],
                "table_descriptions": {},
                "column_meta": {},
                "notes": None,
                "notes_hash": "",
            },
        )
        map_obj = SchemaMigrationMap(
            version=1,
            action=MIGRATION_MAP_ACTION_REMAP,
            table_renames=(SchemaMigrationMapEntry(entry_type="table", from_name="orders", to_name="sales_orders"),),
            column_renames=(),
            dropped_tables=(),
            dropped_columns=(),
            added_tables=(),
            added_columns=(),
        )

        def _touch_sidecar(*_a, **_k):
            schema_path.write_bytes(b"mutated-sidecar")

        def _boom(*_a, **_k):
            raise RuntimeError("boom")

        monkeypatch.setattr("aetherdialect._templates.migrate_sidecar_for_diff", _touch_sidecar)
        monkeypatch.setattr("aetherdialect._templates.TemplateOps.apply_structural_migration_from_map", _boom)
        with pytest.raises(RuntimeError, match="boom"):
            TemplateOps.apply_schema_migration_map(map_obj, str(tmp_path), schema, schema_path)
        assert schema_path.read_bytes() == b"original-sidecar"
        snap = MainExecutionOps.load_aetherspace_snapshot(str(tmp_path), "sales")
        assert snap is not None
        assert snap["columns"] == ["orders.amount"]


class TestAuxiliaryArtifacts:
    def test_soft_refresh_clears_simulation_caches(self, tmp_path) -> None:
        d = str(tmp_path)
        open(os.path.join(d, "qsim_summary_v1.json.gz"), "wb").close()
        open(os.path.join(d, SEED_WARMUP_CACHE_ZIP), "wb").close()
        refresh_migration_simulation_caches(d)
        assert not os.path.isfile(os.path.join(d, "qsim_summary_v1.json.gz"))
        assert not os.path.isfile(os.path.join(d, SEED_WARMUP_CACHE_ZIP))

    def test_destructive_wipe_clears_write_queue_and_aetherspaces(self, tmp_path) -> None:
        d = str(tmp_path)
        os.makedirs(os.path.join(d, "aetherspaces"), exist_ok=True)
        open(os.path.join(d, "aetherspaces", "sales.json"), "w", encoding="utf-8").write("{}")
        open(os.path.join(d, WRITE_QUEUE_FILENAME), "w", encoding="utf-8").write("{}\n")
        wipe_versioned_artifacts(d)
        assert not os.path.isdir(os.path.join(d, "aetherspaces"))
        assert not os.path.isfile(os.path.join(d, WRITE_QUEUE_FILENAME))

    def test_apply_migration_soft_refresh_clears_qsim(self, tmp_path) -> None:
        schema = _schema(
            {"orders": _table("orders", {"order_id": _col("order_id", pk=True), "amount": _col("amount")})},
            structural="s_new",
            effective="eff_new",
            profiling="p_new",
            scope="sc",
        )
        t = _make_template("T0001", "orders", "amount")
        _seed_store(str(tmp_path), schema, {"T0001": t}, structural="s_old")
        open(os.path.join(str(tmp_path), "qsim_summary_v1.json.gz"), "wb").close()
        diff = SchemaDiff(per_table={"orders": TableDiff(dropped_columns=("amount",))})
        TemplateOps.apply_migration_policy(str(tmp_path), schema, schema_diff=diff)
        assert not os.path.isfile(os.path.join(str(tmp_path), "qsim_summary_v1.json.gz"))

    def test_soft_refresh_migrates_aetherspace_snapshots(self, tmp_path) -> None:
        assert callable(MainExecutionOps.apply_structural_migration_to_persisted_scopes)

        old_schema = _schema(
            {"orders": _table("orders", {"order_id": _col("order_id", pk=True), "amount": _col("amount")})},
            structural="s_old",
            effective="eff_old",
            scope="sc",
        )
        new_schema = _schema(
            {"orders": _table("orders", {"order_id": _col("order_id", pk=True)})},
            structural="s_new",
            effective="eff_new",
            scope="sc",
        )
        t = _make_template("T0001", "orders", "order_id")
        _seed_store(str(tmp_path), new_schema, {"T0001": t}, structural="s_old")
        MainExecutionOps.save_aetherspace_snapshot(
            str(tmp_path),
            "sales",
            {
                "version": AETHERSPACE_ARTIFACT_VERSION,
                "tables": ["orders"],
                "columns": ["orders.order_id", "orders.amount"],
                "deny_objects": [],
                "deny_columns": [],
                "table_descriptions": {},
                "column_meta": {},
                "notes": None,
                "notes_hash": "",
            },
        )
        diff = diff_schemas(old_schema, new_schema)
        TemplateOps.apply_migration_policy(str(tmp_path), new_schema, previous_schema=old_schema, schema_diff=diff)
        snap = MainExecutionOps.load_aetherspace_snapshot(str(tmp_path), "sales")
        assert snap is not None
        assert snap["columns"] == ["orders.order_id"]
        assert os.path.isdir(os.path.join(str(tmp_path), "aetherspaces"))
