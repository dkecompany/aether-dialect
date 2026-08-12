"""Deny-only aetherspace scope and space snapshot migration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._constants import AETHERSPACE_ARTIFACT_VERSION
from aetherdialect._contracts_base import EngineContext, NormalizedExpr, SpaceContext
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaDiff, SchemaGraph, TableDiff, TableMetadata
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._schema_graph import assert_consumer_sql_in_scope, assert_intent_in_scope
from aetherdialect._templates_ops import TemplateOps


def _column(name: str, *, data_type: str = "integer") -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type=data_type, sensitivity="none")


def _table(name: str, *, columns: dict[str, ColumnMetadata] | None = None) -> TableMetadata:
    cols = columns or {"id": _column("id")}
    return TableMetadata(name=name, columns=cols, primary_key=["id"], foreign_keys=[])


def _two_table_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "allowed": _table("allowed"),
            "secret": _table("secret"),
        },
        join_paths_multi={},
        effective_structural_hash="eff_t30",
    )


def _deny_only_snapshot(schema: SchemaGraph) -> dict[str, object]:
    return MainExecutionOps.subset_graph_for_space(schema, SpaceContext(deny_objects=frozenset({"secret"})))


def _secret_table_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["secret"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("secret.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


@pytest.mark.fast
def test_deny_only_space_snapshot_leaves_tables_empty() -> None:
    schema = _two_table_schema()
    snap = _deny_only_snapshot(schema)
    assert snap["deny_objects"] == ["secret"]
    assert snap["tables"] == []


@pytest.mark.fast
def test_deny_only_space_intent_gate_rejects_denied_table() -> None:
    schema = _two_table_schema()
    snap = _deny_only_snapshot(schema)
    tables, columns = MainExecutionOps.space_allowed_sets_from_snapshot(snap)
    deny_objects, _ = MainExecutionOps.space_deny_sets_from_snapshot(snap)
    intent = _secret_table_intent()
    gate_ran = bool(tables or columns or deny_objects)
    assert gate_ran
    in_scope = assert_intent_in_scope(intent, tables, columns, schema, deny_tables=deny_objects)
    assert not in_scope


@pytest.mark.fast
def test_deny_only_space_sql_scope_rejects_denied_table() -> None:
    schema = _two_table_schema()
    snap = _deny_only_snapshot(schema)
    tables, _ = MainExecutionOps.space_allowed_sets_from_snapshot(snap)
    deny_objects, _ = MainExecutionOps.space_deny_sets_from_snapshot(snap)
    sql = "SELECT id FROM secret"
    ctx = EngineContext(deny_objects=deny_objects)
    dialect = MagicMock()
    dialect.sqlglot_dialect = "postgres"
    allowed = assert_consumer_sql_in_scope(
        sql,
        dialect,
        ctx,
        schema,
        tables if tables else None,
    )
    assert not allowed


def _base_snapshot(**overrides: object) -> dict[str, object]:
    snap: dict[str, object] = {
        "version": AETHERSPACE_ARTIFACT_VERSION,
        "tables": [],
        "columns": [],
        "deny_objects": [],
        "deny_columns": [],
        "table_descriptions": {},
        "column_meta": {},
        "notes": None,
        "notes_hash": "",
    }
    snap.update(overrides)
    return snap


@pytest.mark.fast
def test_space_migration_remaps_deny_objects_on_table_rename(tmp_path) -> None:
    engine_dir = str(tmp_path)
    MainExecutionOps.save_aetherspace_snapshot(
        engine_dir,
        "deny_rename",
        _base_snapshot(deny_objects=["secret"]),
    )
    MainExecutionOps.apply_structural_migration_to_aetherspace_snapshots(
        engine_dir,
        table_renames=(("secret", "classified"),),
    )
    loaded = MainExecutionOps.load_aetherspace_snapshot(engine_dir, "deny_rename")
    assert loaded is not None
    assert loaded["deny_objects"] == ["classified"]
    assert "secret" not in loaded["deny_objects"]


@pytest.mark.fast
def test_space_migration_remaps_deny_columns_on_column_rename(tmp_path) -> None:
    engine_dir = str(tmp_path)
    MainExecutionOps.save_aetherspace_snapshot(
        engine_dir,
        "deny_col_rename",
        _base_snapshot(
            tables=["secret"],
            columns=["secret.id"],
            deny_columns=["secret.id"],
        ),
    )
    MainExecutionOps.apply_structural_migration_to_aetherspace_snapshots(
        engine_dir,
        column_renames=(("secret", "id", "identifier"),),
    )
    loaded = MainExecutionOps.load_aetherspace_snapshot(engine_dir, "deny_col_rename")
    assert loaded is not None
    assert loaded["deny_columns"] == ["secret.identifier"]
    assert "secret.id" not in loaded["deny_columns"]


@pytest.mark.fast
def test_retype_schema_diff_updates_space_column_meta(tmp_path) -> None:
    engine_dir = str(tmp_path)
    MainExecutionOps.save_aetherspace_snapshot(
        engine_dir,
        "retype_space",
        _base_snapshot(
            tables=["orders"],
            columns=["orders.amount"],
            column_meta={"orders.amount": {"description": "legacy total", "value_type": "integer"}},
        ),
    )
    schema_diff = SchemaDiff(
        per_table={
            "orders": TableDiff(
                retyped_columns=(("amount", "integer", "varchar"),),
                value_type_changed_columns=(("amount", "integer", "varchar"),),
            )
        }
    )
    TemplateOps.apply_structural_migration_from_schema_diff(engine_dir, schema_diff)
    loaded = MainExecutionOps.load_aetherspace_snapshot(engine_dir, "retype_space")
    assert loaded is not None
    meta = loaded["column_meta"]["orders.amount"]
    assert meta.get("value_type") == "varchar"
