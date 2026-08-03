"""T11: cross-table column moves are detected or surfaced with explicit operator guidance."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import ColumnRole, MigrationTier
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import diff_schemas, schema_diff_cross_table_limitation_note
from aetherdialect._templates import export_schema_migration_map_skeleton


def _col(name: str, data_type: str = "integer") -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type=data_type,
        value_type="integer" if data_type in {"integer", "int4"} else data_type,
        role=ColumnRole.NUMERIC_MEASURE.value,
    )


def _table(name: str, cols: dict[str, ColumnMetadata]) -> TableMetadata:
    return TableMetadata(name=name, columns=cols, foreign_keys=[], primary_key="")


def _schema(tables: dict[str, TableMetadata]) -> SchemaGraph:
    return SchemaGraph(tables=tables, join_paths_multi={})


@pytest.mark.fast
def test_cross_table_column_move_detected_not_drop_add() -> None:
    old = _schema(
        {
            "orders": _table("orders", {"id": _col("id"), "amount": _col("amount")}),
            "archive": _table("archive", {"id": _col("id")}),
        }
    )
    new = _schema(
        {
            "orders": _table("orders", {"id": _col("id")}),
            "archive": _table("archive", {"id": _col("id"), "amount": _col("amount")}),
        }
    )
    diff = diff_schemas(old, new)
    assert diff.cross_table_column_moves == (("orders", "amount", "archive", "amount"),)
    assert "orders" not in diff.per_table or diff.per_table["orders"].dropped_columns == ()
    assert "archive" not in diff.per_table or diff.per_table["archive"].added_columns == ()


@pytest.mark.fast
def test_ambiguous_cross_table_move_keeps_drop_add_and_emits_limitation() -> None:
    old = _schema(
        {
            "a": _table("a", {"id": _col("id"), "x": _col("x")}),
            "b": _table("b", {"id": _col("id"), "x": _col("x")}),
            "c": _table("c", {"id": _col("id")}),
        }
    )
    new = _schema(
        {
            "a": _table("a", {"id": _col("id")}),
            "b": _table("b", {"id": _col("id")}),
            "c": _table("c", {"id": _col("id"), "x": _col("x")}),
        }
    )
    diff = diff_schemas(old, new)
    assert diff.cross_table_column_moves == ()
    dropped = {t: td.dropped_columns for t, td in diff.per_table.items() if td.dropped_columns}
    added = {t: td.added_columns for t, td in diff.per_table.items() if td.added_columns}
    assert dropped
    assert added
    note = schema_diff_cross_table_limitation_note(diff)
    assert note is not None
    assert "cross-table" in note.lower()


@pytest.mark.fast
def test_migration_skeleton_includes_cross_table_limitation_when_needed(tmp_path) -> None:
    old = _schema(
        {
            "a": _table("a", {"id": _col("id"), "x": _col("x")}),
            "b": _table("b", {"id": _col("id"), "x": _col("x")}),
            "c": _table("c", {"id": _col("id")}),
        }
    )
    new = _schema(
        {
            "a": _table("a", {"id": _col("id")}),
            "b": _table("b", {"id": _col("id")}),
            "c": _table("c", {"id": _col("id"), "x": _col("x")}),
        }
    )
    diff = diff_schemas(old, new)
    path = export_schema_migration_map_skeleton(
        tmp_path,
        tier=MigrationTier.DESTRUCTIVE,
        schema_diff=diff,
        rename_plan=None,
        previous_schema=old,
        schema=new,
    )
    text = path.read_text(encoding="utf-8")
    assert "cross-table" in text.lower()
