"""diff_schemas must surface nullability/uniqueness; empty diff with hash mismatch is fatal."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import SchemaInvariantError
from aetherdialect._contracts_schema import ColumnMetadata, ColumnRole, SchemaDiff, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import (
    diff_schemas,
    raise_if_schema_diff_covers_structural_change,
    tables_structural_payload,
)
from aetherdialect._utils import structural_hash_fp


def _col(
    name: str,
    *,
    nullable: bool = True,
    unique: bool = False,
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="integer",
        value_type="integer",
        is_nullable=nullable,
        is_unique=unique,
        role=ColumnRole.NUMERIC_MEASURE.value,
    )


def _table(name: str, cols: dict[str, ColumnMetadata]) -> TableMetadata:
    return TableMetadata(name=name, columns=cols, foreign_keys=[], primary_key="")


def _schema(tables: dict[str, TableMetadata]) -> SchemaGraph:
    return SchemaGraph(tables=tables, join_paths_multi={})


@pytest.mark.fast
def test_nullability_flip_changes_structural_hash_and_diff_not_empty() -> None:
    old = _schema({"t": _table("t", {"a": _col("a", nullable=True)})})
    new = _schema({"t": _table("t", {"a": _col("a", nullable=False)})})
    h_old = structural_hash_fp(tables_structural_payload(old.tables))
    h_new = structural_hash_fp(tables_structural_payload(new.tables))
    assert h_old != h_new
    diff = diff_schemas(old, new)
    assert not diff.is_empty
    assert diff.per_table["t"].nullability_changed_columns == ("a",)


@pytest.mark.fast
def test_uniqueness_flip_changes_structural_hash_and_diff_not_empty() -> None:
    old = _schema({"t": _table("t", {"a": _col("a", unique=False)})})
    new = _schema({"t": _table("t", {"a": _col("a", unique=True)})})
    h_old = structural_hash_fp(tables_structural_payload(old.tables))
    h_new = structural_hash_fp(tables_structural_payload(new.tables))
    assert h_old != h_new
    diff = diff_schemas(old, new)
    assert not diff.is_empty
    assert diff.per_table["t"].uniqueness_changed_columns == ("a",)


@pytest.mark.fast
def test_empty_diff_with_structural_hash_mismatch_raises() -> None:
    old = _schema({"t": _table("t", {"a": _col("a", nullable=True)})})
    new = _schema({"t": _table("t", {"a": _col("a", nullable=False)})})
    with pytest.raises(SchemaInvariantError, match="empty diff"):
        raise_if_schema_diff_covers_structural_change(old, new, SchemaDiff())
