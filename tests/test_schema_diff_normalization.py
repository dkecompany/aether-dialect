"""Semantically identical catalog types must not surface as redeclared_columns or hash drift."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, ColumnRole, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import diff_schemas, tables_structural_payload
from aetherdialect._utils import data_type_to_value_type, structural_hash_fp


def _col(name: str, data_type: str) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type=data_type,
        value_type=data_type_to_value_type(data_type),
        role=ColumnRole.NUMERIC_MEASURE.value,
    )


def _table(name: str, cols: dict[str, ColumnMetadata]) -> TableMetadata:
    return TableMetadata(name=name, columns=cols, foreign_keys=[], primary_key="")


def _schema(tables: dict[str, TableMetadata]) -> SchemaGraph:
    return SchemaGraph(tables=tables, join_paths_multi={})


@pytest.mark.fast
def test_int4_vs_integer_produces_empty_diff() -> None:
    old = _schema({"t": _table("t", {"x": _col("x", "int4")})})
    new = _schema({"t": _table("t", {"x": _col("x", "integer")})})
    diff = diff_schemas(old, new)
    assert diff.is_empty


@pytest.mark.fast
def test_varchar_vs_character_varying_produces_empty_diff() -> None:
    old = _schema({"t": _table("t", {"x": _col("x", "varchar(255)")})})
    new = _schema({"t": _table("t", {"x": _col("x", "character varying(255)")})})
    diff = diff_schemas(old, new)
    assert diff.is_empty


@pytest.mark.fast
def test_case_only_type_identifier_produces_empty_diff() -> None:
    old = _schema({"t": _table("t", {"x": _col("x", "INTEGER")})})
    new = _schema({"t": _table("t", {"x": _col("x", "integer")})})
    diff = diff_schemas(old, new)
    assert diff.is_empty


@pytest.mark.fast
def test_semantic_type_aliases_share_structural_hash() -> None:
    old = _schema({"t": _table("t", {"x": _col("x", "int4")})})
    new = _schema({"t": _table("t", {"x": _col("x", "integer")})})
    h_old = structural_hash_fp(tables_structural_payload(old.tables))
    h_new = structural_hash_fp(tables_structural_payload(new.tables))
    assert h_old == h_new


@pytest.mark.fast
def test_integer_to_bigint_is_redeclared() -> None:
    old = _schema({"t": _table("t", {"x": _col("x", "integer")})})
    new = _schema({"t": _table("t", {"x": _col("x", "bigint")})})
    diff = diff_schemas(old, new)
    assert diff.per_table["t"].redeclared_columns == (("x", "integer", "bigint"),)
    assert diff.per_table["t"].retyped_columns == ()


@pytest.mark.fast
def test_varchar_to_text_is_redeclared() -> None:
    old = _schema({"t": _table("t", {"x": _col("x", "varchar(50)")})})
    new = _schema({"t": _table("t", {"x": _col("x", "text")})})
    diff = diff_schemas(old, new)
    assert diff.per_table["t"].redeclared_columns == (("x", "varchar(50)", "text"),)
