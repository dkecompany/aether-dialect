"""Tests for timezone-aware vs naive temporal column metadata and validation."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_reflect import tables_meta_to_schema_graph
from aetherdialect._validation_rules import validate_join_path_key_types


@pytest.mark.fast
def test_aware_and_naive_are_incompatible() -> None:
    """Join-key validation refuses aware vs naive timestamp columns by name."""
    schema = SchemaGraph(
        tables={
            "a": TableMetadata(
                name="a",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                    "ts": ColumnMetadata(name="ts", data_type="timestamptz", sensitivity="none"),
                },
                primary_key=["id"],
                foreign_keys=[],
            ),
            "b": TableMetadata(
                name="b",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                    "ts": ColumnMetadata(name="ts", data_type="timestamp", sensitivity="none"),
                },
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
    )
    assert schema.tables["a"].columns["ts"].is_timezone_aware is True
    assert schema.tables["b"].columns["ts"].is_timezone_aware is False

    sig = ["a.id->b.id", "a.ts->b.ts"]
    issues = validate_join_path_key_types(sig, schema, "main query")
    assert any(i.severity == "error" and "timezone" in i.message.lower() for i in issues)


@pytest.mark.fast
def test_mysql_timestamp_and_datetime_differ() -> None:
    """MySQL TIMESTAMP is timezone-aware while DATETIME is naive and they are incompatible."""
    meta = {
        "events": {
            "column_names_original": ["ts_col", "dt_col"],
            "column_types": ["TIMESTAMP", "DATETIME"],
            "primary_keys": ["ts_col"],
            "foreign_keys": [],
        },
    }
    sg = tables_meta_to_schema_graph(meta, engine="mysql")
    ts_col = sg.tables["events"].columns["ts_col"]
    dt_col = sg.tables["events"].columns["dt_col"]
    assert ts_col.is_timezone_aware is True
    assert dt_col.is_timezone_aware is False
    assert ts_col.value_type == "date"
    assert dt_col.value_type == "date"

    restored_ts = ColumnMetadata.from_dict(ts_col.to_dict())
    restored_dt = ColumnMetadata.from_dict(dt_col.to_dict())
    assert restored_ts.is_timezone_aware is True
    assert restored_dt.is_timezone_aware is False

    schema = SchemaGraph(
        tables={
            "left_t": TableMetadata(
                name="left_t",
                columns={"created_at": ts_col},
                primary_key=["created_at"],
                foreign_keys=[],
            ),
            "right_t": TableMetadata(
                name="right_t",
                columns={"created_at": dt_col},
                primary_key=["created_at"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
    )
    issues = validate_join_path_key_types(["left_t.created_at->right_t.created_at"], schema, "main query")
    assert any(i.severity == "error" and "timezone" in i.message.lower() for i in issues)
