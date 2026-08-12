"""JSON column typing must treat json and jsonb identically for operator assignment."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, ColumnRole, SchemaGraph, TableMetadata
from aetherdialect._schema_profile import assign_column_ops


@pytest.mark.fast
def test_jsonb_takes_the_json_path() -> None:
    col = ColumnMetadata(
        name="meta",
        data_type="jsonb",
        value_type="string",
        role=ColumnRole.CATEGORICAL.value,
        distinct_count=10,
    )
    table = TableMetadata(name="records", columns={"meta": col}, foreign_keys=[], primary_key="")
    schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"records": table})

    assign_column_ops(schema)

    assert col.element_type == "string"
    assert col.valid_where_ops[0] == "contains"
    assert "like" not in col.valid_where_ops
