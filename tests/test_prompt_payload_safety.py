"""Prompt schema payloads must never include samples from complex column types."""

from __future__ import annotations

import json

import pytest

from aetherdialect._constants import GROUND_FIELDS, SCHEMA_FIELD_SAMPLES
from aetherdialect._contracts_base import SchemaInvariantError
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    SchemaGraph,
    TableMetadata,
)


def _complex_column_graph() -> SchemaGraph:
    sample_kwargs = {
        "distinct_count": 42,
        "frequent_values": ["leaked"],
        "min_val": "00",
        "max_val": "ff",
        "value_overlap_sample": ["overlap"],
    }
    cols = {
        "id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True),
        "payload": ColumnMetadata(name="payload", data_type="bytea", value_type="binary", **sample_kwargs),
        "metadata": ColumnMetadata(name="metadata", data_type="jsonb", value_type="json", **sample_kwargs),
        "tags": ColumnMetadata(name="tags", data_type="text[]", value_type="array", **sample_kwargs),
    }
    table = TableMetadata(
        name="records",
        columns=cols,
        primary_key=["id"],
        foreign_keys=[],
    )
    return SchemaGraph(
        tables={"records": table},
        join_paths_multi={},
        effective_structural_hash="test",
    )


@pytest.mark.fast
def test_complex_column_values_never_in_payload() -> None:
    graph = _complex_column_graph()
    payload = json.loads(graph.schema_payload_json(GROUND_FIELDS, owner_master_scope=True))
    columns = payload["records"]["columns"]
    for name in ("payload", "metadata", "tags"):
        col_body = columns[name]
        assert SCHEMA_FIELD_SAMPLES not in col_body
        assert "frequent_values" not in col_body
        assert "min_val" not in col_body
        assert "max_val" not in col_body
        assert "distinct_count" not in col_body

    scalar_col = columns["id"]
    assert scalar_col.get("type") == "integer"

    forced = ColumnMetadata(
        name="forced",
        data_type="jsonb",
        value_type="json",
        distinct_count=1,
        frequent_values=["x"],
    )
    with pytest.raises(SchemaInvariantError, match="prompt sample"):
        forced.assert_prompt_value_scalar(
            table_name="records",
            contribution="prompt sample",
        )
