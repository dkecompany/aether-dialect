"""Binary columns must never be scanned during profiling or appear in prompt samples."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect._constants_runtime import GROUND_FIELDS, SCHEMA_FIELD_SAMPLES
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect
from aetherdialect._schema_profile import _build_column_profile_for_llm, _profile_column
from aetherdialect._utils import data_type_to_value_type


def _recording_engine() -> tuple[MagicMock, list[str]]:
    executed: list[str] = []
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    def fake_execute(sql: Any) -> MagicMock:
        executed.append(str(sql))
        result = MagicMock()
        if "COUNT(DISTINCT" in str(sql):
            result.fetchone.return_value = (10, 5, 0)
        elif "MAX(freq)" in str(sql):
            result.fetchone.return_value = (2,)
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = fake_execute
    return engine, executed


def _sql_references_column(statements: list[str], column: str, dialect: DuckDBDialect) -> list[str]:
    quoted = dialect.quote_identifier(column).lower()
    bare = column.lower()
    hits: list[str] = []
    for stmt in statements:
        low = stmt.lower()
        if quoted in low or f" {bare} " in f" {low} " or f".{bare}" in low:
            hits.append(stmt)
    return hits


@pytest.mark.fast
@pytest.mark.parametrize("data_type", ["bytea", "blob", "binary", "varbinary", "image", "bytes"])
def test_binary_column_not_sampled(data_type: str) -> None:
    assert data_type_to_value_type(data_type) == "binary"
    engine, executed = _recording_engine()
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    col = ColumnMetadata(name="payload", data_type=data_type)

    _profile_column(dialect, engine, col, "attachments", row_count=10)

    assert col.profile_skipped_reason == "binary"
    assert _sql_references_column(executed, "payload", dialect) == []
    assert col.distinct_count == 0
    assert col.frequent_values == []
    assert col.value_overlap_sample == []
    assert col.min_val is None
    assert col.max_val is None
    assert col.profile_failed is False


@pytest.mark.fast
def test_binary_column_absent_from_prompt_samples() -> None:
    col = ColumnMetadata(
        name="payload",
        data_type="bytea",
        value_type="binary",
        distinct_count=99,
        frequent_values=["deadbeef"],
        min_val="00",
        max_val="ff",
        value_overlap_sample=["aa"],
    )
    profile = _build_column_profile_for_llm(col)
    assert "profile_hints" not in profile

    table = TableMetadata(
        name="attachments",
        columns={"payload": col, "id": ColumnMetadata(name="id", data_type="integer", is_primary_key=True)},
        primary_key=["id"],
        foreign_keys=[],
    )
    graph = SchemaGraph(tables={"attachments": table}, join_paths_multi={}, effective_structural_hash="test")
    payload = json.loads(graph.schema_payload_json(GROUND_FIELDS, owner_master_scope=True))
    col_body = payload["attachments"]["columns"]["payload"]
    assert SCHEMA_FIELD_SAMPLES not in col_body
    assert "frequent_values" not in col_body
    assert "min_val" not in col_body
    assert "max_val" not in col_body
    assert "distinct_count" not in col_body
    assert col_body.get("type") == "binary"
