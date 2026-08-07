"""Tests for profiling zero-row tables without inflating row counts or ratios."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect
from aetherdialect._intent_repair import best_descriptive_columns
from aetherdialect._schema_catalog import _profile_table
from aetherdialect._schema_graph import recompute_join_paths_multi


@pytest.mark.fast
def test_zero_row_table_reports_zero_and_unknown_ratios() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    def fake_execute(stmt: object) -> MagicMock:
        sql = str(stmt)
        result = MagicMock()
        if "SELECT COUNT(*)" in sql and "COUNT(DISTINCT" not in sql:
            result.scalar.return_value = 0
            result.fetchone.return_value = (0,)
        elif "COUNT(DISTINCT" in sql:
            result.fetchone.return_value = (0, 0, 0)
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = fake_execute

    table = TableMetadata(
        name="empty_table",
        columns={
            "status": ColumnMetadata(
                name="status",
                data_type="varchar",
                value_type="string",
                sensitivity="none",
            )
        },
        primary_key=[],
        foreign_keys=[],
    )
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    _profile_table(dialect, engine, table)

    assert table.row_count == 0
    col = table.columns["status"]
    assert col.distinct_ratio is None
    assert col.null_ratio is None

    schema = SchemaGraph(
        tables={"empty_table": table},
        join_paths_multi=recompute_join_paths_multi({"empty_table": table}),
        schema_graph_id="sg_empty",
        effective_structural_hash="eff_empty",
    )
    col.distinct_count = 5
    assert best_descriptive_columns("empty_table", schema, set()) == []
