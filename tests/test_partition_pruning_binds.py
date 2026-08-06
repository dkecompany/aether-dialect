"""Partition pruning predicates keep bind tokens on the execute path."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import NormalizedExpr, PredicateGroup, WhereParam
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._core_utils import reconcile_execute_bind_params
from aetherdialect._dialect_postgres import PostgresDialect


def _schema_with_partition(table: str, partition_cols: list[str]) -> SchemaGraph:
    cols: dict[str, ColumnMetadata] = {"id": ColumnMetadata(name="id", data_type="integer")}
    for col in partition_cols:
        cols[col] = ColumnMetadata(name=col, data_type="date")
    meta = TableMetadata(
        name=table,
        columns=cols,
        foreign_keys=[],
        primary_key="id",
        partition_columns=partition_cols,
    )
    return SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={table: meta})


@pytest.mark.fast
def test_pruning_predicate_keeps_bind_token() -> None:
    dialect = PostgresDialect.__new__(PostgresDialect)
    schema = _schema_with_partition("events", ["dt"])
    intent = RuntimeIntent(
        tables=["events"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("events.dt"),
                    op="=",
                    param_key="p1",
                    raw_value=None,
                )
            ]
        ),
        param_values={"p1": "2024-01-15"},
    )
    sql = "SELECT * FROM events"
    pruned = dialect.inject_pruning_predicates(sql, schema=schema, intent=intent)
    assert ":p1" in pruned
    assert "2024-01-15" not in pruned
    bind_map = reconcile_execute_bind_params(pruned, intent.param_values)
    assert bind_map is not None
    assert bind_map["p1"] == "2024-01-15"
