"""Aggregates over no rows: COUNT is zero, SUM/AVG/MIN/MAX are null."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import MulGroup, NormalizedExpr
from aetherdialect._contracts_core import ResidualSpec, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._federation import aggregate_identity_row_for_residual
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import build_deterministic_sql


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="integer", value_type="integer")


@pytest.mark.fast
def test_sum_over_no_rows_is_null_on_both_paths() -> None:
    parent = TableMetadata(name="a", columns={"id": _col("id")}, primary_key=["id"], foreign_keys=[])
    child = TableMetadata(
        name="c",
        columns={"id": _col("id"), "a_id": _col("a_id"), "amount": _col("amount")},
        primary_key=["id"],
        foreign_keys=[FKEdge(src_table="c", src_cols=["a_id"], dst_table="a", dst_cols=["id"])],
    )
    tables = {"a": parent, "c": child}
    schema = SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        effective_structural_hash="h",
    )
    intent = RuntimeIntent(
        tables=["a", "c"],
        grain="grouped",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("a.id")),
            SelectCol(expr=NormalizedExpr(agg_func="sum", add_groups=[MulGroup(multiply=["c.amount"])])),
        ],
        group_by_cols=[NormalizedExpr.from_column("a.id")],
        order_by_cols=[],
        where=None,
        preserve_tables=["a"],
        chosen_join_path_signature=["a.id->c.a_id"],
    )
    sql = build_deterministic_sql(intent, schema=schema, dialect=DialectRegistry.get("sqlite"))
    assert "COALESCE(SUM" not in sql.upper()

    residual = ResidualSpec(
        select_cols=(
            SelectCol(expr=NormalizedExpr(agg_func="count", add_groups=[MulGroup(multiply=["c.id"])])),
            SelectCol(expr=NormalizedExpr(agg_func="sum", add_groups=[MulGroup(multiply=["c.amount"])])),
        )
    )
    identity = aggregate_identity_row_for_residual(residual)
    assert identity[0] == 0
    assert identity[1] is None
