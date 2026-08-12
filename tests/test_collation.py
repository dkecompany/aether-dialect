"""Deterministic COLLATE pinning for coordinator and limit-fed member ordering."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import NormalizedExpr, OrderByCol
from aetherdialect._contracts_core import ResidualSpec, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._federation_plan import render_federation_residual_sql
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import _build_deterministic_select_block, build_deterministic_sql


def _text_table_graph() -> SchemaGraph:
    tables = {
        "t": TableMetadata(
            name="t",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", value_type="integer", sensitivity="none"),
                "name": ColumnMetadata(
                    name="name",
                    data_type="varchar",
                    value_type="string",
                    sensitivity="none",
                ),
            },
            primary_key=["id"],
            foreign_keys=[],
        ),
    }
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


@pytest.mark.fast
def test_coordinator_ordering_is_collation_pinned() -> None:
    schema = _text_table_graph()
    residual = ResidualSpec(
        select_cols=(SelectCol(expr=NormalizedExpr.from_column("t.name")),),
        order_by_cols=(OrderByCol(expr=NormalizedExpr.from_column("t.name"), direction="ASC"),),
    )
    sql = render_federation_residual_sql("SELECT * FROM joined", residual, schema=schema)
    assert "COLLATE" in sql.upper()
    assert " COLLATE C " in sql.upper() or sql.upper().endswith(" COLLATE C NULLS LAST")

    int_residual = ResidualSpec(
        select_cols=(SelectCol(expr=NormalizedExpr.from_column("t.id")),),
        order_by_cols=(OrderByCol(expr=NormalizedExpr.from_column("t.id"), direction="ASC"),),
    )
    int_sql = render_federation_residual_sql("SELECT * FROM joined", int_residual, schema=schema)
    assert "COLLATE" not in int_sql.upper()


@pytest.mark.fast
@pytest.mark.parametrize("engine", ["postgresql", "mysql"])
def test_limit_ordering_pins_collation_on_members(engine: str) -> None:
    schema = _text_table_graph()
    dialect = DialectRegistry.get_class(engine).__new__(DialectRegistry.get_class(engine))
    collation_token = '"C"' if engine == "postgresql" else "utf8mb4_bin"

    with_limit = _build_deterministic_select_block(
        [SelectCol(expr=NormalizedExpr.from_column("t.name"))],
        ["t"],
        [],
        [OrderByCol(expr=NormalizedExpr.from_column("t.name"), direction="ASC")],
        None,
        None,
        5,
        "many",
        dialect,
        schema=schema,
    )
    assert "COLLATE" in with_limit.upper()
    assert collation_token in with_limit

    without_limit = _build_deterministic_select_block(
        [SelectCol(expr=NormalizedExpr.from_column("t.name"))],
        ["t"],
        [],
        [OrderByCol(expr=NormalizedExpr.from_column("t.name"), direction="ASC")],
        None,
        None,
        None,
        "many",
        dialect,
        schema=schema,
    )
    assert "COLLATE" not in without_limit.upper()

    int_with_limit = _build_deterministic_select_block(
        [SelectCol(expr=NormalizedExpr.from_column("t.id"))],
        ["t"],
        [],
        [OrderByCol(expr=NormalizedExpr.from_column("t.id"), direction="ASC")],
        None,
        None,
        5,
        "many",
        dialect,
        schema=schema,
    )
    assert "COLLATE" not in int_with_limit.upper()

    intent = RuntimeIntent(
        tables=["t"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.name"))],
        group_by_cols=[],
        order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("t.name"), direction="ASC")],
        where=None,
        limit=3,
    )
    full_sql = build_deterministic_sql(intent, schema=schema, dialect=dialect)
    assert "COLLATE" in full_sql.upper()
    assert collation_token in full_sql
