"""Case-only identifier collisions are refused at schema index build time."""

from __future__ import annotations

import pytest

from aetherdialect._contracts_base import EngineContext, SchemaInvariantError
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._schema_graph import apply_deny_objects_filter, validate_scope_against_graph
from aetherdialect._schema_reflect import resolve_graph_table_name


def _col(name: str) -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type="varchar", value_type="string")


def _table(name: str, columns: dict[str, ColumnMetadata]) -> TableMetadata:
    return TableMetadata(name=name, columns=columns, primary_key=[], foreign_keys=[])


@pytest.mark.fast
def test_case_only_duplicate_tables_refused() -> None:
    sg = SchemaGraph(
        tables={
            "Orders": _table("Orders", {"id": _col("id")}),
            "orders": _table("orders", {"id": _col("id")}),
        },
        join_paths_multi={},
        effective_structural_hash="x",
    )
    ctx = EngineContext(deny_objects=frozenset({"orders"}))
    with pytest.raises(SchemaInvariantError, match=r"Orders.*orders.*cannot be distinguished"):
        apply_deny_objects_filter(sg, ctx)


@pytest.mark.fast
def test_case_only_duplicate_columns_refused() -> None:
    sg = SchemaGraph(
        tables={
            "customers": _table(
                "customers",
                {
                    "Email": _col("Email"),
                    "email": _col("email"),
                },
            ),
        },
        join_paths_multi={},
        effective_structural_hash="x",
    )
    ctx = EngineContext(deny_columns=frozenset({"customers.email"}))
    with pytest.raises(SchemaInvariantError, match=r"Email.*email.*cannot be distinguished"):
        validate_scope_against_graph(sg, ctx)


@pytest.mark.fast
def test_resolve_graph_table_name_refuses_case_only_duplicates() -> None:
    graph_tables = {"Orders", "orders"}
    with pytest.raises(SchemaInvariantError, match=r"Orders.*orders.*cannot be distinguished"):
        resolve_graph_table_name("Orders", graph_tables)
