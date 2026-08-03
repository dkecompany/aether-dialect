"""Tests that single-member federation rendering matches the direct engine path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aetherdialect._contracts_base import NormalizedExpr, WhereParam, predicate_group_from_list
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import get_dialect_class
from aetherdialect._federation import parse_federation_manifest, plan_federated_intent
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import build_deterministic_sql

_MEMBER_ENGINES = (
    "duckdb",
    "sqlite",
    "postgresql",
    "mysql",
    "mariadb",
    "sqlserver",
    "snowflake",
    "bigquery",
    "redshift",
    "databricks",
)

_MANIFEST = {
    "federation_id": "fed_render_parity",
    "sources": [
        {"source_id": "a", "engine": "duckdb", "role": "owner"},
        {"source_id": "b", "engine": "duckdb", "role": "owner"},
    ],
    "table_namespace": {"left_t": "a", "right_t": "b"},
    "cross_source_joins": [],
}


def _member_graph(table: str, source_id: str) -> SchemaGraph:
    tables = {
        table: TableMetadata(
            name=table,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
            source_id=source_id,
        )
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        profiling_hash="test-profiled",
    )


@pytest.mark.fast
@pytest.mark.parametrize("engine", _MEMBER_ENGINES)
def test_single_member_federated_sql_matches_direct_render(engine: str) -> None:
    manifest = parse_federation_manifest(_MANIFEST, include_derived_roster=True)
    member_schema = _member_graph("left_t", "a")
    composite = SchemaGraph(
        tables={
            "left_t": member_schema.tables["left_t"],
            "right_t": TableMetadata(
                name="right_t",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
                source_id="b",
            ),
        },
        join_paths_multi={},
    )
    intent = RuntimeIntent(
        tables=["left_t"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("left_t.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=predicate_group_from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("left_t.id"),
                    op=">",
                    value_type="integer",
                    param_key="p1",
                    raw_value=0,
                ),
            ]
        ),
        param_values={"p1": 0},
    )
    plan = plan_federated_intent(intent, composite, manifest, member_graphs={"a": member_schema})
    assert len(plan.steps) == 1
    dialect_cls = get_dialect_class(engine)
    dialect = dialect_cls.__new__(dialect_cls)
    if engine == "databricks":
        dialect.config = SimpleNamespace(CATALOG="test_catalog", SCHEMA="test_schema")
    direct_sql = build_deterministic_sql(intent, schema=member_schema, dialect=dialect)
    federated_sql = build_deterministic_sql(plan.steps[0].sub_intent, schema=member_schema, dialect=dialect)
    assert federated_sql == direct_sql
