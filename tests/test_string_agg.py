"""Tests for ordered string aggregation rendering and capability gates."""

from aetherdialect._contracts_base import MulGroup, NormalizedExpr, OrderByCol
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._dialect_sqlglot_engines import DatabricksDialect, MySQLDialect, SnowflakeDialect, SQLiteDialect
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import (
    parse_federation_manifest,
    stamp_federation_member_graph,
)
from aetherdialect._federation_plan import plan_federated_intent
from aetherdialect._schema_graph import compute_database_feature_capability, recompute_join_paths_multi
from aetherdialect._sql_gen import _render_group_sql
from aetherdialect._validation_shape import validate_select_cols_schema

_CROSS_SOURCE_MANIFEST = {
    "federation_id": "fed_cross",
    "sources": [
        {"source_id": "a", "engine": "postgresql", "role": "owner"},
        {"source_id": "b", "engine": "postgresql", "role": "owner"},
    ],
    "table_namespace": {"ta": "a", "tb": "b"},
    "cross_source_joins": [{"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"}],
}


def _eligibility_graph(table: str, source_id: str) -> SchemaGraph:
    table_meta = TableMetadata(
        name=table,
        columns={
            "id": ColumnMetadata(name="id", data_type="integer"),
            "name": ColumnMetadata(name="name", data_type="text"),
        },
        primary_key=["id"],
        foreign_keys=[],
        source_id=source_id,
    )
    tables = {table: table_meta}
    return SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))


def _string_agg_group(*, with_order: bool = False, column: str = "customers.name") -> MulGroup:
    order = [OrderByCol(expr=NormalizedExpr.from_column("orders.id"), direction="asc")] if with_order else []
    return MulGroup(
        multiply=[NormalizedExpr.from_column(column)],
        agg_func="string_agg",
        agg_sep_param_key="sep",
        agg_order_by=order,
    )


class TestStringAggRendering:
    def test_postgresql_renders_string_agg_with_order(self) -> None:
        out = _render_group_sql(_string_agg_group(with_order=True), PostgresDialect.__new__(PostgresDialect))
        assert "STRING_AGG(customers.name, :sep ORDER BY orders.id ASC)" in out.replace('"', "")

    def test_mysql_renders_group_concat_with_separator(self) -> None:
        out = _render_group_sql(_string_agg_group(with_order=True), MySQLDialect.__new__(MySQLDialect))
        assert "GROUP_CONCAT(customers.name ORDER BY orders.id ASC SEPARATOR :sep)" in out.replace("`", "")

    def test_snowflake_renders_listagg_within_group(self) -> None:
        out = _render_group_sql(_string_agg_group(with_order=True), SnowflakeDialect.__new__(SnowflakeDialect))
        assert "LISTAGG(" in out and "WITHIN GROUP" in out
        assert '"customers"."name"' in out or "customers.name" in out.replace('"', "")
        assert '"orders"."id"' in out or "orders.id" in out.replace('"', "")

    def test_databricks_renders_unordered_collect_list(self) -> None:
        out = _render_group_sql(_string_agg_group(with_order=True), DatabricksDialect.__new__(DatabricksDialect))
        compact = out.replace("`", "").replace(" ", "")
        assert "array_join(collect_list(customers.name),:sep)" in compact

    def test_sqlite_renders_bare_group_concat(self) -> None:
        out = _render_group_sql(_string_agg_group(with_order=True), SQLiteDialect.__new__(SQLiteDialect))
        assert out.replace('"', "") == "GROUP_CONCAT(customers.name)"

    def test_sqlite_separator_emitted(self) -> None:
        out = _render_group_sql(_string_agg_group(with_order=False), SQLiteDialect.__new__(SQLiteDialect))
        assert out.replace('"', "") == "GROUP_CONCAT(customers.name, :sep)"


class TestOrderedStringAggCapability:
    def test_sqlite_member_graph_lacks_ordered_string_agg_capability(self) -> None:
        graph = SchemaGraph(
            tables={"t": TableMetadata(name="t", columns={}, primary_key=[], foreign_keys=[])},
            join_paths_multi={},
        )
        stamp_federation_member_graph(graph, federation_id="fed", source_id="local", engine="sqlite")
        cap = compute_database_feature_capability(graph)
        assert cap.supports_ordered_string_agg is False

    def test_ordered_string_agg_on_sqlite_graph_is_rejected(self, typed_schema) -> None:
        graph = SchemaGraph(
            tables=typed_schema.tables,
            join_paths_multi=typed_schema.join_paths_multi,
            federation_membership={"federation_id": "fed", "source_id": "local", "engine": "sqlite"},
        )
        sc = SelectCol(expr=NormalizedExpr(add_groups=[_string_agg_group(with_order=True)]))
        issues = validate_select_cols_schema([sc], graph, set(graph.tables), context="main")
        assert any("ordered string_agg" in i.message for i in issues)

    def test_cross_source_string_agg_is_ineligible(self) -> None:
        manifest = parse_federation_manifest(_CROSS_SOURCE_MANIFEST, include_derived_roster=True)
        composite = compose_composite_graph(
            {"a": _eligibility_graph("ta", "a"), "b": _eligibility_graph("tb", "b")},
            manifest,
        )
        intent = RuntimeIntent(
            tables=["ta", "tb"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[_string_agg_group(column="ta.name")]))],
            group_by_cols=[NormalizedExpr.from_column("ta.id")],
            order_by_cols=[],
            where=None,
            having=None,
            param_values={"sep": ","},
        )
        plan = plan_federated_intent(intent, composite, manifest)
        assert plan.ineligible_reason == "cross-source aggregate not supported: string_agg(ta.name)"
