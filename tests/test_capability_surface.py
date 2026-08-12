"""Cross-surface coverage for IR capabilities."""

from typing import cast

from sqlglot import exp, parse_one

from aetherdialect._constants_runtime import COMPOSE_SUPPORTED_CAPABILITIES, INTENT_SCHEMA
from aetherdialect._contracts_base import MulGroup, NormalizedExpr, OrderByCol
from aetherdialect._contracts_core import SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata, WindowRegistryStep, WindowSpec
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_manifest import (
    parse_federation_manifest,
    stamp_federation_member_graph,
)
from aetherdialect._intent_expr import _order_by_col_from_obc
from aetherdialect._schema_graph import compute_database_feature_capability, recompute_join_paths_multi
from aetherdialect._sql_to_intent_sqlglot import _aggregate_to_expr, _sort_clause, _window_def_to_spec
from aetherdialect._validation_shape import (
    validate_order_by_cols_schema,
    validate_select_cols_schema,
    validate_window_spec_schema,
)


def _mysql_graph() -> SchemaGraph:
    graph = SchemaGraph(
        tables={
            "t": TableMetadata(
                name="t",
                columns={
                    "n": ColumnMetadata(name="n", data_type="integer"),
                    "name": ColumnMetadata(name="name", data_type="text"),
                },
                primary_key=["n"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
    )
    stamp_federation_member_graph(graph, federation_id="fed", source_id="local", engine="mysql")
    return graph


class TestOrderByNullPlacementSurface:
    def test_compose_json_preserves_nulls_on_order_by_col(self) -> None:
        obc = _order_by_col_from_obc({"expr": "t.n", "direction": "desc", "nulls": "last"})
        assert obc.nulls == "last"
        assert obc.direction == "DESC"

    def test_main_query_nulls_last_round_trips_through_sql_import(self) -> None:
        order_node = parse_one("SELECT 1 FROM t ORDER BY t.n ASC NULLS LAST").args["order"]
        cols = _sort_clause(order_node, "postgres", {"t": "t"}, "t", {}, lambda: "p1", [], None)
        assert cols is not None
        assert cols[0].nulls == "last"

    def test_invalid_nulls_value_is_rejected(self, simple_schema) -> None:
        cols = [OrderByCol(expr=NormalizedExpr.from_column("customers.id"), direction="ASC", nulls=cast(str, "middle"))]
        issues = validate_order_by_cols_schema(cols, simple_schema, {"customers"}, context="main")
        assert any("nulls must be" in i.message for i in issues)


class TestExtendedWindowImportSurface:
    def test_percent_rank_round_trips_through_sql_import(self) -> None:
        window = parse_one("SELECT PERCENT_RANK() OVER (ORDER BY t.x) FROM t").find(exp.Window)
        ws = _window_def_to_spec(window, window.this, "postgres", {"t": "t"}, "t", {}, lambda: "p1", [], None)
        assert ws is not None
        assert ws.function == "percent_rank"
        assert ws.argument is None

    def test_cume_dist_round_trips_through_sql_import(self) -> None:
        window = parse_one("SELECT CUME_DIST() OVER (ORDER BY t.x) FROM t").find(exp.Window)
        ws = _window_def_to_spec(window, window.this, "postgres", {"t": "t"}, "t", {}, lambda: "p1", [], None)
        assert ws is not None
        assert ws.function == "cume_dist"

    def test_nth_value_round_trips_through_sql_import(self) -> None:
        window = parse_one("SELECT NTH_VALUE(t.y, 2) OVER (ORDER BY t.x) FROM t").find(exp.Window)
        ws = _window_def_to_spec(window, window.this, "postgres", {"t": "t"}, "t", {}, lambda: "p1", [], None)
        assert ws is not None
        assert ws.function == "nth_value"
        assert ws.numeric_argument == 2

    def test_cume_dist_rejects_column_argument(self, typed_schema) -> None:
        ws = WindowSpec(
            function="cume_dist",
            argument=NormalizedExpr.from_column("orders.amount"),
            order_by=[OrderByCol(expr=NormalizedExpr.from_column("orders.id"), direction="ASC")],
        )
        wr = [WindowRegistryStep(registry_id="w01", window_spec=ws)]
        sc = SelectCol(expr=NormalizedExpr(column_ref="w01"))
        issues = validate_window_spec_schema([sc], typed_schema, {}, "main", window_registry=wr)
        assert any("must not carry an argument" in i.message for i in issues)


class TestAggregateImportSurface:
    def test_string_agg_round_trips_through_sql_import(self) -> None:
        node = parse_one("SELECT STRING_AGG(t.name, ',' ORDER BY t.id) FROM t").find(exp.GroupConcat)
        expr = _aggregate_to_expr(node, "postgres", {"t": "t"}, "t", {}, lambda: "s1", None)
        assert expr is not None
        g = expr.add_groups[0]
        assert g.agg_func == "string_agg"
        assert g.agg_sep_param_key == "s1"
        assert len(g.agg_order_by) == 1

    def test_stddev_round_trips_through_sql_import(self) -> None:
        node = parse_one("SELECT STDDEV_SAMP(t.x) FROM t").find(exp.StddevSamp)
        expr = _aggregate_to_expr(node, "postgres", {"t": "t"}, "t", {}, lambda: "s1", None)
        assert expr is not None
        assert expr.add_groups[0].agg_func == "stddev"

    def test_median_round_trips_through_sql_import(self) -> None:
        node = parse_one("SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.x) FROM t").find(exp.WithinGroup)
        expr = _aggregate_to_expr(node, "postgres", {"t": "t"}, "t", {}, lambda: "s1", None)
        assert expr is not None
        assert expr.add_groups[0].agg_func == "median"


class TestMedianCapabilitySurface:
    def test_median_on_mysql_graph_is_rejected_by_schema(self) -> None:
        graph = _mysql_graph()
        sc = SelectCol(
            expr=NormalizedExpr(add_groups=[MulGroup(multiply=[NormalizedExpr.from_column("t.n")], agg_func="median")])
        )
        issues = validate_select_cols_schema([sc], graph, set(graph.tables), context="main")
        assert any("median is not supported" in i.message for i in issues)

    def test_mysql_member_graph_lacks_median_capability(self) -> None:
        cap = compute_database_feature_capability(_mysql_graph())
        assert cap.supports_median is False


class TestComposeSchemaSurface:
    def test_intent_schema_declares_order_by_nulls(self) -> None:
        props = INTENT_SCHEMA["properties"]["order_by_cols"]["items"]["oneOf"][1]["properties"]
        assert props["nulls"]["enum"] == ["first", "last"]

    def test_compose_capabilities_mention_extended_features(self) -> None:
        joined = " ".join(COMPOSE_SUPPORTED_CAPABILITIES).lower()
        for token in ("string_agg", "stddev", "median", "ntile", "percent_rank", "nulls"):
            assert token in joined


_CROSS_SOURCE_MANIFEST = {
    "federation_id": "fed_cross",
    "sources": [
        {"source_id": "a", "engine": "postgresql", "role": "owner"},
        {"source_id": "b", "engine": "databricks", "role": "owner"},
    ],
    "table_namespace": {"ta": "a", "tb": "b"},
    "cross_source_joins": [{"left": "ta.id", "right": "tb.id", "kind": "inner", "logical_key": "id"}],
}


def _eligibility_graph(table: str, source_id: str, engine: str) -> SchemaGraph:
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
    graph = SchemaGraph(tables=tables, join_paths_multi=recompute_join_paths_multi(tables))
    stamp_federation_member_graph(graph, federation_id="fed_cross", source_id=source_id, engine=engine)
    return graph


class TestOrderedStringAggFederationSurface:
    def test_federated_composite_lacks_ordered_string_agg_when_databricks_member_present(self) -> None:
        manifest = parse_federation_manifest(_CROSS_SOURCE_MANIFEST, include_derived_roster=True)
        composite = compose_composite_graph(
            {"a": _eligibility_graph("ta", "a", "postgresql"), "b": _eligibility_graph("tb", "b", "databricks")},
            manifest,
        )
        cap = composite.database_feature_capability
        assert cap.supports_ordered_string_agg is False
