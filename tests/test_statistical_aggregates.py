"""Tests for stddev, variance, and median aggregate rendering and gates."""

from aetherdialect._contracts_base import DatabaseFeatureCapability, MulGroup, NormalizedExpr
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import SchemaGraph, TableMetadata
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect
from aetherdialect._federation import _federation_ir_capability_reason, stamp_federation_member_graph
from aetherdialect._schema_graph import compute_database_feature_capability
from aetherdialect._sql_gen import _render_group_sql


def _agg_group(func: str, column: str = "t.amount") -> MulGroup:
    return MulGroup(multiply=[NormalizedExpr.from_column(column)], agg_func=func)


class TestStatisticalAggregateRendering:
    def test_stddev_renders_sample_form(self) -> None:
        out = _render_group_sql(_agg_group("stddev"), PostgresDialect.__new__(PostgresDialect))
        assert "STDDEV_SAMP(t.amount)" in out.replace('"', "")

    def test_variance_renders_sample_form(self) -> None:
        out = _render_group_sql(_agg_group("variance"), PostgresDialect.__new__(PostgresDialect))
        assert "VAR_SAMP(t.amount)" in out.replace('"', "")

    def test_postgresql_median_renders_percentile_cont(self) -> None:
        out = _render_group_sql(_agg_group("median"), PostgresDialect.__new__(PostgresDialect))
        assert "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY" in out

    def test_duckdb_median_renders_native_call(self) -> None:
        out = _render_group_sql(_agg_group("median"), DuckDBDialect.__new__(DuckDBDialect))
        assert "median(t.amount)" in out.replace('"', "")


class TestMedianCapability:
    def test_mysql_member_graph_lacks_median_capability(self) -> None:
        graph = SchemaGraph(
            tables={"t": TableMetadata(name="t", columns={}, primary_key=[], foreign_keys=[])},
            join_paths_multi={},
        )
        stamp_federation_member_graph(graph, federation_id="fed", source_id="local", engine="mysql")
        assert compute_database_feature_capability(graph).supports_median is False

    def test_federation_refuses_median_when_member_lacks_support(self) -> None:
        cap = DatabaseFeatureCapability(
            table_count=1,
            fk_edge_count=0,
            has_numeric_measures=True,
            has_date_columns=False,
            has_array_columns=False,
            has_categorical_columns=False,
            max_tables_on_any_join_path=1,
            max_fk_chain_depth=0,
            has_self_referential_fk=False,
            tables_supporting_self_join=frozenset(),
            has_window_capable_table_sets=False,
            aggregatable_columns_by_table={},
            date_columns_by_table={},
            array_columns_by_table={},
            supports_median=False,
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr(add_groups=[_agg_group("median", "t.amount")]))],
            group_by_cols=[NormalizedExpr.from_column("t.id")],
            order_by_cols=[],
            where=None,
        )
        assert _federation_ir_capability_reason(intent, cap) == "median is not supported by all federation members"
