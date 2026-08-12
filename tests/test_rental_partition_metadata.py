"""Unit tests for synthetic rental partition metadata and DuckDB pruning injection."""

from __future__ import annotations

from live_tests.mydb_profile import apply_synthetic_rental_partition_metadata

from aetherdialect._contracts_base import (
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect


def _where_param(col: str, op: str, param_key: str, value) -> WhereParam:
    return WhereParam(
        left_expr=NormalizedExpr.from_column(col),
        op=op,
        param_key=param_key,
        raw_value=value,
    )


def _rental_shop_graph(*, partition_columns: list[str] | None = None) -> SchemaGraph:
    rental = TableMetadata(
        name="rental",
        columns={
            "rental_id": ColumnMetadata(name="rental_id", data_type="integer"),
            "rental_date": ColumnMetadata(name="rental_date", data_type="date"),
        },
        foreign_keys=[],
        primary_key="rental_id",
        partition_columns=list(partition_columns or []),
    )
    return SchemaGraph(
        join_paths_multi={},
        effective_structural_hash="",
        tables={"rental": rental},
    )


def _duckdb_shell() -> DuckDBDialect:
    return DuckDBDialect.__new__(DuckDBDialect)


class TestApplySyntheticRentalPartitionMetadata:
    def test_tags_rental_date_when_unset(self) -> None:
        sg = _rental_shop_graph()
        apply_synthetic_rental_partition_metadata(sg)
        assert sg.tables["rental"].partition_columns == ["rental_date"]

    def test_no_op_when_rental_missing(self) -> None:
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={})
        apply_synthetic_rental_partition_metadata(sg)
        assert sg.tables == {}

    def test_no_op_when_partition_columns_already_set(self) -> None:
        sg = _rental_shop_graph(partition_columns=["other_col"])
        apply_synthetic_rental_partition_metadata(sg)
        assert sg.tables["rental"].partition_columns == ["other_col"]


class TestDuckDBInjectPruningPredicates:
    def test_injects_equality_predicate_on_rental_date(self) -> None:
        sg = _rental_shop_graph(partition_columns=["rental_date"])
        intent = RuntimeIntent(
            tables=["rental"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=PredicateGroup.from_list([_where_param("rental.rental_date", "=", "p1", None)]),
            param_values={"p1": "2023-07-15"},
        )
        sql = "SELECT * FROM rental"
        result = _duckdb_shell().inject_pruning_predicates(sql, schema=sg, intent=intent)
        assert "WHERE" in result.upper()
        assert ":p1" in result
        assert "2023-07-15" not in result
        assert '"rental"."rental_date"' in result or "rental.rental_date" in result.lower()

    def test_unchanged_without_partition_columns(self) -> None:
        sg = _rental_shop_graph(partition_columns=[])
        intent = RuntimeIntent(
            tables=["rental"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=PredicateGroup.from_list([_where_param("rental.rental_date", "=", "p1", None)]),
            param_values={"p1": "2023-07-15"},
        )
        sql = "SELECT * FROM rental WHERE rental_date = '2023-07-15'"
        result = _duckdb_shell().inject_pruning_predicates(sql, schema=sg, intent=intent)
        assert result == sql
