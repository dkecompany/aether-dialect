"""Parameterized pruning-predicate injection tests across warehouse dialects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from aetherdialect._contracts_base import (
    NormalizedExpr,
    WhereParam,
    predicate_group_from_list,
)
from aetherdialect._contracts_core import RuntimeIntent
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._dialect_sqlglot_engines import (
    BigQueryDialect,
    DatabricksDialect,
    DuckDBDialect,
    MariaDBDialect,
    MySQLDialect,
    RedshiftDialect,
    SnowflakeDialect,
    SQLServerDialect,
)
from aetherdialect._dialect_sqlglot_helper import (
    append_required_partition_filter_guard,
    table_requires_pruning_filter,
)
from aetherdialect._schema_build import (
    enrich_postgresql_partition_columns,
    merge_ddl_partition_columns_into_schema_graph,
    parse_mysql_partition_columns,
)


def _where_param(col: str, op: str, param_key: str | None = None, raw_value=None) -> WhereParam:
    """Build a WhereParam for partition injection tests."""
    expr = NormalizedExpr.from_column(col)
    return WhereParam(left_expr=expr, op=op, param_key=param_key or "", raw_value=raw_value)


def _schema_with_partition(table: str, partition_cols: list[str]) -> SchemaGraph:
    """Build a SchemaGraph whose sole table exposes ``partition_columns``."""
    cols: dict[str, ColumnMetadata] = {"id": ColumnMetadata(name="id", data_type="integer")}
    for col in partition_cols:
        cols[col] = ColumnMetadata(name=col, data_type="date" if col == "dt" else "string")
    meta = TableMetadata(
        name=table,
        columns=cols,
        foreign_keys=[],
        primary_key="id",
        partition_columns=partition_cols,
    )
    return SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={table: meta})


def _schema_with_redshift_keys(table: str, sortkey: list[str], distkey: str | None = None) -> SchemaGraph:
    """Build a SchemaGraph with Redshift sort/dist metadata."""
    cols: dict[str, ColumnMetadata] = {"id": ColumnMetadata(name="id", data_type="integer")}
    for col in sortkey + ([distkey] if distkey else []):
        if col not in cols:
            cols[col] = ColumnMetadata(name=col, data_type="date" if col == "dt" else "string")
    meta = TableMetadata(
        name=table,
        columns=cols,
        foreign_keys=[],
        primary_key="id",
        sortkey=list(sortkey),
        distkey=distkey,
    )
    return SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={table: meta})


def _schema_with_clustering(table: str, clustering_key: str) -> SchemaGraph:
    """Build a SchemaGraph with Snowflake clustering metadata."""
    meta = TableMetadata(
        name=table,
        columns={
            "id": ColumnMetadata(name="id", data_type="integer"),
            clustering_key: ColumnMetadata(name=clustering_key, data_type="date"),
        },
        foreign_keys=[],
        primary_key="id",
        clustering_key=clustering_key,
    )
    return SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={table: meta})


def _dialect_shell(dialect_cls: type) -> Any:
    """Return an uninitialized dialect instance for pure SQL helper tests."""
    return dialect_cls.__new__(dialect_cls)


def _equality_intent(table: str, col: str, value: str) -> RuntimeIntent:
    """Build a row-level intent filtering ``table.col`` to ``value``."""
    return RuntimeIntent(
        tables=[table],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=predicate_group_from_list([_where_param(f"{table}.{col}", "=", "p1", None)]),
        param_values={"p1": value},
    )


PRUNING_DIALECTS: list[tuple[str, type, Callable[[], Any], str]] = [
    ("databricks", DatabricksDialect, lambda: _dialect_shell(DatabricksDialect), "`events`.`dt` = '2024-01-15'"),
    ("duckdb", DuckDBDialect, lambda: _dialect_shell(DuckDBDialect), '"events"."dt" = \'2024-01-15\''),
    ("postgres", PostgresDialect, lambda: _dialect_shell(PostgresDialect), '"events"."dt" = \'2024-01-15\''),
    ("mysql", MySQLDialect, lambda: _dialect_shell(MySQLDialect), "`events`.`dt` = '2024-01-15'"),
    ("mariadb", MariaDBDialect, lambda: _dialect_shell(MariaDBDialect), "`events`.`dt` = '2024-01-15'"),
    ("sqlserver", SQLServerDialect, lambda: _dialect_shell(SQLServerDialect), "[events].[dt] = '2024-01-15'"),
    ("bigquery", BigQueryDialect, lambda: _dialect_shell(BigQueryDialect), "`events`.`dt` = '2024-01-15'"),
]


@pytest.mark.parametrize(("dialect_id", "dialect_cls", "dialect_factory", "expected_fragment"), PRUNING_DIALECTS)
def test_inject_equality_partition_predicate(
    dialect_id: str,
    dialect_cls: type,
    dialect_factory: Callable[[], Any],
    expected_fragment: str,
) -> None:
    """Each pruning-capable dialect injects an equality predicate on partition columns."""
    _ = dialect_id, dialect_cls
    schema = _schema_with_partition("events", ["dt"])
    intent = _equality_intent("events", "dt", "2024-01-15")
    sql = "SELECT * FROM events"
    result = dialect_factory().inject_pruning_predicates(sql, schema=schema, intent=intent)
    assert expected_fragment in result
    assert "WHERE" in result.upper()


def test_redshift_inject_sortkey_predicate() -> None:
    """Redshift injects predicates for sortkey columns."""
    schema = _schema_with_redshift_keys("sales", ["dt"])
    intent = _equality_intent("sales", "dt", "2024-01-01")
    sql = "SELECT * FROM sales"
    result = _dialect_shell(RedshiftDialect).inject_pruning_predicates(sql, schema=schema, intent=intent)
    assert '"sales"."dt"' in result or "sales.dt" in result.lower()
    assert "2024-01-01" in result


def test_snowflake_inject_cluster_predicate() -> None:
    """Snowflake injects predicates for clustering key columns."""
    schema = _schema_with_clustering("events", "dt")
    intent = _equality_intent("events", "dt", "2024-06-01")
    sql = "SELECT * FROM events"
    result = _dialect_shell(SnowflakeDialect).inject_pruning_predicates(sql, schema=schema, intent=intent)
    assert "2024-06-01" in result
    assert "WHERE" in result.upper()


@pytest.mark.parametrize(("dialect_id", "dialect_cls", "dialect_factory", "expected_fragment"), PRUNING_DIALECTS)
def test_predicate_already_present_unchanged(
    dialect_id: str,
    dialect_cls: type,
    dialect_factory: Callable[[], Any],
    expected_fragment: str,
) -> None:
    """SQL is unchanged when the partition predicate is already present."""
    _ = dialect_id, dialect_cls
    schema = _schema_with_partition("events", ["dt"])
    intent = _equality_intent("events", "dt", "2024-01-15")
    sql = f"SELECT * FROM events WHERE {expected_fragment}"
    result = dialect_factory().inject_pruning_predicates(sql, schema=schema, intent=intent)
    assert result == sql


def test_no_partition_columns_unchanged() -> None:
    """SQL unchanged when table has no partition metadata."""
    meta = TableMetadata(
        name="plain",
        columns={"id": ColumnMetadata(name="id", data_type="integer")},
        foreign_keys=[],
        primary_key="id",
        partition_columns=[],
    )
    schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"plain": meta})
    intent = _equality_intent("plain", "id", "1")
    sql = "SELECT * FROM plain WHERE id = 1"
    result = _dialect_shell(DatabricksDialect).inject_pruning_predicates(sql, schema=schema, intent=intent)
    assert result == sql


class TestParseMysqlPartitionColumns:
    """Tests for ``parse_mysql_partition_columns``."""

    def test_single_column_expression(self) -> None:
        assert parse_mysql_partition_columns("year(`created_at`)") == ["created_at"]

    def test_range_columns_multi(self) -> None:
        assert parse_mysql_partition_columns("RANGE COLUMNS (region, dt)") == ["region", "dt"]

    def test_empty_expression(self) -> None:
        assert parse_mysql_partition_columns("") == []


class TestMergeDdlPartitionColumns:
    """Tests for ``merge_ddl_partition_columns_into_schema_graph``."""

    def test_merges_partition_columns_from_ddl(self) -> None:
        sg = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "events": TableMetadata(
                    name="events",
                    columns={
                        "id": ColumnMetadata(name="id", data_type="integer"),
                        "dt": ColumnMetadata(name="dt", data_type="date"),
                    },
                    foreign_keys=[],
                    primary_key="id",
                ),
            },
        )
        merge_ddl_partition_columns_into_schema_graph(
            sg,
            {"events": {"partition_columns": ["dt"]}},
        )
        assert sg.tables["events"].partition_columns == ["dt"]


class TestEnrichPostgresqlPartitionColumns:
    """Tests for ``enrich_postgresql_partition_columns``."""

    def test_populates_from_catalog_rows(self, monkeypatch) -> None:
        sg = _schema_with_partition("sales", [])
        sg.tables["sales"].partition_columns = []

        class FakeResult:
            def fetchall(self) -> list[tuple[str, str]]:
                return [("sales", "dt")]

        class FakeConn:
            def __enter__(self) -> FakeConn:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def execute(self, *_args: object, **_kwargs: object) -> FakeResult:
                return FakeResult()

        class FakeEngine:
            def connect(self) -> FakeConn:
                return FakeConn()

        enrich_postgresql_partition_columns(FakeEngine(), sg, schema_name="public")
        assert sg.tables["sales"].partition_columns == ["dt"]


class TestRequiredPartitionFilterGuard:
    """Tests for shared mandatory partition-filter guard helper."""

    def test_table_requires_pruning_filter(self) -> None:
        meta = TableMetadata(
            name="t",
            columns={},
            foreign_keys=[],
            primary_key=[],
            require_partition_filter=True,
        )
        assert table_requires_pruning_filter(meta) is True

    def test_appends_default_guard_predicate(self) -> None:
        meta = TableMetadata(
            name="events",
            columns={"dt": ColumnMetadata(name="dt", data_type="date")},
            foreign_keys=[],
            primary_key=[],
            partition_columns=["dt"],
            require_partition_filter=True,
        )
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"events": meta})
        intent = RuntimeIntent(
            tables=["events"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            where=None,
        )
        sql = "SELECT * FROM events"
        out = append_required_partition_filter_guard(
            sql,
            schema=schema,
            intent=intent,
            sqlglot_dialect="bigquery",
            column_selector=lambda m: list(m.partition_columns),
            default_predicate_sql=lambda _t, _c: "`events`.`dt` >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)",
        )
        assert "DATE_SUB" in out
        assert "WHERE" in out.upper()
