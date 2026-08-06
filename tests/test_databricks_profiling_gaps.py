"""Tests for Databricks profiling gap fixes (PERCENT sampling, overlap eligibility)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_schema import ColumnMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._dialect_sqlglot_engines import DatabricksDialect
from aetherdialect._schema_catalog import (
    _column_value_overlap_eligible,
    _profile_column_spark,
    _profile_column_sql_connector,
)
from aetherdialect._schema_graph import _fk_containment_validates


def _spark_row(**values: object) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda _self, key: values[key]
    row.get = lambda key, default=None: values.get(key, default)
    return row


def _indexed_spark_row(*values: object) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda _self, key: values[key]
    return row


def test_profile_column_spark_uses_percent_tablesample() -> None:
    spark = MagicMock()
    captured: list[str] = []

    def fake_sql(sql: str) -> MagicMock:
        captured.append(sql)
        result = MagicMock()
        if "MAX(freq) AS mx" in sql:
            result.collect.return_value = [_spark_row(mx=1)]
        elif " AS v " in sql and "GROUP BY" in sql:
            result.collect.return_value = []
        elif "as cnt" in sql.lower() and "count(distinct" in sql.lower():
            result.collect.return_value = [_spark_row(cnt=200_000, dist=50_000, nulls=0)]
        elif "MIN(" in sql:
            result.collect.return_value = [_indexed_spark_row(1, 999)]
        elif "GROUP BY" in sql and "ORDER BY COUNT(*) DESC" in sql:
            result.collect.return_value = []
        elif "MAX(freq)" in sql:
            result.collect.return_value = [_spark_row(mx=1)]
        elif "SELECT DISTINCT" in sql:
            result.collect.return_value = []
        else:
            result.collect.return_value = []
        return result

    spark.sql.side_effect = fake_sql
    col = ColumnMetadata(name="order_id", data_type="bigint", value_type="integer")
    dialect = DatabricksDialect.__new__(DatabricksDialect)
    _profile_column_spark(
        spark,
        "cat",
        "sch",
        col,
        "orders",
        row_count=200_000,
        sample_threshold=100_000,
        sample_size=10_000,
        dialect=dialect,
    )
    assert captured
    assert any("PERCENT" in sql for sql in captured)
    assert not any("ROWS" in sql for sql in captured)


def test_profile_column_sql_connector_uses_percent_tablesample() -> None:
    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    captured: list[str] = []

    def fake_execute(sql: str) -> None:
        captured.append(sql)
        if "COUNT(*)" in sql:
            cursor.description = [("cnt",), ("dist",), ("nulls",)]
        elif "MIN(" in sql:
            cursor.description = [("mn",), ("mx",)]
        elif "MAX(freq)" in sql:
            cursor.description = [("mx",)]
        else:
            cursor.description = [("v",)]

    def fake_fetchall() -> list:
        sql = captured[-1]
        if "COUNT(*)" in sql:
            return [(200_000, 50_000, 0)]
        if "MIN(" in sql:
            return [(1, 999)]
        if "MAX(freq)" in sql:
            return [(1,)]
        return []

    cursor.execute.side_effect = fake_execute
    cursor.fetchall.side_effect = fake_fetchall
    col = ColumnMetadata(name="order_id", data_type="bigint", value_type="integer")
    dialect = DatabricksDialect.__new__(DatabricksDialect)
    _profile_column_sql_connector(
        connection,
        "cat",
        "sch",
        col,
        "orders",
        row_count=200_000,
        sample_threshold=100_000,
        sample_size=10_000,
        dialect=dialect,
    )
    assert captured
    assert any("PERCENT" in sql for sql in captured)
    assert not any("ROWS" in sql for sql in captured)


def test_value_overlap_eligible_for_high_cardinality_primary_key() -> None:
    col = ColumnMetadata(
        name="id",
        data_type="bigint",
        value_type="integer",
        is_primary_key=True,
        distinct_count=PolicyConfig.VALUE_OVERLAP_SAMPLE_LIMIT * 100,
    )
    assert _column_value_overlap_eligible(col) is True


@pytest.mark.parametrize(
    "engine",
    [
        "mysql",
        "redshift",
        "snowflake",
        "sqlserver",
        "bigquery",
        "databricks",
        "duckdb",
        "sqlite",
    ],
)
def test_profiling_stats_sample_suffix_defined_per_engine(engine: str) -> None:
    dialect = DialectRegistry.get_class(engine).__new__(DialectRegistry.get_class(engine))
    suffix = dialect.profiling_stats_sample_suffix(
        use_sample=True,
        row_count=200_000,
        sample_size=10_000,
        random_seed=1,
    )
    assert suffix
    if engine in ("mysql", "redshift", "sqlite"):
        assert suffix.startswith("WHERE ")
    elif engine in ("bigquery", "sqlserver"):
        assert suffix.upper().startswith("ORDER BY")
    elif engine == "duckdb":
        assert "SAMPLE" in suffix.upper()
    else:
        assert "SAMPLE" in suffix.upper() or "TABLESAMPLE" in suffix.upper()
    assert isinstance(dialect.profiling_stats_use_subquery_when_sampling(), bool)


def test_fk_containment_uses_samples_for_high_cardinality_key_columns() -> None:
    child = ColumnMetadata(
        name="customer_id",
        data_type="bigint",
        value_type="integer",
        is_foreign_key=True,
        distinct_count=1_000_000,
        value_overlap_sample=["1", "2", "3", "4", "5"],
    )
    parent = ColumnMetadata(
        name="id",
        data_type="bigint",
        value_type="integer",
        is_primary_key=True,
        distinct_count=1_000_000,
        value_overlap_sample=["1", "2", "3", "4", "5", "6"],
    )
    assert _column_value_overlap_eligible(child) is True
    assert _column_value_overlap_eligible(parent) is True
    assert _fk_containment_validates(child, parent) is True


def test_profile_table_spark_surfaces_count_probe_failure() -> None:
    from aetherdialect._contracts_base import ConfigError
    from aetherdialect._contracts_schema import TableMetadata
    from aetherdialect._schema_catalog import _profile_table_spark

    spark = MagicMock()
    spark.sql.side_effect = RuntimeError("connection lost")
    table = TableMetadata(
        name="orders",
        columns={"id": ColumnMetadata(name="id", data_type="bigint", value_type="integer")},
        primary_key=["id"],
        foreign_keys=[],
    )
    dialect = DatabricksDialect.__new__(DatabricksDialect)
    with pytest.raises(ConfigError, match="schema profiling failed for orders"):
        _profile_table_spark(spark, "cat", "sch", table, dialect=dialect)
