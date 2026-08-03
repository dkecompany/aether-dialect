"""Profiling deep queries must respect sampling and a schema-wide deep- query budget."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect
from aetherdialect._schema_catalog import (
    _build_frequent_values_sql,
    _build_minmax_sql,
    _build_mode_sql,
    _profile_column,
    _resolve_profiling_sample_params,
    profile_schema,
)


def _recording_engine() -> tuple[MagicMock, list[str]]:
    executed: list[str] = []
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    def fake_execute(sql: Any) -> MagicMock:
        executed.append(str(sql))
        result = MagicMock()
        stmt = str(sql)
        if "COUNT(*)" in stmt and "COUNT(DISTINCT" not in stmt and "GROUP BY" not in stmt:
            result.scalar.return_value = 250_000
            result.fetchone.return_value = (250_000,)
        elif "COUNT(*)" in stmt and "COUNT(DISTINCT" in stmt:
            result.fetchone.return_value = (10_000, 5, 0)
        elif "MIN(" in stmt or "MAX(" in stmt:
            result.fetchone.return_value = (1, 9)
        elif "GROUP BY" in stmt and "ORDER BY COUNT" in stmt:
            result.fetchall.return_value = [("alpha",), ("beta",)]
        elif "SELECT MAX(c) FROM" in stmt:
            result.fetchone.return_value = (3,)
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = fake_execute
    return engine, executed


def _deep_profile_sql(statements: list[str]) -> list[str]:
    return [
        stmt
        for stmt in statements
        if ("MIN(" in stmt or "MAX(" in stmt)
        or ("GROUP BY" in stmt and "ORDER BY COUNT" in stmt)
        or "SELECT MAX(c) FROM" in stmt
    ]


@pytest.mark.fast
def test_resolve_profiling_sample_params_view_uses_ordered_limit() -> None:
    dialect = PostgresDialect.__new__(PostgresDialect)
    sample_clause, use_subquery = _resolve_profiling_sample_params(
        dialect,
        use_sample=True,
        row_count=250_000,
        sample_size=PolicyConfig.PROFILING_SAMPLE_SIZE,
        table_kind="view",
    )
    assert sample_clause.startswith("ORDER BY")
    assert f"LIMIT {PolicyConfig.PROFILING_SAMPLE_SIZE}" in sample_clause
    assert use_subquery is True


@pytest.mark.fast
def test_postgres_view_deep_profile_sql_uses_sampled_subquery() -> None:
    dialect = PostgresDialect.__new__(PostgresDialect)
    sample_clause, use_subquery = _resolve_profiling_sample_params(
        dialect,
        use_sample=True,
        row_count=250_000,
        sample_size=PolicyConfig.PROFILING_SAMPLE_SIZE,
        table_kind="view",
    )
    qcol, qtbl = '"status"', '"big_view"'
    minmax_sql = _build_minmax_sql(qcol, qtbl, sample_clause=sample_clause, use_subquery=use_subquery)
    freq_sql = _build_frequent_values_sql(qcol, qtbl, 10, sample_clause=sample_clause, use_subquery=use_subquery)
    mode_sql = _build_mode_sql(qcol, qtbl, sample_clause=sample_clause, use_subquery=use_subquery)
    for sql in (minmax_sql, freq_sql, mode_sql):
        assert f"LIMIT {PolicyConfig.PROFILING_SAMPLE_SIZE}" in sql
        assert "FROM (SELECT" in sql


@pytest.mark.fast
def test_profile_column_deep_queries_include_sample_on_large_table() -> None:
    engine, executed = _recording_engine()
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    col = ColumnMetadata(
        name="status",
        data_type="varchar",
        value_type="string",
        distinct_count=5,
    )
    _profile_column(dialect, engine, col, "events", row_count=250_000)
    deep_sql = _deep_profile_sql(executed)
    assert deep_sql
    assert all("USING SAMPLE" in stmt or f"LIMIT {PolicyConfig.PROFILING_SAMPLE_SIZE}" in stmt for stmt in deep_sql)


@pytest.mark.fast
def test_profile_schema_deep_query_budget_caps_expensive_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PolicyConfig, "PROFILING_SCHEMA_DEEP_QUERY_BUDGET", 2)
    engine, executed = _recording_engine()
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    table = TableMetadata(
        name="wide",
        columns={
            "a": ColumnMetadata(name="a", data_type="varchar", value_type="string", distinct_count=3),
            "b": ColumnMetadata(name="b", data_type="varchar", value_type="string", distinct_count=3),
            "c": ColumnMetadata(name="c", data_type="varchar", value_type="string", distinct_count=3),
        },
        primary_key=[],
        foreign_keys=[],
    )
    schema = SchemaGraph(tables={table.name: table}, join_paths_multi={}, created_at="")

    profile_schema(engine, schema, dialect)

    deep_sql = _deep_profile_sql(executed)
    assert len(deep_sql) == 2
