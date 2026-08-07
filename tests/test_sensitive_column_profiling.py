"""Sensitive hidden/restricted columns must never be read from the database during profiling."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import SensitivityClassification
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect_sqlglot_engines import DuckDBDialect
from aetherdialect._schema_catalog import _profile_column, _profile_table, profile_schema
from aetherdialect._schema_graph import _profile_table_clone


def _recording_engine() -> tuple[MagicMock, list[str]]:
    executed: list[str] = []
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    def fake_execute(sql: Any) -> MagicMock:
        executed.append(str(sql))
        result = MagicMock()
        if "COUNT(*)" in str(sql) and "COUNT(DISTINCT" not in str(sql):
            result.scalar.return_value = 10
            result.fetchone.return_value = (10,)
        elif "COUNT(DISTINCT" in str(sql):
            result.fetchone.return_value = (10, 5, 0)
        elif "MAX(freq)" in str(sql):
            result.fetchone.return_value = (2,)
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = fake_execute
    return engine, executed


def _users_table(*, email_sensitivity: str) -> TableMetadata:
    email = ColumnMetadata(name="email", data_type="varchar", value_type="string")
    SensitivityClassification.apply_to(email, email_sensitivity)
    return TableMetadata(
        name="users",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="identifier",
                is_primary_key=True,
            ),
            "email": email,
        },
        primary_key=["id"],
        foreign_keys=[],
    )


def _sql_references_column(statements: list[str], column: str, dialect: DuckDBDialect) -> list[str]:
    quoted = dialect.quote_identifier(column).lower()
    bare = column.lower()
    hits: list[str] = []
    for stmt in statements:
        low = stmt.lower()
        if quoted in low or f" {bare} " in f" {low} " or f".{bare}" in low:
            hits.append(stmt)
    return hits


@pytest.mark.fast
@pytest.mark.parametrize("sensitivity", ["hidden", "restricted"])
def test_profile_column_never_queries_sensitive_column(sensitivity: str) -> None:
    engine, executed = _recording_engine()
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    table = _users_table(email_sensitivity=sensitivity)
    col = table.columns["email"]

    _profile_column(dialect, engine, col, table.name, row_count=10)

    assert col.sensitivity != SensitivityClassification.NONE
    assert _sql_references_column(executed, "email", dialect) == []


@pytest.mark.fast
@pytest.mark.parametrize("sensitivity", ["hidden", "restricted"])
def test_profile_table_skips_sensitive_columns(sensitivity: str) -> None:
    engine, executed = _recording_engine()
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    table = _users_table(email_sensitivity=sensitivity)

    _profile_table(dialect, engine, table)

    assert _sql_references_column(executed, "email", dialect) == []
    assert _sql_references_column(executed, "id", dialect) != []


@pytest.mark.fast
def test_profile_schema_skips_sensitive_columns() -> None:
    engine, executed = _recording_engine()
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    table = _users_table(email_sensitivity="hidden")
    schema = SchemaGraph(tables={table.name: table}, join_paths_multi={}, created_at="")

    profile_schema(engine, schema, dialect)

    assert _sql_references_column(executed, "email", dialect) == []
    assert _sql_references_column(executed, "id", dialect) != []


@pytest.mark.fast
def test_profile_table_clone_skips_sensitive_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    table = _users_table(email_sensitivity="hidden")
    engine, executed = _recording_engine()
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    dialect.engine = engine
    monkeypatch.setattr("aetherdialect._schema_graph.apply_column_roles_llm", lambda *_a, **_k: None)
    monkeypatch.setattr("aetherdialect._schema_graph.apply_boolean_coercion_pass", lambda *_a, **_k: None)
    monkeypatch.setattr("aetherdialect._schema_graph.assign_column_ops", lambda *_a, **_k: None)

    with patch.object(DuckDBDialect, "profile_schema_dispatch", wraps=dialect.profile_schema_dispatch):
        clone = _profile_table_clone(dialect, table, notes_content=None)

    assert clone is not None
    assert _sql_references_column(executed, "email", dialect) == []


@pytest.mark.fast
def test_composite_descriptive_profiling_skips_sensitive_name_columns() -> None:
    engine, executed = _recording_engine()
    dialect = DuckDBDialect.__new__(DuckDBDialect)
    first_name = ColumnMetadata(name="first_name", data_type="varchar", value_type="string")
    SensitivityClassification.apply_to(first_name, SensitivityClassification.HIDDEN)
    last_name = ColumnMetadata(name="last_name", data_type="varchar", value_type="string")
    table = TableMetadata(
        name="people",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="identifier",
                is_primary_key=True,
            ),
            "first_name": first_name,
            "last_name": last_name,
        },
        primary_key=["id"],
        foreign_keys=[],
        row_count=10,
    )

    _profile_table(dialect, engine, table)

    assert _sql_references_column(executed, "first_name", dialect) == []
