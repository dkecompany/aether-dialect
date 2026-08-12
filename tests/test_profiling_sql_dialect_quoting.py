"""Tests for dialect-correct profiling SQL builders."""

from __future__ import annotations

import pytest

from aetherdialect._constants import QUALIFIED_TABLE_REF_ENGINES
from aetherdialect._dialect import DialectRegistry
from aetherdialect._dialect_postgres import PostgresDialect
from aetherdialect._dialect_sqlglot_engines import (
    BigQueryDialect,
    DatabricksDialect,
    DuckDBDialect,
    MySQLDialect,
    RedshiftDialect,
    SnowflakeDialect,
    SQLiteDialect,
    SQLServerDialect,
)
from aetherdialect._schema_profile import (
    _build_profile_stats_sql,
    _build_value_overlap_sample_sql,
)


def _uninit(cls: type) -> object:
    return cls.__new__(cls)


@pytest.mark.parametrize(
    ("dialect_cls", "quote_marker", "cast_fragment"),
    [
        (PostgresDialect, '"', "AS TEXT"),
        (RedshiftDialect, '"', "AS TEXT"),
        (MySQLDialect, "`", "AS CHAR"),
        (SQLServerDialect, "[", "AS NVARCHAR(4000)"),
        (BigQueryDialect, "`", "AS STRING"),
        (SnowflakeDialect, '"', "AS VARCHAR"),
        (DatabricksDialect, "`", "AS STRING"),
    ],
)
def test_semantic_sample_sql_uses_dialect_quoting_and_cast(
    dialect_cls: type,
    quote_marker: str,
    cast_fragment: str,
) -> None:
    dialect = _uninit(dialect_cls)
    qcol = dialect.quote_identifier("order_id")
    qtbl = dialect.quote_identifier("orders")
    assert quote_marker in qcol
    sql = _build_value_overlap_sample_sql(dialect, qcol, qtbl, 100)
    assert qcol in sql
    assert qtbl in sql
    assert cast_fragment in sql
    assert "LIMIT 100" in sql


def test_mysql_profile_stats_uses_backticks() -> None:
    dialect = _uninit(MySQLDialect)
    qcol = dialect.quote_identifier("status")
    qtbl = dialect.quote_identifier("orders")
    sql = _build_profile_stats_sql(qcol, qtbl, use_sample=False, sample_clause="", use_subquery=False)
    assert qcol in sql
    assert qtbl in sql
    assert '"' not in sql


def test_postgres_overlap_sql_uses_double_quotes() -> None:
    dialect = _uninit(PostgresDialect)
    qcol = dialect.quote_identifier("status")
    qtbl = dialect.quote_identifier("orders")
    sql = _build_value_overlap_sample_sql(dialect, qcol, qtbl, 20)
    assert '"status"' in sql
    assert '"orders"' in sql
    assert "ORDER BY" in sql
    assert "LIMIT 20" in sql


@pytest.mark.parametrize("engine", sorted(QUALIFIED_TABLE_REF_ENGINES))
def test_qualified_table_ref_includes_table_name(engine: str) -> None:
    """Engines with catalog qualification emit the bare table name in qualified refs."""
    cls = DialectRegistry.get_class(engine)
    dialect = _uninit(cls)
    ref = dialect.qualified_table_ref("orders")
    assert "orders" in ref.lower()


@pytest.mark.parametrize(
    ("dialect_cls", "quote_marker"),
    [
        (DuckDBDialect, '"'),
        (SQLiteDialect, '"'),
    ],
)
def test_embedded_engine_profile_stats_use_dialect_quotes(
    dialect_cls: type,
    quote_marker: str,
) -> None:
    """DuckDB and SQLite profiling SQL uses dialect-local identifier quoting."""
    dialect = _uninit(dialect_cls)
    qcol = dialect.quote_identifier("status")
    qtbl = dialect.quote_identifier("orders")
    assert quote_marker in qcol
    sql = _build_profile_stats_sql(qcol, qtbl, use_sample=False, sample_clause="", use_subquery=False)
    assert qcol in sql
    assert qtbl in sql


def test_noop_query_log_engines_return_source() -> None:
    """DuckDB and SQLite expose documented no-op query-log sources."""
    for engine in ("duckdb", "sqlite"):
        assert engine in DialectRegistry.get_registered_engines()
        dialect = _uninit(DialectRegistry.get_class(engine))
        src = dialect.query_log_source()
        assert src is not None
        assert src.is_available(None) is False


def test_profile_composite_descriptive_uses_qualified_table_ref() -> None:
    from unittest.mock import MagicMock

    from aetherdialect._contracts_schema import ColumnMetadata, TableMetadata
    from aetherdialect._schema_profile import _profile_composite_descriptive

    captured: list[str] = []
    dialect = MagicMock()
    dialect.qualified_table_ref.return_value = '"ext_schema"."contacts"'
    dialect.quote_identifier.side_effect = lambda name: f'"{name}"'

    class _Conn:
        def execute(self, statement, *_args, **_kwargs):
            captured.append(str(statement))
            return self

        def scalar(self):
            return 2

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class _Engine:
        def connect(self):
            return _Conn()

    table = TableMetadata(
        name="contacts",
        columns={
            "first_name": ColumnMetadata(
                name="first_name",
                data_type="varchar",
                sensitivity="none",
                value_type="string",
            ),
            "last_name": ColumnMetadata(
                name="last_name",
                data_type="varchar",
                sensitivity="none",
                value_type="string",
            ),
        },
        primary_key=[],
        foreign_keys=[],
        row_count=2,
    )
    _profile_composite_descriptive(dialect, _Engine(), table)
    dialect.qualified_table_ref.assert_called_once_with("contacts", kind=table.kind)
    assert captured
    assert '"ext_schema"."contacts"' in captured[0]
