"""Tests for batched result-backend fetch implementations."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from aetherdialect._dialect_sqlglot_engines import (
    BigQueryClientBackend,
    BigQueryStorageBackend,
    DatabricksConnectorBackend,
    DatabricksSparkBackend,
    DatabricksSqlAlchemyBackend,
    DuckDBNativeBackend,
    MySQLConnectorBackend,
    RedshiftConnectorBackend,
    SnowflakeArrowBackend,
    SQLiteNativeBackend,
)
from aetherdialect._dialect_sqlglot_helper import SqlAlchemyResultBackend, SqlServerResultBackend


@pytest.fixture(autouse=True)
def _ensure_google_cloud_bigquery_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep BigQuery backends importable even if earlier tests stubbed ``google.cloud``."""
    google_mod = sys.modules.get("google")
    if google_mod is None:
        google_mod = types.ModuleType("google")
        monkeypatch.setitem(sys.modules, "google", google_mod)
    cloud_mod = sys.modules.get("google.cloud")
    if cloud_mod is None or not hasattr(cloud_mod, "bigquery"):
        cloud_mod = types.ModuleType("google.cloud")
        monkeypatch.setitem(sys.modules, "google.cloud", cloud_mod)
        google_mod.cloud = cloud_mod
    bq_mod = sys.modules.get("google.cloud.bigquery")
    if bq_mod is None:
        bq_mod = types.ModuleType("google.cloud.bigquery")
        monkeypatch.setitem(sys.modules, "google.cloud.bigquery", bq_mod)
    if not hasattr(bq_mod, "ScalarQueryParameter"):
        bq_mod.ScalarQueryParameter = MagicMock(name="ScalarQueryParameter")
    if not hasattr(bq_mod, "QueryJobConfig"):
        bq_mod.QueryJobConfig = MagicMock(name="QueryJobConfig")
    cloud_mod.bigquery = bq_mod


def _drain_batches(backend: Any, batch_rows: int = 2) -> list[tuple[tuple[Any, ...], ...]]:
    return list(
        backend.fetch_rows_batched(
            "SELECT 1",
            None,
            batch_rows=batch_rows,
            max_rows=None,
            max_bytes=None,
        )
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "factory",
    [
        lambda: SqlAlchemyResultBackend(_sqlalchemy_engine_mock([(1,), (2,), (3,)]), dialect_name="duckdb"),
        lambda: SqlServerResultBackend(_sqlalchemy_engine_mock([(4,)]), dialect_name="SQLServerDialect"),
        lambda: DatabricksConnectorBackend(_connector_connection_mock([(5,), (6,)])),
        lambda: DatabricksSparkBackend(_spark_mock([(7,), (8,)])),
        lambda: DatabricksSqlAlchemyBackend(_sqlalchemy_engine_mock([(9,)]), dialect_name="DatabricksDialect"),
        lambda: BigQueryClientBackend(_bigquery_client_mock([(10,), (11,)])),
        lambda: BigQueryStorageBackend(_bigquery_client_mock([(12,)]), MagicMock()),
        lambda: SnowflakeArrowBackend(connection=_snowflake_connection_mock([(13,), (14,)])),
        lambda: MySQLConnectorBackend(_connector_connection_mock([(15,)]), reopen=None),
        lambda: RedshiftConnectorBackend(_connector_connection_mock([(16,)]), reopen=None),
        lambda: DuckDBNativeBackend(_duckdb_connection_mock([(17,), (18,)]), reopen=None),
        lambda: SQLiteNativeBackend(_connector_connection_mock([(19,)]), reopen=None),
    ],
    ids=[
        "sqlalchemy",
        "sqlserver",
        "databricks_connector",
        "databricks_spark",
        "databricks_sqlalchemy",
        "bigquery_client",
        "bigquery_storage",
        "snowflake_arrow",
        "mysql_connector",
        "redshift_connector",
        "duckdb_native",
        "sqlite_native",
    ],
)
def test_every_backend_implements_batched_fetch(factory: Any) -> None:
    backend = factory()
    batches = _drain_batches(backend, batch_rows=2)
    assert batches
    assert all(isinstance(batch, tuple) for batch in batches)
    assert sum(len(batch) for batch in batches) >= 1


def _sqlalchemy_engine_mock(rows: list[tuple[Any, ...]]) -> MagicMock:
    engine = MagicMock()
    conn = MagicMock()
    result = MagicMock()
    result.fetchmany.side_effect = [rows[:2], rows[2:], []]
    conn.execute.return_value = result
    engine.connect.return_value.__enter__.return_value = conn
    engine.begin.return_value.__enter__.return_value = conn
    return engine


def _connector_connection_mock(rows: list[tuple[Any, ...]]) -> MagicMock:
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchmany.side_effect = [rows[:2], rows[2:], []]
    conn.cursor.return_value = cursor
    return conn


def _spark_mock(rows: list[tuple[Any, ...]]) -> MagicMock:
    spark = MagicMock()
    df = MagicMock()
    df.collect.return_value = rows
    spark.sql.return_value = df
    return spark


def _bigquery_client_mock(rows: list[tuple[Any, ...]]) -> MagicMock:
    client = MagicMock()
    job = MagicMock()

    class _Row:
        def __init__(self, values: tuple[Any, ...]) -> None:
            self._values = values

        def values(self) -> tuple[Any, ...]:
            return self._values

    job.result.return_value = iter([_Row(row) for row in rows])
    client.query.return_value = job
    return client


def _snowflake_connection_mock(rows: list[tuple[Any, ...]]) -> MagicMock:
    conn = MagicMock()
    cursor = MagicMock(spec=["execute", "fetchmany", "close"])
    cursor.fetchmany.side_effect = [rows[:2], rows[2:], []]
    conn.cursor.return_value = cursor
    return conn


def _duckdb_connection_mock(rows: list[tuple[Any, ...]]) -> MagicMock:
    conn = MagicMock()
    result = MagicMock()
    result.fetchmany.side_effect = [rows[:2], rows[2:], []]
    conn.execute.return_value = result
    return conn
